#!/usr/bin/env python3
"""
HF combined worker:
- loads a HuggingFace model/tokenizer once
- for each queue config, runs all math + all EQ prompts in one mixed batched generation pass
- writes combined, math-only, and eq-only result pickles

This mirrors the queue/results protocol of scripts/run_exllama_math_eq_combined_worker.py,
but uses the HF worker stack (layer duplication wrappers, model loading) used by
src.workers.math_worker / src.workers.eq_worker (as orchestrated by scripts/beam_search.py).
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from pathlib import Path
from typing import Any

# Add repo root to path for imports
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PADDING_MODE_MASKED = "masked"
PADDING_MODE_INPROMPT_SPACE = "inprompt_space"


def _save_pickle_result(path: Path, config_key: tuple[int, ...], value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("wb") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                import pickle

                pickle.dump({}, f)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    with path.open("r+b") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            import pickle

            f.seek(0)
            data = pickle.load(f)
            data[config_key] = value
            f.seek(0)
            f.truncate()
            pickle.dump(data, f)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _resolve_prompt_pad_id(tokenizer, prompt_pad_id: int | None) -> int:
    if prompt_pad_id is not None:
        return int(prompt_pad_id)
    try:
        space_ids = tokenizer(" ", add_special_tokens=False)["input_ids"]
        if space_ids:
            return int(space_ids[-1])
    except Exception:
        pass
    return int(tokenizer.pad_token_id or tokenizer.eos_token_id)


def run_combined_single_pass_hf(
    *,
    model: Any,
    mixed_items: list[dict[str, Any]],
    tokenizer: Any,
    batch_size: int,
    max_new_tokens: int,
    padding_mode: str,
    prompt_pad_id: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    from src.workers.eq_worker import calculate_eq_score, extract_emotion_scores
    from src.workers.math_worker import calculate_score, extract_integers
    from src.workers.model_utils import strip_thinking

    default_pad_id = int(tokenizer.pad_token_id or tokenizer.eos_token_id)
    pad_id = int(prompt_pad_id) if padding_mode == PADDING_MODE_INPROMPT_SPACE else default_pad_id

    math_scores: list[float] = []
    math_responses: list[dict[str, Any]] = []
    eq_scores: list[float] = []
    eq_responses: list[dict[str, Any]] = []

    for i in range(0, len(mixed_items), batch_size):
        batch = mixed_items[i : i + batch_size]
        if not batch:
            continue

        max_len = max(int(row["input_ids"].shape[1]) for row in batch)
        batch_input_ids = []
        batch_attention_masks = []
        batch_meta: list[dict[str, Any]] = []

        for row in batch:
            input_ids = row["input_ids"]
            attention_mask = row["attention_mask"]

            pad_length = max_len - int(input_ids.shape[1])
            if pad_length > 0:
                input_ids = torch.cat(
                    [
                        torch.full(
                            (1, pad_length),
                            pad_id,
                            device=input_ids.device,
                            dtype=input_ids.dtype,
                        ),
                        input_ids,
                    ],
                    dim=1,
                )
                if padding_mode == PADDING_MODE_INPROMPT_SPACE:
                    left_mask = torch.ones(
                        (1, pad_length),
                        device=attention_mask.device,
                        dtype=attention_mask.dtype,
                    )
                else:
                    left_mask = torch.zeros(
                        (1, pad_length),
                        device=attention_mask.device,
                        dtype=attention_mask.dtype,
                    )
                attention_mask = torch.cat([left_mask, attention_mask], dim=1)

            batch_input_ids.append(input_ids)
            batch_attention_masks.append(attention_mask)
            batch_meta.append(row)

        batch_input_ids = torch.cat(batch_input_ids, dim=0)
        batch_attention_masks = torch.cat(batch_attention_masks, dim=0)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_masks,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )

        for output, meta in zip(outputs, batch_meta):
            generated_ids = output[max_len:]
            raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            stripped_text = strip_thinking(raw_text)

            if meta["task"] == "math":
                answer = meta["answer"]
                qid = meta["qid"]

                stripped_integers = extract_integers(stripped_text)
                integers = stripped_integers
                has_valid_final_answer = len(stripped_integers) > 0
                fallback_used = False
                if len(integers) == 0:
                    integers = extract_integers(raw_text)
                    fallback_used = len(integers) > 0

                if len(integers) == 0:
                    question_score = 0.0
                else:
                    question_score = max(float(calculate_score(answer, v)) for v in integers)

                math_scores.append(question_score)
                math_responses.append(
                    {
                        "qid": qid,
                        "raw_output": raw_text,
                        "stripped_output": stripped_text,
                        "extracted": integers,
                        "has_valid_final_answer": has_valid_final_answer,
                        "fallback_used": fallback_used,
                        "reference": answer,
                        "score": question_score,
                    }
                )
            else:
                reference = meta["reference"]
                qid = meta["qid"]
                predicted, confidence = extract_emotion_scores(stripped_text)
                question_score = float(calculate_eq_score(predicted, reference, confidence))
                eq_scores.append(question_score)
                eq_responses.append(
                    {
                        "qid": qid,
                        "raw_output": raw_text,
                        "extracted": predicted,
                        "confidence": confidence,
                        "reference": reference,
                        "score": question_score,
                    }
                )

    math_valid_count = sum(1 for r in math_responses if r.get("has_valid_final_answer"))
    math_fallback_count = sum(1 for r in math_responses if r.get("fallback_used"))

    math_result = {
        "score": (sum(math_scores) / len(math_scores)) if math_scores else 0.0,
        "valid_final_answer_count": int(math_valid_count),
        "valid_final_answer_rate": (math_valid_count / len(math_responses)) if math_responses else 0.0,
        "fallback_used_count": int(math_fallback_count),
        "fallback_used_rate": (math_fallback_count / len(math_responses)) if math_responses else 0.0,
        "responses": math_responses,
    }
    eq_result = {
        "score": (sum(eq_scores) / len(eq_scores)) if eq_scores else 0.0,
        "responses": eq_responses,
    }
    return math_result, eq_result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HF combined math+EQ worker using one mixed generation pass per config."
    )
    parser.add_argument(
        "--queue-file",
        required=True,
        help="Path to shared queue JSON (canonical layers entries or legacy key entries).",
    )
    parser.add_argument("--combined-results-file", required=True, help="Path to combined results pickle.")
    parser.add_argument("--math-results-file", required=True, help="Path to math-only results pickle.")
    parser.add_argument("--eq-results-file", required=True, help="Path to eq-only results pickle.")

    parser.add_argument("--model-path", required=True, help="HF model path or directory.")
    parser.add_argument("--math-dataset-path", default="datasets/math_16.json")
    parser.add_argument("--eq-dataset-path", default="datasets/eq_16.json")

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--math-max-new", type=int, default=64)
    parser.add_argument("--eq-max-new", type=int, default=128)

    parser.add_argument(
        "--padding-mode",
        type=str,
        choices=[PADDING_MODE_MASKED, PADDING_MODE_INPROMPT_SPACE],
        default=PADDING_MODE_MASKED,
    )
    parser.add_argument("--prompt-pad-id", type=int, default=None)
    parser.add_argument("--math-use-no-think-prefix", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eq-use-no-think-prefix", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--adaptive-batch-retry", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-batch-size", type=int, default=1)
    parser.add_argument("--max-retries-per-phase", type=int, default=8)

    parser.add_argument("--attention-impl", type=str, default="eager", choices=["eager", "flash_attention_2", "sdpa"])
    parser.add_argument("--device-map", type=str, default="cuda:0")
    parser.add_argument("--max-memory-json", type=str, default=None)
    parser.add_argument("--cpu-offload", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--offload-folder", type=str, default=None)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--worker-id", type=str, default=None)

    args = parser.parse_args()

    # Defer heavy deps until after argparse (so `--help` works without torch installed).
    import torch
    from src.core.layer_config import layer_spec_string, parse_queue_entry_layers
    from src.core.layer_duplicator import build_model_with_layers
    from src.core.layer_duplicator_moe import build_model_with_layers_moe
    from src.workers.batch_control import adaptive_batch_execute
    from src.workers.eq_worker import pretokenize_eq_dataset
    from src.workers.math_worker import pretokenize_dataset
    from src.workers.model_utils import (
        is_moe_model,
        load_model_and_tokenizer,
        parse_device_map_arg,
        parse_max_memory_json,
    )
    from src.workers.shared_queue import SharedWorkQueue

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.min_batch_size < 1:
        raise ValueError("--min-batch-size must be >= 1")
    if args.max_retries_per_phase < 0:
        raise ValueError("--max-retries-per-phase must be >= 0")
    if args.math_max_new < 1 or args.eq_max_new < 1:
        raise ValueError("--math-max-new/--eq-max-new must be >= 1")

    try:
        resolved_device_map = parse_device_map_arg(args.device_map)
    except Exception as exc:
        raise ValueError(f"Invalid --device-map value: {exc}") from exc
    try:
        resolved_max_memory = parse_max_memory_json(args.max_memory_json)
    except Exception as exc:
        raise ValueError(f"Invalid --max-memory-json value: {exc}") from exc

    if args.worker_id is None:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
        args.worker_id = f"HF-COMB-{cuda_visible}"

    max_new_tokens = int(max(args.math_max_new, args.eq_max_new))

    print("=" * 80)
    print(f"HF Combined Worker [{args.worker_id}]")
    print("=" * 80)
    print(f"Queue file:            {args.queue_file}")
    print(f"Combined results file: {args.combined_results_file}")
    print(f"Math results file:     {args.math_results_file}")
    print(f"EQ results file:       {args.eq_results_file}")
    print(f"Model:                 {args.model_path}")
    print(f"Math dataset:          {args.math_dataset_path}")
    print(f"EQ dataset:            {args.eq_dataset_path}")
    print(f"Batch size:            {args.batch_size}")
    print(f"Math max_new:          {args.math_max_new}")
    print(f"EQ max_new:            {args.eq_max_new}")
    print(f"Mixed max_new:         {max_new_tokens}")
    print(f"Padding mode:          {args.padding_mode}")
    print(f"Math no_think prefix:  {args.math_use_no_think_prefix}")
    print(f"EQ no_think prefix:    {args.eq_use_no_think_prefix}")
    print(
        f"Adaptive retry:        {args.adaptive_batch_retry} "
        f"(min={args.min_batch_size}, max_retries={args.max_retries_per_phase})"
    )
    print(f"Trust remote:          {args.trust_remote_code}")
    print(f"Local files only:      {args.local_files_only}")
    print(f"Device map:            {resolved_device_map}")
    if resolved_max_memory is not None:
        print(f"Max memory:            {resolved_max_memory}")
    print(f"CPU offload:           {args.cpu_offload} (folder={args.offload_folder})")

    with open(args.math_dataset_path, "r") as f:
        math_dataset = json.load(f)
    with open(args.eq_dataset_path, "r") as f:
        eq_dataset = json.load(f)

    print("Loading tokenizer + model once...")
    attn_impl = args.attention_impl if args.attention_impl != "eager" else None
    tokenizer, base_model, load_meta = load_model_and_tokenizer(
        model_path=args.model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        torch_dtype=torch.bfloat16,
        device_map=resolved_device_map,
        attn_implementation=attn_impl,
        max_memory=resolved_max_memory,
        cpu_offload=args.cpu_offload,
        offload_folder=args.offload_folder,
    )
    num_layers = int(load_meta["num_layers"])
    print(
        f"Loader: {load_meta['loader']} | architectures={load_meta['architectures']} | text_stack={load_meta['text_stack']}"
    )
    print(f"Model has {num_layers} text layers")

    model_is_moe = bool(is_moe_model(base_model))
    build_duplicated_model = build_model_with_layers_moe if model_is_moe else build_model_with_layers
    print(f"Model type: {'MoE' if model_is_moe else 'Dense'}")

    print("Pre-tokenizing math dataset...")
    tokenized_math = pretokenize_dataset(
        math_dataset,
        tokenizer,
        base_model.device,
        use_no_think_prefix=args.math_use_no_think_prefix,
    )
    print("Pre-tokenizing EQ dataset...")
    tokenized_eq = pretokenize_eq_dataset(
        eq_dataset,
        tokenizer,
        base_model.device,
        use_no_think_prefix=args.eq_use_no_think_prefix,
    )

    mixed_items: list[dict[str, Any]] = []
    for qid, cached in tokenized_math.items():
        mixed_items.append(
            {
                "task": "math",
                "qid": qid,
                "input_ids": cached["input_ids"],
                "attention_mask": cached["attention_mask"],
                "answer": cached["answer"],
            }
        )
    for qid, cached in tokenized_eq.items():
        mixed_items.append(
            {
                "task": "eq",
                "qid": qid,
                "input_ids": cached["input_ids"],
                "attention_mask": cached["attention_mask"],
                "reference": cached["reference"],
            }
        )

    prompt_pad_id = _resolve_prompt_pad_id(tokenizer, args.prompt_pad_id)
    queue = SharedWorkQueue(args.queue_file, args.combined_results_file)

    def run_with_retry(run_model):
        exec_result = adaptive_batch_execute(
            lambda batch: run_combined_single_pass_hf(
                model=run_model,
                mixed_items=mixed_items,
                tokenizer=tokenizer,
                batch_size=batch,
                max_new_tokens=max_new_tokens,
                padding_mode=args.padding_mode,
                prompt_pad_id=prompt_pad_id,
            ),
            initial_batch_size=args.batch_size,
            min_batch_size=args.min_batch_size,
            max_retries=args.max_retries_per_phase,
            enabled=args.adaptive_batch_retry,
            phase_name="combined",
            on_retry=lambda msg: print(f"[{args.worker_id}] {msg}"),
        )
        return exec_result.result, exec_result.batch_size, exec_result.retries

    while True:
        entry = queue.get_next_config()
        if entry is None:
            print("Queue empty. Exiting.")
            break

        try:
            parsed_entry = parse_queue_entry_layers(num_layers, entry)
        except Exception as exc:
            print(f"[{args.worker_id}] Invalid queue entry {entry!r}: {exc}")
            continue

        config_key = parsed_entry["layer_key"]
        layer_indices = parsed_entry["layers"]
        config_spec = parsed_entry["spec"]
        remaining, completed = queue.get_queue_status()

        print(
            f"\n[{args.worker_id}] Running config {config_spec} ({layer_spec_string(layer_indices)}) "
            f"(remaining={remaining}, completed={completed})"
        )

        t0 = time.time()
        run_model = build_duplicated_model(base_model, list(layer_indices))
        (math_result, eq_result), used_batch, retries = run_with_retry(run_model)
        elapsed = time.time() - t0

        combined_result = {
            "config_key": config_key,
            "config_layers": list(layer_indices),
            "config_spec": config_spec,
            "elapsed": float(elapsed),
            "mode": "single_pass_all_hf",
            "num_prompts": int(len(mixed_items)),
            "batch_size": int(used_batch),
            "adaptive_retries": int(retries),
            "math_score": float(math_result["score"]),
            "eq_score": float(eq_result["score"]),
            "combined_score": 0.5 * (float(math_result["score"]) + float(eq_result["score"])),
            "math_valid_final_answer_count": int(math_result["valid_final_answer_count"]),
            "math_valid_final_answer_rate": float(math_result["valid_final_answer_rate"]),
            "math_fallback_used_count": int(math_result["fallback_used_count"]),
            "math_fallback_used_rate": float(math_result["fallback_used_rate"]),
        }

        queue.save_result(config_key, combined_result)
        _save_pickle_result(Path(args.math_results_file), config_key, math_result)
        _save_pickle_result(Path(args.eq_results_file), config_key, eq_result)

        print(
            f"[{args.worker_id}] math={math_result['score']:.4f} "
            f"eq={eq_result['score']:.4f} combined={combined_result['combined_score']:.4f} "
            f"valid={math_result['valid_final_answer_count']}/{len(math_result['responses'])} "
            f"batch={used_batch} retries={retries} elapsed={elapsed:.1f}s"
        )

        del run_model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

