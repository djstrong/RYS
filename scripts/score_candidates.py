#!/usr/bin/env python3
"""Score large candidate pools with trained surrogate models and keep top-K."""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.surrogate_utils import counts_from_csv, counts_to_csv, relative_overhead_from_counts


@dataclass
class ScoredCandidate:
    pred_final: float
    record: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score candidate count-vectors with surrogate models.")
    parser.add_argument("--candidates-file", required=True, help="CSV from generate_candidates.py")
    parser.add_argument("--model-method", required=True, help="Path to model_method_b.json")
    parser.add_argument("--model-math", required=True, help="Path to model_math_delta.json")
    parser.add_argument("--model-eq", required=True, help="Path to model_eq_delta.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-layers", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--heap-multiplier",
        type=int,
        default=20,
        help="Internal heap size = top_k * heap_multiplier.",
    )
    parser.add_argument("--lambda-overhead", type=float, default=0.5)
    parser.add_argument("--min-pred-math-delta", type=float, default=-0.02)
    parser.add_argument("--min-pred-eq-delta", type=float, default=-0.02)
    parser.add_argument("--progress-every", type=int, default=250_000)
    return parser.parse_args()


def _load_model(path: Path) -> xgb.Booster:
    model = xgb.Booster()
    model.load_model(str(path))
    return model


def _parse_candidate_row(row: dict[str, str], num_layers: int) -> tuple[list[int], float]:
    counts_raw = row.get("counts_csv")
    if counts_raw:
        counts = counts_from_csv(counts_raw, expected_len=num_layers)
    else:
        cols = [f"c{i}" for i in range(num_layers)]
        if not all(c in row for c in cols):
            raise ValueError("Candidate row requires either counts_csv or c0..cN columns.")
        counts = [int(row[c]) for c in cols]

    overhead_raw = row.get("relative_overhead", "").strip()
    if overhead_raw:
        overhead = float(overhead_raw)
    else:
        overhead = relative_overhead_from_counts(counts, num_layers)
    return counts, overhead


def _evaluate_batch(
    *,
    features: list[list[int]],
    rows: list[dict[str, Any]],
    model_method: xgb.Booster,
    model_math: xgb.Booster,
    model_eq: xgb.Booster,
    lambda_overhead: float,
    min_pred_math_delta: float,
    min_pred_eq_delta: float,
) -> list[ScoredCandidate]:
    X = np.asarray(features, dtype=np.float32)
    dm = xgb.DMatrix(X)
    pm = model_method.predict(dm).astype(np.float32)
    pmd = model_math.predict(dm).astype(np.float32)
    ped = model_eq.predict(dm).astype(np.float32)
    out: list[ScoredCandidate] = []
    for idx, raw in enumerate(rows):
        pred_math = float(pmd[idx])
        pred_eq = float(ped[idx])
        if pred_math < min_pred_math_delta or pred_eq < min_pred_eq_delta:
            continue
        overhead = float(raw["relative_overhead"])
        pred_method = float(pm[idx])
        pred_final = pred_method - (lambda_overhead * overhead)
        rec = dict(raw)
        rec.update(
            {
                "pred_method_b": pred_method,
                "pred_math_delta": pred_math,
                "pred_eq_delta": pred_eq,
                "pred_final": pred_final,
            }
        )
        out.append(ScoredCandidate(pred_final=pred_final, record=rec))
    return out


def _heap_push(heap: list[tuple[float, int, dict[str, Any]]], item: ScoredCandidate, max_size: int, seq: int) -> int:
    payload = (float(item.pred_final), seq, item.record)
    if len(heap) < max_size:
        heapq.heappush(heap, payload)
    elif payload[0] > heap[0][0]:
        heapq.heapreplace(heap, payload)
    return seq + 1


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.heap_multiplier < 1:
        raise ValueError("--heap-multiplier must be >= 1")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_method = _load_model(Path(args.model_method))
    model_math = _load_model(Path(args.model_math))
    model_eq = _load_model(Path(args.model_eq))

    heap_cap = args.top_k * args.heap_multiplier
    heap: list[tuple[float, int, dict[str, Any]]] = []
    seq = 0
    seen_rows = 0
    accepted = 0

    features: list[list[int]] = []
    rows: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal features, rows, seq, accepted
        if not rows:
            return
        scored = _evaluate_batch(
            features=features,
            rows=rows,
            model_method=model_method,
            model_math=model_math,
            model_eq=model_eq,
            lambda_overhead=args.lambda_overhead,
            min_pred_math_delta=args.min_pred_math_delta,
            min_pred_eq_delta=args.min_pred_eq_delta,
        )
        accepted += len(scored)
        for item in scored:
            seq = _heap_push(heap, item, heap_cap, seq)
        features = []
        rows = []

    with Path(args.candidates_file).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            seen_rows += 1
            counts, overhead = _parse_candidate_row(raw, args.num_layers)
            rows.append(
                {
                    "candidate_id": raw.get("candidate_id", str(seen_rows - 1)),
                    "source": raw.get("source", ""),
                    "extra_layers": int(sum(counts) - args.num_layers),
                    "relative_overhead": float(overhead),
                    "counts_csv": counts_to_csv(counts),
                }
            )
            features.append(counts)
            if len(rows) >= args.batch_size:
                flush()
            if args.progress_every > 0 and seen_rows % args.progress_every == 0:
                print(f"scanned={seen_rows} accepted={accepted} heap={len(heap)}")
    flush()

    # Sort descending, dedupe by counts, keep top_k.
    ranked = sorted(heap, key=lambda x: x[0], reverse=True)
    unique: list[dict[str, Any]] = []
    seen_counts: set[str] = set()
    for _, _, rec in ranked:
        sig = str(rec["counts_csv"])
        if sig in seen_counts:
            continue
        seen_counts.add(sig)
        unique.append(rec)
        if len(unique) >= args.top_k:
            break

    top_csv = out_dir / "top_candidates.csv"
    top_json = out_dir / "top_candidates.json"
    summary_json = out_dir / "score_summary.json"

    with top_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "rank",
            "candidate_id",
            "source",
            "extra_layers",
            "relative_overhead",
            "pred_method_b",
            "pred_math_delta",
            "pred_eq_delta",
            "pred_final",
            "counts_csv",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, rec in enumerate(unique, start=1):
            writer.writerow({"rank": rank, **rec})

    with top_json.open("w", encoding="utf-8") as f:
        json.dump(unique, f, indent=2)

    summary = {
        "candidates_file": args.candidates_file,
        "rows_scanned": seen_rows,
        "rows_accepted_after_filters": accepted,
        "heap_size": len(heap),
        "top_k_requested": args.top_k,
        "top_k_written": len(unique),
        "lambda_overhead": args.lambda_overhead,
        "min_pred_math_delta": args.min_pred_math_delta,
        "min_pred_eq_delta": args.min_pred_eq_delta,
        "model_method": args.model_method,
        "model_math": args.model_math,
        "model_eq": args.model_eq,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote: {top_csv}")
    print(f"Wrote: {top_json}")
    print(f"Wrote: {summary_json}")


if __name__ == "__main__":
    main()
