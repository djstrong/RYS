#!/usr/bin/env python3
"""Generate large count-vector candidate pools for surrogate scoring."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.surrogate_utils import counts_from_csv, counts_to_csv, relative_overhead_from_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate count-vector relayer candidates.")
    parser.add_argument("--out-file", required=True, help="Output CSV path.")
    parser.add_argument("--num-candidates", type=int, default=1_000_000)
    parser.add_argument("--num-layers", type=int, default=64)
    parser.add_argument("--max-extra-layers", type=int, default=24)
    parser.add_argument(
        "--max-repeat-per-layer",
        type=int,
        default=8,
        help="Maximum extra repeats per layer beyond baseline 1x.",
    )
    parser.add_argument(
        "--anchor-file",
        default=None,
        help="Optional CSV with `counts_csv` column to bias generation near known good configs.",
    )
    parser.add_argument(
        "--anchor-prob",
        type=float,
        default=0.45,
        help="Probability of sampling an anchor mutation instead of a random vector.",
    )
    parser.add_argument(
        "--mutation-steps",
        type=int,
        default=8,
        help="Max random +/- adjustments for each anchor mutation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=100_000)
    return parser.parse_args()


def _load_anchors(path: Path, *, num_layers: int) -> list[list[int]]:
    if not path.exists():
        raise FileNotFoundError(f"Anchor file not found: {path}")
    anchors: list[list[int]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "counts_csv" not in (reader.fieldnames or []):
            raise ValueError(f"Anchor file missing counts_csv column: {path}")
        for row in reader:
            raw = str(row.get("counts_csv", "")).strip()
            if not raw:
                continue
            try:
                counts = counts_from_csv(raw, expected_len=num_layers)
            except ValueError:
                continue
            anchors.append(counts)
    return anchors


def _random_counts(
    rng: random.Random,
    *,
    num_layers: int,
    max_extra_layers: int,
    max_repeat_per_layer: int,
) -> list[int]:
    counts = [1] * num_layers
    cap = 1 + max_repeat_per_layer
    capacity = num_layers * max_repeat_per_layer
    extra = min(rng.randint(0, max_extra_layers), capacity)
    if extra == 0:
        return counts
    eligible = list(range(num_layers))
    for _ in range(extra):
        if not eligible:
            break
        idx = rng.choice(eligible)
        counts[idx] += 1
        if counts[idx] >= cap:
            eligible.remove(idx)
    return counts


def _trim_to_overhead(
    rng: random.Random,
    counts: list[int],
    *,
    num_layers: int,
    max_extra_layers: int,
) -> None:
    extra = sum(counts) - num_layers
    if extra <= max_extra_layers:
        return
    while extra > max_extra_layers:
        candidates = [idx for idx, c in enumerate(counts) if c > 1]
        if not candidates:
            return
        idx = rng.choice(candidates)
        counts[idx] -= 1
        extra -= 1


def _mutate_anchor(
    rng: random.Random,
    anchor: list[int],
    *,
    num_layers: int,
    max_extra_layers: int,
    max_repeat_per_layer: int,
    mutation_steps: int,
) -> list[int]:
    cap = 1 + max_repeat_per_layer
    out = list(anchor)
    steps = rng.randint(1, max(1, mutation_steps))
    for _ in range(steps):
        do_add = rng.random() < 0.65
        if do_add:
            addable = [i for i, c in enumerate(out) if c < cap]
            if not addable:
                continue
            idx = rng.choice(addable)
            out[idx] += 1
        else:
            removable = [i for i, c in enumerate(out) if c > 1]
            if not removable:
                continue
            idx = rng.choice(removable)
            out[idx] -= 1
    _trim_to_overhead(rng, out, num_layers=num_layers, max_extra_layers=max_extra_layers)
    return out


def main() -> None:
    args = parse_args()
    if args.num_candidates < 1:
        raise ValueError("--num-candidates must be >= 1")
    if args.max_extra_layers < 0:
        raise ValueError("--max-extra-layers must be >= 0")
    if args.max_repeat_per_layer < 0:
        raise ValueError("--max-repeat-per-layer must be >= 0")
    if args.anchor_prob < 0.0 or args.anchor_prob > 1.0:
        raise ValueError("--anchor-prob must be in [0,1]")

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    anchors: list[list[int]] = [[1] * args.num_layers]
    if args.anchor_file:
        anchors.extend(_load_anchors(Path(args.anchor_file), num_layers=args.num_layers))

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["candidate_id", "source", "extra_layers", "relative_overhead", "counts_csv"],
        )
        writer.writeheader()

        for idx in range(args.num_candidates):
            if anchors and rng.random() < args.anchor_prob:
                base = rng.choice(anchors)
                counts = _mutate_anchor(
                    rng,
                    base,
                    num_layers=args.num_layers,
                    max_extra_layers=args.max_extra_layers,
                    max_repeat_per_layer=args.max_repeat_per_layer,
                    mutation_steps=args.mutation_steps,
                )
                source = "anchor_mutation"
            else:
                counts = _random_counts(
                    rng,
                    num_layers=args.num_layers,
                    max_extra_layers=args.max_extra_layers,
                    max_repeat_per_layer=args.max_repeat_per_layer,
                )
                source = "random"

            extra = sum(counts) - args.num_layers
            writer.writerow(
                {
                    "candidate_id": idx,
                    "source": source,
                    "extra_layers": extra,
                    "relative_overhead": relative_overhead_from_counts(counts, args.num_layers),
                    "counts_csv": counts_to_csv(counts),
                }
            )

            if args.progress_every > 0 and (idx + 1) % args.progress_every == 0:
                print(f"generated={idx + 1}/{args.num_candidates}")

    print(f"Wrote: {out_path}")
    print(f"Candidates: {args.num_candidates}")


if __name__ == "__main__":
    main()
