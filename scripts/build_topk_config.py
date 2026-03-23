#!/usr/bin/env python3
"""Build worker config + manifest from scored top candidate count-vectors."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.surrogate_utils import count_vector_to_layers, counts_from_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert scored top candidates into worker config format.")
    parser.add_argument("--top-candidates-csv", required=True, help="CSV from score_candidates.py")
    parser.add_argument("--out-config", required=True, help="Output config file consumed by math/eq workers")
    parser.add_argument("--out-manifest", required=True, help="Manifest JSON with metadata for each config")
    parser.add_argument("--num-layers", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--include-baseline", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")

    rows = _load_rows(Path(args.top_candidates_csv))
    selected = rows[: args.top_k]

    out_config = Path(args.out_config)
    out_manifest = Path(args.out_manifest)
    out_config.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    config_lines: list[str] = []

    if args.include_baseline:
        config_lines.append("0,0")
        manifest_rows.append(
            {
                "rank": 0,
                "config_line": "0,0",
                "layer_key": list(range(args.num_layers)),
                "extra_layers": 0,
                "relative_overhead": 0.0,
                "pred_method_b": 0.0,
                "pred_math_delta": 0.0,
                "pred_eq_delta": 0.0,
                "pred_final": 0.0,
                "source": "forced_baseline",
                "candidate_id": "baseline",
            }
        )

    for rank, row in enumerate(selected, start=1):
        counts = counts_from_csv(str(row.get("counts_csv", "")), expected_len=args.num_layers)
        layers = count_vector_to_layers(counts, args.num_layers)
        config_line = "layers:" + ",".join(str(x) for x in layers)
        config_lines.append(config_line)
        manifest_rows.append(
            {
                "rank": rank,
                "config_line": config_line,
                "layer_key": layers,
                "extra_layers": int(sum(counts) - args.num_layers),
                "relative_overhead": float(row.get("relative_overhead", 0.0)),
                "pred_method_b": float(row.get("pred_method_b", 0.0)),
                "pred_math_delta": float(row.get("pred_math_delta", 0.0)),
                "pred_eq_delta": float(row.get("pred_eq_delta", 0.0)),
                "pred_final": float(row.get("pred_final", 0.0)),
                "source": str(row.get("source", "")),
                "candidate_id": str(row.get("candidate_id", rank)),
                "counts_csv": str(row.get("counts_csv", "")),
            }
        )

    with out_config.open("w", encoding="utf-8") as f:
        f.write("# Surrogate-selected relayer configs\n")
        f.write("# Format: legacy i,j;... or layers:a,b,c,...\n")
        for line in config_lines:
            f.write(line + "\n")

    payload = {
        "source_csv": str(Path(args.top_candidates_csv)),
        "num_layers": args.num_layers,
        "top_k_requested": args.top_k,
        "include_baseline": bool(args.include_baseline),
        "selected_count": len(manifest_rows),
        "selected_rows": manifest_rows,
    }
    out_manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote: {out_config}")
    print(f"Wrote: {out_manifest}")
    print(f"Configs written: {len(config_lines)}")


if __name__ == "__main__":
    main()
