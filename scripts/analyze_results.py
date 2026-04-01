#!/usr/bin/env python3
"""Balanced (Method B) math+EQ analysis for a single run."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.heatmaps import generate_heatmap  # noqa: E402
from src.core.layer_config import ij_to_layers, layer_spec_string  # noqa: E402
from src.utils.math_eq_analysis import (  # noqa: E402
    METHOD_BALANCED,
    build_balanced_rows,
    choose_baseline,
    infer_eq_scale,
    load_scores,
    rank_balanced_rows,
    relayer_string,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze math+EQ results using balanced z-delta (Method B).",
    )
    parser.add_argument("--math-scores", required=True, help="Path to math scores pickle.")
    parser.add_argument("--eq-scores", required=True, help="Path to EQ scores pickle.")
    parser.add_argument("--out-dir", required=True, help="Directory for outputs.")
    parser.add_argument(
        "--eq-scale-policy",
        choices=["auto_to_unit", "none"],
        default="auto_to_unit",
        help="EQ scaling policy. auto_to_unit divides likely 0-100 EQ runs by 100.",
    )
    parser.add_argument(
        "--baseline-policy",
        choices=["canonical_or_proxy", "canonical_only"],
        default="canonical_or_proxy",
        help="How to choose baseline key when canonical baseline is missing.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Base model layer count. Recommended for robust legacy (i,j) conversion.",
    )
    parser.add_argument("--top-n", type=int, default=10, help="Top N configs.")
    parser.add_argument("--title", default=None, help="Optional title prefix.")
    parser.add_argument(
        "--plot-scatter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write balanced_scatter_delta.png (math vs EQ deltas).",
    )
    parser.add_argument(
        "--plot-heatmap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write balanced_heatmap_score.png (Method B score on an i,j grid).",
    )
    return parser.parse_args()


def row_to_output(row: dict[str, Any], num_layers: int) -> dict[str, Any]:
    key = tuple(row["key"])
    if len(key) == 2:
        i, j = key
        relayer = relayer_string(i, j, num_layers)
    else:
        relayer = layer_spec_string(key)
    return {
        "config": list(key),
        "relayer": relayer,
        "method": METHOD_BALANCED,
        "method_score": row[METHOD_BALANCED],
        "math_score": row["math_score"],
        "eq_score": row["eq_score"],
        "math_delta": row["math_delta"],
        "eq_delta": row["eq_delta"],
        "rank": row["rank"],
    }


def write_top_outputs(
    rows: list[dict[str, Any]],
    top_n: int,
    out_dir: Path,
    num_layers: int,
) -> list[dict[str, Any]]:
    ranked = rank_balanced_rows(rows)
    top_rows = [row_to_output(row, num_layers) for row in ranked[:top_n]]

    json_path = out_dir / f"top{top_n}_{METHOD_BALANCED}.json"
    csv_path = out_dir / f"top{top_n}_{METHOD_BALANCED}.csv"
    with json_path.open("w") as f:
        json.dump(top_rows, f, indent=2)

    fields = [
        "config",
        "relayer",
        "method",
        "method_score",
        "math_score",
        "eq_score",
        "math_delta",
        "eq_delta",
        "rank",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in top_rows:
            row_out = dict(row)
            row_out["config"] = str(tuple(row["config"]))
            writer.writerow(row_out)

    return top_rows


def _make_centered_norm(values: list[float]) -> TwoSlopeNorm | None:
    if not values:
        return None
    vmin = float(min(values))
    vmax = float(max(values))
    if vmin >= 0.0 or vmax <= 0.0:
        return None
    return TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)


def _key_to_ij(num_layers: int, key: tuple[int, ...]) -> tuple[int, int] | None:
    """Map a result key to legacy (i, j) when it matches a single-block relayer config."""
    if len(key) == 2:
        return int(key[0]), int(key[1])
    layers_list = list(int(x) for x in key)
    if ij_to_layers(num_layers, 0, 0) == layers_list:
        return (0, 0)
    for j in range(num_layers + 1):
        for i in range(j):
            if ij_to_layers(num_layers, i, j) == layers_list:
                return (i, j)
    return None


def write_balanced_heatmap(
    rows: list[dict[str, Any]],
    num_layers: int,
    title: str,
    out_path: Path,
) -> bool:
    """Fill an (i, j) grid with Method B scores. Baseline (0, 0) is always 0."""
    heatmap_scores: dict[tuple[int, int], float] = {(0, 0): 0.0}
    for row in rows:
        key = tuple(int(x) for x in row["key"])
        ij = _key_to_ij(num_layers, key)
        if ij is None:
            continue
        heatmap_scores[ij] = float(row[METHOD_BALANCED])
    if len(heatmap_scores) <= 1:
        return False
    norm = _make_centered_norm(list(heatmap_scores.values()))
    generate_heatmap(
        heatmap_scores,
        title,
        out_path,
        num_layers,
        mask_missing=False,
        norm=norm,
    )
    return True


def infer_num_layers_from_keys(keys: set[tuple[int, ...]]) -> int:
    if not keys:
        raise ValueError("Cannot infer num_layers from empty key set.")
    canonical_candidates = [len(k) for k in keys if tuple(range(len(k))) == k]
    if canonical_candidates:
        return max(canonical_candidates)
    if all(len(k) == 2 for k in keys):
        return max(int(k[1]) for k in keys)
    max_idx = max(max(k) for k in keys if k)
    return int(max_idx + 1)


def plot_balanced_scatter(
    rows: list[dict[str, Any]],
    title: str,
    out_path: Path,
    top_rows: list[dict[str, Any]],
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    math_delta = np.array([row["math_delta"] for row in rows], dtype=float)
    eq_delta = np.array([row["eq_delta"] for row in rows], dtype=float)
    scores = np.array([row[METHOD_BALANCED] for row in rows], dtype=float)

    ax.axhline(0.0, color="gray", linewidth=1)
    ax.axvline(0.0, color="gray", linewidth=1)
    sc = ax.scatter(math_delta, eq_delta, c=scores, cmap="viridis", s=18, alpha=0.75)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(f"{METHOD_BALANCED} score")

    if top_rows:
        top_points = np.array([[row["math_delta"], row["eq_delta"]] for row in top_rows], dtype=float)
        ax.scatter(
            top_points[:, 0],
            top_points[:, 1],
            s=64,
            facecolors="none",
            edgecolors="red",
            linewidths=1.4,
            label=f"Top {len(top_rows)}",
        )

    ax.scatter([0.0], [0.0], s=85, marker="*", color="black", label="Baseline")
    ax.set_xlabel("Math delta")
    ax.set_ylabel("EQ delta")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    math_path = Path(args.math_scores)
    eq_path = Path(args.eq_scores)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    math_probe = load_scores(math_path)
    eq_probe = load_scores(eq_path)
    if not math_probe:
        raise ValueError(f"No math scores loaded from {math_path}")
    if not eq_probe:
        raise ValueError(f"No EQ scores loaded from {eq_path}")
    common_probe = set(math_probe.keys()) & set(eq_probe.keys())
    if not common_probe:
        raise ValueError("No common keys between math and EQ scores.")

    num_layers = args.num_layers if args.num_layers is not None else infer_num_layers_from_keys(common_probe)
    math_scores = load_scores(math_path, num_layers=num_layers, prefer_legacy_ij=True)
    eq_scores_raw = load_scores(eq_path, num_layers=num_layers, prefer_legacy_ij=True)

    eq_scores, eq_scale_info = infer_eq_scale(eq_scores_raw, args.eq_scale_policy)
    common_keys = set(math_scores.keys()) & set(eq_scores.keys())
    if not common_keys:
        raise ValueError("No common keys between math and EQ scores.")

    baseline_key, baseline_source, warnings = choose_baseline(
        common_keys,
        args.baseline_policy,
        num_layers=num_layers,
    )
    rows, meta, row_warnings = build_balanced_rows(math_scores, eq_scores, baseline_key)
    warnings.extend(row_warnings)
    if not rows:
        raise ValueError("No non-baseline configs available after merge.")

    ranked_rows = rank_balanced_rows(rows)
    for rank, row in enumerate(ranked_rows, start=1):
        row["rank"] = rank

    title_prefix = args.title or math_path.parent.name

    analysis_scores: dict[tuple[int, int], dict[str, Any]] = {
        baseline_key: {
            "key": baseline_key,
            "math_score": meta["baseline_math"],
            "eq_score": meta["baseline_eq"],
            "math_delta": 0.0,
            "eq_delta": 0.0,
            METHOD_BALANCED: 0.0,
            "rank": 0,
        }
    }
    for row in rows:
        analysis_scores[row["key"]] = row

    with (out_dir / "analysis_scores.pkl").open("wb") as f:
        pickle.dump(analysis_scores, f)

    top_rows = write_top_outputs(rows, args.top_n, out_dir, num_layers)
    if args.plot_scatter:
        plot_balanced_scatter(
            rows,
            f"{title_prefix} - {METHOD_BALANCED} (delta scatter)",
            out_dir / "balanced_scatter_delta.png",
            top_rows,
        )
    if args.plot_heatmap:
        wrote = write_balanced_heatmap(
            rows,
            num_layers,
            f"{title_prefix} - {METHOD_BALANCED} score heatmap (i, j)",
            out_dir / "balanced_heatmap_score.png",
        )
        if not wrote:
            warnings.append(
                "Skipped balanced heatmap: no keys mapped to single-block (i, j) "
                "(need legacy (i,j) keys or canonical layer lists from the standard single-block sweep)."
            )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "math_scores_path": str(math_path),
        "eq_scores_path": str(eq_path),
        "title": title_prefix,
        "method": METHOD_BALANCED,
        "eq_scale_policy": args.eq_scale_policy,
        "eq_scale_applied": eq_scale_info.applied,
        "eq_scale_factor": eq_scale_info.factor,
        "eq_scale_reason": eq_scale_info.reason,
        "baseline_policy": args.baseline_policy,
        "baseline_source": baseline_source,
        "baseline_key": list(baseline_key),
        "common_config_count": len(common_keys),
        "nonbaseline_config_count": len(rows),
        "num_layers": num_layers,
        "warnings": warnings,
        "top1": top_rows[0] if top_rows else None,
    }
    with (out_dir / "balanced_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[{title_prefix}] top1={top_rows[0]['config']} score={top_rows[0]['method_score']:.6f} "
        f"dm={top_rows[0]['math_delta']:.6f} de={top_rows[0]['eq_delta']:.6f}"
    )


if __name__ == "__main__":
    main()
