#!/usr/bin/env python3
"""Generate heatmaps for repeat-x8 per-layer duplication sweeps."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


def load_score_map(results_path: Path) -> dict[str, float]:
    with results_path.open("rb") as f:
        raw = pickle.load(f)

    scores: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(key, tuple):
            key_str = ",".join(str(int(x)) for x in key)
        elif isinstance(key, list):
            key_str = ",".join(str(int(x)) for x in key)
        else:
            key_str = str(key)

        score = value.get("score") if isinstance(value, dict) else value
        scores[key_str] = float(score)

    return scores


def load_manifest(manifest_path: Path) -> dict:
    with manifest_path.open("r") as f:
        return json.load(f)


def build_repeat_grid(manifest: dict, scores: dict[str, float]) -> tuple[np.ndarray, float]:
    num_layers = int(manifest["num_layers"])
    entries = manifest["entries"]
    max_extra = max(int(entry.get("extra_repeats", 0)) for entry in entries)

    grid = np.full((max_extra, num_layers), np.nan, dtype=float)
    baseline = np.nan

    for entry in entries:
        key = str(entry["layer_indices_key"])
        if key not in scores:
            continue

        score = scores[key]
        extra = int(entry.get("extra_repeats", 0))
        layer = entry.get("layer")

        if extra == 0 or layer is None:
            baseline = score
            continue

        if 1 <= extra <= max_extra and 0 <= int(layer) < num_layers:
            grid[extra - 1, int(layer)] = score

    if not np.isfinite(baseline):
        baseline = float(np.nanmean(grid))

    return grid, float(baseline)


def best_cell(grid: np.ndarray) -> tuple[int, int] | None:
    if not np.isfinite(grid).any():
        return None
    idx = int(np.nanargmax(grid))
    return tuple(np.unravel_index(idx, grid.shape))


def _safe_two_slope(vmin: float, vcenter: float, vmax: float) -> TwoSlopeNorm | None:
    if not (np.isfinite(vmin) and np.isfinite(vcenter) and np.isfinite(vmax)):
        return None
    if vmin >= vcenter:
        vmin = vcenter - 1e-6
    if vmax <= vcenter:
        vmax = vcenter + 1e-6
    if vmin >= vmax:
        return None
    return TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)


def _positive_sigma(deltas: np.ndarray) -> float:
    pos = deltas[deltas > 0]
    if pos.size >= 2:
        sigma = float(np.std(pos))
        if sigma > 0:
            return sigma

    nonneg = deltas[deltas >= 0]
    if nonneg.size >= 2:
        sigma = float(np.std(nonneg))
        if sigma > 0:
            return sigma

    sigma = float(np.std(deltas))
    if sigma > 0:
        return sigma
    return 1e-9


def compute_asym_diff_bounds(
    diff: np.ndarray,
    *,
    k_pos: float,
    k_neg: float,
    tail_q: float,
    a_min: float,
    a_max: float,
) -> tuple[float, float]:
    finite = diff[np.isfinite(diff)]
    if finite.size == 0:
        return -1.0, 1.0

    eps = 1e-9
    best = float(np.max(finite))
    sigma_pos = _positive_sigma(finite)
    pos = finite[finite > 0]
    neg_mag = -finite[finite < 0]

    q_pct = float(np.clip(tail_q, 0.0, 1.0) * 100.0)
    q_pos = float(np.percentile(pos, q_pct)) if pos.size else max(best, 0.0)
    q_neg = float(np.percentile(neg_mag, q_pct)) if neg_mag.size else q_pos

    raw_ratio = q_neg / max(q_pos, eps) if q_pos > 0 else 1.0
    asym_ratio = float(np.clip(raw_ratio, a_min, a_max))

    # Full positive range must remain visible; negative side is clipped asymmetrically.
    pos_span = max(k_pos * sigma_pos, q_pos, best + eps, eps)
    neg_span_raw = max(k_neg * sigma_pos * asym_ratio, q_neg, eps)
    # Enforce asymmetric clipping: negative span cannot exceed the allowed
    # positive-multiple envelope.
    neg_span_cap = max(a_max * pos_span, eps)
    neg_span = min(neg_span_raw, neg_span_cap)

    diff_vmin = -neg_span
    diff_vmax = pos_span
    if diff_vmin >= diff_vmax:
        span = max(abs(diff_vmin), abs(diff_vmax), 1e-6)
        diff_vmin, diff_vmax = -span, span

    return diff_vmin, diff_vmax


def plot_grid(
    grid: np.ndarray,
    *,
    title: str,
    subtitle: str,
    cbar_label: str,
    out_path: Path,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    norm: TwoSlopeNorm | None = None,
    mark: tuple[int, int] | None = None,
    cbar_ticks: list[float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.2))
    cmap_obj = plt.cm.get_cmap(cmap).copy()
    cmap_obj.set_bad("#B0B0B0")

    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
    )

    num_repeats, num_layers = grid.shape
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Extra repeats")
    ax.set_yticks(range(num_repeats))
    ax.set_yticklabels([str(i + 1) for i in range(num_repeats)])

    step = max(1, num_layers // 8)
    xticks = list(range(0, num_layers, step))
    if xticks[-1] != num_layers - 1:
        xticks.append(num_layers - 1)
    ax.set_xticks(xticks)

    if mark is not None:
        ax.plot(
            mark[1],
            mark[0],
            "o",
            markersize=9,
            markeredgecolor="lime",
            markerfacecolor="none",
            markeredgewidth=2,
        )

    ax.set_title(title)
    ax.text(
        0.0,
        1.02,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    if cbar_ticks:
        cbar.set_ticks(cbar_ticks)
    cbar.set_label(cbar_label)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot repeat-x8 heatmaps from manifest + results.")
    parser.add_argument("--manifest", required=True, help="Path to repeatx8_manifest.json")
    parser.add_argument("--results", required=True, help="Path to math/eq repeatx8 results pickle")
    parser.add_argument("--model", required=True, help="Model name for plot titles")
    parser.add_argument("--task", required=True, choices=["math", "eq"], help="Task label")
    parser.add_argument("--out-dir", default=None, help="Output directory (defaults to results file parent)")
    parser.add_argument("--clip-tail-q", type=float, default=0.95, help="Tail quantile used by clipped bounds.")
    parser.add_argument("--clip-k-pos", type=float, default=3.0, help="Positive SD multiplier for clipped bounds.")
    parser.add_argument("--clip-k-neg", type=float, default=2.0, help="Negative SD multiplier before asymmetry factor.")
    parser.add_argument("--clip-a-min", type=float, default=1.0, help="Minimum asymmetry factor.")
    parser.add_argument("--clip-a-max", type=float, default=6.0, help="Maximum asymmetry factor.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    results_path = Path(args.results)
    out_dir = Path(args.out_dir) if args.out_dir else results_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path)
    score_map = load_score_map(results_path)
    grid, baseline = build_repeat_grid(manifest, score_map)
    diff = grid - baseline

    mark = best_cell(grid)
    if mark is not None:
        best_score = float(grid[mark])
        delta = best_score - baseline
        summary = (
            f"Baseline: {baseline:.4f} | "
            f"Best: layer {mark[1]} +{mark[0] + 1} repeats => {best_score:.4f} ({delta:+.4f})"
        )
    else:
        summary = f"Baseline: {baseline:.4f}"

    score_min = float(np.nanmin(grid)) if np.isfinite(grid).any() else 0.0
    score_max = float(np.nanmax(grid)) if np.isfinite(grid).any() else 1.0
    if score_min == score_max:
        score_min -= 1e-6
        score_max += 1e-6

    max_abs = float(np.nanmax(np.abs(diff))) if np.isfinite(diff).any() else 1.0
    max_abs = max(max_abs, 1e-6)

    diff_vmin_clip, diff_vmax_clip = compute_asym_diff_bounds(
        diff,
        k_pos=args.clip_k_pos,
        k_neg=args.clip_k_neg,
        tail_q=args.clip_tail_q,
        a_min=args.clip_a_min,
        a_max=args.clip_a_max,
    )
    raw_norm_clip = _safe_two_slope(baseline + diff_vmin_clip, baseline, baseline + diff_vmax_clip)
    diff_norm_clip = _safe_two_slope(diff_vmin_clip, 0.0, diff_vmax_clip)
    diff_ticks = [
        float(diff_vmin_clip),
        float(diff_vmin_clip * 0.5),
        0.0,
        float(diff_vmax_clip * 0.5),
        float(diff_vmax_clip),
    ]
    # Drop near-duplicate ticks after rounding noise.
    dedup: list[float] = []
    for t in diff_ticks:
        if not dedup or abs(t - dedup[-1]) > 1e-9:
            dedup.append(t)
    diff_ticks = dedup

    model_task = f"{args.model} {args.task.upper()}".strip()

    plot_grid(
        grid,
        title=f"{model_task} - Repeat-x8 Layer Scores",
        subtitle=summary,
        cbar_label="Score",
        out_path=out_dir / f"heatmap_repeatx8_{args.task}.png",
        cmap="viridis",
        vmin=score_min,
        vmax=score_max,
        mark=mark,
    )
    plot_grid(
        diff,
        title=f"{model_task} - Difference from Baseline",
        subtitle=summary,
        cbar_label="Score delta",
        out_path=out_dir / f"heatmap_repeatx8_{args.task}_diff.png",
        cmap="RdBu_r",
        norm=_safe_two_slope(-max_abs, 0.0, max_abs),
        mark=mark,
    )
    plot_grid(
        grid,
        title=f"{model_task} - Repeat-x8 Scores (Clipped)",
        subtitle=summary,
        cbar_label="Score",
        out_path=out_dir / f"heatmap_repeatx8_{args.task}_clipped.png",
        cmap="viridis",
        norm=raw_norm_clip,
        mark=mark,
    )
    plot_grid(
        diff,
        title=f"{model_task} - Difference from Baseline (Clipped)",
        subtitle=summary,
        cbar_label="Score delta",
        out_path=out_dir / f"heatmap_repeatx8_{args.task}_diff_clipped.png",
        cmap="RdBu_r",
        norm=diff_norm_clip,
        mark=mark,
        cbar_ticks=diff_ticks,
    )


if __name__ == "__main__":
    main()
