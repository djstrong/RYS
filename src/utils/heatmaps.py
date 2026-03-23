"""Heatmap helpers for relayer scan results."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def load_results(filepath: str | Path) -> dict[tuple[int, int], float]:
    """Load results from a pickle file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def results_to_image(
    results: dict[tuple[int, int], float],
    num_layers: int,
    *,
    mask_missing: bool = False,
) -> np.ndarray:
    """Convert an `(i, j)` score dict into a dense image matrix."""
    size = num_layers + 1
    baseline = results.get((0, 0), np.mean(list(results.values())))
    if mask_missing:
        img = np.full((size, size), np.nan)
    else:
        img = np.full((size, size), baseline * 0.8)

    for (i, j), score in results.items():
        if 0 <= i < size and 0 <= j < size:
            img[i, j] = score
    return img


def _marker_labels(results: dict[tuple[int, int], float]) -> tuple[float, tuple[int, int], float, str, str]:
    baseline = results.get((0, 0), np.mean(list(results.values())))
    best_key = max(results.keys(), key=lambda k: results[k])
    best_score = results[best_key]
    delta = best_score - baseline
    baseline_label = (
        f"Baseline (0,0): {baseline:.4f}"
        if (0, 0) in results
        else f"Baseline (mean): {baseline:.4f}"
    )
    best_label = f"Best: {best_key} {best_score:.4f} (+{delta:.4f})"
    return baseline, best_key, best_score, baseline_label, best_label


def generate_heatmap(
    results: dict[tuple[int, int], float],
    title: str,
    output_path: str | Path,
    num_layers: int,
    *,
    cmap: str = "viridis",
    mask_missing: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Any | None = None,
    show_baseline_marker: bool = True,
    show_best_marker: bool = True,
) -> None:
    """Generate and save an absolute-score heatmap."""
    img = results_to_image(results, num_layers, mask_missing=mask_missing)
    _, best_key, _, baseline_label, best_label = _marker_labels(results)

    fig, ax = plt.subplots(figsize=(12, 10))
    if mask_missing:
        cmap_obj = plt.cm.get_cmap(cmap).copy()
        cmap_obj.set_bad(color="#B0B0B0")
        im = ax.imshow(img, cmap=cmap_obj, origin="upper", aspect="equal", vmin=vmin, vmax=vmax, norm=norm)
    else:
        im = ax.imshow(img, cmap=cmap, origin="upper", aspect="equal", vmin=vmin, vmax=vmax, norm=norm)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Benchmark Score", rotation=270, labelpad=20)
    ax.set_xlabel("j (end of duplication range)", fontsize=12)
    ax.set_ylabel("i (start of duplication range)", fontsize=12)
    ax.set_title(title, fontsize=14)

    tick_interval = max(1, num_layers // 10)
    ticks = range(0, num_layers + 1, tick_interval)
    ax.set_xticks(list(ticks))
    ax.set_yticks(list(ticks))

    if show_baseline_marker:
        ax.plot(0, 0, "o", markersize=12, markeredgecolor="red", markerfacecolor="none", markeredgewidth=2, label=baseline_label)
    if show_best_marker and best_key != (0, 0):
        ax.plot(best_key[1], best_key[0], "o", markersize=12, markeredgecolor="lime", markerfacecolor="none", markeredgewidth=2, label=best_label)
    if show_baseline_marker or (show_best_marker and best_key != (0, 0)):
        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap to {output_path}")


def generate_difference_heatmap(
    results: dict[tuple[int, int], float],
    title: str,
    output_path: str | Path,
    num_layers: int,
    *,
    mask_missing: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Any | None = None,
    show_baseline_marker: bool = True,
    show_best_marker: bool = True,
) -> None:
    """Generate and save a difference-from-baseline heatmap."""
    baseline, best_key, _, baseline_label, best_label = _marker_labels(results)
    diff_results = {k: v - baseline for k, v in results.items()}
    img = results_to_image(diff_results, num_layers, mask_missing=mask_missing)

    if not mask_missing:
        fill_value = baseline * 0.8 - baseline
        img[img == fill_value] = 0

    fig, ax = plt.subplots(figsize=(12, 10))
    if norm is None and (vmin is None or vmax is None):
        if mask_missing:
            vmax = np.nanmax(np.abs(img))
            if not np.isfinite(vmax):
                vmax = 1.0
            vmin = -vmax
        else:
            vmax = max(abs(img.min()), abs(img.max()))
            vmin = -vmax

    if mask_missing:
        cmap_obj = plt.cm.get_cmap("RdBu_r").copy()
        cmap_obj.set_bad(color="#B0B0B0")
        im = ax.imshow(img, cmap=cmap_obj, origin="upper", aspect="equal", vmin=vmin, vmax=vmax, norm=norm)
    else:
        im = ax.imshow(img, cmap="RdBu_r", origin="upper", aspect="equal", vmin=vmin, vmax=vmax, norm=norm)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Score Difference from Baseline", rotation=270, labelpad=20)
    ax.set_xlabel("j (end of duplication range)", fontsize=12)
    ax.set_ylabel("i (start of duplication range)", fontsize=12)
    ax.set_title(title, fontsize=14)

    tick_interval = max(1, num_layers // 10)
    ticks = range(0, num_layers + 1, tick_interval)
    ax.set_xticks(list(ticks))
    ax.set_yticks(list(ticks))

    if show_baseline_marker:
        ax.plot(0, 0, "o", markersize=12, markeredgecolor="black", markerfacecolor="none", markeredgewidth=2, label=baseline_label)
    if show_best_marker and best_key != (0, 0):
        ax.plot(best_key[1], best_key[0], "o", markersize=12, markeredgecolor="lime", markerfacecolor="none", markeredgewidth=2, label=best_label)
    if show_baseline_marker or (show_best_marker and best_key != (0, 0)):
        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved difference heatmap to {output_path}")


def print_summary(results: dict[tuple[int, int], float], name: str) -> None:
    """Print summary statistics for a result table."""
    print(f"\n=== {name} Summary ===")
    baseline = results.get((0, 0), 0.0)
    print(f"Baseline score (0,0): {baseline:.4f}")

    scores = list(results.values())
    print(f"Score range: {min(scores):.4f} - {max(scores):.4f}")
    print(f"Mean score: {np.mean(scores):.4f}")
    print(f"Std dev: {np.std(scores):.4f}")

    best_key = max(results.keys(), key=lambda k: results[k])
    print(f"Best config: {best_key} with score {results[best_key]:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render score and delta heatmaps from an `(i, j)` results pickle.")
    parser.add_argument("--results-file", required=True, help="Path to a pickle mapping `(i, j)` to scores.")
    parser.add_argument("--output-dir", required=True, help="Directory to write the heatmaps into.")
    parser.add_argument("--num-layers", type=int, required=True, help="Base model layer count.")
    parser.add_argument("--title-prefix", default="Relayer scan", help="Plot title prefix.")
    parser.add_argument("--mask-missing", action="store_true", help="Show missing cells as gray rather than dimmed.")
    parser.add_argument("--hide-baseline-marker", action="store_true", help="Do not draw the baseline marker.")
    parser.add_argument("--hide-best-marker", action="store_true", help="Do not draw the best-config marker.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(args.results_file)
    print_summary(results, args.title_prefix)

    generate_heatmap(
        results,
        f"{args.title_prefix} - Scores",
        output_dir / "heatmap_scores.png",
        args.num_layers,
        mask_missing=args.mask_missing,
        show_baseline_marker=not args.hide_baseline_marker,
        show_best_marker=not args.hide_best_marker,
    )
    generate_difference_heatmap(
        results,
        f"{args.title_prefix} - Difference from Baseline",
        output_dir / "heatmap_diff.png",
        args.num_layers,
        mask_missing=args.mask_missing,
        show_baseline_marker=not args.hide_baseline_marker,
        show_best_marker=not args.hide_best_marker,
    )


if __name__ == "__main__":
    main()
