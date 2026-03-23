"""Shared utilities for surrogate-guided relayer candidate search."""

from __future__ import annotations

import math
from typing import Iterable


def extract_score(raw: object) -> float | None:
    """Extract a numeric score from scalar/dict result payloads."""
    if isinstance(raw, dict):
        if "score" in raw:
            raw = raw["score"]
        elif "math_score" in raw:
            raw = raw["math_score"]
        elif "eq_score" in raw:
            raw = raw["eq_score"]
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def key_to_count_vector(layer_key: Iterable[int], num_layers: int) -> list[int]:
    """Convert an expanded layer execution key into per-layer counts."""
    counts = [0] * num_layers
    for raw_idx in layer_key:
        idx = int(raw_idx)
        if idx < 0 or idx >= num_layers:
            raise ValueError(f"Layer index {idx} outside [0,{num_layers}).")
        counts[idx] += 1
    return counts


def count_vector_to_layers(counts: Iterable[int], num_layers: int) -> list[int]:
    """Decode per-layer counts into canonical expanded layer order."""
    values = [int(v) for v in counts]
    if len(values) != num_layers:
        raise ValueError(f"Expected {num_layers} counts, got {len(values)}.")
    layers: list[int] = []
    for idx, count in enumerate(values):
        if count < 0:
            raise ValueError(f"Count for layer {idx} must be >= 0, got {count}.")
        layers.extend([idx] * count)
    return layers


def counts_to_csv(counts: Iterable[int]) -> str:
    """Serialize counts as comma-separated integers."""
    return ",".join(str(int(v)) for v in counts)


def counts_from_csv(raw: str, *, expected_len: int | None = None) -> list[int]:
    """Parse comma-separated count vector text."""
    text = str(raw).strip()
    if not text:
        raise ValueError("Empty counts CSV string.")
    parts = [p.strip() for p in text.split(",")]
    if not parts:
        raise ValueError("Counts CSV has no entries.")
    values = [int(p) for p in parts]
    if expected_len is not None and len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} counts, got {len(values)}.")
    return values


def relative_overhead_from_counts(counts: Iterable[int], num_layers: int) -> float:
    """Compute relative compute overhead implied by count vector."""
    values = [int(v) for v in counts]
    if len(values) != num_layers:
        raise ValueError(f"Expected {num_layers} counts, got {len(values)}.")
    extra_layers = sum(values) - num_layers
    return float(extra_layers / float(num_layers))


def stable_quantile_bins(values: list[float], bins: int) -> list[int]:
    """Assign values to stable quantile bins with duplicate-edge handling."""
    if bins < 1:
        raise ValueError("bins must be >= 1")
    if not values:
        return []
    sorted_idx = sorted(range(len(values)), key=lambda i: values[i])
    out = [0] * len(values)
    for rank, idx in enumerate(sorted_idx):
        frac = (rank + 0.5) / float(len(values))
        bucket = min(bins - 1, int(math.floor(frac * bins)))
        out[idx] = bucket
    return out

