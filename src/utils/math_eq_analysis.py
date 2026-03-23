"""Shared math+EQ analysis helpers for balanced z-delta ranking."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from src.core.layer_config import (
    layer_spec_string,
    legacy_key_to_ij,
    normalize_to_layers,
    parse_layer_list_string,
)


METHOD_BALANCED = "balanced_z_delta"


@dataclass
class EqScaleInfo:
    factor: float
    reason: str
    applied: bool


def normalize_key(
    raw_key: Any,
    *,
    num_layers: int | None = None,
    prefer_legacy_ij: bool = True,
) -> tuple[int, ...] | None:
    # Optional legacy `(i,j)` parsing path first.
    if prefer_legacy_ij and num_layers is not None:
        legacy = legacy_key_to_ij(raw_key)
        if legacy is not None:
            i, j = legacy
            if (i, j) == (0, 0):
                return tuple(range(num_layers))
            if 0 <= i < j <= num_layers:
                return tuple(list(range(0, j)) + list(range(i, num_layers)))

    # Canonical explicit layer lists / spec strings.
    try:
        layers = normalize_to_layers(num_layers, raw_key) if num_layers is not None else None
    except Exception:
        layers = None
    if layers is not None:
        return tuple(int(x) for x in layers)

    # Fallback for already-canonical tuple/list keys when num_layers is unknown.
    if isinstance(raw_key, (tuple, list)) and raw_key:
        try:
            return tuple(int(x) for x in raw_key)
        except (TypeError, ValueError):
            return None
    if isinstance(raw_key, str):
        raw = raw_key.strip()
        legacy = legacy_key_to_ij(raw)
        if legacy is not None:
            return tuple(legacy)
        if raw.lower().startswith("layers:"):
            try:
                return tuple(parse_layer_list_string(raw))
            except Exception:
                return None
    return None


def load_scores(
    path: Path,
    *,
    num_layers: int | None = None,
    prefer_legacy_ij: bool = True,
) -> dict[tuple[int, ...], float]:
    with path.open("rb") as f:
        data = pickle.load(f)

    out: dict[tuple[int, ...], float] = {}
    for raw_key, raw_val in data.items():
        key = normalize_key(
            raw_key,
            num_layers=num_layers,
            prefer_legacy_ij=prefer_legacy_ij,
        )
        if key is None:
            continue
        if isinstance(raw_val, dict) and "score" in raw_val:
            score = raw_val["score"]
        else:
            score = raw_val
        try:
            out[key] = float(score)
        except (TypeError, ValueError):
            continue
    return out


def infer_eq_scale(
    eq_scores: dict[tuple[int, ...], float],
    policy: str,
) -> tuple[dict[tuple[int, ...], float], EqScaleInfo]:
    if policy == "none":
        return eq_scores, EqScaleInfo(factor=1.0, reason="disabled", applied=False)
    if not eq_scores:
        return eq_scores, EqScaleInfo(factor=1.0, reason="empty_scores", applied=False)

    vals = np.array(list(eq_scores.values()), dtype=float)
    baseline = eq_scores.get((0, 0))
    if baseline is None:
        canonical_candidates = [k for k in eq_scores if tuple(range(len(k))) == k]
        if canonical_candidates:
            # Prefer the longest canonical sequence as baseline guess.
            best_key = max(canonical_candidates, key=len)
            baseline = eq_scores.get(best_key)
    p95 = float(np.percentile(vals, 95))
    if (baseline is not None and baseline > 1.5) or p95 > 1.5:
        scaled = {k: v * 0.01 for k, v in eq_scores.items()}
        reason = (
            f"converted_from_percent_scale baseline={baseline:.4f} p95={p95:.4f}"
            if baseline is not None
            else f"converted_from_percent_scale p95={p95:.4f}"
        )
        return scaled, EqScaleInfo(factor=0.01, reason=reason, applied=True)

    return eq_scores, EqScaleInfo(
        factor=1.0,
        reason=f"already_unit_scale p95={p95:.4f}",
        applied=False,
    )


def choose_baseline(
    common_keys: set[tuple[int, ...]],
    policy: str,
    *,
    num_layers: int | None = None,
) -> tuple[tuple[int, ...], str, list[str]]:
    warnings: list[str] = []
    if num_layers is not None:
        canonical = tuple(range(num_layers))
        if canonical in common_keys:
            return canonical, "canonical", warnings

    # Backward compatibility with raw `(0,0)` keys if present.
    if (0, 0) in common_keys:
        return (0, 0), "legacy_00", warnings

    if policy == "canonical_only":
        raise ValueError("Missing canonical baseline layer sequence in common keys.")

    # Fallback heuristic: prefer the shortest sequence, then lexicographic.
    proxy = min(common_keys, key=lambda k: (len(k), k))
    warnings.append(f"Missing canonical baseline; using fallback proxy baseline {layer_spec_string(proxy)}.")
    return proxy, "fallback_proxy", warnings


def safe_z(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(values.mean()) if values.size else 0.0
    std = float(values.std()) if values.size else 1.0
    if std < 1e-12:
        std = 1.0
    return (values - mean) / std, mean, std


def build_balanced_rows(
    math_scores: dict[tuple[int, ...], float],
    eq_scores: dict[tuple[int, ...], float],
    baseline_key: tuple[int, ...],
) -> tuple[list[dict[str, Any]], dict[str, float], list[str]]:
    warnings: list[str] = []
    common = sorted(set(math_scores.keys()) & set(eq_scores.keys()))
    baseline_math = math_scores[baseline_key]
    baseline_eq = eq_scores[baseline_key]

    rows: list[dict[str, Any]] = []
    for key in common:
        if key == baseline_key:
            continue
        math_score = math_scores[key]
        eq_score = eq_scores[key]
        rows.append(
            {
                "key": key,
                "math_score": math_score,
                "eq_score": eq_score,
                "math_delta": math_score - baseline_math,
                "eq_delta": eq_score - baseline_eq,
            }
        )

    math_delta_arr = np.array([row["math_delta"] for row in rows], dtype=float)
    eq_delta_arr = np.array([row["eq_delta"] for row in rows], dtype=float)

    if rows:
        z_math, math_mean, math_std = safe_z(math_delta_arr)
        z_eq, eq_mean, eq_std = safe_z(eq_delta_arr)
        if math_std == 1.0 and np.std(math_delta_arr) < 1e-12:
            warnings.append("Math deltas have near-zero variance; using std=1.0 for z-score.")
        if eq_std == 1.0 and np.std(eq_delta_arr) < 1e-12:
            warnings.append("EQ deltas have near-zero variance; using std=1.0 for z-score.")
    else:
        z_math = np.array([], dtype=float)
        z_eq = np.array([], dtype=float)
        math_mean = 0.0
        math_std = 1.0
        eq_mean = 0.0
        eq_std = 1.0

    for idx, row in enumerate(rows):
        row[METHOD_BALANCED] = float(z_math[idx] + z_eq[idx])

    metadata = {
        "baseline_math": baseline_math,
        "baseline_eq": baseline_eq,
        "math_delta_mean": math_mean,
        "math_delta_std": math_std,
        "eq_delta_mean": eq_mean,
        "eq_delta_std": eq_std,
    }
    return rows, metadata, warnings


def rank_balanced_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (row[METHOD_BALANCED], row["eq_delta"], row["math_delta"]),
        reverse=True,
    )


def relayer_string(i: int, j: int, num_layers: int) -> str:
    if (i, j) == (0, 0):
        return "baseline"
    return f"[[0-{j}]-[{i + 1}-{num_layers}]]"
