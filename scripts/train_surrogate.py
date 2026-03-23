#!/usr/bin/env python3
"""Train surrogate regressors for relayer ranking."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.math_eq_analysis import (
    METHOD_BALANCED,
    build_balanced_rows,
    choose_baseline,
    infer_eq_scale,
    normalize_key,
)
from src.utils.surrogate_utils import (
    extract_score,
    key_to_count_vector,
    stable_quantile_bins,
)


TIMESTAMP_PATTERNS = (
    re.compile(r"(20\d{6}_\d{6})"),
    re.compile(r"(20\d{6}_\d{4})"),
    re.compile(r"(20\d{6})"),
)


@dataclass(frozen=True)
class ScoreRecord:
    score: float
    source: str
    source_path: str
    priority: tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train surrogate ranking model for relayer configs.")
    parser.add_argument(
        "--beam-math-results",
        default="results/beam-search/beam_math_results.pkl",
    )
    parser.add_argument(
        "--beam-eq-results",
        default="results/beam-search/beam_eq_results.pkl",
    )
    parser.add_argument("--single-block-math-results", default="results/math_results.pkl")
    parser.add_argument("--single-block-eq-results", default="results/eq_results.pkl")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-layers", type=int, default=64)
    parser.add_argument("--holdout-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gate-spearman", type=float, default=0.60)
    parser.add_argument("--eq-scale-policy", choices=["auto_to_unit", "none"], default="auto_to_unit")
    parser.add_argument(
        "--baseline-policy",
        choices=["canonical_or_proxy", "canonical_only"],
        default="canonical_or_proxy",
    )
    parser.add_argument("--max-retries", type=int, default=3, help="Additional hyperparameter retries after trial 1.")
    parser.add_argument(
        "--include-beam",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include beam results in training merge.",
    )
    parser.add_argument(
        "--include-single-block",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include root single-block scores in training merge.",
    )
    return parser.parse_args()


def _extract_timestamp_code(path: Path) -> int:
    text = str(path)
    for pattern in TIMESTAMP_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        raw = match.group(1).replace("_", "")
        try:
            return int(raw)
        except ValueError:
            continue
    return 0


def _source_rank(source: str) -> int:
    # Prefer calibrated beam signals over single-block fallback.
    if source == "beam":
        return 3
    if source == "single_block":
        return 2
    return 1


def load_score_records(
    path: Path,
    *,
    source: str,
    num_layers: int,
    prefer_legacy_ij: bool = True,
) -> dict[tuple[int, ...], ScoreRecord]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {path}, got {type(payload).__name__}")

    ts_code = _extract_timestamp_code(path)
    mtime_code = int(path.stat().st_mtime)
    merged: dict[tuple[int, ...], ScoreRecord] = {}
    for raw_key, raw_val in payload.items():
        key = normalize_key(
            raw_key,
            num_layers=num_layers,
            prefer_legacy_ij=prefer_legacy_ij,
        )
        if key is None:
            continue
        score = extract_score(raw_val)
        if score is None:
            continue
        priority = (_source_rank(source), ts_code, mtime_code)
        rec = ScoreRecord(
            score=float(score),
            source=source,
            source_path=str(path),
            priority=priority,
        )
        prev = merged.get(key)
        if prev is None or rec.priority > prev.priority:
            merged[key] = rec
    return merged


def merge_record_maps(
    maps: list[dict[tuple[int, ...], ScoreRecord]],
) -> dict[tuple[int, ...], ScoreRecord]:
    out: dict[tuple[int, ...], ScoreRecord] = {}
    for mp in maps:
        for key, rec in mp.items():
            prev = out.get(key)
            if prev is None or rec.priority > prev.priority:
                out[key] = rec
    return out


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(values.shape[0], dtype=float)
    return ranks


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[0] != b.shape[0]:
        raise ValueError("spearman_corr requires arrays with same length.")
    if a.shape[0] < 2:
        return 0.0
    ra = _rankdata(a)
    rb = _rankdata(b)
    da = ra - ra.mean()
    db = rb - rb.mean()
    denom = float(np.sqrt((da * da).sum()) * np.sqrt((db * db).sum()))
    if denom < 1e-12:
        return 0.0
    return float((da * db).sum() / denom)


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def stratified_holdout_indices(y: np.ndarray, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n = int(y.shape[0])
    if n < 3:
        idx = np.arange(n, dtype=int)
        return idx[: max(1, n - 1)], idx[max(1, n - 1) :]

    target = max(1, int(round(n * frac)))
    target = min(target, n - 1)

    bins_count = min(10, max(2, n // 20))
    bins = stable_quantile_bins([float(v) for v in y], bins=bins_count)
    groups: dict[int, list[int]] = {}
    for idx, b in enumerate(bins):
        groups.setdefault(int(b), []).append(idx)

    rng = np.random.default_rng(seed)
    for g in groups.values():
        rng.shuffle(g)

    allocations: dict[int, int] = {}
    leftovers: list[tuple[float, int]] = []
    used = 0
    for b, g in groups.items():
        raw = frac * len(g)
        base = int(np.floor(raw))
        max_take = max(0, len(g) - 1)
        base = min(base, max_take)
        allocations[b] = base
        used += base
        leftovers.append((raw - base, b))

    remaining = target - used
    leftovers.sort(reverse=True)
    while remaining > 0:
        changed = False
        for _, b in leftovers:
            g = groups[b]
            if allocations[b] < max(0, len(g) - 1):
                allocations[b] += 1
                remaining -= 1
                changed = True
                if remaining <= 0:
                    break
        if not changed:
            break

    holdout: list[int] = []
    for b, g in groups.items():
        holdout.extend(g[: allocations[b]])

    if not holdout:
        largest = max(groups.values(), key=len)
        holdout.append(largest[0])

    holdout_idx = np.array(sorted(set(holdout)), dtype=int)
    mask = np.ones(n, dtype=bool)
    mask[holdout_idx] = False
    train_idx = np.arange(n, dtype=int)[mask]
    if train_idx.size == 0:
        train_idx = holdout_idx[:-1]
        holdout_idx = holdout_idx[-1:]
    return train_idx, holdout_idx


def _trial_params(base_seed: int, retries: int) -> list[dict[str, Any]]:
    presets: list[dict[str, Any]] = [
        {
            "num_boost_round": 500,
            "max_depth": 6,
            "eta": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },
        {
            "num_boost_round": 700,
            "max_depth": 4,
            "eta": 0.08,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
        },
        {
            "num_boost_round": 900,
            "max_depth": 8,
            "eta": 0.03,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
        },
        {
            "num_boost_round": 350,
            "max_depth": 6,
            "eta": 0.1,
            "subsample": 0.95,
            "colsample_bytree": 0.95,
        },
    ]
    count = min(len(presets), 1 + max(0, retries))
    out: list[dict[str, Any]] = []
    for idx in range(count):
        p = dict(presets[idx])
        p.update(
            {
                "objective": "reg:squarederror",
                "seed": base_seed + idx,
                "nthread": -1,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "eval_metric": "rmse",
            }
        )
        out.append(p)
    return out


def _fit_model(X: np.ndarray, y: np.ndarray, params: dict[str, Any]) -> xgb.Booster:
    dtrain = xgb.DMatrix(X, label=y)
    train_params = {k: v for k, v in params.items() if k != "num_boost_round"}
    rounds = int(params.get("num_boost_round", 300))
    model = xgb.train(params=train_params, dtrain=dtrain, num_boost_round=rounds)
    return model


def write_feature_importance(path: Path, model: xgb.Booster, num_layers: int) -> None:
    gain = model.get_score(importance_type="gain")
    cover = model.get_score(importance_type="cover")
    weight = model.get_score(importance_type="weight")
    rows: list[dict[str, Any]] = []
    for idx in range(num_layers):
        fk = f"f{idx}"
        rows.append(
            {
                "feature": fk,
                "layer_idx": idx,
                "gain": float(gain.get(fk, 0.0)),
                "cover": float(cover.get(fk, 0.0)),
                "weight": float(weight.get(fk, 0.0)),
            }
        )
    rows.sort(key=lambda r: r["gain"], reverse=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "layer_idx", "gain", "cover", "weight"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not args.include_beam and not args.include_single_block:
        raise ValueError("At least one source family must be enabled.")
    if args.holdout_frac <= 0.0 or args.holdout_frac >= 0.5:
        raise ValueError("--holdout-frac must be in (0, 0.5).")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    math_maps: list[dict[tuple[int, ...], ScoreRecord]] = []
    eq_maps: list[dict[tuple[int, ...], ScoreRecord]] = []
    source_files: list[str] = []

    if args.include_beam:
        beam_math = Path(args.beam_math_results)
        beam_eq = Path(args.beam_eq_results)
        math_maps.append(load_score_records(beam_math, source="beam", num_layers=args.num_layers))
        eq_maps.append(load_score_records(beam_eq, source="beam", num_layers=args.num_layers))
        source_files.extend([str(beam_math), str(beam_eq)])
    if args.include_single_block:
        sb_math = Path(args.single_block_math_results)
        sb_eq = Path(args.single_block_eq_results)
        math_maps.append(load_score_records(sb_math, source="single_block", num_layers=args.num_layers))
        eq_maps.append(load_score_records(sb_eq, source="single_block", num_layers=args.num_layers))
        source_files.extend([str(sb_math), str(sb_eq)])

    merged_math = merge_record_maps(math_maps)
    merged_eq = merge_record_maps(eq_maps)
    math_scores = {k: rec.score for k, rec in merged_math.items()}
    eq_raw = {k: rec.score for k, rec in merged_eq.items()}
    eq_scores, eq_scale = infer_eq_scale(eq_raw, policy=args.eq_scale_policy)

    common_keys = set(math_scores) & set(eq_scores)
    if not common_keys:
        raise RuntimeError("No common keys between merged math and EQ sources.")

    baseline_key, baseline_source, baseline_warnings = choose_baseline(
        common_keys,
        args.baseline_policy,
        num_layers=args.num_layers,
    )
    rows, meta, row_warnings = build_balanced_rows(math_scores, eq_scores, baseline_key)

    # Add explicit baseline row for model anchoring.
    rows_with_baseline: list[dict[str, Any]] = list(rows)
    rows_with_baseline.append(
        {
            "key": baseline_key,
            "math_score": float(meta["baseline_math"]),
            "eq_score": float(meta["baseline_eq"]),
            "math_delta": 0.0,
            "eq_delta": 0.0,
            METHOD_BALANCED: 0.0,
        }
    )

    keys: list[tuple[int, ...]] = [tuple(int(x) for x in r["key"]) for r in rows_with_baseline]
    X = np.array([key_to_count_vector(k, args.num_layers) for k in keys], dtype=np.float32)
    y_method = np.array([float(r[METHOD_BALANCED]) for r in rows_with_baseline], dtype=np.float32)
    y_math_delta = np.array([float(r["math_delta"]) for r in rows_with_baseline], dtype=np.float32)
    y_eq_delta = np.array([float(r["eq_delta"]) for r in rows_with_baseline], dtype=np.float32)

    train_idx, holdout_idx = stratified_holdout_indices(y_method, args.holdout_frac, args.seed)
    X_train = X[train_idx]
    X_hold = X[holdout_idx]

    trials = _trial_params(args.seed, args.max_retries)
    trial_reports: list[dict[str, Any]] = []
    best_trial: dict[str, Any] | None = None

    for idx, params in enumerate(trials, start=1):
        model_method = _fit_model(X_train, y_method[train_idx], params)
        model_math = _fit_model(X_train, y_math_delta[train_idx], params)
        model_eq = _fit_model(X_train, y_eq_delta[train_idx], params)

        dhold = xgb.DMatrix(X_hold)
        pred_method = model_method.predict(dhold).astype(np.float32)
        pred_math = model_math.predict(dhold).astype(np.float32)
        pred_eq = model_eq.predict(dhold).astype(np.float32)

        report = {
            "trial": idx,
            "params": params,
            "spearman_method_b": spearman_corr(y_method[holdout_idx], pred_method),
            "mae_method_b": mae(y_method[holdout_idx], pred_method),
            "spearman_math_delta": spearman_corr(y_math_delta[holdout_idx], pred_math),
            "mae_math_delta": mae(y_math_delta[holdout_idx], pred_math),
            "spearman_eq_delta": spearman_corr(y_eq_delta[holdout_idx], pred_eq),
            "mae_eq_delta": mae(y_eq_delta[holdout_idx], pred_eq),
            "model_method": model_method,
            "model_math": model_math,
            "model_eq": model_eq,
            "pred_method": pred_method,
            "pred_math": pred_math,
            "pred_eq": pred_eq,
        }
        trial_reports.append(report)
        if best_trial is None or report["spearman_method_b"] > float(best_trial["spearman_method_b"]):
            best_trial = report

    assert best_trial is not None

    # Persist best models.
    model_method_path = out_dir / "model_method_b.json"
    model_math_path = out_dir / "model_math_delta.json"
    model_eq_path = out_dir / "model_eq_delta.json"
    best_trial["model_method"].save_model(str(model_method_path))
    best_trial["model_math"].save_model(str(model_math_path))
    best_trial["model_eq"].save_model(str(model_eq_path))

    # Diagnostics.
    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_layers": args.num_layers,
        "holdout_frac": args.holdout_frac,
        "seed": args.seed,
        "gate_spearman": args.gate_spearman,
        "gate_passed": bool(best_trial["spearman_method_b"] >= args.gate_spearman),
        "best_trial": int(best_trial["trial"]),
        "train_rows": int(train_idx.size),
        "holdout_rows": int(holdout_idx.size),
        "rows_total": int(X.shape[0]),
        "sources": source_files,
        "baseline_key": list(baseline_key),
        "baseline_source": baseline_source,
        "baseline_math": float(meta["baseline_math"]),
        "baseline_eq": float(meta["baseline_eq"]),
        "eq_scale_applied": bool(eq_scale.applied),
        "eq_scale_factor": float(eq_scale.factor),
        "eq_scale_reason": eq_scale.reason,
        "baseline_warnings": baseline_warnings,
        "row_warnings": row_warnings,
        "trial_metrics": [
            {
                k: v
                for k, v in trial.items()
                if k
                not in {
                    "model_method",
                    "model_math",
                    "model_eq",
                    "pred_method",
                    "pred_math",
                    "pred_eq",
                }
            }
            for trial in trial_reports
        ],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Pred-vs-true for holdout.
    with (out_dir / "pred_vs_true_holdout.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "row_idx",
            "layer_key",
            "split",
            "y_method_b",
            "pred_method_b",
            "y_math_delta",
            "pred_math_delta",
            "y_eq_delta",
            "pred_eq_delta",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        hold_set = set(int(i) for i in holdout_idx.tolist())
        dall = xgb.DMatrix(X)
        pred_method_all = best_trial["model_method"].predict(dall).astype(np.float32)
        pred_math_all = best_trial["model_math"].predict(dall).astype(np.float32)
        pred_eq_all = best_trial["model_eq"].predict(dall).astype(np.float32)
        for idx, key in enumerate(keys):
            writer.writerow(
                {
                    "row_idx": idx,
                    "layer_key": str(key),
                    "split": "holdout" if idx in hold_set else "train",
                    "y_method_b": float(y_method[idx]),
                    "pred_method_b": float(pred_method_all[idx]),
                    "y_math_delta": float(y_math_delta[idx]),
                    "pred_math_delta": float(pred_math_all[idx]),
                    "y_eq_delta": float(y_eq_delta[idx]),
                    "pred_eq_delta": float(pred_eq_all[idx]),
                }
            )

    # Training table with provenance.
    with (out_dir / "training_table.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "layer_key",
            "math_score",
            "eq_score",
            "math_delta",
            "eq_delta",
            METHOD_BALANCED,
            "math_source",
            "eq_source",
            "math_source_path",
            "eq_source_path",
            "is_single_block",
            "extra_layers",
            "relative_overhead",
            "counts_csv",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        row_map = {tuple(int(x) for x in r["key"]): r for r in rows_with_baseline}
        for key in keys:
            row = row_map[key]
            counts = key_to_count_vector(key, args.num_layers)
            math_rec = merged_math.get(key)
            eq_rec = merged_eq.get(key)
            writer.writerow(
                {
                    "layer_key": str(key),
                    "math_score": float(row["math_score"]),
                    "eq_score": float(row["eq_score"]),
                    "math_delta": float(row["math_delta"]),
                    "eq_delta": float(row["eq_delta"]),
                    METHOD_BALANCED: float(row[METHOD_BALANCED]),
                    "math_source": "" if math_rec is None else math_rec.source,
                    "eq_source": "" if eq_rec is None else eq_rec.source,
                    "math_source_path": "" if math_rec is None else math_rec.source_path,
                    "eq_source_path": "" if eq_rec is None else eq_rec.source_path,
                    "is_single_block": bool(key == tuple(range(args.num_layers))),
                    "extra_layers": int(len(key) - args.num_layers),
                    "relative_overhead": float((len(key) - args.num_layers) / float(args.num_layers)),
                    "counts_csv": ",".join(str(v) for v in counts),
                }
            )

    np.savez_compressed(
        out_dir / "feature_matrix.npz",
        X=X.astype(np.int16),
        y_method=y_method,
        y_math_delta=y_math_delta,
        y_eq_delta=y_eq_delta,
        keys=np.array([",".join(str(x) for x in key) for key in keys], dtype=object),
        train_idx=train_idx.astype(np.int32),
        holdout_idx=holdout_idx.astype(np.int32),
    )

    label_summary = {
        "rows": int(X.shape[0]),
        "method_mean": float(np.mean(y_method)),
        "method_std": float(np.std(y_method)),
        "math_delta_mean": float(np.mean(y_math_delta)),
        "math_delta_std": float(np.std(y_math_delta)),
        "eq_delta_mean": float(np.mean(y_eq_delta)),
        "eq_delta_std": float(np.std(y_eq_delta)),
    }
    (out_dir / "label_summary.json").write_text(json.dumps(label_summary, indent=2), encoding="utf-8")
    write_feature_importance(out_dir / "feature_importance.csv", best_trial["model_method"], args.num_layers)

    if not metrics["gate_passed"]:
        report = (
            "# Surrogate validation gate failed\n\n"
            f"- Gate threshold (Spearman): `{args.gate_spearman:.4f}`\n"
            f"- Best trial: `{best_trial['trial']}`\n"
            f"- Best holdout Spearman: `{best_trial['spearman_method_b']:.6f}`\n\n"
            "See `metrics.json` and `pred_vs_true_holdout.csv` for diagnostics.\n"
        )
        (out_dir / "gate_fail_report.md").write_text(report, encoding="utf-8")
        raise SystemExit(2)

    print(f"Wrote: {out_dir / 'metrics.json'}")
    print(f"Wrote: {model_method_path}")
    print(f"Wrote: {model_math_path}")
    print(f"Wrote: {model_eq_path}")
    print(f"Gate passed: spearman={best_trial['spearman_method_b']:.6f}")


if __name__ == "__main__":
    main()
