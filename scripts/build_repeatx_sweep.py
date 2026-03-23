#!/usr/bin/env python3
"""Build baseline + per-layer repeat sweep configs and manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.layer_config import (
    expand_multi_block_config,
    parse_blocks_string,
    parse_layer_list_string,
    validate_layers,
)

def build_layer_indices(base_layers: list[int], layer: int | None, extra_repeats: int) -> list[int]:
    out = list(base_layers)
    if layer is None or extra_repeats <= 0:
        return out
    if layer not in out:
        return out
    # Insert repeats after the right-most occurrence to make behavior deterministic
    # when the base config already contains duplicated layers.
    insert_at = len(out) - 1 - out[::-1].index(layer) + 1
    return out[:insert_at] + [layer] * extra_repeats + out[insert_at:]


def load_base_layers(args: argparse.Namespace) -> tuple[list[int], str]:
    if args.base_spec and args.base_layers_file:
        raise ValueError("Use at most one of --base-spec and --base-layers-file.")

    if args.base_spec:
        raw = args.base_spec.strip()
        if raw in {"0,0", "(0,0)", "(0, 0)"}:
            layers = list(range(args.num_layers))
            return layers, "baseline:(0,0)"
        blocks = parse_blocks_string(raw)
        layers = expand_multi_block_config(args.num_layers, blocks)
        validate_layers(args.num_layers, layers)
        return layers, f"blocks:{raw}"

    if args.base_layers_file:
        path = Path(args.base_layers_file)
        if not path.exists():
            raise FileNotFoundError(f"--base-layers-file not found: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            row = line.strip()
            if not row or row.startswith("#"):
                continue
            layers = parse_layer_list_string(row)
            validate_layers(args.num_layers, layers)
            return layers, f"layers-file:{path}"
        raise ValueError(f"No usable layer line found in --base-layers-file: {path}")

    layers = list(range(args.num_layers))
    return layers, "baseline:range"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create repeat sweep config: baseline + each layer repeated 1..N extra times."
    )
    parser.add_argument("--num-layers", type=int, required=True, help="Number of transformer layers.")
    parser.add_argument(
        "--max-extra-repeats",
        type=int,
        default=8,
        help="Max extra repeats per layer (default: 8).",
    )
    parser.add_argument("--config-out", required=True, help="Output config file path.")
    parser.add_argument("--manifest-out", required=True, help="Output manifest JSON path.")
    parser.add_argument("--title", default="repeat-layer sweep", help="Manifest title/label.")
    parser.add_argument(
        "--base-spec",
        default=None,
        help=(
            "Optional base block spec to apply repeats on top of "
            "(e.g. '24,31;8,15;51,52')."
        ),
    )
    parser.add_argument(
        "--base-layers-file",
        default=None,
        help="Optional file containing one canonical layer line ('layers:...') for the base config.",
    )
    args = parser.parse_args()

    if args.num_layers < 1:
        raise ValueError("--num-layers must be >= 1")
    if args.max_extra_repeats < 1:
        raise ValueError("--max-extra-repeats must be >= 1")

    config_out = Path(args.config_out)
    manifest_out = Path(args.manifest_out)
    config_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    base_layers, base_source = load_base_layers(args)
    entries: list[dict] = []
    lines: list[str] = []

    # Baseline
    baseline_layers = build_layer_indices(base_layers, None, 0)
    baseline_key = ",".join(str(x) for x in baseline_layers)
    entries.append(
        {
            "idx": 0,
            "name": "baseline",
            "layer": None,
            "extra_repeats": 0,
            "total_occurrences": 1,
            "layer_indices_key": baseline_key,
        }
    )
    lines.append(f"layers:{baseline_key}")

    idx = 1
    for layer in range(args.num_layers):
        if layer not in base_layers:
            continue
        for extra in range(1, args.max_extra_repeats + 1):
            layer_indices = build_layer_indices(base_layers, layer, extra)
            key = ",".join(str(x) for x in layer_indices)
            entries.append(
                {
                    "idx": idx,
                    "name": f"layer{layer:02d}_x{extra + 1}",
                    "layer": layer,
                    "extra_repeats": extra,
                    "total_occurrences": extra + 1,
                    "layer_indices_key": key,
                }
            )
            lines.append(f"layers:{key}")
            idx += 1

    with config_out.open("w") as f:
        f.write(f"# {args.title}\n")
        f.write(
            f"# base + each layer repeated with 1..{args.max_extra_repeats} extra copies\n"
        )
        f.write(f"# base_source: {base_source}\n")
        for line in lines:
            f.write(line + "\n")

    manifest = {
        "num_layers": args.num_layers,
        "num_configs": len(entries),
        "base_source": base_source,
        "base_layers_key": baseline_key,
        "layout": f"base + per-layer extra repeats 1..{args.max_extra_repeats}",
        "entries": entries,
    }
    with manifest_out.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote config: {config_out} ({len(lines)} configs)")
    print(f"Wrote manifest: {manifest_out}")


if __name__ == "__main__":
    main()
