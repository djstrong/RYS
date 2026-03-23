"""Canonical layer-configuration helpers.

This module defines a canonical representation for relayer configs:
`layer_indices` (a list/tuple of layer ids used for the forward pass).

Legacy `(i, j)` configs are supported as shorthand and can be converted into
canonical layer indices for backward compatibility.
"""

from __future__ import annotations

import re
from typing import Any


def baseline_layers(num_layers: int) -> list[int]:
    """Return the baseline (unmodified) layer order."""
    return list(range(num_layers))


def is_baseline_layers(layer_indices: list[int] | tuple[int, ...], num_layers: int) -> bool:
    """True when `layer_indices` matches the original model layer order."""
    if len(layer_indices) != num_layers:
        return False
    return all(idx == pos for pos, idx in enumerate(layer_indices))


def validate_layers(
    num_layers: int,
    layer_indices: list[int] | tuple[int, ...],
    *,
    allow_empty: bool = False,
) -> None:
    """Validate canonical layer indices."""
    if not allow_empty and len(layer_indices) == 0:
        raise ValueError("Layer list must be non-empty.")
    for idx in layer_indices:
        if idx < 0 or idx >= num_layers:
            raise ValueError(f"Layer index {idx} out of range [0, {num_layers}).")


def layer_key(layer_indices: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    """Canonical hashable key for result dicts."""
    return tuple(int(x) for x in layer_indices)


def layer_spec_string(layer_indices: list[int] | tuple[int, ...]) -> str:
    """Serialize layer list as text."""
    return "layers:" + ",".join(str(int(x)) for x in layer_indices)


def parse_layer_list_string(layers_str: str) -> list[int]:
    """Parse `layers:0,1,2` or `0,1,2` into a list of layer indices."""
    raw = layers_str.strip()
    if raw.lower().startswith("layers:"):
        raw = raw.split(":", 1)[1].strip()
    if not raw:
        raise ValueError("Empty layer list string.")
    items = [part.strip() for part in raw.split(",") if part.strip()]
    if not items:
        raise ValueError("Layer list string has no values.")
    return [int(x) for x in items]


def parse_blocks_string(blocks_str: str) -> tuple[tuple[int, int], ...]:
    """Parse block string like `3,6;4,6` into `((3, 6), (4, 6))`."""
    raw = blocks_str.strip()
    if raw.lower().startswith("blocks:"):
        raw = raw.split(":", 1)[1].strip()
    if not raw:
        raise ValueError("Empty block specification.")

    blocks: list[tuple[int, int]] = []
    for pair in raw.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        if pair.startswith("(") and pair.endswith(")"):
            pair = pair[1:-1].strip()
        parts = [p.strip() for p in pair.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Invalid block format: {pair}. Expected 'start,end'.")
        blocks.append((int(parts[0]), int(parts[1])))

    if not blocks:
        raise ValueError("No blocks parsed from specification.")
    return tuple(blocks)


def validate_block(num_layers: int, block: tuple[int, int]) -> None:
    """Validate a single `(i, j)` block."""
    i, j = int(block[0]), int(block[1])
    if i == 0 and j == 0:
        return
    if i < 0 or j < 0 or i >= j or j > num_layers:
        raise ValueError(f"Invalid block {(i, j)} for {num_layers} layers.")


def expand_single_block(num_layers: int, block: tuple[int, int]) -> list[int]:
    """Expand one `(i, j)` shorthand into explicit layer indices."""
    i, j = int(block[0]), int(block[1])
    validate_block(num_layers, (i, j))
    if i == 0 and j == 0:
        return baseline_layers(num_layers)
    return list(range(0, j)) + list(range(i, num_layers))


def expand_multi_block_config(num_layers: int, blocks: tuple[tuple[int, int], ...]) -> list[int]:
    """Expand multiple `(i, j)` blocks with sequential insertion semantics."""
    if not blocks:
        return baseline_layers(num_layers)

    result = expand_single_block(num_layers, blocks[0])
    for raw_block in blocks[1:]:
        i, j = int(raw_block[0]), int(raw_block[1])
        validate_block(num_layers, (i, j))
        if i == 0 and j == 0:
            continue
        insert_layers = list(range(i, j))
        if not insert_layers:
            continue
        try:
            insert_pos = len(result) - 1 - result[::-1].index(j - 1) + 1
        except ValueError:
            continue
        result = result[:insert_pos] + insert_layers + result[insert_pos:]
    return result


def ij_to_layers(num_layers: int, i: int, j: int) -> list[int]:
    """Convert one legacy `(i, j)` key to explicit layer indices."""
    return expand_single_block(num_layers, (int(i), int(j)))


_LEGACY_KEY_RE = re.compile(r"^\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?$")


def legacy_key_to_ij(raw_key: Any) -> tuple[int, int] | None:
    """Try parsing legacy `(i, j)` key from tuple/list/string."""
    if isinstance(raw_key, tuple) and len(raw_key) == 2:
        try:
            return int(raw_key[0]), int(raw_key[1])
        except (TypeError, ValueError):
            return None
    if isinstance(raw_key, list) and len(raw_key) == 2:
        try:
            return int(raw_key[0]), int(raw_key[1])
        except (TypeError, ValueError):
            return None
    if isinstance(raw_key, str):
        m = _LEGACY_KEY_RE.match(raw_key.strip())
        if m:
            return int(m.group(1)), int(m.group(2))
    return None


def legacy_key_to_layers(num_layers: int, raw_key: Any) -> list[int] | None:
    """Convert legacy key into canonical layer list when possible."""
    ij = legacy_key_to_ij(raw_key)
    if ij is None:
        return None
    i, j = ij
    return ij_to_layers(num_layers, i, j)


def normalize_to_layers(num_layers: int, raw_spec: Any) -> list[int]:
    """Normalize layers/blocks/legacy specs into canonical layer indices."""
    if isinstance(raw_spec, dict):
        if "layers" in raw_spec:
            layers = [int(x) for x in raw_spec["layers"]]
            validate_layers(num_layers, layers)
            return layers
        if "layer_indices" in raw_spec:
            layers = [int(x) for x in raw_spec["layer_indices"]]
            validate_layers(num_layers, layers)
            return layers
        if "blocks" in raw_spec:
            blocks = parse_blocks_string(str(raw_spec["blocks"]))
            layers = expand_multi_block_config(num_layers, blocks)
            validate_layers(num_layers, layers)
            return layers
        if "key" in raw_spec:
            layers = legacy_key_to_layers(num_layers, raw_spec["key"])
            if layers is None:
                raise ValueError(f"Could not parse legacy key: {raw_spec['key']!r}")
            validate_layers(num_layers, layers)
            return layers
        if "spec" in raw_spec:
            return normalize_to_layers(num_layers, raw_spec["spec"])

    if isinstance(raw_spec, str):
        spec = raw_spec.strip()
        if spec.lower().startswith("layers:"):
            layers = parse_layer_list_string(spec)
            validate_layers(num_layers, layers)
            return layers
        if spec.lower().startswith("blocks:") or ";" in spec or "," in spec:
            # First try legacy key (single pair). If that fails, parse blocks.
            layers = legacy_key_to_layers(num_layers, spec)
            if layers is not None:
                validate_layers(num_layers, layers)
                return layers
            blocks = parse_blocks_string(spec)
            layers = expand_multi_block_config(num_layers, blocks)
            validate_layers(num_layers, layers)
            return layers
        raise ValueError(f"Unsupported config spec string: {raw_spec!r}")

    if isinstance(raw_spec, (list, tuple)):
        if len(raw_spec) == 0:
            raise ValueError("Empty sequence config.")
        # Legacy `(i, j)` shorthand.
        legacy_layers = legacy_key_to_layers(num_layers, raw_spec)
        if legacy_layers is not None:
            validate_layers(num_layers, legacy_layers)
            return legacy_layers
        # Canonical explicit layer list.
        try:
            layers = [int(x) for x in raw_spec]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid layer sequence: {raw_spec!r}") from exc
        validate_layers(num_layers, layers)
        return layers

    raise ValueError(f"Unsupported config spec type: {type(raw_spec).__name__}")


def parse_queue_entry_layers(num_layers: int, entry: dict[str, Any]) -> dict[str, Any]:
    """Parse one queue entry into canonical layer config fields.

    Returns dict with:
      - idx: int
      - layers: list[int]
      - layer_key: tuple[int, ...]
      - spec: str
      - legacy_key: tuple[int, int] | None
      - source: str
    """
    if not isinstance(entry, dict):
        raise ValueError(f"Queue entry must be a dict, got {type(entry).__name__}")

    idx = int(entry.get("idx", -1))
    source = "unknown"
    legacy_key = None

    if "layers" in entry:
        layers = [int(x) for x in entry["layers"]]
        source = "layers"
    elif "layer_indices" in entry:
        layers = [int(x) for x in entry["layer_indices"]]
        source = "layer_indices"
    elif "key" in entry:
        legacy_key = legacy_key_to_ij(entry["key"])
        if legacy_key is None:
            raise ValueError(f"Invalid legacy queue key: {entry['key']!r}")
        layers = ij_to_layers(num_layers, legacy_key[0], legacy_key[1])
        source = "legacy_key"
    elif "spec" in entry:
        layers = normalize_to_layers(num_layers, entry["spec"])
        source = "spec"
    else:
        raise ValueError("Queue entry missing layers/key/spec fields.")

    validate_layers(num_layers, layers)
    spec = str(entry.get("spec") or layer_spec_string(layers))

    return {
        "idx": idx,
        "layers": layers,
        "layer_key": layer_key(layers),
        "spec": spec,
        "legacy_key": legacy_key,
        "source": source,
    }

