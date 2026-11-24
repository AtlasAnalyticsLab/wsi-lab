"""Shared immutable rollout sequence and small measurement helpers."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Set, Tuple, Union


ABSTRACT_PRIORITY_SEED = 1
PAN_PREFERENCE = ("right", "down", "left", "up")
EXPECTED_QUOTAS = {"pan": 50, "zoom_in": 25, "zoom_out": 25}


def abstract_priorities(slide_ordinal: int) -> Tuple[str, ...]:
    """Return the backend-independent priority order for one manifest row."""
    if slide_ordinal < 0:
        raise ValueError("slide_ordinal must be non-negative")
    priorities = [
        action_class
        for action_class, count in EXPECTED_QUOTAS.items()
        for _ in range(count)
    ]
    random.Random(ABSTRACT_PRIORITY_SEED + slide_ordinal).shuffle(priorities)
    result = tuple(priorities)
    validate_abstract_sequence(result)
    return result


def validate_abstract_sequence(sequence: Sequence[str]) -> None:
    counts = {name: sequence.count(name) for name in EXPECTED_QUOTAS}
    if len(sequence) != 100 or counts != EXPECTED_QUOTAS:
        raise AssertionError(f"Invalid abstract action protocol: length={len(sequence)}, counts={counts}")


def percentile(values: Sequence[Union[float, int]], percentile_value: float) -> float:
    """Return a linearly interpolated percentile without a NumPy dependency."""
    if not values:
        return math.nan
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile_value / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def latency_summary_ns(values: Sequence[int]) -> Dict[str, Union[float, int]]:
    total = sum(values)
    return {
        "count": len(values),
        "total_ns": total,
        "mean_ns": total / len(values) if values else math.nan,
        "median_ns": percentile(values, 50),
        "p95_ns": percentile(values, 95),
        "min_ns": min(values) if values else math.nan,
        "max_ns": max(values) if values else math.nan,
        "steps_per_second_step_only": (len(values) * 1e9 / total) if total else math.nan,
    }


def observation_nbytes(value: Any, _seen: Union[Set[int], None] = None) -> int:
    """Count array-like payload bytes while avoiding duplicate object references."""
    seen = set() if _seen is None else _seen
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    if hasattr(value, "nbytes"):
        return int(value.nbytes)
    if isinstance(value, Mapping):
        return sum(observation_nbytes(item, seen) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(observation_nbytes(item, seen) for item in value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    return 0


def directory_snapshot(root: Union[str, Path]) -> Dict[str, Union[int, str]]:
    """Return logical and allocated storage beneath a generated-state root."""
    root_path = Path(root)
    result: Dict[str, Union[int, str]] = {
        "root": str(root_path.resolve()),
        "file_count": 0,
        "directory_count": 0,
        "symlink_count": 0,
        "logical_bytes": 0,
        "allocated_bytes": 0,
    }
    if not root_path.exists():
        return result
    for path in [root_path, *root_path.rglob("*")]:
        stat = path.lstat()
        # Filesystems allocate metadata blocks to directories and symlinks too.
        # Count each entry exactly once so this agrees with `du --block-size=1`.
        result["allocated_bytes"] += stat.st_blocks * 512
        if path.is_symlink():
            result["symlink_count"] += 1
        elif path.is_dir():
            result["directory_count"] += 1
        elif path.is_file():
            result["file_count"] += 1
            result["logical_bytes"] += stat.st_size
    return result
