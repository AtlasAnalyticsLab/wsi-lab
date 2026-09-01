#!/usr/bin/env python3
"""Read-only post-hoc tissue audit for accepted WSI-Lab benchmark steps.

The online benchmark already persists the post-transition ``cell_id`` for
every environment call.  Native processed grids are the authoritative mapping
from that ID to the viewport's level-0 bounds and AtlasPatch-derived tissue
fields, so no WSI decoding or environment replay is needed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Direct-file execution is supported in addition to ``python -m``.
if __package__ in {None, ""}:  # pragma: no cover - operator convenience
    _REPOSITORY = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_REPOSITORY))

from wsi_lab.constants import NAVIGATION_MAGNIFICATIONS
from wsi_lab.preprocessing.artifacts import ArtifactStore


@dataclass(frozen=True, slots=True)
class CellTissueRegion:
    """The authoritative persisted native row for one navigation cell."""

    slide_id: str
    cell_id: int
    magnification: float
    depth: int
    row: int
    column: int
    level0_bounds: tuple[int, int, int, int]
    clipped_level0_bounds: tuple[int, int, int, int]
    tissue_fraction: float
    tissue_valid: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _GridRange:
    magnification: float
    offset: int
    count: int

    @property
    def stop(self) -> int:
        return self.offset + self.count


class ProcessedGridLookup:
    """Resolve global cell IDs through native mmap grids without WSI reads."""

    def __init__(self, processed_root: str | Path) -> None:
        self.processed_root = Path(processed_root).expanduser().resolve()
        self._store = ArtifactStore(self.processed_root)
        self._metadata: dict[str, dict[str, Any]] = {}
        self._ranges: dict[str, tuple[_GridRange, ...]] = {}
        self._arrays: dict[tuple[str, float], np.ndarray] = {}

    def _slide_metadata(self, slide_id: str) -> dict[str, Any]:
        if slide_id not in self._metadata:
            self._metadata[slide_id] = self._store.load_metadata(slide_id)
        return self._metadata[slide_id]

    def _grid_ranges(self, slide_id: str) -> tuple[_GridRange, ...]:
        if slide_id in self._ranges:
            return self._ranges[slide_id]
        metadata = self._slide_metadata(slide_id)
        levels = tuple(
            sorted(
                (
                    _GridRange(
                        magnification=float(description["magnification"]),
                        offset=int(description["offset"]),
                        count=int(description["shape"][0]),
                    )
                    for description in metadata["grids"].values()
                ),
                key=lambda item: item.offset,
            )
        )
        if not levels or any(level.count <= 0 for level in levels):
            raise ValueError(f"{slide_id} has no non-empty persisted navigation levels")
        expected = int(metadata["cell_id_offset"])
        for level in levels:
            if level.offset != expected:
                raise ValueError(
                    f"{slide_id} grid ID ranges are not contiguous at {level.magnification:g}x"
                )
            expected = level.stop
        if expected != int(metadata["cell_id_offset"]) + int(metadata["num_cells"]):
            raise ValueError(f"{slide_id} grid ranges disagree with persisted num_cells")
        self._ranges[slide_id] = levels
        return levels

    def _grid(self, slide_id: str, magnification: float) -> np.ndarray:
        key = (slide_id, float(magnification))
        if key not in self._arrays:
            self._arrays[key] = self._store.load_grid(
                slide_id, float(magnification), mmap=True
            )
        return self._arrays[key]

    def lookup(
        self,
        slide_id: str,
        cell_id: int,
        *,
        expected_magnification: float | None = None,
        expected_depth: int | None = None,
    ) -> CellTissueRegion:
        """Return the one native grid row addressed by ``slide_id/cell_id``."""

        key = int(cell_id)
        matches = [level for level in self._grid_ranges(slide_id) if level.offset <= key < level.stop]
        if len(matches) != 1:
            raise KeyError(f"cell {key} does not map to exactly one grid for {slide_id}")
        level = matches[0]
        array = self._grid(slide_id, level.magnification)
        local = key - level.offset
        row = array[local]
        stored_id = int(row["cell_id"])
        stored_magnification = float(row["magnification"])
        if stored_id != key:
            raise ValueError(f"grid row ID {stored_id} does not match requested cell {key}")
        if not math.isclose(stored_magnification, level.magnification, abs_tol=1e-7):
            raise ValueError(f"grid row magnification changed for {slide_id}/{key}")
        if expected_magnification is not None and not math.isclose(
            stored_magnification, float(expected_magnification), abs_tol=1e-7
        ):
            raise ValueError(
                f"recorded magnification {expected_magnification} disagrees with native "
                f"{stored_magnification} for {slide_id}/{key}"
            )
        try:
            depth = NAVIGATION_MAGNIFICATIONS.index(stored_magnification)
        except ValueError as exc:
            raise ValueError(
                f"unknown native magnification {stored_magnification} for {slide_id}/{key}"
            ) from exc
        if expected_depth is not None and depth != int(expected_depth):
            raise ValueError(
                f"recorded depth {expected_depth} disagrees with native {depth} "
                f"for {slide_id}/{key}"
            )
        fraction = float(row["tissue_fraction"])
        valid_raw = int(row["tissue_valid"])
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError(f"invalid tissue fraction {fraction} for {slide_id}/{key}")
        if valid_raw not in {0, 1}:
            raise ValueError(f"invalid tissue_valid value {valid_raw} for {slide_id}/{key}")
        x0 = int(row["level0_x0"])
        y0 = int(row["level0_y0"])
        x1 = int(row["level0_x1"])
        y1 = int(row["level0_y1"])
        clipped_x1 = int(row["clipped_x1"])
        clipped_y1 = int(row["clipped_y1"])
        if not (x0 < clipped_x1 <= x1 and y0 < clipped_y1 <= y1):
            raise ValueError(f"invalid native level-0 bounds for {slide_id}/{key}")
        return CellTissueRegion(
            slide_id=slide_id,
            cell_id=key,
            magnification=stored_magnification,
            depth=depth,
            row=int(row["row"]),
            column=int(row["column"]),
            # These nominal bounds are NavigationCell.bounds and therefore the
            # exact region passed to the native viewport reader. Edge cells may
            # extend past the WSI and are white-padded by the reader.
            level0_bounds=(x0, y0, x1, y1),
            # Native tissue fractions use the clipped region projected into the
            # persisted 224x224 binary tissue-mask coordinate frame.
            clipped_level0_bounds=(x0, y0, clipped_x1, clipped_y1),
            tissue_fraction=fraction,
            tissue_valid=bool(valid_raw),
        )

    def close(self) -> None:
        # Clearing references is sufficient for normal use. Explicitly closing
        # memmaps prevents long-lived importing processes from retaining FDs.
        for array in self._arrays.values():
            mapped = getattr(array, "_mmap", None)
            if mapped is not None:
                mapped.close()
        self._arrays.clear()

    def __enter__(self) -> ProcessedGridLookup:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def _fraction_summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    ordered = sorted(float(value) for value in values)

    def percentile(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "minimum": ordered[0],
        "p05": percentile(0.05),
        "median": statistics.median(ordered),
        "p95": percentile(0.95),
        "maximum": ordered[-1],
    }


def _visit_summary(visits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fractions = [float(visit["tissue_fraction"]) for visit in visits]
    valid = sum(bool(visit["tissue_valid"]) for visit in visits)
    threshold_counts = {
        "atlaspatch_fraction_at_least_0.01": sum(value >= 0.01 for value in fractions),
        "atlaspatch_fraction_at_least_0.10": sum(value >= 0.10 for value in fractions),
        "atlaspatch_fraction_at_least_0.50": sum(value >= 0.50 for value in fractions),
    }
    return {
        "accepted_post_step_visits": len(visits),
        "atlaspatch_any_overlap_visits": valid,
        "atlaspatch_any_overlap_fraction": valid / len(visits) if visits else None,
        "sensitivity_positive_visits": threshold_counts,
        "tissue_valid_visits": valid,
        "tissue_valid_visit_fraction": valid / len(visits) if visits else None,
        "tissue_fraction": _fraction_summary(fractions),
    }


def accepted_post_step_regions(
    online_result: Mapping[str, Any],
    lookup: ProcessedGridLookup,
    *,
    expected_accepted_per_slide: int | None = 100,
) -> list[dict[str, Any]]:
    """Map accepted post-step states; resets and rejected/no-op calls are excluded."""

    if online_result.get("backend") != "wsi-lab" or online_result.get("phase") != "online":
        raise ValueError("expected a WSI-Lab online result")
    if online_result.get("status") != "complete":
        raise ValueError("online result is not complete")
    repetition = int(online_result["repetition"])
    visits: list[dict[str, Any]] = []
    for slide in online_result["slides"]:
        slide_id = str(slide["slide_id"])
        current_cell = int(slide["reset"]["cell_id"])
        accepted_indices: list[int] = []
        for step in slide["steps"]:
            if str(step["slide_id"]) != slide_id:
                raise ValueError(f"step/slide ID mismatch inside {slide_id}")
            if int(step["before_cell_id"]) != current_cell:
                raise ValueError(f"broken native state chain inside {slide_id}")
            current_cell = int(step["cell_id"])
            if not bool(step["accepted"]):
                continue
            if bool(step["invalid_action"]) or not bool(step["state_changed"]):
                raise ValueError(f"accepted flag is inconsistent for {slide_id}")
            accepted_index = int(step["index"])
            accepted_indices.append(accepted_index)
            native = lookup.lookup(
                slide_id,
                current_cell,
                expected_magnification=float(step["magnification"]),
                expected_depth=int(step["depth"]),
            )
            visits.append(
                {
                    "repetition": repetition,
                    "ordinal": int(slide["ordinal"]),
                    "slide_id": slide_id,
                    "accepted_index": accepted_index,
                    "requested_class": str(step["requested_class"]),
                    "concrete_action": str(step["concrete_action"]),
                    **native.to_dict(),
                }
            )
        if accepted_indices != list(range(len(accepted_indices))):
            raise ValueError(f"accepted indices are not contiguous for {slide_id}")
        if (
            expected_accepted_per_slide is not None
            and len(accepted_indices) != expected_accepted_per_slide
        ):
            raise ValueError(
                f"{slide_id} has {len(accepted_indices)} accepted steps; expected "
                f"{expected_accepted_per_slide}"
            )
    return visits


def tissue_report(
    online_result: Mapping[str, Any],
    lookup: ProcessedGridLookup,
    *,
    include_visits: bool = False,
    expected_accepted_per_slide: int | None = 100,
) -> dict[str, Any]:
    """Build visit-weighted and unique-region summaries from persisted rows."""

    visits = accepted_post_step_regions(
        online_result,
        lookup,
        expected_accepted_per_slide=expected_accepted_per_slide,
    )
    unique = list(
        {
            (str(visit["slide_id"]), int(visit["cell_id"])): visit for visit in visits
        }.values()
    )
    by_magnification: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for visit in visits:
        by_magnification[f"{float(visit['magnification']):g}"].append(visit)
    trace_payload = [
        [str(visit["slide_id"]), int(visit["accepted_index"]), int(visit["cell_id"])]
        for visit in visits
    ]
    result: dict[str, Any] = {
        "schema_version": 1,
        "backend": "wsi-lab",
        "phase": "post_hoc_visited_tissue",
        "source_repetition": int(online_result["repetition"]),
        "processed_root": str(lookup.processed_root),
        "semantics": {
            "included_states": "accepted post-step cells only",
            "reset_state_included": False,
            "rejected_or_no_op_states_included": False,
            "level0_bounds": "native nominal NavigationCell.bounds used for viewport reads",
            "clipped_level0_bounds": "nominal bounds clipped to the WSI right/bottom edges",
            "tissue_fraction": (
                "fraction of projected overview-mask pixels marked tissue inside the "
                "clipped cell region"
            ),
            "tissue_valid": "at least one projected overview-mask pixel is tissue",
            "primary_metric": "atlaspatch_any_overlap (tissue_valid)",
            "sensitivity_thresholds": [0.01, 0.10, 0.50],
            "tumor_annotation_metric": False,
        },
        "trace_sha256": hashlib.sha256(
            json.dumps(trace_payload, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "visit_weighted": _visit_summary(visits),
        "unique_cell_weighted": _visit_summary(unique),
        "by_magnification_visit_weighted": {
            key: _visit_summary(value)
            for key, value in sorted(by_magnification.items(), key=lambda item: float(item[0]))
        },
    }
    if include_visits:
        result["visits"] = visits
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read online result {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"online result is not a JSON object: {path}")
    return value


def _default_processed_root(result: Mapping[str, Any]) -> Path:
    try:
        workspace = Path(str(result["prepared_workspace"]["root"]))
    except (KeyError, TypeError) as exc:
        raise ValueError("online result has no prepared workspace root") from exc
    return workspace.expanduser().resolve() / "artifacts/processed"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online-result", type=Path, required=True)
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=None,
        help="Override <prepared_workspace>/artifacts/processed for a relocated result.",
    )
    parser.add_argument(
        "--include-visits",
        action="store_true",
        help="Include every accepted mapped row rather than summary fields only.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = _load_json(args.online_result.expanduser().resolve())
    processed_root = (
        args.processed_root.expanduser().resolve()
        if args.processed_root is not None
        else _default_processed_root(result)
    )
    with ProcessedGridLookup(processed_root) as lookup:
        report = tissue_report(result, lookup, include_visits=bool(args.include_visits))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
