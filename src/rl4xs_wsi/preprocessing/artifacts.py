"""Crash-safe, memory-mappable preprocessing artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from rl4xs_wsi.constants import SCHEMA_VERSION, magnification_token
from rl4xs_wsi.errors import ArtifactCompatibilityError
from rl4xs_wsi.preprocessing.atlaspatch import sha256_file
from rl4xs_wsi.preprocessing.geometry import GRID_DTYPE, GridLevel, NavigationGeometry
from rl4xs_wsi.preprocessing.mapping import MappingSummary
from rl4xs_wsi.preprocessing.tissue import TissueMask
from rl4xs_wsi.utils.atomic import atomic_json_dump
from rl4xs_wsi.wsi.reader import OverviewTransform


@dataclass(frozen=True, slots=True)
class SlideArtifactPaths:
    root: Path
    metadata: Path
    overview: Path
    tissue_mask: Path
    grids: Path
    lock: Path


@dataclass(frozen=True, slots=True)
class SlideOffset:
    slide_id: str
    slide_index: int
    offset: int
    num_cells: int
    level0_width: int
    level0_height: int


def atomic_save_npy(path: str | Path, value: np.ndarray) -> None:
    """Write a non-pickled NumPy artifact using same-directory replacement."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".npy", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.save(stream, value, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


@contextmanager
def _advisory_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover - Windows fallback
            pass
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except ImportError:  # pragma: no cover - Windows fallback
            pass
        os.close(descriptor)


class ArtifactStore:
    """One fingerprinted preprocessing directory.

    ``root`` is the stage directory itself (the directory containing the
    pipeline's ``manifest.json``), not the workspace root.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    @classmethod
    def from_workspace(
        cls,
        workspace_root: str | Path,
        dataset: str,
        fingerprint_value: str,
    ) -> ArtifactStore:
        return cls(Path(workspace_root) / "processed" / dataset / fingerprint_value)

    @property
    def offsets_path(self) -> Path:
        return self.root / "cell_offsets.json"

    @property
    def index_path(self) -> Path:
        return self.root / "slides.json"

    def slide_paths(self, slide_id: str) -> SlideArtifactPaths:
        if not slide_id or Path(slide_id).name != slide_id or slide_id in {".", ".."}:
            raise ValueError(f"unsafe slide ID: {slide_id!r}")
        root = self.root / "slides" / slide_id
        return SlideArtifactPaths(
            root=root,
            metadata=root / "metadata.json",
            overview=root / "overview.npy",
            tissue_mask=root / "tissue_mask.npy",
            grids=root / "grids",
            lock=root / ".write.lock",
        )

    def generated_tissue_path(self, slide_id: str) -> Path:
        if not slide_id or Path(slide_id).name != slide_id:
            raise ValueError(f"unsafe slide ID: {slide_id!r}")
        return self.root / "generated_tissue" / f"{slide_id}.npy"

    def write_offsets(
        self,
        offsets: Mapping[str, SlideOffset],
        *,
        input_fingerprint: str,
    ) -> None:
        ordered = [
            asdict(value) for value in sorted(offsets.values(), key=lambda item: item.slide_index)
        ]
        previous_index = -1
        previous_end = 0
        total_cells = 0
        for item in ordered:
            slide_index = int(item["slide_index"])
            offset = int(item["offset"])
            num_cells = int(item["num_cells"])
            if slide_index <= previous_index:
                raise ValueError("slide indices in the sparse offset table must be unique")
            if offset < previous_end or num_cells <= 0:
                raise ValueError("sparse slide cell-ID ranges must be positive and non-overlapping")
            if offset > np.iinfo(np.int64).max - (num_cells - 1):
                raise ValueError("slide cell-ID range exceeds int64")
            previous_index = slide_index
            previous_end = offset + num_cells
            total_cells += num_cells
        atomic_json_dump(
            self.offsets_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
                "input_fingerprint": input_fingerprint,
                "total_cells": total_cells,
                "max_cell_id_exclusive": previous_end,
                "slides": ordered,
            },
        )

    def load_offsets(
        self, *, expected_input_fingerprint: str | None = None
    ) -> dict[str, SlideOffset]:
        try:
            raw = json.loads(self.offsets_path.read_text(encoding="utf-8"))
            if (
                raw.get("status") != "complete"
                or int(raw.get("schema_version", -1)) != SCHEMA_VERSION
            ):
                raise ArtifactCompatibilityError(
                    f"cell-offset table is incomplete: {self.offsets_path}"
                )
            if (
                expected_input_fingerprint is not None
                and raw.get("input_fingerprint") != expected_input_fingerprint
            ):
                raise ArtifactCompatibilityError(f"cell-offset table is stale: {self.offsets_path}")
            items = list(raw["slides"])
            slide_ids = [str(item["slide_id"]) for item in items]
            if len(slide_ids) != len(set(slide_ids)):
                raise ArtifactCompatibilityError(
                    f"cell-offset table contains duplicate slide IDs: {self.offsets_path}"
                )
            result = {
                str(item["slide_id"]): SlideOffset(
                    slide_id=str(item["slide_id"]),
                    slide_index=int(item["slide_index"]),
                    offset=int(item["offset"]),
                    num_cells=int(item["num_cells"]),
                    level0_width=int(item["level0_width"]),
                    level0_height=int(item["level0_height"]),
                )
                for item in items
            }
        except ArtifactCompatibilityError:
            raise
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise ArtifactCompatibilityError(
                f"invalid cell-offset table {self.offsets_path}: {exc}"
            ) from exc
        ordered = list(result.values())
        previous_index = -1
        previous_end = 0
        total_cells = 0
        for item in ordered:
            if (
                item.slide_index <= previous_index
                or item.offset < previous_end
                or item.num_cells <= 0
                or item.offset > np.iinfo(np.int64).max - (item.num_cells - 1)
            ):
                raise ArtifactCompatibilityError(
                    f"invalid sparse cell-offset table: {self.offsets_path}"
                )
            previous_index = item.slide_index
            previous_end = item.offset + item.num_cells
            total_cells += item.num_cells
        if total_cells != int(raw.get("total_cells", -1)):
            raise ArtifactCompatibilityError(f"cell-offset total mismatch: {self.offsets_path}")
        if previous_end != int(raw.get("max_cell_id_exclusive", previous_end)):
            raise ArtifactCompatibilityError(f"cell-offset maximum mismatch: {self.offsets_path}")
        return result

    def write_slide(
        self,
        *,
        geometry: NavigationGeometry,
        overview: np.ndarray,
        transform: OverviewTransform,
        tissue_mask: TissueMask,
        start_cell_id: int,
        label: int,
        slide_path: str | Path,
        input_fingerprint: str,
        embedding_sources: Mapping[float, Mapping[str, Any]],
        mapping_summaries: Mapping[float, MappingSummary],
        overwrite: bool = False,
    ) -> dict[str, Any]:
        paths = self.slide_paths(geometry.slide_id)
        with _advisory_lock(paths.lock):
            if not overwrite and self.is_slide_complete(
                geometry.slide_id, expected_input_fingerprint=input_fingerprint
            ):
                return self.load_metadata(
                    geometry.slide_id, expected_input_fingerprint=input_fingerprint
                )
            overview_array = np.asarray(overview, dtype=np.uint8)
            if overview_array.shape != (transform.canvas_height, transform.canvas_width, 3):
                raise ValueError("overview does not match its persisted transform")
            mask_array = np.asarray(tissue_mask.values)
            if mask_array.dtype != np.bool_ or mask_array.shape != overview_array.shape[:2]:
                raise ValueError("tissue mask must be boolean and aligned with overview")
            if start_cell_id not in {
                int(value) for value in geometry.level(geometry.magnifications[0]).cells["cell_id"]
            }:
                raise ValueError("start_cell_id is not in the coarsest grid")

            atomic_json_dump(
                paths.metadata,
                {
                    "schema_version": SCHEMA_VERSION,
                    "artifact_type": "processed_slide",
                    "status": "incomplete",
                    "input_fingerprint": input_fingerprint,
                    "slide_id": geometry.slide_id,
                },
            )
            atomic_save_npy(paths.overview, overview_array)
            atomic_save_npy(paths.tissue_mask, mask_array)
            grids: dict[str, dict[str, Any]] = {}
            for magnification, level in geometry.levels.items():
                token = magnification_token(magnification)
                grid_path = paths.grids / f"{token}.npy"
                cells = np.asarray(level.cells)
                if cells.dtype != GRID_DTYPE:
                    raise ValueError(f"grid dtype mismatch at {magnification:g}x")
                atomic_save_npy(grid_path, cells)
                grids[token] = {
                    "magnification": magnification,
                    "path": str(grid_path.relative_to(paths.root)),
                    "shape": list(cells.shape),
                    "dtype": cells.dtype.descr,
                    "size": grid_path.stat().st_size,
                    "sha256": sha256_file(grid_path),
                    "rows": level.rows,
                    "columns": level.columns,
                    "offset": level.offset,
                    "footprint_level0": level.footprint_level0,
                }
            metadata: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "processed_slide",
                "status": "complete",
                "completed_at": datetime.now(UTC).isoformat(),
                "input_fingerprint": input_fingerprint,
                "slide_id": geometry.slide_id,
                "slide_index": geometry.slide_index,
                "slide_path": str(Path(slide_path).expanduser().resolve()),
                "label": int(label),
                "level0_width": geometry.level0_width,
                "level0_height": geometry.level0_height,
                "level0_magnification": geometry.level0_magnification,
                "cell_id_offset": geometry.cell_id_offset,
                "num_cells": geometry.num_cells,
                "start_cell_id": int(start_cell_id),
                "overview_transform": transform.to_dict(),
                "overview": self._array_description(paths.overview, overview_array, paths.root),
                "tissue_mask": {
                    **self._array_description(paths.tissue_mask, mask_array, paths.root),
                    "source": tissue_mask.source,
                    "threshold": tissue_mask.threshold,
                    "source_path": str(tissue_mask.source_path)
                    if tissue_mask.source_path
                    else None,
                    "source_page": tissue_mask.source_page,
                    "provenance": dict(tissue_mask.provenance),
                },
                "grids": grids,
                "embedding_sources": {
                    magnification_token(float(key)): dict(value)
                    for key, value in embedding_sources.items()
                },
                "mapping_summaries": {
                    magnification_token(float(key)): asdict(value)
                    for key, value in mapping_summaries.items()
                },
            }
            # Completion metadata is written last; partial arrays are ignored.
            atomic_json_dump(paths.metadata, metadata)
            return metadata

    @staticmethod
    def _array_description(path: Path, value: np.ndarray, root: Path) -> dict[str, Any]:
        return {
            "path": str(path.relative_to(root)),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    def is_slide_complete(
        self,
        slide_id: str,
        *,
        expected_input_fingerprint: str | None = None,
    ) -> bool:
        try:
            self.load_metadata(slide_id, expected_input_fingerprint=expected_input_fingerprint)
        except ArtifactCompatibilityError:
            return False
        return True

    def load_metadata(
        self,
        slide_id: str,
        *,
        expected_input_fingerprint: str | None = None,
    ) -> dict[str, Any]:
        paths = self.slide_paths(slide_id)
        try:
            raw = json.loads(paths.metadata.read_text(encoding="utf-8"))
            if raw.get("status") != "complete" or raw.get("artifact_type") != "processed_slide":
                raise ArtifactCompatibilityError(f"processed slide is incomplete: {paths.metadata}")
            if int(raw.get("schema_version", -1)) != SCHEMA_VERSION:
                raise ArtifactCompatibilityError(
                    f"processed slide schema is incompatible: {paths.metadata}"
                )
            if raw.get("slide_id") != slide_id:
                raise ArtifactCompatibilityError(f"processed slide ID mismatch: {paths.metadata}")
            if (
                expected_input_fingerprint is not None
                and raw.get("input_fingerprint") != expected_input_fingerprint
            ):
                raise ArtifactCompatibilityError(f"processed slide is stale: {paths.metadata}")
            descriptions = [raw["overview"], raw["tissue_mask"], *raw["grids"].values()]
            for description in descriptions:
                artifact = paths.root / str(description["path"])
                if not artifact.is_file() or artifact.stat().st_size != int(description["size"]):
                    raise ArtifactCompatibilityError(
                        f"processed array missing or truncated: {artifact}"
                    )
        except ArtifactCompatibilityError:
            raise
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise ArtifactCompatibilityError(
                f"invalid processed-slide metadata {paths.metadata}: {exc}"
            ) from exc
        return raw

    def _load_array(
        self, slide_id: str, description: Mapping[str, Any], *, mmap: bool
    ) -> np.ndarray:
        paths = self.slide_paths(slide_id)
        path = paths.root / str(description["path"])
        try:
            array = np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ArtifactCompatibilityError(f"cannot load processed array {path}: {exc}") from exc
        expected_shape = tuple(int(value) for value in description["shape"])
        if array.shape != expected_shape:
            raise ArtifactCompatibilityError(
                f"processed array shape mismatch {array.shape} != {expected_shape}: {path}"
            )
        return array

    def load_overview(self, slide_id: str, *, mmap: bool = True) -> np.ndarray:
        metadata = self.load_metadata(slide_id)
        array = self._load_array(slide_id, metadata["overview"], mmap=mmap)
        if array.dtype != np.uint8:
            raise ArtifactCompatibilityError(f"overview dtype is {array.dtype}, expected uint8")
        return array

    def load_tissue_mask(self, slide_id: str, *, mmap: bool = True) -> np.ndarray:
        metadata = self.load_metadata(slide_id)
        array = self._load_array(slide_id, metadata["tissue_mask"], mmap=mmap)
        if array.dtype != np.bool_:
            raise ArtifactCompatibilityError(f"tissue mask dtype is {array.dtype}, expected bool")
        return array

    def load_grid(self, slide_id: str, magnification: float, *, mmap: bool = True) -> np.ndarray:
        metadata = self.load_metadata(slide_id)
        token = magnification_token(float(magnification))
        try:
            description = metadata["grids"][token]
        except KeyError as exc:
            raise ArtifactCompatibilityError(
                f"processed slide {slide_id} has no {magnification:g}x grid"
            ) from exc
        array = self._load_array(slide_id, description, mmap=mmap)
        if array.dtype != GRID_DTYPE:
            raise ArtifactCompatibilityError(
                f"grid dtype is incompatible at {magnification:g}x for {slide_id}"
            )
        return array

    def load_geometry(self, slide_id: str, *, mmap: bool = True) -> NavigationGeometry:
        metadata = self.load_metadata(slide_id)
        levels: dict[float, GridLevel] = {}
        for token, description in sorted(
            metadata["grids"].items(), key=lambda item: float(item[1]["magnification"])
        ):
            del token
            magnification = float(description["magnification"])
            levels[magnification] = GridLevel(
                magnification=magnification,
                footprint_level0=int(description["footprint_level0"]),
                rows=int(description["rows"]),
                columns=int(description["columns"]),
                offset=int(description["offset"]),
                cells=self.load_grid(slide_id, magnification, mmap=mmap),
            )
        return NavigationGeometry(
            slide_id=slide_id,
            slide_index=int(metadata["slide_index"]),
            level0_width=int(metadata["level0_width"]),
            level0_height=int(metadata["level0_height"]),
            level0_magnification=float(metadata["level0_magnification"]),
            cell_id_offset=int(metadata["cell_id_offset"]),
            levels=levels,
        )

    def write_index(
        self,
        slide_ids: list[str] | tuple[str, ...],
        *,
        merge: bool = False,
    ) -> None:
        selected = list(dict.fromkeys(slide_ids))
        if merge and self.index_path.is_file():
            try:
                raw = json.loads(self.index_path.read_text(encoding="utf-8"))
                if raw.get("status") != "complete":
                    raise ValueError("existing index is incomplete")
                selected = list(
                    dict.fromkeys([str(item["slide_id"]) for item in raw["slides"]] + selected)
                )
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                raise ArtifactCompatibilityError(
                    f"cannot merge invalid processed-slide index {self.index_path}: {exc}"
                ) from exc
        entries = []
        for slide_id in selected:
            metadata = self.load_metadata(slide_id)
            entries.append(
                {
                    "slide_id": slide_id,
                    "slide_index": metadata["slide_index"],
                    "metadata": str(self.slide_paths(slide_id).metadata.relative_to(self.root)),
                    "input_fingerprint": metadata["input_fingerprint"],
                    "cell_id_offset": metadata["cell_id_offset"],
                    "num_cells": metadata["num_cells"],
                }
            )
        entries.sort(key=lambda item: int(item["slide_index"]))
        slide_indices = [int(item["slide_index"]) for item in entries]
        if len(slide_indices) != len(set(slide_indices)):
            raise ArtifactCompatibilityError("processed-slide index has duplicate slide indices")
        atomic_json_dump(
            self.index_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
                "slides": entries,
            },
        )


__all__ = [
    "ArtifactStore",
    "SlideArtifactPaths",
    "SlideOffset",
    "atomic_save_npy",
]
