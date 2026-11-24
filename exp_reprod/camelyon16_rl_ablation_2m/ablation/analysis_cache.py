"""Analysis-only tissue and tumor helpers for the ablation study.

Nothing in this module is an environment input.  The cache is deliberately
separate from preprocessing fingerprints and contains only reporting data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
TUMOR_LABEL = 2
NON_EXHAUSTIVE_CAMELYON16_TUMOR_SLIDES = frozenset(
    {
        "tumor_010",
        "tumor_015",
        "tumor_018",
        "tumor_020",
        "tumor_025",
        "tumor_029",
        "tumor_033",
        "tumor_034",
        "tumor_044",
        "tumor_046",
        "tumor_051",
        "tumor_054",
        "tumor_055",
        "tumor_056",
        "tumor_067",
        "tumor_079",
        "tumor_085",
        "tumor_092",
        "tumor_095",
        "tumor_110",
    }
)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )


@dataclass(frozen=True, slots=True)
class RasterTransform:
    """Aspect-preserving level-0 to analysis-canvas transform."""

    source_width: int
    source_height: int
    canvas_width: int
    canvas_height: int
    content_x0: int
    content_y0: int
    content_width: int
    content_height: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RasterTransform:
        return cls(
            **{
                name: int(value[name])
                for name in (
                    "source_width",
                    "source_height",
                    "canvas_width",
                    "canvas_height",
                    "content_x0",
                    "content_y0",
                    "content_width",
                    "content_height",
                )
            }
        )

    @property
    def scale_x(self) -> float:
        return self.content_width / self.source_width

    @property
    def scale_y(self) -> float:
        return self.content_height / self.source_height

    def project_point(self, x: float, y: float) -> tuple[float, float]:
        return self.content_x0 + x * self.scale_x, self.content_y0 + y * self.scale_y

    def project_bounds(self, bounds: Sequence[int | float]) -> tuple[float, float, float, float]:
        if len(bounds) != 4:
            raise ValueError("bounds must contain four coordinates")
        x0, y0 = self.project_point(float(bounds[0]), float(bounds[1]))
        x1, y1 = self.project_point(float(bounds[2]), float(bounds[3]))
        return x0, y0, x1, y1

    def raster_bounds(
        self, bounds: Sequence[int | float], *, ensure_nonempty: bool = True
    ) -> tuple[int, int, int, int]:
        x0f, y0f, x1f, y1f = self.project_bounds(bounds)
        x0 = max(0, min(self.canvas_width, math.floor(x0f)))
        y0 = max(0, min(self.canvas_height, math.floor(y0f)))
        x1 = max(0, min(self.canvas_width, math.ceil(x1f)))
        y1 = max(0, min(self.canvas_height, math.ceil(y1f)))
        if ensure_nonempty and x0 < self.canvas_width and y0 < self.canvas_height:
            x1 = min(self.canvas_width, max(x0 + 1, x1))
            y1 = min(self.canvas_height, max(y0 + 1, y1))
        return x0, y0, x1, y1


@dataclass(frozen=True, slots=True)
class SlideAnalysis:
    """Small read-only reporting sidecar for one slide."""

    slide_id: str
    transform: RasterTransform
    navigation_roi_level0: tuple[int, int, int, int]
    tissue_mask: np.ndarray | None
    tumor_overview_mask: np.ndarray | None
    tumor_cell_ids: np.ndarray
    tumor_cell_fractions: np.ndarray
    annotation_available: bool
    annotation_exhaustive: bool
    tumor_mask_level: int | None = None
    tumor_mask_downsample_x: float | None = None
    tumor_mask_downsample_y: float | None = None

    def __post_init__(self) -> None:
        ids = np.asarray(self.tumor_cell_ids, dtype=np.int64)
        fractions = np.asarray(self.tumor_cell_fractions, dtype=np.float32)
        if ids.ndim != 1 or fractions.shape != ids.shape:
            raise ValueError("tumor cell IDs and fractions must be aligned 1-D arrays")
        if len(ids) and (np.any(ids[1:] <= ids[:-1]) or np.any(fractions <= 0.0)):
            raise ValueError("tumor cell IDs must be sorted/unique with positive fractions")
        if np.any(fractions > 1.0):
            raise ValueError("tumor cell fractions must lie in (0,1]")
        if self.tissue_mask is not None:
            tissue = np.asarray(self.tissue_mask)
            if tissue.dtype != np.bool_ or tissue.shape != (
                self.transform.canvas_height,
                self.transform.canvas_width,
            ):
                raise ValueError("tissue mask must be a canvas-aligned boolean array")
        if self.tumor_overview_mask is not None:
            tumor = np.asarray(self.tumor_overview_mask)
            if tumor.dtype != np.bool_ or tumor.shape != (
                self.transform.canvas_height,
                self.transform.canvas_width,
            ):
                raise ValueError("tumor overview mask must be a canvas-aligned boolean array")
        object.__setattr__(self, "tumor_cell_ids", ids)
        object.__setattr__(self, "tumor_cell_fractions", fractions)

    def tumor_fraction(self, cell_id: int) -> float:
        index = int(np.searchsorted(self.tumor_cell_ids, int(cell_id)))
        if index >= len(self.tumor_cell_ids) or int(self.tumor_cell_ids[index]) != int(cell_id):
            return 0.0
        return float(self.tumor_cell_fractions[index])

    def tissue_fraction(self, bounds: Sequence[int | float]) -> float | None:
        if self.tissue_mask is None:
            return None
        x0, y0, x1, y1 = self.transform.raster_bounds(bounds)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        return float(np.mean(self.tissue_mask[y0:y1, x0:x1]))

    def tissue_coverage(self, visited_bounds: Iterable[Sequence[int | float]]) -> float | None:
        if self.tissue_mask is None:
            return None
        eligible = np.asarray(self.tissue_mask, dtype=np.bool_).copy()
        rx0, ry0, rx1, ry1 = self.transform.raster_bounds(
            self.navigation_roi_level0, ensure_nonempty=False
        )
        roi = np.zeros_like(eligible)
        roi[ry0:ry1, rx0:rx1] = True
        eligible &= roi
        denominator = int(eligible.sum())
        if denominator == 0:
            return None
        visited = np.zeros_like(eligible)
        for bounds in visited_bounds:
            x0, y0, x1, y1 = self.transform.raster_bounds(bounds)
            if x1 > x0 and y1 > y0:
                visited[y0:y1, x0:x1] = True
        return float(np.count_nonzero(visited & eligible) / denominator)

    def tumor_coverage(self, visited_bounds: Iterable[Sequence[int | float]]) -> float | None:
        """Annotated tumor pixels covered by any view; misses require exhaustive annotation."""

        if (
            self.tumor_overview_mask is None
            or not self.annotation_available
            or not self.annotation_exhaustive
        ):
            return None
        tumor = np.asarray(self.tumor_overview_mask, dtype=np.bool_)
        denominator = int(tumor.sum())
        if denominator == 0:
            return None
        visited = np.zeros_like(tumor)
        for bounds in visited_bounds:
            x0, y0, x1, y1 = self.transform.raster_bounds(bounds)
            if x1 > x0 and y1 > y0:
                visited[y0:y1, x0:x1] = True
        return float(np.count_nonzero(visited & tumor) / denominator)


class AnalysisCache:
    """Filesystem-backed provider suitable for environment wrappers."""

    def __init__(self, root: str | Path) -> None:
        resolved = Path(root).expanduser().resolve()
        # Callers may pass either the manifest root or its ``slides`` child.
        self.root = resolved / "slides" if (resolved / "slides").is_dir() else resolved
        self._last: tuple[str, SlideAnalysis] | None = None

    def paths(self, slide_id: str) -> tuple[Path, Path]:
        if not slide_id or Path(slide_id).name != slide_id:
            raise ValueError(f"unsafe slide ID: {slide_id!r}")
        return self.root / f"{slide_id}.json", self.root / f"{slide_id}.npz"

    def save(self, analysis: SlideAnalysis) -> None:
        metadata_path, arrays_path = self.paths(analysis.slide_id)
        arrays_path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=arrays_path.parent,
                prefix=f".{arrays_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(
                    handle,
                    tissue_mask=(
                        np.asarray(analysis.tissue_mask, dtype=np.bool_)
                        if analysis.tissue_mask is not None
                        else np.empty((0, 0), dtype=np.bool_)
                    ),
                    tumor_overview_mask=(
                        np.asarray(analysis.tumor_overview_mask, dtype=np.bool_)
                        if analysis.tumor_overview_mask is not None
                        else np.empty((0, 0), dtype=np.bool_)
                    ),
                    tumor_cell_ids=analysis.tumor_cell_ids,
                    tumor_cell_fractions=analysis.tumor_cell_fractions,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, arrays_path)
        except BaseException:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise
        _atomic_json(
            metadata_path,
            {
                "schema_version": SCHEMA_VERSION,
                "slide_id": analysis.slide_id,
                "overview_transform": asdict(analysis.transform),
                "navigation_roi_level0": list(analysis.navigation_roi_level0),
                "has_tissue_mask": analysis.tissue_mask is not None,
                "has_tumor_overview_mask": analysis.tumor_overview_mask is not None,
                "annotation_available": analysis.annotation_available,
                "annotation_exhaustive": analysis.annotation_exhaustive,
                "tumor_mask_level": analysis.tumor_mask_level,
                "tumor_mask_downsample_x": analysis.tumor_mask_downsample_x,
                "tumor_mask_downsample_y": analysis.tumor_mask_downsample_y,
                "arrays": arrays_path.name,
            },
        )
        self._last = (analysis.slide_id, analysis)

    def load(self, slide_id: str) -> SlideAnalysis | None:
        if self._last is not None and self._last[0] == slide_id:
            return self._last[1]
        metadata_path, arrays_path = self.paths(slide_id)
        if not metadata_path.is_file() or not arrays_path.is_file():
            return None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("slide_id") != slide_id:
            raise ValueError(f"incompatible analysis cache entry: {metadata_path}")
        with np.load(arrays_path, allow_pickle=False) as arrays:
            tissue = np.asarray(arrays["tissue_mask"], dtype=np.bool_)
            tumor_overview = np.asarray(arrays["tumor_overview_mask"], dtype=np.bool_)
            result = SlideAnalysis(
                slide_id=slide_id,
                transform=RasterTransform.from_mapping(metadata["overview_transform"]),
                navigation_roi_level0=tuple(
                    int(value) for value in metadata["navigation_roi_level0"]
                ),
                tissue_mask=tissue if bool(metadata["has_tissue_mask"]) else None,
                tumor_overview_mask=(
                    tumor_overview if bool(metadata["has_tumor_overview_mask"]) else None
                ),
                tumor_cell_ids=np.asarray(arrays["tumor_cell_ids"], dtype=np.int64),
                tumor_cell_fractions=np.asarray(arrays["tumor_cell_fractions"], dtype=np.float32),
                annotation_available=bool(metadata["annotation_available"]),
                annotation_exhaustive=bool(metadata["annotation_exhaustive"]),
                tumor_mask_level=(
                    None
                    if metadata.get("tumor_mask_level") is None
                    else int(metadata["tumor_mask_level"])
                ),
                tumor_mask_downsample_x=(
                    None
                    if metadata.get("tumor_mask_downsample_x") is None
                    else float(metadata["tumor_mask_downsample_x"])
                ),
                tumor_mask_downsample_y=(
                    None
                    if metadata.get("tumor_mask_downsample_y") is None
                    else float(metadata["tumor_mask_downsample_y"])
                ),
            )
        self._last = (slide_id, result)
        return result

    __call__ = load


def tumor_binary(mask: np.ndarray, *, tumor_label: int = TUMOR_LABEL) -> np.ndarray:
    """Return the annotated tumor class; value 1 is normal tissue, not tumor."""

    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"tumor mask must be 2-D, got {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError("CAMELYON16 tumor masks must contain integer labels")
    return np.asarray(array == int(tumor_label), dtype=np.bool_)


def read_tiff_mask_page(path: str | Path, *, level: int = 5) -> tuple[np.ndarray, dict[str, Any]]:
    """Read one explicit TIFF pyramid page without opening the WSI."""

    try:
        import tifffile
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise ImportError("tifffile is required for CAMELYON16 tumor masks") from exc
    source = Path(path).expanduser().resolve()
    with tifffile.TiffFile(source) as handle:
        if level < 0 or level >= len(handle.pages):
            raise ValueError(f"mask level {level} is unavailable in {source}")
        page = handle.pages[level]
        array = np.asarray(page.asarray())
        metadata = {
            "path": str(source),
            "level": int(level),
            "width": int(page.imagewidth),
            "height": int(page.imagelength),
            "pages": len(handle.pages),
        }
    if array.shape != (metadata["height"], metadata["width"]):
        raise ValueError(f"decoded TIFF page shape mismatch: {source}")
    return array, metadata


def _structured_cell_arrays(cells: Any) -> tuple[np.ndarray, ...]:
    array = np.asarray(cells)
    names = set(array.dtype.names or ())
    required = {"cell_id", "level0_x0", "level0_y0", "level0_x1", "level0_y1"}
    if array.ndim == 1 and required.issubset(names):
        return tuple(
            np.asarray(array[name], dtype=np.int64)
            for name in (
                "cell_id",
                "level0_x0",
                "level0_y0",
                "level0_x1",
                "level0_y1",
            )
        )
    records = list(cells)

    def value(record: Any, name: str) -> int:
        if isinstance(record, Mapping):
            return int(record[name])
        return int(getattr(record, name))

    return tuple(
        np.asarray([value(record, name) for record in records], dtype=np.int64)
        for name in ("cell_id", "level0_x0", "level0_y0", "level0_x1", "level0_y1")
    )


def index_tumor_cells(
    labeled_mask: np.ndarray,
    cell_levels: Iterable[Any],
    *,
    source_width: int,
    source_height: int,
    tumor_label: int = TUMOR_LABEL,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted cell IDs and annotated-tumor fractions using integral sums."""

    tumor = tumor_binary(labeled_mask, tumor_label=tumor_label)
    height, width = tumor.shape
    if source_width <= 0 or source_height <= 0:
        raise ValueError("source dimensions must be positive")
    downsample_x = source_width / width
    downsample_y = source_height / height
    integral = np.pad(tumor.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    hit_ids: list[np.ndarray] = []
    hit_fractions: list[np.ndarray] = []
    for cells in cell_levels:
        cell_id, x0_raw, y0_raw, x1_raw, y1_raw = _structured_cell_arrays(cells)
        x0 = np.clip(np.floor(x0_raw / downsample_x).astype(np.int64), 0, width)
        y0 = np.clip(np.floor(y0_raw / downsample_y).astype(np.int64), 0, height)
        x1 = np.clip(np.ceil(x1_raw / downsample_x).astype(np.int64), 0, width)
        y1 = np.clip(np.ceil(y1_raw / downsample_y).astype(np.int64), 0, height)
        sums = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
        area = np.maximum(1, (x1 - x0) * (y1 - y0))
        hit = sums > 0
        hit_ids.append(cell_id[hit])
        hit_fractions.append((sums[hit] / area[hit]).astype(np.float32))
    if not hit_ids or not any(len(values) for values in hit_ids):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    ids = np.concatenate(hit_ids)
    fractions = np.concatenate(hit_fractions)
    order = np.argsort(ids, kind="stable")
    ids = ids[order]
    fractions = fractions[order]
    if len(ids) > 1 and np.any(ids[1:] == ids[:-1]):
        raise ValueError("cell levels contain duplicate cell IDs")
    return ids, fractions


def project_tumor_mask_to_overview(
    labeled_mask: np.ndarray,
    transform: RasterTransform,
    *,
    tumor_label: int = TUMOR_LABEL,
) -> np.ndarray:
    """Project annotation pixels to a canvas while preserving tiny tumor foci."""

    tumor = tumor_binary(labeled_mask, tumor_label=tumor_label)
    output = np.zeros((transform.canvas_height, transform.canvas_width), dtype=np.bool_)
    rows, columns = np.nonzero(tumor)
    if not len(rows):
        return output
    downsample_x = transform.source_width / tumor.shape[1]
    downsample_y = transform.source_height / tumor.shape[0]
    x = transform.content_x0 + np.floor((columns + 0.5) * downsample_x * transform.scale_x)
    y = transform.content_y0 + np.floor((rows + 0.5) * downsample_y * transform.scale_y)
    x = np.clip(x.astype(np.int64), 0, transform.canvas_width - 1)
    y = np.clip(y.astype(np.int64), 0, transform.canvas_height - 1)
    output[y, x] = True
    return output


def _record_value(record: Mapping[str, Any] | Any, name: str, default: Any = None) -> Any:
    return (
        record.get(name, default) if isinstance(record, Mapping) else getattr(record, name, default)
    )


def _stable_rank(seed: int, slide_id: str) -> str:
    return hashlib.sha256(f"{int(seed)}\0{slide_id}".encode()).hexdigest()


def select_monitor_cohort(
    records: Iterable[Mapping[str, Any] | Any],
    *,
    cohort_size: int = 30,
    positive_slides: int = 15,
    seed: int = 20_260_831,
) -> tuple[dict[str, Any], ...]:
    """Select a fixed balanced train cohort with exhaustive positive annotations."""

    if cohort_size <= 0 or positive_slides <= 0 or positive_slides >= cohort_size:
        raise ValueError("cohort must contain positive and benign slides")
    normalized: list[dict[str, Any]] = []
    for record in records:
        slide_id = str(_record_value(record, "slide_id"))
        label = int(_record_value(record, "label"))
        split = str(_record_value(record, "split", "train"))
        available = bool(_record_value(record, "annotation_available", label == 1))
        exhaustive = bool(
            _record_value(
                record,
                "annotation_exhaustive",
                slide_id not in NON_EXHAUSTIVE_CAMELYON16_TUMOR_SLIDES,
            )
        )
        if split != "train" or label not in (0, 1):
            continue
        normalized.append(
            {
                "slide_id": slide_id,
                "label": label,
                "center": _record_value(record, "center"),
                "tumor_class": _record_value(record, "tumor_class"),
                "annotation_available": available,
                "annotation_exhaustive": exhaustive,
            }
        )
    positives = [
        item
        for item in normalized
        if item["label"] == 1 and item["annotation_available"] and item["annotation_exhaustive"]
    ]
    negatives = [item for item in normalized if item["label"] == 0]
    benign_slides = cohort_size - positive_slides
    if len(positives) < positive_slides or len(negatives) < benign_slides:
        raise ValueError(
            "insufficient train slides for requested balanced/exhaustively annotated cohort"
        )
    positives.sort(key=lambda item: _stable_rank(seed, item["slide_id"]))
    negatives.sort(key=lambda item: _stable_rank(seed, item["slide_id"]))
    selected = positives[:positive_slides] + negatives[:benign_slides]
    selected.sort(key=lambda item: _stable_rank(seed + 1, item["slide_id"]))
    return tuple(selected)


def write_monitor_cohort(
    path: str | Path,
    cohort: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    positive_slides: int,
) -> Path:
    destination = Path(path).expanduser().resolve()
    _atomic_json(
        destination,
        {
            "schema_version": SCHEMA_VERSION,
            "selection": {
                "split": "train",
                "cohort_size": len(cohort),
                "positive_slides": int(positive_slides),
                "seed": int(seed),
                "positive_annotation_requirement": "available_and_exhaustive",
            },
            "slides": [dict(item) for item in cohort],
        },
    )
    return destination


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_analysis_cache(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    tumor_mask_level: int = 5,
    cohort_size: int = 30,
    positive_slides: int = 15,
    cohort_seed: int = 20_260_831,
) -> dict[str, Any]:
    """Build train-only reporting sidecars on a local machine or server."""

    from wsi_lab.config import load_config
    from wsi_lab.data.factory import create_dataset_adapter
    from wsi_lab.data.splits import DatasetSplit
    from wsi_lab.pipeline.planner import artifact_paths, stage_fingerprints
    from wsi_lab.pipeline.stages import Stage
    from wsi_lab.preprocessing.artifacts import ArtifactStore

    config = load_config(config_path)
    if config.dataset.name != "camelyon16":
        raise ValueError("this study analysis cache is CAMELYON16-only")
    adapter = create_dataset_adapter(config.dataset, config.embeddings)
    split_path = config.workspace.root / "splits" / "camelyon16_split.json"
    split = DatasetSplit.load(split_path)
    fingerprints = stage_fingerprints(config)
    preprocessing_manifest = artifact_paths(config, fingerprints)[Stage.PREPROCESSING]
    store = ArtifactStore(preprocessing_manifest.parent)
    train_inputs: list[tuple[str, Any, Any]] = []
    missing_sidecars: list[str] = []
    for slide_id in split.ids("train"):
        assignment = split.assignment(slide_id)
        record = adapter.get_slide(slide_id)
        train_inputs.append((slide_id, assignment, record))
        if assignment.label == 1 and (
            record.tumor_mask_path is None or record.annotation_path is None
        ):
            missing_sidecars.append(slide_id)
    if missing_sidecars:
        raise ValueError(
            "every positive training slide requires its canonical tumor mask and XML "
            f"annotation; missing={missing_sidecars[:8]}"
        )

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    cache = AnalysisCache(destination / "slides")
    entries: dict[str, dict[str, Any]] = {}
    cohort_records: list[dict[str, Any]] = []
    for slide_id, assignment, record in train_inputs:
        metadata = store.load_metadata(slide_id)
        transform = RasterTransform.from_mapping(metadata["overview_transform"])
        grids = tuple(
            store.load_grid(slide_id, magnification, mmap=True)
            for magnification in (0.625, 1.25, 2.5, 5.0, 10.0, 20.0)
        )
        annotation_available = bool(
            assignment.label == 1
            and record.tumor_mask_path is not None
            and record.annotation_path is not None
        )
        annotation_exhaustive = bool(
            annotation_available and slide_id not in NON_EXHAUSTIVE_CAMELYON16_TUMOR_SLIDES
        )
        tumor_ids = np.empty(0, dtype=np.int64)
        tumor_fractions = np.empty(0, dtype=np.float32)
        tumor_overview: np.ndarray | None = None
        downsample_x: float | None = None
        downsample_y: float | None = None
        if annotation_available:
            assert record.tumor_mask_path is not None
            labeled_mask, mask_metadata = read_tiff_mask_page(
                record.tumor_mask_path, level=tumor_mask_level
            )
            tumor_ids, tumor_fractions = index_tumor_cells(
                labeled_mask,
                grids,
                source_width=transform.source_width,
                source_height=transform.source_height,
            )
            tumor_overview = project_tumor_mask_to_overview(labeled_mask, transform)
            downsample_x = transform.source_width / int(mask_metadata["width"])
            downsample_y = transform.source_height / int(mask_metadata["height"])
        navigation_roi = tuple(int(value) for value in metadata["navigation_roi_level0"])
        analysis = SlideAnalysis(
            slide_id=slide_id,
            transform=transform,
            navigation_roi_level0=navigation_roi,
            tissue_mask=np.asarray(store.load_tissue_mask(slide_id, mmap=False), dtype=np.bool_),
            tumor_overview_mask=tumor_overview,
            tumor_cell_ids=tumor_ids,
            tumor_cell_fractions=tumor_fractions,
            annotation_available=annotation_available,
            annotation_exhaustive=annotation_exhaustive,
            tumor_mask_level=tumor_mask_level if annotation_available else None,
            tumor_mask_downsample_x=downsample_x,
            tumor_mask_downsample_y=downsample_y,
        )
        cache.save(analysis)
        cache_json, cache_npz = cache.paths(slide_id)
        entries[slide_id] = {
            "cache_json": str(cache_json.relative_to(destination)),
            "cache_npz": str(cache_npz.relative_to(destination)),
            "label": int(assignment.label),
            "annotation_available": annotation_available,
            "annotation_exhaustive": annotation_exhaustive,
            "tumor_cells": len(tumor_ids),
        }
        cohort_records.append(
            {
                "slide_id": slide_id,
                "label": int(assignment.label),
                "split": "train",
                "center": record.center,
                "tumor_class": record.tumor_class,
                "annotation_available": annotation_available,
                "annotation_exhaustive": annotation_exhaustive,
            }
        )
    cohort = select_monitor_cohort(
        cohort_records,
        cohort_size=cohort_size,
        positive_slides=positive_slides,
        seed=cohort_seed,
    )
    cohort_path = write_monitor_cohort(
        destination / "monitor_slides.json",
        cohort,
        seed=cohort_seed,
        positive_slides=positive_slides,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "scope": "training-only-analysis",
        "dataset": "camelyon16",
        "train_only": True,
        "split_fingerprint": split.split_fingerprint,
        "preprocessing_fingerprint": fingerprints[Stage.PREPROCESSING],
        "tumor_label": TUMOR_LABEL,
        "tumor_mask_level": int(tumor_mask_level),
        "slides_indexed": len(entries),
        "positive_slides_indexed": sum(
            int(item["annotation_available"]) for item in entries.values()
        ),
        "annotation_non_exhaustive_slides": sorted(
            slide_id
            for slide_id, item in entries.items()
            if item["annotation_available"] and not item["annotation_exhaustive"]
        ),
        "entries": entries,
        "cohort": {
            "path": str(cohort_path.relative_to(destination)),
            "sha256": _sha256_file(cohort_path),
            "cohort_size": len(cohort),
            "positive_slides": int(positive_slides),
            "seed": int(cohort_seed),
        },
    }
    _atomic_json(destination / "manifest.json", manifest)
    return manifest


def verify_analysis_cache(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Read-only compatibility check used before study execution."""

    from wsi_lab.config import load_config
    from wsi_lab.data.splits import DatasetSplit
    from wsi_lab.pipeline.planner import stage_fingerprints
    from wsi_lab.pipeline.stages import Stage

    config = load_config(config_path)
    split = DatasetSplit.load(config.workspace.root / "splits" / "camelyon16_split.json")
    destination = Path(output_dir).expanduser().resolve()
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fingerprints = stage_fingerprints(config)
    required = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "scope": "training-only-analysis",
        "dataset": "camelyon16",
        "train_only": True,
        "split_fingerprint": split.split_fingerprint,
        "preprocessing_fingerprint": fingerprints[Stage.PREPROCESSING],
        "tumor_label": TUMOR_LABEL,
        "tumor_mask_level": 5,
        "slides_indexed": len(split.ids("train")),
    }
    mismatches = {
        name: (manifest.get(name), expected)
        for name, expected in required.items()
        if manifest.get(name) != expected
    }
    if mismatches:
        raise ValueError(f"analysis cache manifest mismatch: {mismatches}")
    entries = manifest.get("entries")
    if not isinstance(entries, Mapping) or set(entries) != set(split.ids("train")):
        raise ValueError("analysis cache entries do not exactly match the train split")
    positive_train_ids = {
        slide_id for slide_id in split.ids("train") if split.assignment(slide_id).label == 1
    }
    invalid_labels = [
        slide_id
        for slide_id, entry in entries.items()
        if int(entry.get("label", -1)) != int(split.assignment(str(slide_id)).label)
    ]
    missing_annotations = [
        slide_id
        for slide_id in sorted(positive_train_ids)
        if not bool(entries[slide_id].get("annotation_available"))
    ]
    if invalid_labels or missing_annotations:
        raise ValueError(
            "analysis cache label/annotation contract mismatch: "
            f"labels={invalid_labels[:8]}, annotations={missing_annotations[:8]}"
        )
    cache = AnalysisCache(destination / "slides")
    for slide_id, entry in entries.items():
        if cache.load(str(slide_id)) is None:
            raise ValueError(f"missing analysis cache entry: {slide_id}")
        for key in ("cache_json", "cache_npz"):
            if not (destination / str(entry[key])).is_file():
                raise ValueError(f"missing analysis cache file: {entry[key]}")
    cohort = manifest.get("cohort")
    if not isinstance(cohort, Mapping):
        raise ValueError("analysis cache has no monitor cohort")
    cohort_contract = {
        "path": "monitor_slides.json",
        "cohort_size": 30,
        "positive_slides": 15,
        "seed": 20_260_831,
    }
    cohort_mismatches = {
        name: (cohort.get(name), expected)
        for name, expected in cohort_contract.items()
        if cohort.get(name) != expected
    }
    if cohort_mismatches:
        raise ValueError(f"analysis monitor cohort contract mismatch: {cohort_mismatches}")
    cohort_path = destination / str(cohort["path"])
    if _sha256_file(cohort_path) != str(cohort["sha256"]):
        raise ValueError("monitor cohort checksum mismatch")
    cohort_document = json.loads(cohort_path.read_text(encoding="utf-8"))
    slides = cohort_document.get("slides")
    if not isinstance(slides, list) or len(slides) != int(cohort["cohort_size"]):
        raise ValueError("monitor cohort size mismatch")
    expected_cohort = select_monitor_cohort(
        (
            {
                "slide_id": str(slide_id),
                "label": int(entry["label"]),
                "split": "train",
                "annotation_available": bool(entry["annotation_available"]),
                "annotation_exhaustive": bool(entry["annotation_exhaustive"]),
            }
            for slide_id, entry in entries.items()
        ),
        cohort_size=30,
        positive_slides=15,
        seed=20_260_831,
    )
    observed_ids = tuple(str(item.get("slide_id", "")) for item in slides)
    expected_ids = tuple(str(item["slide_id"]) for item in expected_cohort)
    if observed_ids != expected_ids:
        raise ValueError("monitor cohort does not match deterministic selection")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--config", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--tumor-mask-level", type=int, default=5)
    build.add_argument("--cohort-size", type=int, default=30)
    build.add_argument("--positive-slides", type=int, default=15)
    build.add_argument("--cohort-seed", type=int, default=20_260_831)
    verify = commands.add_parser("verify")
    verify.add_argument("--config", required=True, type=Path)
    verify.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "build":
        result = build_analysis_cache(
            arguments.config,
            arguments.output,
            tumor_mask_level=arguments.tumor_mask_level,
            cohort_size=arguments.cohort_size,
            positive_slides=arguments.positive_slides,
            cohort_seed=arguments.cohort_seed,
        )
    else:
        result = verify_analysis_cache(arguments.config, arguments.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = [
    "NON_EXHAUSTIVE_CAMELYON16_TUMOR_SLIDES",
    "TUMOR_LABEL",
    "AnalysisCache",
    "RasterTransform",
    "SlideAnalysis",
    "build_analysis_cache",
    "index_tumor_cells",
    "project_tumor_mask_to_overview",
    "read_tiff_mask_page",
    "select_monitor_cohort",
    "tumor_binary",
    "verify_analysis_cache",
    "write_monitor_cohort",
]


if __name__ == "__main__":
    raise SystemExit(main())
