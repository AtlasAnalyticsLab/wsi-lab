"""End-to-end geometric preprocessing and environment artifact adapter."""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from wsi_lab.constants import (
    CELL_ID_SLIDE_BLOCK_SIZE,
    LEVEL0_FOOTPRINTS,
    NAVIGATION_MAGNIFICATIONS,
)
from wsi_lab.data.base import DatasetAdapter
from wsi_lab.data.embeddings.base import EmbeddingMetadata, EmbeddingProvider
from wsi_lab.data.embeddings.hdf5 import HDF5EmbeddingProvider
from wsi_lab.data.splits import DatasetSplit
from wsi_lab.errors import ArtifactCompatibilityError, PreprocessingError, TestSplitAccessError
from wsi_lab.preprocessing.artifacts import ArtifactStore, SlideOffset
from wsi_lab.preprocessing.atlaspatch import (
    AtlasPatchOptions,
    ensure_mpp_override_csv,
)
from wsi_lab.preprocessing.geometry import GeometryBuilder
from wsi_lab.preprocessing.mapping import MappingSummary, map_embeddings_to_grid
from wsi_lab.preprocessing.tissue import (
    TissueMask,
    apply_tissue_to_geometry,
    choose_start_cell,
    load_or_create_tissue_mask,
)
from wsi_lab.utils.fingerprints import file_fingerprint, fingerprint
from wsi_lab.wsi.reader import OpenSlideReader, OverviewTransform, SlideMetadata, WSIReader


@dataclass(frozen=True, slots=True)
class PreprocessingOptions:
    overview_size: int = 224
    mapping_tolerance: int = 0
    tissue_source: str = "atlaspatch"
    tissue_threshold: float = 0.01
    tissue_rebuild: bool = False
    strict_mapping: bool = True
    atlaspatch: AtlasPatchOptions = field(default_factory=AtlasPatchOptions)
    tissue_cache_root: Path | None = None

    def __post_init__(self) -> None:
        if self.overview_size < 1:
            raise ValueError("overview_size must be positive")
        if self.mapping_tolerance < 0:
            raise ValueError("mapping_tolerance cannot be negative")
        if not 0.0 <= self.tissue_threshold <= 1.0:
            raise ValueError("tissue_threshold must lie in [0,1]")
        if self.tissue_source not in {"auto", "existing", "atlaspatch"}:
            raise ValueError(f"unsupported tissue_source: {self.tissue_source}")

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        tissue_cache_root: str | Path | None = None,
        mpp_csv: str | Path | None = None,
    ) -> PreprocessingOptions:
        tissue = config.tissue
        workspace = config.workspace
        tissue_source = tissue.source
        configured_root = (
            Path(tissue_cache_root).expanduser().resolve()
            if tissue_cache_root is not None
            else Path(workspace.root).expanduser().resolve() / "tissue" / str(config.dataset.name)
        )
        return cls(
            overview_size=int(config.environment.overview_size),
            mapping_tolerance=int(config.embeddings.mapping_tolerance),
            tissue_source=str(getattr(tissue_source, "value", tissue_source)),
            tissue_threshold=float(tissue.threshold),
            tissue_rebuild=bool(tissue.rebuild),
            atlaspatch=AtlasPatchOptions.from_config(tissue.atlaspatch, mpp_csv=mpp_csv),
            tissue_cache_root=configured_root,
        )


@dataclass(frozen=True, slots=True)
class SlidePreprocessingResult:
    slide_id: str
    artifact_path: Path
    input_fingerprint: str
    reused: bool
    cell_id_offset: int
    num_cells: int
    start_cell_id: int
    mapping_summaries: Mapping[float, MappingSummary]


class Preprocessor:
    """Generate dense geometry and attach sparse HDF5 source-row mappings."""

    def __init__(
        self,
        adapter: DatasetAdapter,
        embeddings: EmbeddingProvider,
        wsi_reader: WSIReader,
        artifacts: ArtifactStore,
        *,
        options: PreprocessingOptions | None = None,
    ) -> None:
        self.adapter = adapter
        self.embeddings = embeddings
        self.wsi_reader = wsi_reader
        self.artifacts = artifacts
        self.options = options or PreprocessingOptions()
        self.geometry_builder = GeometryBuilder()
        self._offsets: dict[str, SlideOffset] | None = None
        self._offset_fingerprint: str | None = None
        self._prepared_tissue: dict[str, tuple[np.ndarray, OverviewTransform, TissueMask]] = {}
        # Checkpoint hashing can be expensive; perform it at most once.
        self._atlaspatch_identity = self.options.atlaspatch.cache_identity()

    def _ensure_offsets(self, required_slide_ids: Sequence[str]) -> dict[str, SlideOffset]:
        """Materialize sparse, deterministic dataset-global cell-ID blocks.

        A slide owns ``adapter_index * CELL_ID_SLIDE_BLOCK_SIZE`` regardless of
        which other slides have been processed.  Arbitrary bounded subsets are
        therefore safe: only explicitly requested HDF5 metadata is inspected,
        while cell IDs remain deterministic and globally non-overlapping.
        """

        records = tuple(self.adapter.iter_slides())
        ordered_ids = tuple(record.slide_id for record in records)
        positions = {slide_id: index for index, slide_id in enumerate(ordered_ids)}
        requested = tuple(dict.fromkeys(required_slide_ids))
        unknown = set(requested).difference(positions)
        if unknown:
            raise KeyError(f"unknown slide IDs: {sorted(unknown)[:8]}")
        offset_fingerprint = fingerprint(
            {
                "navigation_magnifications": NAVIGATION_MAGNIFICATIONS,
                "level0_footprints": LEVEL0_FOOTPRINTS,
                "allocation": "fixed-slide-index-block-v1",
                "slide_block_size": CELL_ID_SLIDE_BLOCK_SIZE,
                "slides_in_adapter_order": ordered_ids,
            }
        )
        self._offset_fingerprint = offset_fingerprint
        if self._offsets is None:
            try:
                offsets = self.artifacts.load_offsets(expected_input_fingerprint=offset_fingerprint)
            except ArtifactCompatibilityError:
                offsets = {}
        else:
            offsets = dict(self._offsets)

        for slide_id, item in offsets.items():
            try:
                expected_index = positions[slide_id]
            except KeyError as exc:
                raise ArtifactCompatibilityError(
                    f"persisted cell offsets contain unknown slide {slide_id}"
                ) from exc
            expected_offset = expected_index * CELL_ID_SLIDE_BLOCK_SIZE
            if item.slide_index != expected_index or item.offset != expected_offset:
                raise ArtifactCompatibilityError(
                    f"persisted cell block disagrees with adapter index for {slide_id}"
                )
        if not requested or set(requested).issubset(offsets):
            self._offsets = offsets
            return offsets

        for slide_id in requested:
            if slide_id in offsets:
                continue
            slide_index = positions[slide_id]
            record = records[slide_index]
            metadata = self.embeddings.metadata(record.slide_id, self.adapter.magnifications[0])
            count = sum(
                math.ceil(metadata.level0_height / LEVEL0_FOOTPRINTS[magnification])
                * math.ceil(metadata.level0_width / LEVEL0_FOOTPRINTS[magnification])
                for magnification in NAVIGATION_MAGNIFICATIONS
            )
            if count >= CELL_ID_SLIDE_BLOCK_SIZE:
                raise PreprocessingError(
                    f"{record.slide_id} requires {count} cells, exceeding its fixed "
                    f"cell-ID block capacity {CELL_ID_SLIDE_BLOCK_SIZE - 1}"
                )
            offset = slide_index * CELL_ID_SLIDE_BLOCK_SIZE
            if offset > np.iinfo(np.int64).max - (count - 1):
                raise PreprocessingError(
                    f"dataset-global cell-ID block for {record.slide_id} exceeds int64"
                )
            offsets[record.slide_id] = SlideOffset(
                slide_id=record.slide_id,
                slide_index=slide_index,
                offset=offset,
                num_cells=count,
                level0_width=metadata.level0_width,
                level0_height=metadata.level0_height,
            )
        self.artifacts.write_offsets(offsets, input_fingerprint=offset_fingerprint)
        self._offsets = dict(sorted(offsets.items(), key=lambda item: item[1].slide_index))
        return self._offsets

    def _embedding_metadata(self, slide_id: str) -> dict[float, EmbeddingMetadata]:
        values = {
            magnification: self.embeddings.metadata(slide_id, magnification)
            for magnification in self.adapter.magnifications
        }
        dimensions = {(item.level0_width, item.level0_height) for item in values.values()}
        base_magnifications = {item.level0_magnification for item in values.values()}
        feature_dimensions = {item.feature_dim for item in values.values()}
        if len(dimensions) != 1:
            raise PreprocessingError(
                f"cross-scale WSI dimensions disagree for {slide_id}: {dimensions}"
            )
        if len(base_magnifications) != 1:
            raise PreprocessingError(
                f"cross-scale level-0 magnifications disagree for {slide_id}: {base_magnifications}"
            )
        if len(feature_dimensions) != 1:
            raise PreprocessingError(
                f"cross-scale feature dimensions disagree for {slide_id}: {feature_dimensions}"
            )
        for magnification, item in values.items():
            expected = LEVEL0_FOOTPRINTS[float(magnification)]
            if item.patch_size_level0 != expected or item.stride_level0 != expected:
                raise PreprocessingError(
                    f"{slide_id} {magnification:g}x footprint/stride "
                    f"{item.patch_size_level0}/{item.stride_level0} != canonical {expected}"
                )
        return values

    def _atlaspatch_options_for_slide(
        self,
        slide_id: str,
        slide_path: Path,
        wsi: SlideMetadata,
        embedding_metadata: Mapping[float, EmbeddingMetadata],
    ) -> tuple[AtlasPatchOptions, tuple[float | None, float | None]]:
        if wsi.mpp_x is not None and wsi.mpp_y is not None:
            if min(wsi.mpp_x, wsi.mpp_y) <= 0 or not np.isclose(
                wsi.mpp_x, wsi.mpp_y, rtol=0.05, atol=0.0
            ):
                raise PreprocessingError(
                    f"invalid or anisotropic WSI MPP for {slide_id}: {wsi.mpp_x}, {wsi.mpp_y}"
                )
            return self.options.atlaspatch, (wsi.mpp_x, wsi.mpp_y)
        available = [item.mpp for item in embedding_metadata.values() if item.mpp is not None]
        if not available or any(value <= 0 or not np.isfinite(value) for value in available):
            raise PreprocessingError(
                f"{slide_id} has no usable WSI MPP and embedding metadata has no valid MPP override"
            )
        if not all(
            np.isclose(available[0], value, rtol=0.01, atol=1e-6) for value in available[1:]
        ):
            raise PreprocessingError(
                f"embedding MPP metadata disagrees across scales for {slide_id}: {available}"
            )
        root = self.options.tissue_cache_root or self.artifacts.root / "generated_tissue"
        csv_path = ensure_mpp_override_csv(
            root / "mpp" / f"{slide_id}.csv",
            slide_path,
            float(available[0]),
        )
        return replace(self.options.atlaspatch, mpp_csv=csv_path), (available[0], available[0])

    def _slide_fingerprint(
        self,
        slide_id: str,
        slide_path: Path,
        offset: SlideOffset,
        metadata: Mapping[float, EmbeddingMetadata],
        atlaspatch_options: AtlasPatchOptions,
        tissue_mask_path: Path | None,
    ) -> str:
        return fingerprint(
            {
                # Topology v2 adds tissue-valid navigation and snapped ROI
                # metadata.  Existing dense-grid artifacts must be rebuilt.
                "schema": 2,
                "slide_id": slide_id,
                "wsi": file_fingerprint(slide_path),
                "embeddings": {
                    magnification: {
                        "file": file_fingerprint(item.path),
                        "feature_key": item.feature_key,
                        "coordinate_key": item.coordinate_key,
                        "rows": item.num_instances,
                        "feature_dim": item.feature_dim,
                        "patch_size_level0": item.patch_size_level0,
                    }
                    for magnification, item in metadata.items()
                },
                "existing_tissue": (
                    file_fingerprint(tissue_mask_path)
                    if tissue_mask_path is not None and tissue_mask_path.is_file()
                    else None
                ),
                "offset": asdict(offset),
                "offset_table": self._offset_fingerprint,
                "options": {
                    "overview_size": self.options.overview_size,
                    "mapping_tolerance": self.options.mapping_tolerance,
                    "tissue_source": self.options.tissue_source,
                    "tissue_threshold": self.options.tissue_threshold,
                    "strict_mapping": self.options.strict_mapping,
                    "atlaspatch": {
                        **self._atlaspatch_identity,
                        "mpp_csv": (
                            file_fingerprint(atlaspatch_options.mpp_csv)
                            if atlaspatch_options.mpp_csv is not None
                            else None
                        ),
                    },
                },
            }
        )

    def _inspect_slide_context(
        self, slide_id: str, *, require_global_offset: bool = True
    ) -> tuple[
        Any,
        SlideOffset | None,
        dict[float, EmbeddingMetadata],
        SlideMetadata,
        float,
        AtlasPatchOptions,
        tuple[float | None, float | None],
    ]:
        record = self.adapter.get_slide(slide_id)
        offset = self._ensure_offsets((slide_id,))[slide_id] if require_global_offset else None
        embedding_metadata = self._embedding_metadata(slide_id)
        wsi = self.wsi_reader.inspect(record.slide_path)
        embedded_dimensions = next(
            iter({(item.level0_width, item.level0_height) for item in embedding_metadata.values()})
        )
        if tuple(wsi.dimensions) != embedded_dimensions:
            raise PreprocessingError(
                f"WSI/embedding dimension mismatch for {slide_id}: {wsi.dimensions} != {embedded_dimensions}"
            )
        level0_magnification = next(
            iter({item.level0_magnification for item in embedding_metadata.values()})
        )
        if wsi.objective_power is not None and not np.isclose(
            wsi.objective_power, level0_magnification, rtol=0.05, atol=0.0
        ):
            raise PreprocessingError(
                f"WSI objective {wsi.objective_power:g}x disagrees with embedding level-0 "
                f"magnification {level0_magnification:g}x for {slide_id}"
            )
        atlaspatch_options, mpp = self._atlaspatch_options_for_slide(
            slide_id, record.slide_path, wsi, embedding_metadata
        )
        return (
            record,
            offset,
            embedding_metadata,
            wsi,
            level0_magnification,
            atlaspatch_options,
            mpp,
        )

    def _materialize_tissue(
        self,
        slide_id: str,
        record: Any,
        atlaspatch_options: AtlasPatchOptions,
        mpp: tuple[float | None, float | None],
    ) -> tuple[np.ndarray, OverviewTransform, TissueMask]:
        cached = self._prepared_tissue.get(slide_id)
        if cached is not None:
            return cached
        overview, transform = self.wsi_reader.read_overview(
            record.slide_path,
            (self.options.overview_size, self.options.overview_size),
        )
        tissue_root = self.options.tissue_cache_root or self.artifacts.root / "generated_tissue"
        tissue = load_or_create_tissue_mask(
            record.slide_path,
            overview,
            transform,
            existing_path=record.tissue_mask_path,
            generated_path=tissue_root / "masks" / f"{slide_id}.npy",
            source=self.options.tissue_source,
            threshold=self.options.tissue_threshold,
            rebuild=self.options.tissue_rebuild,
            atlaspatch_options=atlaspatch_options,
            mpp=mpp,
        )
        result = (overview, transform, tissue)
        self._prepared_tissue[slide_id] = result
        return result

    def prepare_tissue_mask(self, slide_id: str) -> TissueMask:
        """Validate spatial metadata and materialize one reusable tissue mask."""

        (
            record,
            _offset,
            _embedding_metadata,
            _wsi,
            _level0_magnification,
            atlaspatch_options,
            mpp,
        ) = self._inspect_slide_context(slide_id, require_global_offset=False)
        _, _, tissue = self._materialize_tissue(slide_id, record, atlaspatch_options, mpp)
        return tissue

    def preprocess_slide(
        self, slide_id: str, *, overwrite: bool = False
    ) -> SlidePreprocessingResult:
        selected_record = self.adapter.get_slide(slide_id)
        if selected_record.label not in {0, 1}:
            raise PreprocessingError(f"cannot preprocess unlabeled slide {slide_id!r} for WSIEnv")
        (
            record,
            offset,
            embedding_metadata,
            wsi,
            level0_magnification,
            atlaspatch_options,
            mpp,
        ) = self._inspect_slide_context(slide_id)
        assert offset is not None
        input_fingerprint = self._slide_fingerprint(
            slide_id,
            record.slide_path,
            offset,
            embedding_metadata,
            atlaspatch_options,
            record.tissue_mask_path,
        )
        if (
            not overwrite
            and not self.options.tissue_rebuild
            and self.artifacts.is_slide_complete(
                slide_id, expected_input_fingerprint=input_fingerprint
            )
        ):
            stored = self.artifacts.load_metadata(
                slide_id, expected_input_fingerprint=input_fingerprint
            )
            reused_summaries = {
                float(key): MappingSummary(**value)
                for key, value in stored["mapping_summaries"].items()
            }
            return SlidePreprocessingResult(
                slide_id=slide_id,
                artifact_path=self.artifacts.slide_paths(slide_id).root,
                input_fingerprint=input_fingerprint,
                reused=True,
                cell_id_offset=int(stored["cell_id_offset"]),
                num_cells=int(stored["num_cells"]),
                start_cell_id=int(stored["start_cell_id"]),
                mapping_summaries=reused_summaries,
            )

        overview, transform, tissue = self._materialize_tissue(
            slide_id, record, atlaspatch_options, mpp
        )
        geometry = self.geometry_builder.build(
            slide_id,
            level0_width=wsi.dimensions[0],
            level0_height=wsi.dimensions[1],
            level0_magnification=level0_magnification,
            cell_id_offset=offset.offset,
            slide_index=offset.slide_index,
        )
        geometry = apply_tissue_to_geometry(geometry, tissue, transform)
        levels = dict(geometry.levels)
        summaries: dict[float, MappingSummary] = {}
        embedding_sources: dict[float, dict[str, Any]] = {}
        for magnification in self.adapter.magnifications:
            level = levels[float(magnification)]
            coordinates = self.embeddings.coordinates(slide_id, magnification)
            mapping = map_embeddings_to_grid(
                level,
                coordinates,
                tolerance=self.options.mapping_tolerance,
                level0_width=geometry.level0_width,
                level0_height=geometry.level0_height,
                strict=self.options.strict_mapping,
            )
            attached = mapping.attach(level)
            embedded = attached["embedding_index"] >= 0
            tissue_overlapping = embedded & (attached["tissue_fraction"] > 0.0)
            mapped_cells = int(np.count_nonzero(embedded))
            overlap_count = int(np.count_nonzero(tissue_overlapping))
            levels[float(magnification)] = replace(level, cells=attached)
            summaries[float(magnification)] = replace(
                mapping.summary,
                tissue_overlapping_cells=overlap_count,
                tissue_overlap_fraction=(overlap_count / mapped_cells if mapped_cells else None),
            )
            item = embedding_metadata[float(magnification)]
            embedding_sources[float(magnification)] = {
                "path": str(item.path),
                "file": file_fingerprint(item.path),
                "feature_key": item.feature_key,
                "coordinate_key": item.coordinate_key,
                "num_instances": item.num_instances,
                "feature_dim": item.feature_dim,
                "source_row_semantics": "embedding_index is the original HDF5 dataset row",
            }
        geometry = replace(geometry, levels=levels)
        start_cell_id = choose_start_cell(geometry)
        self.artifacts.write_slide(
            geometry=geometry,
            overview=overview,
            transform=transform,
            tissue_mask=tissue,
            start_cell_id=start_cell_id,
            label=record.label,
            slide_path=record.slide_path,
            input_fingerprint=input_fingerprint,
            embedding_sources=embedding_sources,
            mapping_summaries=summaries,
            overwrite=True,
        )
        return SlidePreprocessingResult(
            slide_id=slide_id,
            artifact_path=self.artifacts.slide_paths(slide_id).root,
            input_fingerprint=input_fingerprint,
            reused=False,
            cell_id_offset=offset.offset,
            num_cells=geometry.num_cells,
            start_cell_id=start_cell_id,
            mapping_summaries=summaries,
        )

    def preprocess(
        self,
        slide_ids: Sequence[str] | None = None,
        *,
        overwrite: bool = False,
        merge_index: bool = False,
    ) -> tuple[SlidePreprocessingResult, ...]:
        selected = tuple(slide_ids) if slide_ids is not None else self.adapter.slide_ids()
        unknown = set(selected).difference(self.adapter.slide_ids())
        if unknown:
            raise KeyError(f"unknown slide IDs: {sorted(unknown)[:8]}")
        self._ensure_offsets(selected)
        results = tuple(
            self.preprocess_slide(slide_id, overwrite=overwrite) for slide_id in selected
        )
        self.artifacts.write_index(selected, merge=merge_index)
        return results


class NavigationCellMapping(Mapping[int, Any]):
    """Lazy direct-index adapter from mmap grid rows to NavigationCell."""

    def __init__(self, arrays: Sequence[np.ndarray]) -> None:
        from wsi_lab.envs.wsi_env import NavigationCell

        self._cell_type = NavigationCell
        self._arrays = tuple(arrays)
        self._starts = tuple(int(array["cell_id"][0]) for array in self._arrays if len(array))
        if len(self._starts) != len(self._arrays):
            raise ArtifactCompatibilityError("empty navigation grid is not supported")
        for previous, following in pairwise(self._arrays):
            if int(previous["cell_id"][-1]) + 1 != int(following["cell_id"][0]):
                raise ArtifactCompatibilityError("navigation grid IDs are not contiguous")

    def __len__(self) -> int:
        return sum(len(array) for array in self._arrays)

    def __iter__(self) -> Iterator[int]:
        for array in self._arrays:
            yield from (int(value) for value in array["cell_id"])

    def __getitem__(self, cell_id: int) -> Any:
        key = int(cell_id)
        index = bisect.bisect_right(self._starts, key) - 1
        if index < 0:
            raise KeyError(cell_id)
        array = self._arrays[index]
        local = key - self._starts[index]
        if local < 0 or local >= len(array) or int(array["cell_id"][local]) != key:
            raise KeyError(cell_id)
        cell = array[local]
        tissue_fraction = float(cell["tissue_fraction"])
        if not np.isfinite(tissue_fraction):
            raise ArtifactCompatibilityError(f"cell {key} has no persisted tissue fraction")
        return self._cell_type(
            cell_id=key,
            magnification=float(cell["magnification"]),
            row=int(cell["row"]),
            column=int(cell["column"]),
            bounds=(
                int(cell["level0_x0"]),
                int(cell["level0_y0"]),
                int(cell["level0_x1"]),
                int(cell["level0_y1"]),
            ),
            parent_id=int(cell["parent_id"]),
            child_ids=(
                int(cell["child_tl_id"]),
                int(cell["child_tr_id"]),
                int(cell["child_bl_id"]),
                int(cell["child_br_id"]),
            ),
            neighbor_ids=(
                int(cell["neighbor_up"]),
                int(cell["neighbor_down"]),
                int(cell["neighbor_left"]),
                int(cell["neighbor_right"]),
            ),
            tissue_fraction=tissue_fraction,
            embedding_index=int(cell["embedding_index"]),
            tissue_valid=bool(cell["tissue_valid"]),
        )


class ProcessedEpisodeSource:
    """Production ``EpisodeSource`` backed by mmap grids and source HDF5 rows."""

    def __init__(
        self,
        adapter: DatasetAdapter,
        embeddings: EmbeddingProvider,
        wsi_reader: WSIReader,
        artifacts: ArtifactStore,
        split: DatasetSplit,
        *,
        allow_test: bool = False,
    ) -> None:
        if split.dataset != adapter.name:
            raise ValueError(f"split dataset {split.dataset!r} != adapter {adapter.name!r}")
        self.adapter = adapter
        self.embeddings = embeddings
        self.wsi_reader = wsi_reader
        self.artifacts = artifacts
        self.split = split
        self.allow_test = allow_test
        self._loaded_cells: dict[str, NavigationCellMapping] = {}

    @classmethod
    def from_config(
        cls,
        config: Any,
        artifacts: ArtifactStore,
        split: DatasetSplit,
        *,
        allow_test: bool = False,
    ) -> ProcessedEpisodeSource:
        from wsi_lab.data.factory import create_dataset_adapter

        adapter = create_dataset_adapter(config.dataset, config.embeddings)
        environment = config.environment
        reader = OpenSlideReader(
            handle_cache_size=int(environment.slide_handle_cache_size),
            viewport_cache_size=int(environment.viewport_cache_size),
            persistent_cache_dir=environment.persistent_rgb_cache,
        )
        return cls(
            adapter,
            HDF5EmbeddingProvider(adapter),
            reader,
            artifacts,
            split,
            allow_test=allow_test,
        )

    def slide_ids(self, split: str) -> Sequence[str]:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"unknown split: {split!r}")
        if split == "test" and not self.allow_test:
            raise TestSplitAccessError("test split access requires allow_test=True")
        return self.split.ids(split)  # type: ignore[arg-type]

    def load_slide(self, slide_id: str) -> Any:
        from wsi_lab.envs.wsi_env import LoadedSlide

        assignment = self.split.assignment(slide_id)
        if assignment.split == "test" and not self.allow_test:
            raise TestSplitAccessError("test split access requires allow_test=True")
        metadata = self.artifacts.load_metadata(slide_id)
        persisted = OverviewTransform.from_dict(metadata["overview_transform"])
        if slide_id not in self._loaded_cells:
            arrays = tuple(
                self.artifacts.load_grid(slide_id, magnification, mmap=True)
                for magnification in NAVIGATION_MAGNIFICATIONS
            )
            self._loaded_cells[slide_id] = NavigationCellMapping(arrays)
        navigation_roi = metadata["navigation_roi_level0"]
        return LoadedSlide(
            slide_id=slide_id,
            label=int(assignment.label),
            overview=self.artifacts.load_overview(slide_id, mmap=True),
            overview_transform=persisted,
            cells=self._loaded_cells[slide_id],
            start_cell_id=int(metadata["start_cell_id"]),
            navigation_roi_level0=(
                int(navigation_roi[0]),
                int(navigation_roi[1]),
                int(navigation_roi[2]),
                int(navigation_roi[3]),
            ),
        )

    def read_viewport(self, slide_id: str, cell: Any, output_size: int) -> np.ndarray:
        record = self.adapter.get_slide(slide_id)
        # Bounds are deliberately nominal: an edge cell retains its canonical
        # footprint and the WSI reader supplies lossless white padding.
        x0, y0, x1, y1 = cell.bounds
        return self.wsi_reader.read_region(
            record.slide_path,
            (int(x0), int(y0), int(x1), int(y1)),
            (int(output_size), int(output_size)),
        )

    def read_embedding(self, slide_id: str, cell: Any) -> np.ndarray:
        source_row = int(cell.embedding_index)
        if source_row < 0:
            raise ValueError(f"cell {cell.cell_id} has no embedding source row")
        features = self.embeddings.features(
            slide_id,
            float(cell.magnification),
            [source_row],
        )
        if features.ndim != 2 or len(features) != 1:
            raise PreprocessingError(
                f"embedding provider returned shape {features.shape} for one persisted source row"
            )
        return np.asarray(features[0]).copy()

    def close(self) -> None:
        self._loaded_cells.clear()
        self.wsi_reader.close()


__all__ = [
    "NavigationCellMapping",
    "PreprocessingOptions",
    "Preprocessor",
    "ProcessedEpisodeSource",
    "SlidePreprocessingResult",
]
