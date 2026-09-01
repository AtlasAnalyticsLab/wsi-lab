#!/usr/bin/env python3
"""Prepare and benchmark WSI-Lab without touching published package code.

Run as a module from the repository root, for example::

    python -m benchmarks.histogym_comparison.wsi.runner validate --cohort pilot

The measured ``prepare`` and ``online`` commands are intentionally separate so
the outer orchestrator can place the agreed manual cache-drop boundary before
the complete WSI-Lab block.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from collections import Counter
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import psutil

# Direct-file execution is supported in addition to ``python -m``.
if __package__ in {None, ""}:  # pragma: no cover - exercised by operator invocation
    _REPOSITORY = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_REPOSITORY))

from benchmarks.histogym_comparison.common.protocol import (
    EXPECTED_QUOTAS,
    directory_snapshot,
    latency_summary_ns,
    observation_nbytes,
)
from benchmarks.histogym_comparison.common.resource_monitor import ResourceMonitor
from benchmarks.histogym_comparison.wsi.cohort import (
    CohortSelection,
    SlideSpec,
    atomic_json_dump,
    json_default,
    load_cohort,
    sha256_file,
)
from benchmarks.histogym_comparison.wsi.episode_source import (
    FixedEmbeddingEpisodeSource,
)
from benchmarks.histogym_comparison.wsi.scheduler import (
    AcceptedActionScheduler,
    abstract_class,
)

MODULE_STARTED_NS = time.perf_counter_ns()
PROCESS_CREATED_EPOCH_NS = int(psutil.Process(os.getpid()).create_time() * 1e9)
HARNESS_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
REPOSITORY_ROOT = Path(
    os.environ.get("WSI_LAB_SOURCE_ROOT", HARNESS_REPOSITORY_ROOT)
).expanduser().resolve()
DEFAULT_MANIFEST = (
    HARNESS_REPOSITORY_ROOT / "benchmarks/histogym_comparison/manifests/cohort.json"
)
DEFAULT_CONFIG = HARNESS_REPOSITORY_ROOT / "configs/camelyon16.yaml"
EXPECTED_OBSERVATION_BYTES_224 = (5 * 224 * 224 + 3 * 224 * 224 + 5) * 4
RESULT_SCHEMA_VERSION = 1
WSI_LAB_SOURCE_VERIFIER = REPOSITORY_ROOT / "exp_reprod/scripts/verify_wsi_source.py"


@dataclass(frozen=True, slots=True)
class WorkspaceLayout:
    root: Path
    marker: Path
    tissue_root: Path
    tissue_context: Path
    processed_root: Path
    raw_root: Path
    results_root: Path

    @classmethod
    def from_root(cls, root: str | Path) -> WorkspaceLayout:
        value = Path(root).expanduser().resolve()
        return cls(
            root=value,
            marker=value / "experiment.json",
            tissue_root=value / "artifacts/tissue",
            tissue_context=value / "artifacts/tissue_context",
            processed_root=value / "artifacts/processed",
            raw_root=value / "raw",
            results_root=value / "results",
        )


class JsonlStream:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream: TextIO | None = None

    def __enter__(self) -> JsonlStream:
        self._stream = self.path.open("x", encoding="utf-8")
        return self

    def write(self, value: Any) -> None:
        if self._stream is None:
            raise RuntimeError("JSONL stream is not open")
        self._stream.write(
            json.dumps(
                value,
                default=json_default,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        self._stream.write("\n")

    def flush(self) -> None:
        if self._stream is not None:
            self._stream.flush()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._stream is not None:
            self._stream.flush()
            self._stream.close()
            self._stream = None


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _verify_source_identity() -> dict[str, Any]:
    try:
        verification = subprocess.run(
            [
                sys.executable,
                str(WSI_LAB_SOURCE_VERIFIER),
                "--repository-root",
                str(REPOSITORY_ROOT),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        raise RuntimeError(f"cannot verify submitted WSI-Lab source: {detail}") from exc
    try:
        return dict(json.loads(verification.stdout))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("WSI-Lab source verifier returned invalid JSON") from exc


def _versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {"python": platform.python_version()}
    for distribution in (
        "wsi-lab",
        "numpy",
        "gymnasium",
        "openslide-python",
        "atlas-patch",
        "psutil",
    ):
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[distribution] = None
    return result


def _provenance(config_path: Path, selection: CohortSelection) -> dict[str, Any]:
    return {
        "created_at_utc": _now(),
        "source_identity_check": _verify_source_identity(),
        "config_sha256": sha256_file(config_path),
        "cohort": selection.to_dict(),
        "versions": _versions(),
    }


def _infer_dataset_root(selection: CohortSelection) -> Path:
    roots = {slide.wsi_path.parent.parent.resolve() for slide in selection.slides}
    if len(roots) != 1:
        raise ValueError(f"manifest WSI paths do not share one dataset root: {sorted(roots)}")
    return roots.pop()


def _load_native_context(config_path: Path, selection: CohortSelection):
    # The checked-in config uses ${CAMELYON16_ROOT}. Deriving it from the fixed
    # manifest makes the private runner self-contained without changing config.
    os.environ.setdefault("CAMELYON16_ROOT", str(_infer_dataset_root(selection)))

    from wsi_lab.config.loader import load_config
    from wsi_lab.data.factory import create_dataset_adapter

    config = load_config(config_path)
    adapter = create_dataset_adapter(config.dataset, config.embeddings)
    environment = config.environment
    tissue_source = str(getattr(config.tissue.source, "value", config.tissue.source))
    if tissue_source != "atlaspatch":
        raise ValueError(f"benchmark requires native AtlasPatch tissue source, got {tissue_source}")
    if bool(config.tissue.rebuild):
        raise ValueError("fresh experiment directories, not tissue.rebuild, define cold preparation")
    if int(environment.overview_size) != 224 or int(environment.viewport_size) != 224:
        raise ValueError("benchmark protocol requires full 224x224 overview and viewport")
    if str(environment.start_position) != "center":
        raise ValueError("benchmark protocol requires the configured center start")
    if bool(environment.geometric_augmentation) or bool(environment.photometric_augmentation):
        raise ValueError("benchmark protocol excludes geometric and photometric augmentation")
    if int(environment.max_episode_steps) <= sum(EXPECTED_QUOTAS.values()):
        raise ValueError("max_episode_steps must exceed 100 accepted transitions")
    if environment.persistent_rgb_cache is not None:
        raise ValueError("persistent RGB caching must be disabled for the agreed protocol")

    expected_ids = set(adapter.slide_ids())
    for slide in selection.slides:
        if slide.slide_id not in expected_ids:
            raise ValueError(f"manifest slide is unknown to native adapter: {slide.slide_id}")
        record = adapter.get_slide(slide.slide_id)
        if record.official_cohort != "development" or record.label != 1:
            raise ValueError(f"{slide.slide_id} is not an annotated development tumor slide")
        if record.annotation_path is None:
            raise ValueError(f"native adapter has no XML annotation for {slide.slide_id}")
        if slide.label != int(record.label):
            raise ValueError(f"label drift for {slide.slide_id}: manifest={slide.label}")
        if slide.wsi_path != record.slide_path.resolve():
            raise ValueError(
                f"WSI path drift for {slide.slide_id}: {slide.wsi_path} != {record.slide_path}"
            )
        if slide.xml_path != record.annotation_path.resolve():
            raise ValueError(
                f"XML path drift for {slide.slide_id}: {slide.xml_path} != {record.annotation_path}"
            )
        if not slide.wsi_path.is_file() or not slide.xml_path.is_file():
            raise FileNotFoundError(f"missing WSI/XML input for {slide.slide_id}")
        for magnification in adapter.magnifications:
            embedding = adapter.embedding_location(slide.slide_id, magnification).path
            if not embedding.is_file():
                raise FileNotFoundError(embedding)
    return config, adapter


def _marker_payload(
    *,
    selection: CohortSelection,
    config_path: Path,
    status: str,
    attempt: int,
    device: str,
    resumed: bool,
) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "backend": "wsi-lab",
        "status": status,
        "attempt": attempt,
        "resumed": resumed,
        "updated_at_utc": _now(),
        "manifest_sha256": selection.manifest_sha256,
        "selection_sha256": selection.selection_sha256,
        "cohort": selection.cohort,
        "slide_ids": [slide.slide_id for slide in selection.slides],
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "atlaspatch_device": device,
        "fresh_measurement_valid": not resumed,
    }


def _initialize_workspace(
    layout: WorkspaceLayout,
    selection: CohortSelection,
    config_path: Path,
    *,
    device: str,
    resume: bool,
) -> tuple[int, bool]:
    existing_entries = list(layout.root.iterdir()) if layout.root.is_dir() else []
    if layout.root.exists() and not layout.root.is_dir():
        raise ValueError(f"workspace root is not a directory: {layout.root}")
    if existing_entries and not resume:
        raise FileExistsError(
            f"fresh preparation workspace is not empty: {layout.root}; "
            "choose a new path or explicitly use --resume after an interrupted experiment"
        )
    attempt = 1
    resumed = False
    if existing_entries:
        if not layout.marker.is_file():
            raise ValueError(f"cannot resume unmarked workspace: {layout.root}")
        previous = json.loads(layout.marker.read_text(encoding="utf-8"))
        for key, expected in (
            ("backend", "wsi-lab"),
            ("manifest_sha256", selection.manifest_sha256),
            ("selection_sha256", selection.selection_sha256),
            ("config_sha256", sha256_file(config_path)),
            ("atlaspatch_device", device),
        ):
            if previous.get(key) != expected:
                raise ValueError(f"resume marker mismatch for {key}: {previous.get(key)!r}")
        if previous.get("status") == "complete":
            raise ValueError("preparation is already complete; do not rerun it")
        attempt = int(previous.get("attempt", 0)) + 1
        resumed = True
    layout.root.mkdir(parents=True, exist_ok=True)
    layout.raw_root.mkdir(parents=True, exist_ok=True)
    layout.results_root.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(
        layout.marker,
        _marker_payload(
            selection=selection,
            config_path=config_path,
            status="running",
            attempt=attempt,
            device=device,
            resumed=resumed,
        ),
    )
    return attempt, resumed


def _verify_prepared_workspace(
    layout: WorkspaceLayout,
    selection: CohortSelection,
    config_path: Path,
) -> dict[str, Any]:
    if not layout.marker.is_file():
        raise FileNotFoundError(f"missing WSI-Lab preparation marker: {layout.marker}")
    marker = json.loads(layout.marker.read_text(encoding="utf-8"))
    if marker.get("status") != "complete":
        raise ValueError(f"WSI-Lab preparation is not complete: {layout.marker}")
    for key, expected in (
        ("backend", "wsi-lab"),
        ("manifest_sha256", selection.manifest_sha256),
        ("selection_sha256", selection.selection_sha256),
        ("config_sha256", sha256_file(config_path)),
    ):
        if marker.get(key) != expected:
            raise ValueError(f"prepared workspace mismatch for {key}: {marker.get(key)!r}")
    if marker.get("slide_ids") != [slide.slide_id for slide in selection.slides]:
        raise ValueError("prepared workspace slide order differs from the fixed manifest")
    return marker


def _path_group_snapshot(paths: list[Path]) -> dict[str, int]:
    unique = sorted({path.resolve() for path in paths if path.exists()}, key=str)
    logical = 0
    allocated = 0
    files = 0
    directories = 0
    for path in unique:
        candidates = [path, *path.rglob("*")] if path.is_dir() else [path]
        for candidate in candidates:
            if candidate.is_dir():
                directories += 1
            elif candidate.is_file():
                stat = candidate.stat()
                logical += stat.st_size
                allocated += stat.st_blocks * 512
                files += 1
    return {
        "file_count": files,
        "directory_count": directories,
        "logical_bytes": logical,
        "allocated_bytes": allocated,
    }


def _combine_phase_resources(phases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Combine sequential monitor blocks while preserving their raw summaries."""

    if not phases:
        return {"phases": {}}
    ordered = list(phases.values())
    wall_seconds = sum(float(item["wall_seconds"]) for item in ordered)
    cpu_seconds = sum(float(item["cpu_seconds"]) for item in ordered)
    logical_cpus = int(ordered[0]["logical_cpu_count"])
    cpu_one_core_percent = cpu_seconds / wall_seconds * 100.0 if wall_seconds else 0.0
    io_keys = set().union(*(item.get("io", {}).keys() for item in ordered))
    io = {
        key: sum(int(item.get("io", {}).get(key, 0)) for item in ordered)
        for key in sorted(io_keys)
    }
    gpu_source = next((item for item in ordered if item.get("gpu_index") is not None), {})
    baseline_rss = int(ordered[0]["baseline_rss_bytes"])
    peak_rss = max(int(item["peak_rss_bytes"]) for item in ordered)
    return {
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "average_cpu_percent_one_core_100": cpu_one_core_percent,
        "average_cpu_percent_machine_normalized": cpu_one_core_percent / logical_cpus,
        "logical_cpu_count": logical_cpus,
        "baseline_rss_bytes": baseline_rss,
        "peak_rss_bytes": peak_rss,
        "peak_incremental_rss_bytes": max(0, peak_rss - baseline_rss),
        "process_samples": sum(int(item["process_samples"]) for item in ordered),
        "io": io,
        "gpu_index": gpu_source.get("gpu_index"),
        "gpu_samples": gpu_source.get("gpu_samples", 0),
        "gpu_baseline_memory_mib": gpu_source.get("gpu_baseline_memory_mib"),
        "gpu_peak_memory_mib": gpu_source.get("gpu_peak_memory_mib"),
        "gpu_peak_incremental_memory_mib": gpu_source.get(
            "gpu_peak_incremental_memory_mib"
        ),
        "gpu_peak_utilization_percent": gpu_source.get("gpu_peak_utilization_percent"),
        "gpu_average_utilization_percent": gpu_source.get(
            "gpu_average_utilization_percent"
        ),
        "gpu_peak_power_watts": gpu_source.get("gpu_peak_power_watts"),
        "phases": phases,
    }


def _preprocessing_options(config: Any, tissue_root: Path, device: str):
    from wsi_lab.preprocessing.pipeline import PreprocessingOptions

    options = PreprocessingOptions.from_config(config, tissue_cache_root=tissue_root)
    return replace(options, atlaspatch=replace(options.atlaspatch, device=device))


def _run_tissue_stage(
    config: Any,
    adapter: Any,
    selection: CohortSelection,
    layout: WorkspaceLayout,
    *,
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    from wsi_lab.data.embeddings.hdf5 import HDF5EmbeddingProvider
    from wsi_lab.preprocessing.artifacts import ArtifactStore
    from wsi_lab.preprocessing.pipeline import Preprocessor
    from wsi_lab.utils.atomic import atomic_json_dump as native_atomic_json_dump
    from wsi_lab.wsi.reader import OpenSlideReader

    setup_started = time.perf_counter_ns()
    reader = OpenSlideReader(
        handle_cache_size=int(config.environment.slide_handle_cache_size),
        viewport_cache_size=0,
    )
    processor = Preprocessor(
        adapter,
        HDF5EmbeddingProvider(adapter),
        reader,
        ArtifactStore(layout.tissue_context),
        options=_preprocessing_options(config, layout.tissue_root, device),
    )
    setup_ns = time.perf_counter_ns() - setup_started
    rows: list[dict[str, Any]] = []
    try:
        for slide in selection.slides:
            started = time.perf_counter_ns()
            tissue = processor.prepare_tissue_mask(slide.slide_id)
            duration_ns = time.perf_counter_ns() - started
            row = {
                "ordinal": slide.ordinal,
                "slide_id": slide.slide_id,
                "duration_ns": duration_ns,
                "source": str(tissue.source),
                "source_path": str(tissue.source_path) if tissue.source_path else None,
                "source_page": tissue.source_page,
                "shape": list(tissue.values.shape),
                "tissue_pixels": int(tissue.values.sum()),
                "provenance": dict(tissue.provenance),
                "storage": None,
            }
            rows.append(row)
    finally:
        close_started = time.perf_counter_ns()
        reader.close()
        close_ns = time.perf_counter_ns() - close_started
    index_entries = [
        {
            "slide_id": row["slide_id"],
            "source": row["source"],
            "source_path": row["source_path"],
            "source_page": row["source_page"],
            "shape": row["shape"],
            "tissue_pixels": row["tissue_pixels"],
            "provenance": row["provenance"],
        }
        for row in rows
    ]
    index_started = time.perf_counter_ns()
    native_atomic_json_dump(
        layout.tissue_root / "index.json",
        {
            "status": "complete",
            "fingerprint": selection.selection_sha256,
            "manifest_sha256": selection.manifest_sha256,
            "scope": "fixed-benchmark-cohort",
            "cohort": selection.cohort,
            "official_test_opened": False,
            "slides": index_entries,
        },
    )
    index_write_ns = time.perf_counter_ns() - index_started
    return rows, {
        "setup_ns": setup_ns,
        "close_ns": close_ns,
        "index_write_ns": index_write_ns,
    }


def _run_navigation_artifact_stage(
    config: Any,
    adapter: Any,
    selection: CohortSelection,
    layout: WorkspaceLayout,
    *,
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    from wsi_lab.data.embeddings.hdf5 import HDF5EmbeddingProvider
    from wsi_lab.preprocessing.artifacts import ArtifactStore
    from wsi_lab.preprocessing.pipeline import Preprocessor
    from wsi_lab.wsi.reader import OpenSlideReader

    write_ns: dict[str, int] = {}
    slide_ns: dict[str, int] = {}

    class TimingStore(ArtifactStore):
        def write_slide(self, *args: Any, **kwargs: Any):
            geometry = kwargs.get("geometry")
            slide_id = str(getattr(geometry, "slide_id", "unknown"))
            started = time.perf_counter_ns()
            try:
                return super().write_slide(*args, **kwargs)
            finally:
                write_ns[slide_id] = time.perf_counter_ns() - started

    class TimingPreprocessor(Preprocessor):
        def preprocess_slide(self, slide_id: str, *, overwrite: bool = False):
            started = time.perf_counter_ns()
            try:
                return super().preprocess_slide(slide_id, overwrite=overwrite)
            finally:
                slide_ns[str(slide_id)] = time.perf_counter_ns() - started

    setup_started = time.perf_counter_ns()
    reader = OpenSlideReader(
        handle_cache_size=int(config.environment.slide_handle_cache_size),
        viewport_cache_size=int(config.environment.viewport_cache_size),
        persistent_cache_dir=None,
    )
    store = TimingStore(layout.processed_root)
    processor = TimingPreprocessor(
        adapter,
        HDF5EmbeddingProvider(adapter),
        reader,
        store,
        options=_preprocessing_options(config, layout.tissue_root, device),
    )
    setup_ns = time.perf_counter_ns() - setup_started
    selected_ids = tuple(slide.slide_id for slide in selection.slides)
    call_started = time.perf_counter_ns()
    try:
        results = processor.preprocess(selected_ids, overwrite=False, write_index=True)
        call_ns = time.perf_counter_ns() - call_started
    finally:
        close_started = time.perf_counter_ns()
        reader.close()
        close_ns = time.perf_counter_ns() - close_started

    rows: list[dict[str, Any]] = []
    by_id = {result.slide_id: result for result in results}
    for slide in selection.slides:
        result = by_id[slide.slide_id]
        total = slide_ns[slide.slide_id]
        writing = write_ns.get(slide.slide_id, 0)
        mapping = {
            f"{float(magnification):g}": asdict(summary)
            for magnification, summary in result.mapping_summaries.items()
        }
        row = {
            "ordinal": slide.ordinal,
            "slide_id": slide.slide_id,
            "duration_ns": total,
            "artifact_write_ns": writing,
            "geometry_mapping_and_other_ns": max(0, total - writing),
            "reused": bool(result.reused),
            "artifact_path": str(result.artifact_path),
            "num_cells": int(result.num_cells),
            "start_cell_id": int(result.start_cell_id),
            "mapping_summaries": mapping,
            "storage": None,
        }
        rows.append(row)
    residual_ns = max(0, call_ns - sum(slide_ns.values()))
    return rows, {
        "setup_ns": setup_ns,
        "native_call_ns": call_ns,
        "offset_index_and_orchestration_ns": residual_ns,
        "close_ns": close_ns,
    }


def run_prepare(args: argparse.Namespace) -> dict[str, Any]:
    operation_started = time.perf_counter_ns()
    _verify_source_identity()
    if not args.device.startswith("cuda:"):
        raise ValueError("AtlasPatch requires an explicit CUDA device")
    if int(args.device.split(":", 1)[1]) != args.gpu_index:
        raise ValueError("CUDA device and GPU monitor index must agree")
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must be unset so AtlasPatch and GPU monitoring use "
            "the same host-visible index"
        )
    config_path = Path(args.config).expanduser().resolve()
    selection_override = getattr(args, "_selection_override", None)
    if selection_override is None:
        selection = load_cohort(args.manifest, cohort=args.cohort, limit=args.limit)
    elif isinstance(selection_override, CohortSelection):
        # Private long-run scenarios prepare only the unique slides referenced
        # by their immutable episode plan.  The normal CLI never supplies this
        # internal override, so legacy measured runs retain their exact path.
        selection = selection_override
    else:
        raise TypeError("_selection_override must be a CohortSelection")
    config, adapter = _load_native_context(config_path, selection)
    layout = WorkspaceLayout.from_root(args.workspace)
    attempt, resumed = _initialize_workspace(
        layout,
        selection,
        config_path,
        device=args.device,
        resume=args.resume,
    )
    raw_path = layout.raw_root / f"preparation_attempt_{attempt}.jsonl"
    if raw_path.exists():
        raise FileExistsError(raw_path)
    pre_measurement_setup_ns = time.perf_counter_ns() - operation_started
    process_start_to_measurement_ns = max(0, time.time_ns() - PROCESS_CREATED_EPOCH_NS)
    active_monitor: ResourceMonitor | None = None
    phase_resources: dict[str, dict[str, Any]] = {}
    resources: dict[str, Any] | None = None
    try:
        active_monitor = ResourceMonitor(
            interval_seconds=float(args.monitor_interval),
            # GPU sampling is confined to the AtlasPatch/tissue phase. It is
            # disabled for geometry/mapping/artifact materialization.
            gpu_index=int(args.gpu_index) if args.gpu_index is not None else None,
        ).start()
        tissue_started = time.perf_counter_ns()
        tissue_rows, tissue_overhead = _run_tissue_stage(
            config,
            adapter,
            selection,
            layout,
            device=args.device,
        )
        tissue_stage_ns = time.perf_counter_ns() - tissue_started
        phase_resources["tissue"] = active_monitor.stop()
        active_monitor = None
        tissue_source_counts = dict(Counter(str(row["source"]) for row in tissue_rows))
        if not resumed:
            unexpected_sources = [
                str(row["slide_id"])
                for row in tissue_rows
                if row["source"] != "atlaspatch"
            ]
            if unexpected_sources:
                raise RuntimeError(
                    "fresh preparation reused or bypassed native AtlasPatch tissue for: "
                    f"{unexpected_sources[:8]}"
                )

        active_monitor = ResourceMonitor(
            interval_seconds=float(args.monitor_interval),
            gpu_index=None,
        ).start()
        navigation_started = time.perf_counter_ns()
        navigation_rows, navigation_overhead = _run_navigation_artifact_stage(
            config,
            adapter,
            selection,
            layout,
            device=args.device,
        )
        navigation_stage_ns = time.perf_counter_ns() - navigation_started
        phase_resources["navigation_artifacts"] = active_monitor.stop()
        active_monitor = None
        if not resumed:
            unexpectedly_reused = [
                str(row["slide_id"]) for row in navigation_rows if bool(row["reused"])
            ]
            if unexpectedly_reused:
                raise RuntimeError(
                    "fresh preparation reused navigation artifacts for: "
                    f"{unexpectedly_reused[:8]}"
                )
        resources = _combine_phase_resources(phase_resources)

        # Filesystem walks are report work, not native preparation. Attach all
        # logical/allocated storage after both timed/resource-monitored phases.
        for row in tissue_rows:
            slide_id = str(row["slide_id"])
            mask_base = layout.tissue_root / "masks" / slide_id
            row["storage"] = _path_group_snapshot(
                [
                    mask_base.with_suffix(".npy"),
                    mask_base.with_suffix(".npy.manifest.json"),
                    mask_base.with_suffix(".npy.lock"),
                    layout.tissue_root / "mpp" / f"{slide_id}.csv",
                ]
            )
        for row in navigation_rows:
            row["storage"] = directory_snapshot(
                layout.processed_root / "slides" / str(row["slide_id"])
            )

        # Benchmark-result I/O is deliberately outside the measured resource
        # block so CPU/I/O metrics describe native preparation rather than the
        # harness serializing its report.
        raw_serialization_started = time.perf_counter_ns()
        with JsonlStream(raw_path) as raw_stream:
            for row in tissue_rows:
                raw_stream.write({"phase": "tissue", **row})
            for row in navigation_rows:
                raw_stream.write({"phase": "navigation_artifacts", **row})
        raw_serialization_ns = time.perf_counter_ns() - raw_serialization_started

        tissue_by_id = {row["slide_id"]: row for row in tissue_rows}
        navigation_by_id = {row["slide_id"]: row for row in navigation_rows}
        slides = [
            {
                "ordinal": slide.ordinal,
                "slide_id": slide.slide_id,
                "tissue": tissue_by_id[slide.slide_id],
                "navigation_artifacts": navigation_by_id[slide.slide_id],
                "total_preprocessing_ns": (
                    int(tissue_by_id[slide.slide_id]["duration_ns"])
                    + int(navigation_by_id[slide.slide_id]["duration_ns"])
                ),
            }
            for slide in selection.slides
        ]
        backend_artifacts = directory_snapshot(layout.root / "artifacts")
        tissue_native_ns = (
            int(tissue_overhead["setup_ns"])
            + sum(int(row["duration_ns"]) for row in tissue_rows)
            + int(tissue_overhead["close_ns"])
            + int(tissue_overhead["index_write_ns"])
        )
        navigation_native_ns = (
            int(navigation_overhead["setup_ns"])
            + int(navigation_overhead["native_call_ns"])
            + int(navigation_overhead["close_ns"])
        )
        total_native_ns = tissue_native_ns + navigation_native_ns
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "backend": "wsi-lab",
            "phase": "preparation",
            "repetition": 0,
            "status": "complete",
            "attempt": attempt,
            "resumed": resumed,
            "fresh_measurement_valid": not resumed,
            "provenance": _provenance(config_path, selection),
            "slides": slides,
            "aggregate": {
                "slide_count": len(slides),
                "tissue_stage_outer_wall_ns": tissue_stage_ns,
                "navigation_artifact_stage_outer_wall_ns": navigation_stage_ns,
                "preparation_outer_wall_ns": tissue_stage_ns + navigation_stage_ns,
                "tissue_native_ns": tissue_native_ns,
                "navigation_artifact_native_ns": navigation_native_ns,
                "total_preprocessing_ns": total_native_ns,
                "total_preprocessing_seconds": total_native_ns / 1e9,
                "tissue_slide_sum_ns": sum(int(row["duration_ns"]) for row in tissue_rows),
                "navigation_slide_sum_ns": sum(
                    int(row["duration_ns"]) for row in navigation_rows
                ),
                "artifact_write_sum_ns": sum(
                    int(row["artifact_write_ns"]) for row in navigation_rows
                ),
                "tissue_overhead": tissue_overhead,
                "navigation_overhead": navigation_overhead,
                "built_slides": sum(not bool(row["reused"]) for row in navigation_rows),
                "reused_slides": sum(bool(row["reused"]) for row in navigation_rows),
                "tissue_source_counts": tissue_source_counts,
                "fresh_atlaspatch_tissue_slides": tissue_source_counts.get("atlaspatch", 0),
                "reused_tissue_slides": sum(
                    count
                    for source_name, count in tissue_source_counts.items()
                    if source_name != "atlaspatch"
                ),
                "backend_generated_storage": backend_artifacts,
                "process_entry_to_result_ns": time.perf_counter_ns() - MODULE_STARTED_NS,
                "pre_measurement_setup_ns": pre_measurement_setup_ns,
                "process_start_to_measurement_ns": process_start_to_measurement_ns,
                "raw_serialization_ns_excluded_from_measurement": raw_serialization_ns,
                "operation_wall_including_report_io_ns": (
                    time.perf_counter_ns() - operation_started
                ),
                "original_embedding_generation_included": False,
            },
            "resources": resources,
            "raw": {"per_slide_jsonl": str(raw_path)},
        }
        result_path = layout.results_root / "preparation.json"
        atomic_json_dump(result_path, result)
        marker = _marker_payload(
            selection=selection,
            config_path=config_path,
            status="complete",
            attempt=attempt,
            device=args.device,
            resumed=resumed,
        )
        marker["preparation_result"] = str(result_path)
        marker["backend_generated_storage"] = backend_artifacts
        atomic_json_dump(layout.marker, marker)
        return result
    except BaseException as exc:
        if resources is None:
            if active_monitor is not None:
                with suppress(BaseException):
                    partial = active_monitor.stop()
                    phase_resources["failed_active_phase"] = partial
            resources = _combine_phase_resources(phase_resources)
        failure = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "backend": "wsi-lab",
            "phase": "preparation",
            "repetition": 0,
            "status": "failed",
            "attempt": attempt,
            "resumed": resumed,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "resources": resources,
            "raw": {"per_slide_jsonl": str(raw_path)},
        }
        atomic_json_dump(layout.results_root / f"preparation_attempt_{attempt}_failed.json", failure)
        marker = _marker_payload(
            selection=selection,
            config_path=config_path,
            status="failed",
            attempt=attempt,
            device=args.device,
            resumed=resumed,
        )
        marker["error_type"] = type(exc).__name__
        marker["error"] = str(exc)
        atomic_json_dump(layout.marker, marker)
        raise


def _construct_environment(config: Any, adapter: Any, selection: CohortSelection, layout: WorkspaceLayout):
    from wsi_lab.data.embeddings.hdf5 import HDF5EmbeddingProvider
    from wsi_lab.data.splits import DatasetSplit, SplitAssignment
    from wsi_lab.envs.wsi_env import ConstantBelief, WSIEnv
    from wsi_lab.preprocessing.artifacts import ArtifactStore
    from wsi_lab.preprocessing.pipeline import ProcessedEpisodeSource
    from wsi_lab.wsi.reader import OpenSlideReader

    assignments = tuple(
        SplitAssignment(
            slide_id=slide.slide_id,
            label=slide.label,
            split="train",
            patient_id=None,
        )
        for slide in selection.slides
    )
    split = DatasetSplit(
        dataset=adapter.name,
        seed=1,
        validation_fraction=0.0,
        strategy="fixed-benchmark-manifest",
        assignments=assignments,
    )
    reader = OpenSlideReader(
        handle_cache_size=int(config.environment.slide_handle_cache_size),
        viewport_cache_size=int(config.environment.viewport_cache_size),
        persistent_cache_dir=None,
    )
    native_source = ProcessedEpisodeSource(
        adapter,
        HDF5EmbeddingProvider(adapter),
        reader,
        ArtifactStore(layout.processed_root),
        split,
    )
    source = FixedEmbeddingEpisodeSource(native_source, embedding_dimension=1)
    belief = ConstantBelief()
    env = WSIEnv(
        source,
        belief,
        split="train",
        evaluation=True,
        overview_size=int(config.environment.overview_size),
        viewport_size=int(config.environment.viewport_size),
        start_position=config.environment.start_position,
        geometric_augmentation=False,
        photometric_augmentation=False,
        max_episode_steps=int(config.environment.max_episode_steps),
        min_diagnostic_evidence=int(config.environment.min_diagnostic_evidence),
        max_visit_count=int(config.environment.max_visit_count),
        reward_type=config.environment.reward_type,
        step_cost=float(config.environment.step_cost),
        belief_reward_beta=float(config.environment.belief_reward_beta),
        lightweight_evaluation=False,
        render_mode=None,
    )
    return env, source


def _depth(magnification: float) -> int:
    from wsi_lab.constants import NAVIGATION_MAGNIFICATIONS

    return NAVIGATION_MAGNIFICATIONS.index(float(magnification))


def _assert_full_observation(observation: dict[str, Any]) -> int:
    expected_shapes = {
        "global": (5, 224, 224),
        "viewport": (3, 224, 224),
        "scalars": (5,),
    }
    observed = {key: tuple(value.shape) for key, value in observation.items()}
    if observed != expected_shapes:
        raise AssertionError(f"full observation shape changed: {observed} != {expected_shapes}")
    size = observation_nbytes(observation)
    if size != EXPECTED_OBSERVATION_BYTES_224:
        raise AssertionError(
            f"full observation bytes changed: {size} != {EXPECTED_OBSERVATION_BYTES_224}"
        )
    belief = float(observation["scalars"][2])
    if belief != 0.5:
        raise AssertionError(f"constant observation belief changed: {belief}")
    return size


def _run_slide_rollout(
    env: Any,
    source: FixedEmbeddingEpisodeSource,
    slide: SlideSpec,
    *,
    seed: int,
) -> dict[str, Any]:
    dummy_returns_before = source.dummy_embedding_returns
    underlying_reads_before = source.underlying_embedding_reads
    outer_reset_through_rollout_started = time.perf_counter_ns()
    reset_cpu_started = time.process_time_ns()
    reset_started = time.perf_counter_ns()
    observation, info = env.reset(
        seed=int(seed) + slide.ordinal,
        options={"slide_id": slide.slide_id},
    )
    reset_latency_ns = time.perf_counter_ns() - reset_started
    reset_cpu_time_ns = time.process_time_ns() - reset_cpu_started
    reset_observation_bytes = _assert_full_observation(observation)
    if info["slide_id"] != slide.slide_id or int(info["actions_taken"]) != 0:
        raise AssertionError(f"reset did not select a fresh episode for {slide.slide_id}")
    if float(info["belief"]) != 0.5:
        raise AssertionError("ConstantBelief must expose 0.5")

    scheduler = AcceptedActionScheduler(slide.ordinal)
    steps: list[dict[str, Any]] = []
    postponements: list[dict[str, Any]] = []
    rollout_started = time.perf_counter_ns()
    attempts = 0
    while not scheduler.complete:
        attempts += 1
        if attempts > 500:
            raise RuntimeError(f"too many no-op retries for {slide.slide_id}")
        selection_started = time.perf_counter_ns()
        decision = scheduler.choose(info)
        selection_latency_ns = time.perf_counter_ns() - selection_started
        before_cell = int(info["cell_id"])
        before_actions = int(info["actions_taken"])
        before_magnification = float(info["magnification"])
        for postponed in decision.postponed_items:
            postponements.append(
                {
                    "event_index": len(postponements),
                    "accepted_actions_so_far": scheduler.accepted,
                    **postponed,
                    "cell_id": before_cell,
                    "magnification": before_magnification,
                    "depth": _depth(before_magnification),
                }
            )
        historical_mask = [bool(value) for value in info["action_mask"]]
        step_cpu_started = time.process_time_ns()
        step_started = time.perf_counter_ns()
        next_observation, reward, terminated, truncated, next_info = env.step(
            int(decision.concrete_action)
        )
        latency_ns = time.perf_counter_ns() - step_started
        cpu_time_ns = time.process_time_ns() - step_cpu_started
        size = _assert_full_observation(next_observation)
        after_cell = int(next_info["cell_id"])
        after_actions = int(next_info["actions_taken"])
        invalid = bool(next_info.get("invalid_action", False))
        state_changed = after_cell != before_cell
        accepted = (not invalid) and state_changed and after_actions == before_actions + 1
        record = {
            "event": "step",
            "backend": "wsi-lab",
            "slide_id": slide.slide_id,
            "ordinal": slide.ordinal,
            "index": scheduler.accepted if accepted else None,
            "accepted_index_candidate": scheduler.accepted,
            "attempt": attempts,
            "attempt_index": attempts - 1,
            "priority_index": decision.priority_index,
            "postponement_count": decision.postponement_count,
            "priority_head_class": decision.priority_head_class,
            "requested_class": decision.requested_class,
            "concrete_action": decision.concrete_action.name.lower(),
            "concrete_action_id": int(decision.concrete_action),
            "pending_priority_index": decision.pending_index,
            "postponed_classes": list(decision.postponed_classes),
            "postponed_items": list(decision.postponed_items),
            "accepted": accepted,
            "invalid_action": invalid,
            "state_changed": state_changed,
            "latency_ns": latency_ns,
            "cpu_time_ns": cpu_time_ns,
            "cpu_ns": cpu_time_ns,
            "selection_latency_ns_excluded": selection_latency_ns,
            "before_cell_id": before_cell,
            "cell_id": after_cell,
            "before_magnification": before_magnification,
            "magnification": float(next_info["magnification"]),
            "before_depth": _depth(before_magnification),
            "depth": _depth(float(next_info["magnification"])),
            "observation_bytes": size,
            "actions_taken": after_actions,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "historical_action_mask": historical_mask,
        }
        steps.append(record)
        if terminated or truncated:
            raise AssertionError("navigation-only 100-step rollout terminated unexpectedly")
        if accepted:
            if abstract_class(decision.concrete_action) != decision.requested_class:
                raise AssertionError("concrete action does not match requested abstract class")
            scheduler.confirm(decision)
        else:
            scheduler.reject_no_op(decision, cell_id=before_cell)
        observation, info = next_observation, next_info
    rollout_wall_ns = time.perf_counter_ns() - rollout_started
    scheduler.verify_complete()
    if source.underlying_embedding_reads != underlying_reads_before:
        raise AssertionError("online HDF5 feature reads changed during the rollout")

    accepted_steps = [step for step in steps if step["accepted"]]
    quotas = Counter(str(step["requested_class"]) for step in accepted_steps)
    if {name: quotas[name] for name in EXPECTED_QUOTAS} != EXPECTED_QUOTAS:
        raise AssertionError(f"rollout quotas changed for {slide.slide_id}: {quotas}")
    accepted_latencies = [int(step["latency_ns"]) for step in accepted_steps]
    all_call_latencies = [int(step["latency_ns"]) for step in steps]
    native_online_ns = reset_latency_ns + sum(all_call_latencies)
    outer_reset_through_rollout_ns = (
        time.perf_counter_ns() - outer_reset_through_rollout_started
    )
    return {
        "ordinal": slide.ordinal,
        "slide_id": slide.slide_id,
        "reset": {
            "latency_ns": reset_latency_ns,
            "cpu_time_ns": reset_cpu_time_ns,
            "cpu_ns": reset_cpu_time_ns,
            "observation_bytes": reset_observation_bytes,
            "cell_id": int(steps[0]["before_cell_id"]),
            "magnification": float(steps[0]["before_magnification"]),
            "depth": int(steps[0]["before_depth"]),
        },
        "priority_order": list(scheduler.initial_priorities),
        "postponements": postponements,
        "steps": steps,
        "outer_wall": {
            "reset_through_rollout_ns": outer_reset_through_rollout_ns,
            "includes_adapter_postponement_and_validation_overhead": True,
            "excludes_environment_construction_and_close": True,
        },
        "summary": {
            "accepted_steps": len(accepted_steps),
            "step_calls": len(steps),
            "no_op_attempts": len(steps) - len(accepted_steps),
            "postponement_events": len(postponements),
            "accepted_quotas": {name: quotas[name] for name in EXPECTED_QUOTAS},
            "accepted_step_latency": latency_summary_ns(accepted_latencies),
            "all_native_step_call_latency": latency_summary_ns(all_call_latencies),
            "accepted_step_cpu_time": latency_summary_ns(
                [int(step["cpu_time_ns"]) for step in accepted_steps]
            ),
            "rollout_wall_ns": rollout_wall_ns,
            "steps_per_second_rollout_wall": len(accepted_steps) * 1e9 / rollout_wall_ns,
            "outer_reset_through_rollout_ns": outer_reset_through_rollout_ns,
            "accepted_steps_per_outer_second_including_adapter": (
                len(accepted_steps) * 1e9 / outer_reset_through_rollout_ns
            ),
            "environment_api_ns_reset_plus_all_step_calls": (
                native_online_ns
            ),
            "native_online_latency_ns": native_online_ns,
            "accepted_steps_per_native_step_second": (
                len(accepted_steps) * 1e9 / sum(all_call_latencies)
            ),
            "steps_per_second_reset_inclusive_environment_api": (
                len(accepted_steps) * 1e9 / native_online_ns
            ),
            "accepted_steps_per_online_second_including_reset": (
                len(accepted_steps) * 1e9 / native_online_ns
            ),
            "magnification_counts_after_step": dict(
                Counter(f"{float(step['magnification']):g}" for step in accepted_steps)
            ),
            "observation_bytes": sorted(
                {int(step["observation_bytes"]) for step in accepted_steps}
            ),
            "dummy_embedding_returns": (
                source.dummy_embedding_returns - dummy_returns_before
            ),
            "underlying_embedding_reads": (
                source.underlying_embedding_reads - underlying_reads_before
            ),
        },
    }


def _online_aggregate(
    slides: list[dict[str, Any]], construction_ns: int, construction_cpu_ns: int
) -> dict[str, Any]:
    accepted_steps = [
        step
        for slide in slides
        for step in slide["steps"]
        if bool(step["accepted"])
    ]
    all_calls = [step for slide in slides for step in slide["steps"]]
    reset_latencies = [int(slide["reset"]["latency_ns"]) for slide in slides]
    reset_cpu_times = [int(slide["reset"]["cpu_time_ns"]) for slide in slides]
    accepted_latencies = [int(step["latency_ns"]) for step in accepted_steps]
    accepted_cpu_times = [int(step["cpu_time_ns"]) for step in accepted_steps]
    all_call_latencies = [int(step["latency_ns"]) for step in all_calls]
    all_call_cpu_times = [int(step["cpu_time_ns"]) for step in all_calls]
    by_class = {
        name: latency_summary_ns(
            [
                int(step["latency_ns"])
                for step in accepted_steps
                if step["requested_class"] == name
            ]
        )
        for name in EXPECTED_QUOTAS
    }
    quotas = Counter(str(step["requested_class"]) for step in accepted_steps)
    expected_total = {
        name: count * len(slides) for name, count in EXPECTED_QUOTAS.items()
    }
    observed_total = {name: quotas[name] for name in EXPECTED_QUOTAS}
    if observed_total != expected_total:
        raise AssertionError(f"aggregate quotas differ: {observed_total} != {expected_total}")
    rollout_ns = sum(int(slide["summary"]["rollout_wall_ns"]) for slide in slides)
    outer_reset_through_rollout_ns = sum(
        int(slide["outer_wall"]["reset_through_rollout_ns"]) for slide in slides
    )
    environment_api_ns = sum(reset_latencies) + sum(
        int(step["latency_ns"]) for step in all_calls
    )
    return {
        "slide_count": len(slides),
        "environment_construction_ns": construction_ns,
        "environment_construction_cpu_ns": construction_cpu_ns,
        "environment_construction_seconds": construction_ns / 1e9,
        "reset_latency": latency_summary_ns(reset_latencies),
        "reset_cpu_time": latency_summary_ns(reset_cpu_times),
        "accepted_step_latency": latency_summary_ns(accepted_latencies),
        "accepted_step_cpu_time": latency_summary_ns(accepted_cpu_times),
        "all_step_call_latency": latency_summary_ns(all_call_latencies),
        "all_step_call_cpu_time": latency_summary_ns(all_call_cpu_times),
        "accepted_step_latency_by_class": by_class,
        "accepted_steps": len(accepted_steps),
        "step_calls": len(all_calls),
        "no_op_attempts": len(all_calls) - len(accepted_steps),
        "postponement_events": sum(len(slide["postponements"]) for slide in slides),
        "accepted_quotas": observed_total,
        "rollout_wall_ns": rollout_ns,
        "steps_per_second_rollout_wall": len(accepted_steps) * 1e9 / rollout_ns,
        "outer_reset_through_rollout_ns": outer_reset_through_rollout_ns,
        "accepted_steps_per_outer_second_including_adapter": (
            len(accepted_steps) * 1e9 / outer_reset_through_rollout_ns
        ),
        "environment_api_ns_reset_plus_all_step_calls": environment_api_ns,
        "native_online_latency_ns": environment_api_ns,
        "environment_api_cpu_ns_reset_plus_all_step_calls": (
            sum(reset_cpu_times) + sum(all_call_cpu_times)
        ),
        "steps_per_second_reset_inclusive_environment_api": (
            len(accepted_steps) * 1e9 / environment_api_ns
        ),
        "accepted_steps_per_native_step_second": (
            len(accepted_steps) * 1e9 / sum(all_call_latencies)
        ),
        "accepted_steps_per_online_second_including_reset": (
            len(accepted_steps) * 1e9 / environment_api_ns
        ),
        "observation_bytes": sorted({int(step["observation_bytes"]) for step in accepted_steps}),
        "render_called": False,
        "diagnosis_actions": 0,
    }


def run_online(args: argparse.Namespace) -> dict[str, Any]:
    operation_started = time.perf_counter_ns()
    _verify_source_identity()
    config_path = Path(args.config).expanduser().resolve()
    selection = load_cohort(args.manifest, cohort=args.cohort, limit=args.limit)
    config, adapter = _load_native_context(config_path, selection)
    layout = WorkspaceLayout.from_root(args.workspace)
    marker = _verify_prepared_workspace(layout, selection, config_path)
    repetition = int(args.repetition)
    result_path = layout.results_root / f"online_repetition_{repetition}.json"
    raw_path = layout.raw_root / f"online_repetition_{repetition}_steps.jsonl"
    if result_path.exists() or raw_path.exists():
        raise FileExistsError(
            f"online repetition {repetition} already has output; use a fresh repetition/workspace"
        )
    pre_measurement_setup_ns = time.perf_counter_ns() - operation_started
    process_start_to_measurement_ns = max(0, time.time_ns() - PROCESS_CREATED_EPOCH_NS)
    monitor = ResourceMonitor(
        interval_seconds=float(args.monitor_interval),
        # Online interaction is CPU/OpenSlide work. Avoid nvidia-smi polling in
        # the latency-sensitive pass; the configured GPU is sampled only during AtlasPatch.
        gpu_index=None,
    ).start()
    resources: dict[str, Any] | None = None
    env = None
    source = None
    try:
        construction_cpu_started = time.process_time_ns()
        construction_started = time.perf_counter_ns()
        env, source = _construct_environment(config, adapter, selection, layout)
        construction_ns = time.perf_counter_ns() - construction_started
        construction_cpu_ns = time.process_time_ns() - construction_cpu_started
        slides: list[dict[str, Any]] = []
        for slide in selection.slides:
            slides.append(
                _run_slide_rollout(
                    env,
                    source,
                    slide,
                    seed=int(args.seed),
                )
            )
        close_cpu_started = time.process_time_ns()
        close_started = time.perf_counter_ns()
        env.close()
        close_ns = time.perf_counter_ns() - close_started
        close_cpu_ns = time.process_time_ns() - close_cpu_started
        env = None
        resources = monitor.stop()

        # Keep report serialization outside the measured block. The per-call
        # dictionaries above contain no observation arrays, only scalar audit
        # data, so retaining them does not extend native viewport lifetimes.
        raw_serialization_started = time.perf_counter_ns()
        with JsonlStream(raw_path) as raw_stream:
            for slide_result in slides:
                for step in slide_result["steps"]:
                    raw_stream.write(step)
        raw_serialization_ns = time.perf_counter_ns() - raw_serialization_started
        aggregate = _online_aggregate(slides, construction_ns, construction_cpu_ns)
        aggregate.update(
            {
                "environment_close_ns": close_ns,
                "environment_close_cpu_ns": close_cpu_ns,
                "operation_wall_including_report_io_ns": (
                    time.perf_counter_ns() - operation_started
                ),
                "dummy_embedding_returns": int(source.dummy_embedding_returns),
                "underlying_embedding_reads": int(source.underlying_embedding_reads),
                "belief_model_loaded": False,
                "belief_checkpoint_loaded": False,
                "belief_value": 0.5,
                "full_observation_generation": True,
                "lightweight_evaluation": False,
                "pre_measurement_setup_ns": pre_measurement_setup_ns,
                "process_start_to_measurement_ns": process_start_to_measurement_ns,
                "raw_serialization_ns_excluded_from_measurement": raw_serialization_ns,
            }
        )
        if source.underlying_embedding_reads != 0:
            raise AssertionError("online HDF5 feature reads must remain zero")
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "backend": "wsi-lab",
            "phase": "online",
            "repetition": repetition,
            "status": "complete",
            "prepared_workspace": {
                "root": str(layout.root),
                "preparation_attempt": int(marker["attempt"]),
                "fresh_measurement_valid": bool(marker["fresh_measurement_valid"]),
            },
            "provenance": _provenance(config_path, selection),
            "slides": slides,
            "aggregate": aggregate,
            "resources": resources,
            "raw": {"steps_jsonl": str(raw_path)},
        }
        atomic_json_dump(result_path, result)
        return result
    except BaseException as exc:
        if env is not None:
            with suppress(BaseException):
                env.close()
        if resources is None:
            with suppress(BaseException):
                resources = monitor.stop()
        failure = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "backend": "wsi-lab",
            "phase": "online",
            "repetition": repetition,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "resources": resources,
            "raw": {"steps_jsonl": str(raw_path)},
        }
        atomic_json_dump(
            layout.results_root / f"online_repetition_{repetition}_failed.json", failure
        )
        raise


def run_validate(args: argparse.Namespace) -> dict[str, Any]:
    _verify_source_identity()
    config_path = Path(args.config).expanduser().resolve()
    selection = load_cohort(args.manifest, cohort=args.cohort, limit=args.limit)
    config, adapter = _load_native_context(config_path, selection)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "backend": "wsi-lab",
        "phase": "validation",
        "repetition": 0,
        "slides": [
            {
                "ordinal": slide.ordinal,
                "slide_id": slide.slide_id,
                "wsi_path": str(slide.wsi_path),
                "xml_path": str(slide.xml_path),
                "label": slide.label,
            }
            for slide in selection.slides
        ],
        "aggregate": {
            "slide_count": len(selection.slides),
            "dataset": adapter.name,
            "overview_size": int(config.environment.overview_size),
            "viewport_size": int(config.environment.viewport_size),
            "max_episode_steps": int(config.environment.max_episode_steps),
            "existing_generated_artifacts_used": False,
        },
        "resources": None,
        "provenance": _provenance(config_path, selection),
    }


def _add_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cohort", choices=("measured", "pilot"), default="measured")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only the first N fixed pilot rows; forbidden for measured runs.",
    )


def _add_monitoring(parser: argparse.ArgumentParser, *, include_gpu: bool) -> None:
    parser.add_argument("--monitor-interval", type=float, default=0.1)
    if include_gpu:
        parser.add_argument(
            "--gpu-index",
            type=int,
            default=1,
            help="Whole-device NVIDIA metrics during AtlasPatch; pass -1 to disable.",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Read-only native input validation.")
    _add_inputs(validate)

    prepare = subparsers.add_parser(
        "prepare", help="Fresh native AtlasPatch and navigation-artifact preparation."
    )
    _add_inputs(prepare)
    _add_monitoring(prepare, include_gpu=True)
    prepare.add_argument("--workspace", type=Path, required=True)
    prepare.add_argument("--device", default="cuda:0")
    prepare.add_argument("--resume", action="store_true")

    online = subparsers.add_parser(
        "online", help="One prepared-state 100-slide online repetition."
    )
    _add_inputs(online)
    _add_monitoring(online, include_gpu=False)
    online.add_argument("--workspace", type=Path, required=True)
    online.add_argument("--repetition", type=int, choices=(1, 2, 3), required=True)
    online.add_argument("--seed", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(args, "monitor_interval") and args.monitor_interval <= 0:
        raise ValueError("--monitor-interval must be positive")
    if hasattr(args, "gpu_index") and args.gpu_index < 0:
        args.gpu_index = None
    if args.command == "validate":
        result = run_validate(args)
    elif args.command == "prepare":
        result = run_prepare(args)
    elif args.command == "online":
        result = run_online(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    summary = {
        "backend": result["backend"],
        "phase": result["phase"],
        "repetition": result["repetition"],
        "status": result.get("status", "validated"),
        "slide_count": len(result["slides"]),
        "aggregate": result["aggregate"],
    }
    print(json.dumps(summary, default=json_default, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
