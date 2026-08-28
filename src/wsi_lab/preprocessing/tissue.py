"""Validated tissue masks aligned to the shared overview coordinate frame."""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from collections.abc import Mapping
from contextlib import nullcontext, suppress
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from wsi_lab.errors import ArtifactCompatibilityError, PreprocessingError
from wsi_lab.preprocessing.artifacts import advisory_lock
from wsi_lab.preprocessing.atlaspatch import (
    ATLASPATCH_MODEL_FILENAME,
    ATLASPATCH_MODEL_REPOSITORY,
    ATLASPATCH_SAM_CONFIG_FILENAME,
    PINNED_SAM2_COMMIT,
    AtlasPatchOptions,
    AtlasPatchRunner,
    installed_sam2_provenance,
    resolve_atlaspatch_checkpoint_identity,
    resolve_atlaspatch_sam_config_identity,
    sha256_file,
)
from wsi_lab.preprocessing.geometry import GridLevel, NavigationGeometry, validate_geometry
from wsi_lab.utils.atomic import atomic_json_dump
from wsi_lab.utils.fingerprints import file_fingerprint, fingerprint
from wsi_lab.wsi.reader import OverviewTransform

LOGGER = logging.getLogger(__name__)
TissueSource = Literal["auto", "existing", "atlaspatch"]


@dataclass(frozen=True, slots=True)
class TissueMask:
    values: NDArray[np.bool_]
    source: str
    threshold: float
    source_path: Path | None = None
    source_page: int | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("tissue mask must have shape [H,W]")
        if self.values.dtype != np.bool_:
            raise TypeError("tissue mask values must be boolean")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("tissue threshold must lie in [0,1]")


def atlaspatch_available(executable: str = "atlaspatch") -> bool:
    """Return whether an AtlasPatch executable can be resolved without importing it."""

    import shutil

    candidate = Path(executable).expanduser()
    return (
        candidate.is_file()
        if candidate.parent != Path(".")
        else shutil.which(executable) is not None
    )


def _normalize_binary_mask(value: np.ndarray, *, path: Path) -> NDArray[np.bool_]:
    """Accept genuinely binary encodings, never arbitrary thresholded classes."""

    array = np.asarray(value)
    while array.ndim > 2 and 1 in (array.shape[0], array.shape[-1]):
        axis = 0 if array.shape[0] == 1 else array.ndim - 1
        array = np.squeeze(array, axis=axis)
    if array.ndim != 2:
        raise PreprocessingError(
            f"tissue mask must be single-channel 2-D, got {array.shape}: {path}"
        )
    if not (np.issubdtype(array.dtype, np.bool_) or np.issubdtype(array.dtype, np.number)):
        raise PreprocessingError(f"tissue mask has unsupported dtype {array.dtype}: {path}")
    if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
        raise PreprocessingError(f"tissue mask contains non-finite values: {path}")
    unique = np.unique(array)
    if len(unique) > 2:
        preview = unique[:8].tolist()
        raise PreprocessingError(
            f"tissue mask is not binary ({len(unique)} distinct values; first={preview}): {path}"
        )
    if len(unique) == 2 and not np.any(unique == 0):
        raise PreprocessingError(
            f"two-valued tissue mask does not encode background as zero ({unique.tolist()}): {path}"
        )
    return np.asarray(array != 0, dtype=np.bool_)


def _compatible_dimensions(
    width: int,
    height: int,
    transform: OverviewTransform,
    *,
    path: Path,
) -> tuple[float, float]:
    if width <= 0 or height <= 0:
        raise PreprocessingError(f"tissue mask has invalid dimensions {width}x{height}: {path}")
    downsample_x = transform.source_width / width
    downsample_y = transform.source_height / height
    if downsample_x < 0.98 or downsample_y < 0.98:
        raise PreprocessingError(
            f"tissue mask is larger than the level-0 WSI ({width}x{height} versus "
            f"{transform.source_width}x{transform.source_height}): {path}"
        )
    relative_error = abs(downsample_x - downsample_y) / max(downsample_x, downsample_y)
    if relative_error > 0.01:
        raise PreprocessingError(
            f"tissue mask aspect/downsample is incompatible with WSI: "
            f"downsample_x={downsample_x:g}, downsample_y={downsample_y:g}: {path}"
        )
    return float(downsample_x), float(downsample_y)


def _read_tiff_mask(
    path: Path, transform: OverviewTransform
) -> tuple[NDArray[np.bool_], int, dict[str, Any]]:
    try:
        import tifffile
    except ImportError as exc:
        raise ImportError("tifffile is required to read TIFF tissue masks") from exc
    target_downsample = max(
        transform.source_width / transform.content_width,
        transform.source_height / transform.content_height,
    )
    try:
        with tifffile.TiffFile(path) as handle:
            candidates: list[tuple[float, int, float, float]] = []
            rejected: list[str] = []
            for index, page in enumerate(handle.pages):
                page_metadata: Any = page
                width = int(page_metadata.imagewidth)
                height = int(page_metadata.imagelength)
                try:
                    downsample_x, downsample_y = _compatible_dimensions(
                        width, height, transform, path=path
                    )
                except PreprocessingError as exc:
                    rejected.append(f"page {index}: {exc}")
                    continue
                average = (downsample_x + downsample_y) / 2.0
                scale_error = abs(np.log(max(average, 1e-12) / target_downsample))
                candidates.append((float(scale_error), index, downsample_x, downsample_y))
            if not candidates:
                detail = "; ".join(rejected[:4])
                raise PreprocessingError(f"TIFF has no WSI-compatible image page: {path}; {detail}")
            _, selected, downsample_x, downsample_y = min(candidates)
            page = handle.pages[selected]
            page_metadata = page
            width = int(page_metadata.imagewidth)
            height = int(page_metadata.imagelength)
            raw = page.asarray()
    except (OSError, ValueError) as exc:
        raise PreprocessingError(f"cannot read tissue mask {path}: {exc}") from exc
    values = _normalize_binary_mask(raw, path=path)
    if values.shape != (height, width):
        raise PreprocessingError(
            f"decoded mask shape {values.shape} disagrees with TIFF page dimensions: {path}"
        )
    return (
        values,
        selected,
        {
            "native_width": width,
            "native_height": height,
            "downsample_x": downsample_x,
            "downsample_y": downsample_y,
        },
    )


def _read_native_mask(
    path: str | Path, transform: OverviewTransform
) -> tuple[NDArray[np.bool_], int | None, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise PreprocessingError(f"tissue mask does not exist: {source}")
    suffix = source.suffix.casefold()
    if suffix in {".tif", ".tiff"}:
        return _read_tiff_mask(source, transform)
    try:
        if suffix == ".npy":
            raw = np.load(source, allow_pickle=False)
        elif suffix == ".png":
            with Image.open(source) as image:
                raw = np.asarray(image)
        else:
            raise PreprocessingError(f"unsupported tissue-mask format {suffix!r}: {source}")
    except (OSError, ValueError) as exc:
        raise PreprocessingError(f"cannot read tissue mask {source}: {exc}") from exc
    values = _normalize_binary_mask(raw, path=source)
    downsample_x, downsample_y = _compatible_dimensions(
        values.shape[1], values.shape[0], transform, path=source
    )
    return (
        values,
        None,
        {
            "native_width": values.shape[1],
            "native_height": values.shape[0],
            "downsample_x": downsample_x,
            "downsample_y": downsample_y,
        },
    )


def _place_content_mask(
    native: NDArray[np.bool_], transform: OverviewTransform
) -> NDArray[np.bool_]:
    image = Image.fromarray(native.astype(np.uint8) * 255, mode="L").resize(
        (transform.content_width, transform.content_height), Image.Resampling.NEAREST
    )
    content = np.asarray(image, dtype=np.uint8) != 0
    canvas = np.zeros((transform.canvas_height, transform.canvas_width), dtype=np.bool_)
    canvas[
        transform.content_y0 : transform.content_y0 + transform.content_height,
        transform.content_x0 : transform.content_x0 + transform.content_width,
    ] = content
    return canvas


def _atomic_save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npy", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.save(stream, value, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def _generated_manifest_path(mask_path: Path) -> Path:
    return mask_path.with_name(f"{mask_path.name}.manifest.json")


def _generated_lock_path(mask_path: Path) -> Path:
    return mask_path.with_name(f"{mask_path.name}.lock")


def _cache_normalized_existing_mask(
    path: Path,
    values: NDArray[np.bool_],
    *,
    slide_path: Path,
    source_path: Path,
    source_page: int | None,
    transform: OverviewTransform,
    provenance: Mapping[str, Any],
) -> None:
    """Atomically publish a normalized copy while leaving dataset data read-only."""

    if path == source_path:
        raise PreprocessingError(
            "normalized tissue cache path must differ from the existing source mask"
        )
    manifest_path = _generated_manifest_path(path)
    source_identity = file_fingerprint(source_path)
    cache_identity = fingerprint(
        {
            "schema_version": 2,
            "slide": file_fingerprint(slide_path),
            "source": source_identity,
            "source_page": source_page,
            "overview_transform": transform.to_dict(),
        }
    )
    atomic_json_dump(
        manifest_path,
        {
            "schema_version": 2,
            "artifact_type": "tissue_mask",
            "cache_kind": "normalized-existing",
            "status": "incomplete",
            "fingerprint": cache_identity,
            "slide_id": slide_path.stem,
        },
    )
    _atomic_save_npy(path, np.asarray(values, dtype=np.bool_))
    atomic_json_dump(
        manifest_path,
        {
            "schema_version": 2,
            "artifact_type": "tissue_mask",
            "cache_kind": "normalized-existing",
            "status": "complete",
            "fingerprint": cache_identity,
            "created_at": datetime.now(UTC).isoformat(),
            "slide_id": slide_path.stem,
            "mask_path": str(path),
            "source": "existing",
            "source_path": str(source_path),
            "source_page": source_page,
            "source_fingerprint": source_identity,
            "overview_transform": transform.to_dict(),
            "normalized_mask": {
                "shape": list(values.shape),
                "dtype": str(values.dtype),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            },
            "provenance": dict(provenance),
        },
    )


def _cache_fingerprint(
    slide_path: Path,
    transform: OverviewTransform,
    options: AtlasPatchOptions,
    checkpoint_identity: Mapping[str, Any] | None = None,
) -> str:
    base = fingerprint(
        {
            "schema_version": 2,
            "slide": file_fingerprint(slide_path),
            "overview_transform": transform.to_dict(),
            "atlaspatch": options.cache_identity(),
        }
    )
    if checkpoint_identity is None:
        return base
    return fingerprint(
        {
            "schema_version": 2,
            "base_fingerprint": base,
            "checkpoint_identity": dict(checkpoint_identity),
        }
    )


def _checkpoint_identity_from_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the stable, path-independent model identity used for a mask."""

    model = provenance.get("segmentation_model")
    if not isinstance(model, Mapping):
        raise PreprocessingError("AtlasPatch provenance lacks segmentation_model identity")
    checkpoint = model.get("checkpoint")
    sam2 = model.get("sam2")
    sam2_config = model.get("sam2_config")
    if (
        not isinstance(checkpoint, Mapping)
        or not isinstance(sam2, Mapping)
        or not isinstance(sam2_config, Mapping)
    ):
        raise PreprocessingError(
            "AtlasPatch provenance lacks checkpoint, SAM2, or SAM2 config identity"
        )
    identity = {
        "repository": model.get("repository"),
        "filename": model.get("filename"),
        "requested_revision": model.get("requested_revision"),
        "resolved_snapshot": checkpoint.get("resolved_snapshot"),
        "size": checkpoint.get("size"),
        "sha256": checkpoint.get("sha256"),
        "sam2_commit": sam2.get("pinned_commit"),
        "sam2_config_filename": sam2_config.get("filename"),
        "sam2_config_size": sam2_config.get("size"),
        "sam2_config_sha256": sam2_config.get("sha256"),
    }
    if identity["repository"] != ATLASPATCH_MODEL_REPOSITORY:
        raise PreprocessingError("AtlasPatch provenance has an unexpected model repository")
    if identity["filename"] != ATLASPATCH_MODEL_FILENAME:
        raise PreprocessingError("AtlasPatch provenance has an unexpected model filename")
    if identity["requested_revision"] != "main":
        raise PreprocessingError("AtlasPatch provenance has an unexpected requested revision")
    if identity["sam2_commit"] != PINNED_SAM2_COMMIT:
        raise PreprocessingError("AtlasPatch provenance has an unexpected SAM2 commit")
    if identity["sam2_config_filename"] != ATLASPATCH_SAM_CONFIG_FILENAME:
        raise PreprocessingError("AtlasPatch provenance has an unexpected SAM2 config")
    if not isinstance(identity["resolved_snapshot"], str) or not identity["resolved_snapshot"]:
        raise PreprocessingError("AtlasPatch provenance lacks a resolved model snapshot")
    if not isinstance(identity["size"], int) or identity["size"] <= 0:
        raise PreprocessingError("AtlasPatch provenance has an invalid checkpoint size")
    if not isinstance(identity["sha256"], str) or len(identity["sha256"]) != 64:
        raise PreprocessingError("AtlasPatch provenance has an invalid checkpoint SHA256")
    if not isinstance(identity["sam2_config_size"], int) or identity["sam2_config_size"] <= 0:
        raise PreprocessingError("AtlasPatch provenance has an invalid SAM2 config size")
    if (
        not isinstance(identity["sam2_config_sha256"], str)
        or len(identity["sam2_config_sha256"]) != 64
    ):
        raise PreprocessingError("AtlasPatch provenance has an invalid SAM2 config SHA256")
    return identity


def _current_checkpoint_identity() -> dict[str, Any] | None:
    """Return the currently cached AtlasPatch model identity without downloading."""

    try:
        checkpoint = resolve_atlaspatch_checkpoint_identity()
    except PreprocessingError:
        return None
    sam2 = installed_sam2_provenance()
    sam2_config = resolve_atlaspatch_sam_config_identity()
    return _checkpoint_identity_from_provenance(
        {
            "segmentation_model": {
                "repository": ATLASPATCH_MODEL_REPOSITORY,
                "filename": ATLASPATCH_MODEL_FILENAME,
                "requested_revision": "main",
                "checkpoint": checkpoint,
                "sam2": sam2,
                "sam2_config": sam2_config,
            }
        }
    )


def _load_generated_mask(
    path: Path,
    *,
    expected_base_fingerprint: str,
    slide_path: Path,
    options: AtlasPatchOptions,
    transform: OverviewTransform,
    threshold: float,
) -> TissueMask:
    manifest_path = _generated_manifest_path(path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ArtifactCompatibilityError(
                f"generated tissue manifest is not complete: {manifest_path}"
            )
        if manifest.get("schema_version") != 2:
            raise ArtifactCompatibilityError(
                f"generated tissue manifest schema is stale: {manifest_path}"
            )
        if manifest.get("base_fingerprint") != expected_base_fingerprint:
            raise ArtifactCompatibilityError(
                f"generated tissue base fingerprint is stale: {manifest_path}"
            )
        provenance = manifest.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ArtifactCompatibilityError(
                f"generated tissue provenance is missing: {manifest_path}"
            )
        try:
            checkpoint_identity = _checkpoint_identity_from_provenance(provenance)
        except PreprocessingError as exc:
            raise ArtifactCompatibilityError(
                f"generated tissue model provenance is invalid: {manifest_path}: {exc}"
            ) from exc
        try:
            current_checkpoint_identity = _current_checkpoint_identity()
        except PreprocessingError as exc:
            raise ArtifactCompatibilityError(
                f"installed AtlasPatch model provenance is invalid: {exc}"
            ) from exc
        if (
            current_checkpoint_identity is not None
            and current_checkpoint_identity != checkpoint_identity
        ):
            raise ArtifactCompatibilityError(
                "generated tissue checkpoint differs from the currently cached "
                f"AtlasPatch model: {manifest_path}"
            )
        expected_fingerprint = _cache_fingerprint(
            slide_path,
            transform,
            options,
            checkpoint_identity,
        )
        if manifest.get("checkpoint_identity") != checkpoint_identity:
            raise ArtifactCompatibilityError(
                f"generated tissue checkpoint identity is inconsistent: {manifest_path}"
            )
        if manifest.get("fingerprint") != expected_fingerprint:
            raise ArtifactCompatibilityError(
                f"generated tissue fingerprint is stale: {manifest_path}"
            )
        values = np.load(path, allow_pickle=False)
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
        raise ArtifactCompatibilityError(
            f"cannot load generated tissue mask {path}: {exc}"
        ) from exc
    if values.dtype != np.bool_ or values.shape != (
        transform.canvas_height,
        transform.canvas_width,
    ):
        raise ArtifactCompatibilityError(
            f"generated tissue mask has incompatible dtype/shape {values.dtype}/{values.shape}: {path}"
        )
    normalized_description = manifest.get("normalized_mask")
    expected_sha256 = (
        normalized_description.get("sha256") if isinstance(normalized_description, dict) else None
    )
    if not expected_sha256 or sha256_file(path) != expected_sha256:
        raise ArtifactCompatibilityError(f"generated tissue mask checksum mismatch: {path}")
    return TissueMask(
        values=np.asarray(values, dtype=np.bool_),
        source="atlaspatch-cache",
        threshold=threshold,
        source_path=path,
        provenance=manifest.get("provenance", {}),
    )


def load_or_create_tissue_mask(
    slide_path: str | Path,
    overview: np.ndarray,
    transform: OverviewTransform,
    *,
    existing_path: str | Path | None = None,
    generated_path: str | Path | None = None,
    source: TissueSource | Any = "atlaspatch",
    threshold: float = 0.01,
    rebuild: bool = False,
    atlaspatch_options: AtlasPatchOptions | Mapping[str, Any] | None = None,
    atlaspatch_runner: AtlasPatchRunner | None = None,
    mpp: tuple[float | None, float | None] | None = None,
    concurrent_writes: bool = False,
) -> TissueMask:
    """Reuse a verified mask or generate one through the AtlasPatch CLI.

    Existing dataset masks remain read-only. Every selected mask is normalized
    to the overview canvas and, when ``generated_path`` is supplied, atomically
    published there with a provenance manifest written last.
    """

    source_name = str(getattr(source, "value", source)).casefold()
    if source_name not in {"auto", "existing", "atlaspatch"}:
        raise ValueError(f"unsupported tissue source: {source_name}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("tissue threshold must lie in [0,1]")
    if source_name == "existing" and rebuild:
        raise ValueError("rebuild=True conflicts with tissue source='existing'")
    overview_array = np.asarray(overview)
    if overview_array.shape != (transform.canvas_height, transform.canvas_width, 3):
        raise ValueError(
            f"overview shape {overview_array.shape} does not match transform canvas "
            f"{transform.canvas_height}x{transform.canvas_width}"
        )
    slide = Path(slide_path).expanduser().resolve()
    if not slide.is_file():
        raise FileNotFoundError(slide)
    existing = Path(existing_path).expanduser().resolve() if existing_path is not None else None
    generated = Path(generated_path).expanduser().resolve() if generated_path is not None else None
    if source_name in {"auto", "existing"} and not rebuild:
        if existing is not None and existing.is_file():
            try:
                native, page, details = _read_native_mask(existing, transform)
            except PreprocessingError:
                if source_name == "existing":
                    raise
                LOGGER.warning(
                    "Ignoring incompatible existing tissue mask; AtlasPatch is required",
                    exc_info=True,
                )
            else:
                normalized = _place_content_mask(native, transform)
                existing_provenance = {
                    "source": file_fingerprint(existing),
                    "source_page": page,
                    "binary_validation": "passed",
                    "alignment": details,
                    "overview_transform": transform.to_dict(),
                }
                if generated is not None:
                    _cache_normalized_existing_mask(
                        generated,
                        normalized,
                        slide_path=slide,
                        source_path=existing,
                        source_page=page,
                        transform=transform,
                        provenance=existing_provenance,
                    )
                return TissueMask(
                    values=normalized,
                    source="existing",
                    threshold=threshold,
                    source_path=existing,
                    source_page=page,
                    provenance=existing_provenance,
                )
        elif source_name == "existing":
            raise PreprocessingError(
                f"existing tissue mask was required but is unavailable: {existing}"
            )

    if atlaspatch_runner is not None and atlaspatch_options is not None:
        raise ValueError("pass atlaspatch_runner or atlaspatch_options, not both")
    if atlaspatch_runner is not None:
        runner = atlaspatch_runner
        options = runner.options
    else:
        if atlaspatch_options is None:
            options = AtlasPatchOptions()
        elif isinstance(atlaspatch_options, AtlasPatchOptions):
            options = atlaspatch_options
        else:
            options = AtlasPatchOptions(**dict(atlaspatch_options))
        runner = AtlasPatchRunner(options)
    if generated is None:
        raise PreprocessingError(
            "AtlasPatch tissue generation requires generated_path so the normalized mask and provenance can be cached"
        )
    expected_base_fingerprint = _cache_fingerprint(slide, transform, options)
    if not rebuild and generated.is_file() and _generated_manifest_path(generated).is_file():
        try:
            return _load_generated_mask(
                generated,
                expected_base_fingerprint=expected_base_fingerprint,
                slide_path=slide,
                options=options,
                transform=transform,
                threshold=threshold,
            )
        except ArtifactCompatibilityError:
            # Another process may currently be publishing this cache. Recheck
            # under the per-mask lock before deciding that AtlasPatch must run.
            pass

    generated.parent.mkdir(parents=True, exist_ok=True)
    generation_lock = (
        advisory_lock(_generated_lock_path(generated)) if concurrent_writes else nullcontext()
    )
    with generation_lock:
        if not rebuild and generated.is_file() and _generated_manifest_path(generated).is_file():
            try:
                return _load_generated_mask(
                    generated,
                    expected_base_fingerprint=expected_base_fingerprint,
                    slide_path=slide,
                    options=options,
                    transform=transform,
                    threshold=threshold,
                )
            except ArtifactCompatibilityError:
                LOGGER.warning(
                    "Generated tissue cache is stale or invalid; AtlasPatch will rerun",
                    exc_info=True,
                )

        atomic_json_dump(
            _generated_manifest_path(generated),
            {
                "schema_version": 2,
                "artifact_type": "tissue_mask",
                "status": "incomplete",
                "base_fingerprint": expected_base_fingerprint,
                "slide_id": slide.stem,
            },
        )
        with tempfile.TemporaryDirectory(
            prefix=f".{slide.stem}.atlaspatch.", dir=generated.parent
        ) as temporary:
            result = runner.run(slide, temporary, mpp=mpp)
            native, page, details = _read_native_mask(result.mask_path, transform)
            normalized = _place_content_mask(native, transform)
            _atomic_save_npy(generated, normalized)
            provenance = {
                **dict(result.provenance),
                "selected_mask_page": page,
                "alignment": details,
                "overview_transform": transform.to_dict(),
                "normalized_shape": list(normalized.shape),
                "normalized_dtype": str(normalized.dtype),
            }
            checkpoint_identity = _checkpoint_identity_from_provenance(provenance)
            complete_fingerprint = _cache_fingerprint(
                slide,
                transform,
                options,
                checkpoint_identity,
            )
            manifest = {
                "schema_version": 2,
                "artifact_type": "tissue_mask",
                "status": "complete",
                "base_fingerprint": expected_base_fingerprint,
                "checkpoint_identity": checkpoint_identity,
                "fingerprint": complete_fingerprint,
                "created_at": datetime.now(UTC).isoformat(),
                "slide_id": slide.stem,
                "mask_path": str(generated),
                "normalized_mask": {
                    "size": generated.stat().st_size,
                    "sha256": sha256_file(generated),
                },
                "provenance": provenance,
            }
            atomic_json_dump(_generated_manifest_path(generated), manifest)
        return TissueMask(
            values=normalized,
            source="atlaspatch",
            threshold=threshold,
            source_path=generated,
            provenance=provenance,
        )


def _tissue_sums_and_areas(
    level: GridLevel,
    mask: TissueMask,
    transform: OverviewTransform,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    values = mask.values.astype(np.int64, copy=False)
    integral = np.pad(values.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
    cells = level.cells
    x0f = transform.content_x0 + cells["level0_x0"] * transform.scale_x
    y0f = transform.content_y0 + cells["level0_y0"] * transform.scale_y
    x1f = transform.content_x0 + cells["clipped_x1"] * transform.scale_x
    y1f = transform.content_y0 + cells["clipped_y1"] * transform.scale_y
    x0 = np.clip(np.floor(x0f).astype(np.int64), 0, transform.canvas_width - 1)
    y0 = np.clip(np.floor(y0f).astype(np.int64), 0, transform.canvas_height - 1)
    x1 = np.clip(np.ceil(x1f).astype(np.int64), 1, transform.canvas_width)
    y1 = np.clip(np.ceil(y1f).astype(np.int64), 1, transform.canvas_height)
    x1 = np.maximum(x1, x0 + 1)
    y1 = np.maximum(y1, y0 + 1)
    sums = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    area = (x1 - x0) * (y1 - y0)
    return sums.astype(np.int64, copy=False), area.astype(np.int64, copy=False)


def tissue_fractions(
    level: GridLevel,
    mask: TissueMask,
    transform: OverviewTransform,
) -> NDArray[np.float32]:
    sums, area = _tissue_sums_and_areas(level, mask, transform)
    return (sums / area).astype(np.float32)


def tissue_bbox_level0(
    mask: TissueMask,
    transform: OverviewTransform,
) -> tuple[int, int, int, int]:
    """Return the exact union-tissue box in canonical level-0 coordinates.

    Mask pixels represent rectangles in the shared overview frame.  Their
    outer edges are mapped back to level-0 coordinates, so disconnected tissue
    islands are enclosed by one box.
    """

    values = np.asarray(mask.values, dtype=np.bool_)
    content_x1 = transform.content_x0 + transform.content_width
    content_y1 = transform.content_y0 + transform.content_height
    content = values[
        transform.content_y0 : content_y1,
        transform.content_x0 : content_x1,
    ]
    positions = np.argwhere(content)
    if not len(positions):
        raise PreprocessingError(
            "tissue mask contains no tissue pixels in the WSI overview content"
        )

    mask_y0, mask_x0 = positions.min(axis=0)
    mask_y1, mask_x1 = positions.max(axis=0) + 1
    raw_x0 = max(0, math.floor(float(mask_x0) / transform.scale_x))
    raw_y0 = max(0, math.floor(float(mask_y0) / transform.scale_y))
    raw_x1 = min(
        transform.source_width,
        math.ceil(float(mask_x1) / transform.scale_x),
    )
    raw_y1 = min(
        transform.source_height,
        math.ceil(float(mask_y1) / transform.scale_y),
    )
    return (
        raw_x0,
        raw_y0,
        raw_x1,
        raw_y1,
    )


def navigation_roi_level0(
    mask: TissueMask,
    transform: OverviewTransform,
    *,
    grid_footprint_level0: int,
) -> tuple[int, int, int, int]:
    """Expand the exact tissue bounds outward to the canonical global grid."""

    if grid_footprint_level0 <= 0:
        raise ValueError("grid footprint must be positive")
    x0, y0, x1, y1 = tissue_bbox_level0(mask, transform)
    footprint = int(grid_footprint_level0)
    return (
        (x0 // footprint) * footprint,
        (y0 // footprint) * footprint,
        math.ceil(x1 / footprint) * footprint,
        math.ceil(y1 / footprint) * footprint,
    )


def _attach_directional_tissue_neighbors(cells: np.ndarray, rows: int, columns: int) -> None:
    """Connect each valid source to the next valid cell along its grid axis."""

    for neighbor_field in (
        "neighbor_up",
        "neighbor_down",
        "neighbor_left",
        "neighbor_right",
    ):
        cells[neighbor_field] = -1
    valid = cells["tissue_valid"].reshape(rows, columns).astype(bool, copy=False)
    ids = cells["cell_id"].reshape(rows, columns)

    # Iterating over axes rather than individual cells keeps preprocessing
    # linear while avoiding several full-grid int64 scratch arrays.
    for row in range(rows):
        columns_with_tissue = np.flatnonzero(valid[row])
        if len(columns_with_tissue) < 2:
            continue
        row_ids = ids[row, columns_with_tissue]
        local = row * columns + columns_with_tissue
        cells["neighbor_left"][local[1:]] = row_ids[:-1]
        cells["neighbor_right"][local[:-1]] = row_ids[1:]
    for column in range(columns):
        rows_with_tissue = np.flatnonzero(valid[:, column])
        if len(rows_with_tissue) < 2:
            continue
        column_ids = ids[rows_with_tissue, column]
        local = rows_with_tissue * columns + column
        cells["neighbor_up"][local[1:]] = column_ids[:-1]
        cells["neighbor_down"][local[:-1]] = column_ids[1:]


def apply_tissue_to_geometry(
    geometry: NavigationGeometry,
    mask: TissueMask,
    transform: OverviewTransform,
) -> NavigationGeometry:
    if (
        transform.source_width != geometry.level0_width
        or transform.source_height != geometry.level0_height
    ):
        raise PreprocessingError("overview transform dimensions do not match navigation geometry")
    expected_mask_shape = (transform.canvas_height, transform.canvas_width)
    if mask.values.shape != expected_mask_shape:
        raise PreprocessingError(
            f"tissue mask shape {mask.values.shape} does not match overview canvas "
            f"{expected_mask_shape}"
        )
    coarsest = geometry.level(geometry.magnifications[0]).footprint_level0
    tissue_bbox = tissue_bbox_level0(mask, transform)
    roi = navigation_roi_level0(
        mask,
        transform,
        grid_footprint_level0=coarsest,
    )
    levels: dict[float, GridLevel] = {}
    for magnification, level in geometry.levels.items():
        cells = level.cells.copy()
        sums, areas = _tissue_sums_and_areas(level, mask, transform)
        fractions = (sums / areas).astype(np.float32)
        cells["tissue_fraction"] = fractions
        cells["tissue_status"] = fractions >= mask.threshold
        cells["tissue_valid"] = sums > 0
        _attach_directional_tissue_neighbors(cells, level.rows, level.columns)
        levels[magnification] = replace(level, cells=cells)
    result = replace(
        geometry,
        levels=levels,
        tissue_bbox_level0=tissue_bbox,
        navigation_roi_level0=roi,
    )
    validate_geometry(result)
    return result


def choose_start_cell(geometry: NavigationGeometry) -> int:
    """Select central coarsest cell, falling back to nearest tissue cell."""

    level = geometry.level(geometry.magnifications[0])
    center_x = geometry.level0_width / 2
    center_y = geometry.level0_height / 2
    cell_center_x = (level.cells["level0_x0"] + level.cells["clipped_x1"]) / 2
    cell_center_y = (level.cells["level0_y0"] + level.cells["clipped_y1"]) / 2
    distance = (cell_center_x - center_x) ** 2 + (cell_center_y - center_y) ** 2
    central = int(np.argmin(distance))
    if bool(level.cells["tissue_valid"][central]):
        return int(level.cells["cell_id"][central])
    tissue = np.flatnonzero(level.cells["tissue_valid"])
    if not len(tissue):
        raise PreprocessingError("navigation geometry has no tissue-valid starting cell")
    nearest = int(tissue[np.argmin(distance[tissue])])
    return int(level.cells["cell_id"][nearest])


__all__ = [
    "TissueMask",
    "apply_tissue_to_geometry",
    "atlaspatch_available",
    "choose_start_cell",
    "load_or_create_tissue_mask",
    "navigation_roi_level0",
    "tissue_bbox_level0",
    "tissue_fractions",
]
