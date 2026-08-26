"""Resumable whole-cohort validation with sparse-aware diagnostics."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from rl4xs_wsi.data.base import DatasetAdapter, EmbeddingLocation, SlideRecord
from rl4xs_wsi.data.embeddings.hdf5 import AtlasPatchHDF5Reader
from rl4xs_wsi.errors import DatasetValidationError
from rl4xs_wsi.schemas import ValidationIssue, ValidationReport
from rl4xs_wsi.utils.atomic import atomic_json_dump
from rl4xs_wsi.utils.fingerprints import fingerprint

_VALIDATOR_SCHEMA_VERSION = 2


class SlideInspector(Protocol):
    def inspect(self, path: str | Path) -> Any: ...


@dataclass(frozen=True, slots=True)
class ValidationOptions:
    feature_chunk_rows: int = 8192
    mapping_tolerance: int = 0
    check_feature_finiteness: bool = True
    check_wsi_existence: bool = True
    check_wsi_readability: bool = True
    allow_boundary_overhang: bool = True

    def __post_init__(self) -> None:
        if self.feature_chunk_rows < 1:
            raise ValueError("feature_chunk_rows must be positive")
        if self.mapping_tolerance < 0:
            raise ValueError("mapping_tolerance cannot be negative")
        if self.check_wsi_readability and not self.check_wsi_existence:
            raise ValueError("WSI readability checking requires WSI existence checking")


def report_has_errors(report: ValidationReport) -> bool:
    return any(item.severity in {"error", "fatal"} for item in report.issues)


class DatasetValidator:
    """Validate required files without imposing a dense embedding hierarchy."""

    def __init__(
        self,
        adapter: DatasetAdapter,
        *,
        reader: AtlasPatchHDF5Reader | None = None,
        slide_inspector: SlideInspector | None = None,
        options: ValidationOptions | None = None,
    ) -> None:
        self.adapter = adapter
        self.reader = reader or AtlasPatchHDF5Reader()
        self.slide_inspector = slide_inspector
        self.options = options or ValidationOptions(
            check_wsi_readability=slide_inspector is not None
        )

    def validate(
        self,
        *,
        report_path: str | Path | None = None,
        raise_on_error: bool = False,
        resume: bool = True,
        slide_ids: Sequence[str] | None = None,
    ) -> ValidationReport:
        """Validate all required artifacts, checkpointing after every file.

        A prior complete *or incomplete* report can be resumed.  Individual
        results are reused only when the validator context and a file's stat
        signature are unchanged.  The feature-finiteness scan is therefore
        not repeated for compatible HDF5 files, while a changed file is always
        checked again.
        """

        issues: list[ValidationIssue] = []
        per_file: dict[str, dict[str, Any]] = {}
        checked_files = 0
        observed_dimensions: set[int] = set()
        if slide_ids is None:
            records = tuple(self.adapter.iter_slides())
        else:
            selected = tuple(str(value) for value in slide_ids)
            if len(selected) != len(set(selected)):
                raise ValueError("slide_ids cannot contain duplicates")
            try:
                records = tuple(self.adapter.get_slide(slide_id) for slide_id in selected)
            except KeyError as exc:
                raise DatasetValidationError(f"unknown selected slide ID: {exc.args[0]}") from exc
        context_fingerprint = fingerprint(
            {
                "validator_schema_version": _VALIDATOR_SCHEMA_VERSION,
                "dataset": self.adapter.name,
                "root": self.adapter.root,
                "selected_slide_ids": [item.slide_id for item in records],
                "magnifications": self.adapter.magnifications,
                "options": self.options,
            }
        )
        report_target = (
            Path(report_path).expanduser().resolve() if report_path is not None else None
        )
        prior = (
            self._load_prior_entries(report_target) if resume and report_target is not None else {}
        )
        expected_keys = {
            key
            for record in records
            for key in (
                f"{record.slide_id}@wsi",
                *(f"{record.slide_id}@{value:g}x" for value in self.adapter.magnifications),
            )
        }
        prior = {
            key: value
            for key, value in prior.items()
            if key in expected_keys
            and isinstance(value.get("_resume"), dict)
            and value["_resume"].get("context_fingerprint") == context_fingerprint
        }
        planned_report_fingerprint = self._report_fingerprint(records, context_fingerprint)

        def checkpoint(*, complete: bool = False) -> ValidationReport:
            current_dimension = (
                next(iter(observed_dimensions)) if len(observed_dimensions) == 1 else None
            )
            result = ValidationReport(
                fingerprint=(
                    self._report_fingerprint(records, context_fingerprint)
                    if complete
                    else planned_report_fingerprint
                ),
                complete=complete,
                checked_files=checked_files,
                feature_dimension=current_dimension,
                issues=issues,
                per_file={**prior, **per_file},
            )
            if report_target is not None:
                atomic_json_dump(report_target, result.model_dump(mode="json"))
            return result

        if report_target is not None:
            checkpoint()
        for record in records:
            slide_key = f"{record.slide_id}@wsi"
            slide_signature = self._slide_signature(record)
            cached = self._cached_result(prior, slide_key, context_fingerprint, slide_signature)
            if cached is None:
                local_issues: list[ValidationIssue] = []
                temporary: dict[str, dict[str, Any]] = {}
                self._validate_slide(record, local_issues, temporary)
                slide_result = temporary[slide_key]
                final_signature = self._slide_signature(record)
                cacheable = final_signature == slide_signature
                if not cacheable:
                    local_issues.append(
                        self._issue(
                            "error",
                            "file_changed_during_validation",
                            "WSI metadata changed while it was being validated; rerun validation",
                            record,
                            path=record.slide_path,
                        )
                    )
                slide_result = self._decorate_result(
                    slide_result,
                    context_fingerprint=context_fingerprint,
                    input_signature=final_signature,
                    issues=local_issues,
                    cacheable=cacheable,
                )
            else:
                slide_result, local_issues = cached
            per_file[slide_key] = slide_result
            issues.extend(local_issues)
            checked_files += int(self.options.check_wsi_existence)
            checkpoint()
            scale_metadata: list[dict[str, Any]] = []
            for magnification in self.adapter.magnifications:
                location = self.adapter.embedding_location(record.slide_id, magnification)
                key = f"{record.slide_id}@{magnification:g}x"
                input_signature = self._embedding_signature(location)
                cached = self._cached_result(prior, key, context_fingerprint, input_signature)
                if cached is None:
                    local_issues = []
                    result = self._validate_embedding(location, record, local_issues)
                    final_signature = self._embedding_signature(location)
                    cacheable = final_signature == input_signature
                    if not cacheable:
                        local_issues.append(
                            self._issue(
                                "error",
                                "file_changed_during_validation",
                                "embedding metadata changed while it was being validated; rerun validation",
                                record,
                                location.magnification,
                                location.path,
                            )
                        )
                    result = self._decorate_result(
                        result,
                        context_fingerprint=context_fingerprint,
                        input_signature=final_signature,
                        issues=local_issues,
                        cacheable=cacheable,
                    )
                else:
                    result, local_issues = cached
                checked_files += 1
                per_file[key] = result
                issues.extend(local_issues)
                if result.get("feature_dim") is not None:
                    observed_dimensions.add(int(result["feature_dim"]))
                if result.get("metadata"):
                    scale_metadata.append(result["metadata"])
                checkpoint()
            self._validate_cross_scale(record, scale_metadata, issues)
            self._validate_wsi_alignment(record, slide_result, scale_metadata, issues)
            checkpoint()
        if len(observed_dimensions) > 1:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="inconsistent_feature_dimension",
                    message=f"configured embeddings expose multiple feature dimensions: {sorted(observed_dimensions)}",
                )
            )
        report = checkpoint(complete=True)
        if raise_on_error and report_has_errors(report):
            counts: dict[str, int] = {}
            for item in issues:
                if item.severity in {"error", "fatal"}:
                    counts[item.code] = counts.get(item.code, 0) + 1
            raise DatasetValidationError(f"dataset validation failed: {counts}")
        return report

    @staticmethod
    def _path_signature(path: Path) -> dict[str, Any]:
        resolved = path.expanduser().resolve()
        try:
            stat = resolved.stat()
        except OSError:
            return {"path": str(resolved), "exists": False}
        return {
            "path": str(resolved),
            "exists": True,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "device": stat.st_dev,
            "inode": stat.st_ino,
        }

    def _slide_signature(self, record: SlideRecord) -> dict[str, Any]:
        return {
            "file": (
                self._path_signature(record.slide_path)
                if self.options.check_wsi_existence
                else {"inspection": "disabled"}
            ),
            "slide_id": record.slide_id,
            "label": record.label,
            "label_name": record.label_name,
            "official_cohort": record.official_cohort,
        }

    def _embedding_signature(self, location: EmbeddingLocation) -> dict[str, Any]:
        return {
            "file": self._path_signature(location.path),
            "slide_id": location.slide_id,
            "magnification": location.magnification,
            "encoder": location.encoder,
            "feature_key": location.feature_key,
            "coordinate_key": location.coordinate_key,
            "passport_key": location.passport_key,
        }

    def _report_fingerprint(
        self,
        records: tuple[SlideRecord, ...],
        context_fingerprint: str,
    ) -> str:
        inventory: list[dict[str, Any]] = []
        for record in records:
            inventory.append(self._slide_signature(record))
            inventory.extend(
                self._embedding_signature(
                    self.adapter.embedding_location(record.slide_id, magnification)
                )
                for magnification in self.adapter.magnifications
            )
        return fingerprint({"context": context_fingerprint, "inventory": inventory})

    @staticmethod
    def _load_prior_entries(path: Path) -> dict[str, dict[str, Any]]:
        if not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            entries = value.get("per_file", {})
            if not isinstance(entries, dict):
                return {}
            return {str(key): item for key, item in entries.items() if isinstance(item, dict)}
        except (OSError, json.JSONDecodeError, TypeError):
            return {}

    @staticmethod
    def _decorate_result(
        result: dict[str, Any],
        *,
        context_fingerprint: str,
        input_signature: dict[str, Any],
        issues: list[ValidationIssue],
        cacheable: bool,
    ) -> dict[str, Any]:
        return {
            **result,
            "_resume": {
                "validator_schema_version": _VALIDATOR_SCHEMA_VERSION,
                "context_fingerprint": context_fingerprint,
                "input_signature": input_signature,
                "complete": cacheable,
                "issues": [item.model_dump(mode="json") for item in issues],
            },
        }

    @staticmethod
    def _cached_result(
        prior: dict[str, dict[str, Any]],
        key: str,
        context_fingerprint: str,
        input_signature: dict[str, Any],
    ) -> tuple[dict[str, Any], list[ValidationIssue]] | None:
        value = prior.get(key)
        if value is None:
            return None
        metadata = value.get("_resume")
        if not isinstance(metadata, dict):
            return None
        if (
            metadata.get("validator_schema_version") != _VALIDATOR_SCHEMA_VERSION
            or metadata.get("context_fingerprint") != context_fingerprint
            or metadata.get("input_signature") != input_signature
            or metadata.get("complete") is not True
        ):
            return None
        try:
            cached_issues = [
                ValidationIssue.model_validate(item) for item in metadata.get("issues", [])
            ]
        except (TypeError, ValueError):
            return None
        return dict(value), cached_issues

    def _validate_slide(
        self,
        record: SlideRecord,
        issues: list[ValidationIssue],
        per_file: dict[str, dict[str, Any]],
    ) -> int:
        if record.label not in {0, 1}:
            issues.append(
                self._issue(
                    "error", "invalid_label", f"label must be binary, got {record.label}", record
                )
            )
        if not self.options.check_wsi_existence:
            per_file[f"{record.slide_id}@wsi"] = {
                "skipped": True,
                "reason": "WSI existence/readability checks disabled",
            }
            return 0
        if not record.slide_path.is_file():
            issues.append(
                self._issue(
                    "error",
                    "missing_wsi",
                    "required WSI file does not exist",
                    record,
                    path=record.slide_path,
                )
            )
            per_file[f"{record.slide_id}@wsi"] = {"exists": False}
            return 1
        result: dict[str, Any] = {"exists": True, "size": record.slide_path.stat().st_size}
        if self.options.check_wsi_readability:
            if self.slide_inspector is None:
                issues.append(
                    self._issue(
                        "error",
                        "wsi_inspector_unavailable",
                        "WSI readability checking was requested but no slide inspector was provided",
                        record,
                        path=record.slide_path,
                    )
                )
            else:
                try:
                    metadata = self.slide_inspector.inspect(record.slide_path)
                    dimensions = getattr(metadata, "dimensions", None)
                    if dimensions is not None:
                        result["dimensions"] = list(dimensions)
                    objective_power = getattr(metadata, "objective_power", None)
                    if objective_power is not None:
                        result["objective_power"] = float(objective_power)
                except Exception as exc:  # backend exceptions are normalized into the report
                    issues.append(
                        self._issue(
                            "error",
                            "unreadable_wsi",
                            f"WSI reader failed: {type(exc).__name__}: {exc}",
                            record,
                            path=record.slide_path,
                        )
                    )
        per_file[f"{record.slide_id}@wsi"] = result
        return 1

    def _validate_embedding(
        self,
        location: EmbeddingLocation,
        record: SlideRecord,
        issues: list[ValidationIssue],
    ) -> dict[str, Any]:
        path = location.path
        if not path.is_file():
            issues.append(
                self._issue(
                    "error",
                    "missing_embedding_file",
                    "required embedding artifact does not exist",
                    record,
                    location.magnification,
                    path,
                )
            )
            return {"exists": False}
        try:
            metadata = self.reader.inspect(location)
            coordinates = self.reader.read_coordinates(location)
        except Exception as exc:
            issues.append(
                self._issue(
                    "error",
                    "invalid_embedding_file",
                    f"embedding reader failed: {type(exc).__name__}: {exc}",
                    record,
                    location.magnification,
                    path,
                )
            )
            return {"exists": True, "readable": False}
        result: dict[str, Any] = {
            "exists": True,
            "readable": True,
            "size": path.stat().st_size,
            "rows": metadata.num_instances,
            "feature_dim": metadata.feature_dim,
            "feature_dtype": metadata.feature_dtype,
            "coordinate_dtype": metadata.coordinate_dtype,
            "metadata": {
                "magnification": metadata.magnification,
                "level0_width": metadata.level0_width,
                "level0_height": metadata.level0_height,
                "level0_magnification": metadata.level0_magnification,
                "patch_size": metadata.patch_size,
                "patch_size_level0": metadata.patch_size_level0,
                "stride_level0": metadata.stride_level0,
            },
        }
        if metadata.num_instances == 0:
            issues.append(
                self._issue(
                    "warning",
                    "empty_embedding_file",
                    "embedding artifact has no instances",
                    record,
                    location.magnification,
                    path,
                )
            )
        if not np.issubdtype(coordinates.dtype, np.integer):
            issues.append(
                self._issue(
                    "error",
                    "non_integer_coordinates",
                    f"coordinate dtype is {coordinates.dtype}",
                    record,
                    location.magnification,
                    path,
                )
            )
            return result
        if not np.issubdtype(np.dtype(metadata.feature_dtype), np.floating):
            issues.append(
                self._issue(
                    "error",
                    "non_floating_features",
                    f"feature dtype is {metadata.feature_dtype}",
                    record,
                    location.magnification,
                    path,
                )
            )
        expected_footprint = round(
            metadata.patch_size * metadata.level0_magnification / metadata.magnification
        )
        if expected_footprint != metadata.patch_size_level0:
            issues.append(
                self._issue(
                    "error",
                    "inconsistent_patch_footprint",
                    f"patch_size_level0={metadata.patch_size_level0}, expected {expected_footprint}",
                    record,
                    location.magnification,
                    path,
                )
            )
        xy = coordinates[:, :2].astype(np.int64, copy=False)
        if len(xy):
            origin_invalid = (
                (xy[:, 0] < 0)
                | (xy[:, 1] < 0)
                | (xy[:, 0] >= metadata.level0_width)
                | (xy[:, 1] >= metadata.level0_height)
            )
            invalid_count = int(np.count_nonzero(origin_invalid))
            result["invalid_origins"] = invalid_count
            if invalid_count:
                issues.append(
                    self._issue(
                        "error",
                        "coordinate_origin_out_of_bounds",
                        f"{invalid_count} coordinate origins lie outside the level-0 WSI",
                        record,
                        location.magnification,
                        path,
                    )
                )
            stride = metadata.stride_level0
            residual_x = np.minimum(xy[:, 0] % stride, (-xy[:, 0]) % stride)
            residual_y = np.minimum(xy[:, 1] % stride, (-xy[:, 1]) % stride)
            off_grid = (residual_x > self.options.mapping_tolerance) | (
                residual_y > self.options.mapping_tolerance
            )
            off_grid_count = int(np.count_nonzero(off_grid))
            result["off_grid"] = off_grid_count
            if off_grid_count:
                issues.append(
                    self._issue(
                        "error",
                        "off_grid_coordinates",
                        f"{off_grid_count} coordinates exceed mapping tolerance "
                        f"{self.options.mapping_tolerance}px",
                        record,
                        location.magnification,
                        path,
                    )
                )
            overhang = (
                (xy[:, 0] + metadata.patch_size_level0 > metadata.level0_width)
                | (xy[:, 1] + metadata.patch_size_level0 > metadata.level0_height)
            ) & ~origin_invalid
            overhang_count = int(np.count_nonzero(overhang))
            result["boundary_overhang"] = overhang_count
            if overhang_count:
                severity = "info" if self.options.allow_boundary_overhang else "error"
                issues.append(
                    self._issue(
                        severity,
                        "partial_boundary_cells",
                        f"{overhang_count} valid grid origins have footprints clipped by the WSI boundary",
                        record,
                        location.magnification,
                        path,
                    )
                )
            self._validate_duplicates(location, coordinates, record, issues, result)
        if self.options.check_feature_finiteness:
            nonfinite = 0
            for _, _, chunk in self.reader.iter_feature_chunks(
                location, self.options.feature_chunk_rows
            ):
                nonfinite += int(chunk.size - np.count_nonzero(np.isfinite(chunk)))
            result["nonfinite_feature_values"] = nonfinite
            if nonfinite:
                issues.append(
                    self._issue(
                        "error",
                        "nonfinite_features",
                        f"feature tensor contains {nonfinite} non-finite values",
                        record,
                        location.magnification,
                        path,
                    )
                )
        return result

    def _validate_duplicates(
        self,
        location: EmbeddingLocation,
        coordinates: np.ndarray,
        record: SlideRecord,
        issues: list[ValidationIssue],
        result: dict[str, Any],
    ) -> None:
        xy = coordinates[:, :2]
        _, inverse, counts = np.unique(xy, axis=0, return_inverse=True, return_counts=True)
        duplicate_groups = np.flatnonzero(counts > 1)
        extra_rows = int(np.sum(counts[duplicate_groups] - 1))
        result["duplicate_groups"] = len(duplicate_groups)
        result["duplicate_extra_rows"] = extra_rows
        if not len(duplicate_groups):
            return
        conflicting = 0
        for group in duplicate_groups:
            indices = np.flatnonzero(inverse == group).astype(np.int64)
            if not np.all(coordinates[indices] == coordinates[indices[0]]):
                conflicting += 1
                continue
            features = self.reader.read_features(location, indices.tolist())
            if not all(np.array_equal(features[0], value) for value in features[1:]):
                conflicting += 1
                continue
            passports = self.reader.read_passports(location, indices.tolist())
            if passports is not None and not all(passports[0] == value for value in passports[1:]):
                conflicting += 1
        result["conflicting_duplicate_groups"] = conflicting
        if conflicting:
            issues.append(
                self._issue(
                    "error",
                    "conflicting_duplicate_coordinates",
                    f"{conflicting}/{len(duplicate_groups)} duplicate coordinate groups differ in data",
                    record,
                    location.magnification,
                    location.path,
                )
            )
        exact = len(duplicate_groups) - conflicting
        if exact:
            issues.append(
                self._issue(
                    "warning",
                    "exact_duplicate_coordinates",
                    f"{exact} exact duplicate coordinate groups ({extra_rows} extra rows) will collapse "
                    "to their lowest source index",
                    record,
                    location.magnification,
                    location.path,
                )
            )

    def _validate_cross_scale(
        self,
        record: SlideRecord,
        metadata: list[dict[str, Any]],
        issues: list[ValidationIssue],
    ) -> None:
        if not metadata:
            return
        dimensions = {(item["level0_width"], item["level0_height"]) for item in metadata}
        level0_mags = {item["level0_magnification"] for item in metadata}
        if len(dimensions) > 1:
            issues.append(
                self._issue(
                    "error", "cross_scale_dimension_mismatch", str(sorted(dimensions)), record
                )
            )
        if len(level0_mags) > 1:
            issues.append(
                self._issue(
                    "error",
                    "cross_scale_base_magnification_mismatch",
                    str(sorted(level0_mags)),
                    record,
                )
            )
        ordered = sorted(metadata, key=lambda item: item["magnification"])
        for lower, higher in pairwise(ordered):
            ratio = higher["magnification"] / lower["magnification"]
            footprint_ratio = lower["patch_size_level0"] / higher["patch_size_level0"]
            if not np.isclose(ratio, footprint_ratio, rtol=0.0, atol=1e-6):
                issues.append(
                    self._issue(
                        "error",
                        "cross_scale_alignment_mismatch",
                        f"{lower['magnification']:g}x->{higher['magnification']:g}x scale ratio {ratio:g} "
                        f"does not match footprint ratio {footprint_ratio:g}",
                        record,
                    )
                )

    def _validate_wsi_alignment(
        self,
        record: SlideRecord,
        slide_result: dict[str, Any],
        metadata: list[dict[str, Any]],
        issues: list[ValidationIssue],
    ) -> None:
        if not metadata or "dimensions" not in slide_result:
            return
        wsi_dimensions = tuple(int(value) for value in slide_result["dimensions"])
        embedded_dimensions = (metadata[0]["level0_width"], metadata[0]["level0_height"])
        if wsi_dimensions != embedded_dimensions:
            issues.append(
                self._issue(
                    "error",
                    "wsi_embedding_dimension_mismatch",
                    f"WSI dimensions {wsi_dimensions} != embedding metadata {embedded_dimensions}",
                    record,
                    path=record.slide_path,
                )
            )
        objective = slide_result.get("objective_power")
        if objective is not None and not np.isclose(
            float(objective), float(metadata[0]["level0_magnification"]), rtol=0.05, atol=0.0
        ):
            issues.append(
                self._issue(
                    "error",
                    "wsi_embedding_base_magnification_mismatch",
                    f"WSI objective {float(objective):g}x != embedding level-0 "
                    f"{float(metadata[0]['level0_magnification']):g}x",
                    record,
                    path=record.slide_path,
                )
            )

    @staticmethod
    def _issue(
        severity: str,
        code: str,
        message: str,
        record: SlideRecord,
        magnification: float | None = None,
        path: str | Path | None = None,
    ) -> ValidationIssue:
        return ValidationIssue(
            severity=severity,
            code=code,
            message=message,
            slide_id=record.slide_id,
            magnification=magnification,
            path=str(path) if path is not None else None,
        )


__all__ = ["DatasetValidator", "ValidationOptions", "report_has_errors"]
