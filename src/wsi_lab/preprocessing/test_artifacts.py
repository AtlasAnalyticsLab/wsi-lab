"""Evaluator-only preparation and publication of official-test artifacts.

Training preprocessing deliberately excludes the official test cohort.  This
module publishes a small, immutable evaluation index over test slide artifacts
without extending or rewriting the development index.  A fingerprint-scoped
lock makes preparation safe when many evaluator jobs start concurrently.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypeAlias

from wsi_lab.constants import SCHEMA_VERSION
from wsi_lab.errors import ArtifactCompatibilityError, PreprocessingError
from wsi_lab.preprocessing.artifacts import ArtifactStore, advisory_lock
from wsi_lab.preprocessing.pipeline import SlidePreprocessingResult
from wsi_lab.utils.atomic import atomic_json_dump
from wsi_lab.utils.fingerprints import fingerprint

AUTHORIZED_TEST_SCOPE = "authorized-official-test-evaluation"
TestArtifactMode: TypeAlias = Literal["auto", "prepare", "require"]
TestArtifactBuilder: TypeAlias = Callable[[tuple[str, ...]], Sequence[SlidePreprocessingResult]]

_MANIFEST_NAME = "manifest.json"
_INDEX_NAME = "slides.json"
_ARTIFACT_TYPE = "authorized_test_evaluation_artifacts"
_INDEX_ARTIFACT_TYPE = "authorized_test_evaluation_index"


def test_artifact_fingerprint(
    *,
    preprocessing_fingerprint: str,
    split_fingerprint: str,
    test_slide_ids: Sequence[str],
) -> str:
    """Return the established fingerprint for an authorized test evaluation."""

    return fingerprint(
        {
            "scope": AUTHORIZED_TEST_SCOPE,
            "preprocessing": preprocessing_fingerprint,
            "split": split_fingerprint,
            "test_slides": tuple(test_slide_ids),
        }
    )


def _validated_slide_ids(test_slide_ids: Sequence[str]) -> tuple[str, ...]:
    slide_ids = tuple(str(slide_id) for slide_id in test_slide_ids)
    if not slide_ids:
        raise PreprocessingError("authorized test artifact preparation requires test slides")
    if len(slide_ids) != len(set(slide_ids)):
        raise PreprocessingError("authorized test artifact slide IDs must be unique")
    unsafe = [
        slide_id
        for slide_id in slide_ids
        if not slide_id or Path(slide_id).name != slide_id or slide_id in {".", ".."}
    ]
    if unsafe:
        raise PreprocessingError(f"unsafe authorized test slide IDs: {unsafe[:8]}")
    return slide_ids


def _read_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreprocessingError(f"cannot read {description} {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise PreprocessingError(f"invalid {description} {path}: expected a JSON object")
    return raw


def _entry_from_metadata(
    store: ArtifactStore,
    slide_id: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        if str(metadata["slide_id"]) != slide_id:
            raise ValueError("slide ID mismatch")
        return {
            "slide_id": slide_id,
            "slide_index": int(metadata["slide_index"]),
            "metadata": str(store.slide_paths(slide_id).metadata.relative_to(store.root)),
            "metadata_fingerprint": fingerprint(dict(metadata)),
            "input_fingerprint": str(metadata["input_fingerprint"]),
            "cell_id_offset": int(metadata["cell_id_offset"]),
            "num_cells": int(metadata["num_cells"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactCompatibilityError(
            f"processed-slide metadata is incomplete for {slide_id!r}: {exc}"
        ) from exc


def _adopt_entries(
    store: ArtifactStore,
    slide_ids: tuple[str, ...],
) -> tuple[dict[str, dict[str, Any]], tuple[str, ...]]:
    entries: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for slide_id in slide_ids:
        try:
            metadata = store.load_metadata(slide_id)
            entries[slide_id] = _entry_from_metadata(store, slide_id, metadata)
        except ArtifactCompatibilityError:
            missing.append(slide_id)
    return entries, tuple(missing)


def _index_payload(
    *,
    evaluation_fingerprint: str,
    preprocessing_fingerprint: str,
    split_fingerprint: str,
    slide_ids: tuple[str, ...],
    entries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    slide_indices = [int(entries[slide_id]["slide_index"]) for slide_id in slide_ids]
    if len(slide_indices) != len(set(slide_indices)):
        raise ArtifactCompatibilityError("test-artifact index has duplicate slide indices")
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": _INDEX_ARTIFACT_TYPE,
        "status": "complete",
        "evaluation_fingerprint": evaluation_fingerprint,
        "scope": AUTHORIZED_TEST_SCOPE,
        "preprocessing_fingerprint": preprocessing_fingerprint,
        "split_fingerprint": split_fingerprint,
        "test_slides": list(slide_ids),
        "slides": [dict(entries[slide_id]) for slide_id in slide_ids],
    }
    return {**body, "fingerprint": fingerprint(body)}


def _validate_complete(
    *,
    manifest_path: Path,
    index_path: Path,
    evaluation_fingerprint: str,
    preprocessing_fingerprint: str,
    split_fingerprint: str,
    slide_ids: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if not manifest_path.exists():
        return None
    manifest = _read_json(manifest_path, description="test-artifact manifest")
    if manifest.get("status") != "complete":
        return None

    expected_manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": _ARTIFACT_TYPE,
        "fingerprint": evaluation_fingerprint,
        "scope": AUTHORIZED_TEST_SCOPE,
        "preprocessing_fingerprint": preprocessing_fingerprint,
        "split_fingerprint": split_fingerprint,
        "test_slides": list(slide_ids),
        "slides": len(slide_ids),
        "index": _INDEX_NAME,
        "training_dependency": False,
        "existing_development_slide_artifacts_modified": False,
        "development_index_modified": False,
    }
    mismatched = [
        key for key, expected in expected_manifest.items() if manifest.get(key) != expected
    ]
    if mismatched:
        raise PreprocessingError(
            "completed test-artifact manifest is incompatible "
            f"({', '.join(mismatched)}): {manifest_path}"
        )
    if not index_path.is_file():
        raise PreprocessingError(
            f"completed test-artifact manifest has no immutable index: {index_path}"
        )
    index = _read_json(index_path, description="test-artifact index")
    try:
        index_fingerprint = str(index["fingerprint"])
        index_body = {key: value for key, value in index.items() if key != "fingerprint"}
        indexed_slides = list(index_body["slides"])
        indexed_ids = [str(item["slide_id"]) for item in indexed_slides]
    except (KeyError, TypeError, ValueError) as exc:
        raise PreprocessingError(f"invalid test-artifact index {index_path}: {exc}") from exc
    expected_index_fields = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": _INDEX_ARTIFACT_TYPE,
        "status": "complete",
        "evaluation_fingerprint": evaluation_fingerprint,
        "scope": AUTHORIZED_TEST_SCOPE,
        "preprocessing_fingerprint": preprocessing_fingerprint,
        "split_fingerprint": split_fingerprint,
        "test_slides": list(slide_ids),
    }
    index_mismatched = [
        key for key, expected in expected_index_fields.items() if index_body.get(key) != expected
    ]
    if set(index_body) != {*expected_index_fields, "slides"}:
        index_mismatched.append("fields")
    if indexed_ids != list(slide_ids):
        index_mismatched.append("slides")
    expected_entry_fields = {
        "slide_id",
        "slide_index",
        "metadata",
        "input_fingerprint",
        "cell_id_offset",
        "num_cells",
    }
    extended_entry_fields = {*expected_entry_fields, "metadata_fingerprint"}
    for slide_id, entry in zip(slide_ids, indexed_slides, strict=False):
        if not isinstance(entry, dict) or frozenset(entry) not in {
            frozenset(expected_entry_fields),
            frozenset(extended_entry_fields),
        }:
            index_mismatched.append("slide fields")
            break
        expected_metadata = str(Path("slides") / slide_id / "metadata.json")
        try:
            valid_entry = (
                entry["metadata"] == expected_metadata
                and bool(str(entry["input_fingerprint"]))
                and int(entry["slide_index"]) >= 0
                and int(entry["cell_id_offset"]) >= 0
                and int(entry["num_cells"]) > 0
                and (
                    "metadata_fingerprint" not in entry or bool(str(entry["metadata_fingerprint"]))
                )
            )
        except (TypeError, ValueError):
            valid_entry = False
        if not valid_entry:
            index_mismatched.append("slide entries")
            break
    try:
        slide_indices = [int(entry["slide_index"]) for entry in indexed_slides]
    except (KeyError, TypeError, ValueError):
        slide_indices = []
    if len(slide_indices) != len(slide_ids) or len(slide_indices) != len(set(slide_indices)):
        index_mismatched.append("slide indices")
    if index_fingerprint != fingerprint(index_body):
        index_mismatched.append("fingerprint")
    if manifest.get("index_fingerprint") != index_fingerprint:
        index_mismatched.append("manifest.index_fingerprint")
    if index_mismatched:
        raise PreprocessingError(
            "completed test-artifact index is incompatible "
            f"({', '.join(dict.fromkeys(index_mismatched))}): {index_path}"
        )
    return manifest, index


def _result(
    *,
    manifest_path: Path,
    index_path: Path,
    manifest_payload: Mapping[str, Any],
    index_payload: Mapping[str, Any],
    evaluation_fingerprint: str,
    preprocessing_fingerprint: str,
    split_fingerprint: str,
    slide_ids: tuple[str, ...],
    built: int,
    reused: int,
    cache_result: Literal["prepared", "reused"],
    requested_mode: TestArtifactMode,
) -> dict[str, Any]:
    return {
        "status": "complete",
        "manifest": str(manifest_path),
        "index": str(index_path),
        "index_fingerprint": str(index_payload["fingerprint"]),
        "fingerprint": evaluation_fingerprint,
        "scope": AUTHORIZED_TEST_SCOPE,
        "preprocessing_fingerprint": preprocessing_fingerprint,
        "split_fingerprint": split_fingerprint,
        "test_slides": list(slide_ids),
        "slides": len(slide_ids),
        "built": built,
        "reused": reused,
        "cache_result": cache_result,
        "requested_mode": requested_mode,
        "started_at": manifest_payload.get("started_at"),
        "completed_at": manifest_payload.get("completed_at"),
        "training_dependency": False,
        # Retain the original public field while exposing the more explicit
        # name used by the immutable-index implementation.
        "development_slide_artifacts_modified": False,
        "shared_index_extended": False,
        "existing_development_slide_artifacts_modified": False,
        "development_index_modified": False,
    }


def load_test_artifact_entries(
    index_path: str | Path,
    *,
    expected_fingerprint: str,
) -> dict[str, dict[str, Any]]:
    """Load one validated immutable index for lazy evaluator-side pinning."""

    path = Path(index_path).expanduser().resolve()
    index = _read_json(path, description="test-artifact index")
    actual_fingerprint = str(index.get("fingerprint", ""))
    body = {key: value for key, value in index.items() if key != "fingerprint"}
    if (
        actual_fingerprint != expected_fingerprint
        or actual_fingerprint != fingerprint(body)
        or index.get("status") != "complete"
        or index.get("artifact_type") != _INDEX_ARTIFACT_TYPE
    ):
        raise PreprocessingError(f"test-artifact index fingerprint is incompatible: {path}")
    try:
        items = list(index["slides"])
        entries = {str(item["slide_id"]): dict(item) for item in items}
    except (KeyError, TypeError, ValueError) as exc:
        raise PreprocessingError(f"invalid test-artifact index {path}: {exc}") from exc
    if len(entries) != len(items):
        raise PreprocessingError(f"test-artifact index has duplicate slide IDs: {path}")
    return entries


def ensure_test_artifacts(
    *,
    store: ArtifactStore,
    preprocessing_fingerprint: str,
    split_fingerprint: str,
    test_slide_ids: Sequence[str],
    mode: TestArtifactMode = "auto",
    builder: TestArtifactBuilder | None = None,
) -> dict[str, Any]:
    """Require or prepare one immutable, fingerprinted official-test index.

    ``require`` is strictly read-only.  ``auto`` and ``prepare`` share the same
    concurrency-safe preparation semantics; the distinct names let callers
    express whether preparation is incidental to evaluation or an explicit
    CLI stage.  The callback receives only slide IDs without completed
    artifacts in ``store``.
    """

    if mode not in {"auto", "prepare", "require"}:
        raise ValueError(f"unknown test-artifact mode: {mode!r}")
    slide_ids = _validated_slide_ids(test_slide_ids)
    evaluation_fingerprint = test_artifact_fingerprint(
        preprocessing_fingerprint=preprocessing_fingerprint,
        split_fingerprint=split_fingerprint,
        test_slide_ids=slide_ids,
    )
    evaluation_root = store.root / "test_evaluations" / evaluation_fingerprint
    manifest_path = evaluation_root / _MANIFEST_NAME
    index_path = evaluation_root / _INDEX_NAME

    complete = _validate_complete(
        manifest_path=manifest_path,
        index_path=index_path,
        evaluation_fingerprint=evaluation_fingerprint,
        preprocessing_fingerprint=preprocessing_fingerprint,
        split_fingerprint=split_fingerprint,
        slide_ids=slide_ids,
    )
    if complete is not None:
        manifest, index = complete
        return _result(
            manifest_path=manifest_path,
            index_path=index_path,
            manifest_payload=manifest,
            index_payload=index,
            evaluation_fingerprint=evaluation_fingerprint,
            preprocessing_fingerprint=preprocessing_fingerprint,
            split_fingerprint=split_fingerprint,
            slide_ids=slide_ids,
            built=0,
            reused=len(slide_ids),
            cache_result="reused",
            requested_mode=mode,
        )
    if mode == "require":
        raise PreprocessingError(
            "required authorized test artifacts are absent or incomplete: "
            f"{manifest_path}; run `wsi-lab prepare-test --config ... --allow-test` first"
        )

    lock_path = store.root / "test_evaluations" / ".locks" / f"{evaluation_fingerprint}.lock"
    with advisory_lock(lock_path):
        # A concurrent process may have completed preparation while this caller
        # waited for the fingerprint lock.
        complete = _validate_complete(
            manifest_path=manifest_path,
            index_path=index_path,
            evaluation_fingerprint=evaluation_fingerprint,
            preprocessing_fingerprint=preprocessing_fingerprint,
            split_fingerprint=split_fingerprint,
            slide_ids=slide_ids,
        )
        if complete is not None:
            manifest, index = complete
            return _result(
                manifest_path=manifest_path,
                index_path=index_path,
                manifest_payload=manifest,
                index_payload=index,
                evaluation_fingerprint=evaluation_fingerprint,
                preprocessing_fingerprint=preprocessing_fingerprint,
                split_fingerprint=split_fingerprint,
                slide_ids=slide_ids,
                built=0,
                reused=len(slide_ids),
                cache_result="reused",
                requested_mode=mode,
            )

        started_at = datetime.now(UTC)
        base_manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": _ARTIFACT_TYPE,
            "fingerprint": evaluation_fingerprint,
            "scope": AUTHORIZED_TEST_SCOPE,
            "preprocessing_fingerprint": preprocessing_fingerprint,
            "split_fingerprint": split_fingerprint,
            "test_slides": list(slide_ids),
            "slides": len(slide_ids),
            "index": _INDEX_NAME,
            "started_at": started_at.isoformat(),
            "training_dependency": False,
            "existing_development_slide_artifacts_modified": False,
            "development_index_modified": False,
        }
        atomic_json_dump(manifest_path, {**base_manifest, "status": "incomplete"})
        try:
            entries, missing = _adopt_entries(store, slide_ids)
            built = 0
            reused = len(entries)
            if missing:
                if builder is None:
                    raise PreprocessingError(
                        "authorized test artifacts are missing and no builder was supplied: "
                        f"{list(missing[:8])}"
                    )
                results = tuple(builder(missing))
                result_ids = tuple(result.slide_id for result in results)
                if len(result_ids) != len(set(result_ids)) or set(result_ids) != set(missing):
                    raise PreprocessingError(
                        "test-artifact builder must return exactly one result for each requested "
                        f"slide; requested={list(missing[:8])}, returned={list(result_ids[:8])}"
                    )
                built = sum(not result.reused for result in results)
                reused += sum(result.reused for result in results)
                built_entries, still_missing = _adopt_entries(store, missing)
                if still_missing:
                    raise PreprocessingError(
                        "test-artifact builder did not publish complete slide artifacts: "
                        f"{list(still_missing[:8])}"
                    )
                entries.update(built_entries)
            if built + reused != len(slide_ids):
                raise PreprocessingError("test-artifact preparation accounting is inconsistent")

            index = _index_payload(
                evaluation_fingerprint=evaluation_fingerprint,
                preprocessing_fingerprint=preprocessing_fingerprint,
                split_fingerprint=split_fingerprint,
                slide_ids=slide_ids,
                entries=entries,
            )
            atomic_json_dump(index_path, index)
            completed_at = datetime.now(UTC)
            complete_manifest = {
                **base_manifest,
                "status": "complete",
                "completed_at": completed_at.isoformat(),
                "index_fingerprint": index["fingerprint"],
                "built": built,
                "reused": reused,
            }
            # Completion is published last.  Valid complete manifests are
            # immutable and are returned before this write path is entered.
            atomic_json_dump(manifest_path, complete_manifest)
        except BaseException as exc:
            atomic_json_dump(
                manifest_path,
                {
                    **base_manifest,
                    "status": "failed",
                    "failed_at": datetime.now(UTC).isoformat(),
                    "error": repr(exc),
                },
            )
            raise

        return _result(
            manifest_path=manifest_path,
            index_path=index_path,
            manifest_payload=complete_manifest,
            index_payload=index,
            evaluation_fingerprint=evaluation_fingerprint,
            preprocessing_fingerprint=preprocessing_fingerprint,
            split_fingerprint=split_fingerprint,
            slide_ids=slide_ids,
            built=built,
            reused=reused,
            cache_result="prepared",
            requested_mode=mode,
        )


__all__ = [
    "AUTHORIZED_TEST_SCOPE",
    "TestArtifactBuilder",
    "TestArtifactMode",
    "ensure_test_artifacts",
    "load_test_artifact_entries",
    "test_artifact_fingerprint",
]
