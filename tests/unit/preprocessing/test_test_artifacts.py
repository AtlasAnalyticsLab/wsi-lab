from __future__ import annotations

import json
import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

import wsi_lab.preprocessing.test_artifacts as test_artifacts
from wsi_lab.errors import ArtifactCompatibilityError, PreprocessingError
from wsi_lab.preprocessing.artifacts import ArtifactStore, advisory_lock
from wsi_lab.preprocessing.pipeline import SlidePreprocessingResult
from wsi_lab.utils.fingerprints import fingerprint


class MemoryMetadataStore(ArtifactStore):
    def __init__(self, root: Path, metadata: dict[str, dict[str, Any]] | None = None) -> None:
        super().__init__(root)
        self.metadata = dict(metadata or {})
        self.load_calls: list[str] = []
        self.fail_on_load = False
        self._metadata_lock = threading.Lock()

    def load_metadata(
        self,
        slide_id: str,
        *,
        expected_input_fingerprint: str | None = None,
    ) -> dict[str, Any]:
        del expected_input_fingerprint
        with self._metadata_lock:
            self.load_calls.append(slide_id)
            if self.fail_on_load:
                raise AssertionError("warm manifest path must not inspect slide metadata")
            try:
                return dict(self.metadata[slide_id])
            except KeyError as exc:
                raise ArtifactCompatibilityError(f"missing {slide_id}") from exc

    def publish(self, slide_id: str, *, slide_index: int) -> None:
        with self._metadata_lock:
            self.metadata[slide_id] = _metadata(slide_id, slide_index)


def _metadata(slide_id: str, slide_index: int) -> dict[str, Any]:
    return {
        "slide_id": slide_id,
        "slide_index": slide_index,
        "input_fingerprint": f"input-{slide_id}",
        "cell_id_offset": slide_index * 1_000,
        "num_cells": 17,
    }


def _probe_process_lock(lock_path: str, outcomes: Any) -> None:
    """Spawn-safe helper that first probes the held lock without blocking."""

    import fcntl
    import os

    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            outcomes.put("blocked")
        else:
            outcomes.put("acquired_without_blocking")
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)
    with advisory_lock(lock_path):
        outcomes.put("acquired_after_release")


def _result(store: ArtifactStore, slide_id: str, slide_index: int, *, reused: bool = False) -> Any:
    return SlidePreprocessingResult(
        slide_id=slide_id,
        artifact_path=store.slide_paths(slide_id).root,
        input_fingerprint=f"input-{slide_id}",
        reused=reused,
        cell_id_offset=slide_index * 1_000,
        num_cells=17,
        start_cell_id=slide_index * 1_000,
        mapping_summaries={},
    )


def _ensure(
    store: MemoryMetadataStore,
    *,
    mode: test_artifacts.TestArtifactMode = "auto",
    split: str = "split-fingerprint",
    builder: test_artifacts.TestArtifactBuilder | None = None,
) -> dict[str, Any]:
    return test_artifacts.ensure_test_artifacts(
        store=store,
        preprocessing_fingerprint="preprocessing-fingerprint",
        split_fingerprint=split,
        test_slide_ids=("test_a", "test_b"),
        mode=mode,
        builder=builder,
    )


def test_fingerprint_preserves_established_payload() -> None:
    expected = fingerprint(
        {
            "scope": "authorized-official-test-evaluation",
            "preprocessing": "preprocessing-fingerprint",
            "split": "split-fingerprint",
            "test_slides": ("test_a", "test_b"),
        }
    )

    actual = test_artifacts.test_artifact_fingerprint(
        preprocessing_fingerprint="preprocessing-fingerprint",
        split_fingerprint="split-fingerprint",
        test_slide_ids=("test_a", "test_b"),
    )

    assert actual == expected


def test_advisory_lock_serializes_independent_processes(tmp_path: Path) -> None:
    pytest.importorskip("fcntl")
    context = multiprocessing.get_context("spawn")
    outcomes = context.Queue()
    lock_path = tmp_path / "shared.lock"
    process = context.Process(
        target=_probe_process_lock,
        args=(str(lock_path), outcomes),
    )

    try:
        with advisory_lock(lock_path):
            process.start()
            assert outcomes.get(timeout=5) == "blocked"
        assert outcomes.get(timeout=5) == "acquired_after_release"
    finally:
        if process.pid is not None:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        outcomes.close()
        outcomes.join_thread()

    assert process.exitcode == 0


def test_prepare_adopts_complete_slides_builds_only_missing_and_warm_path_never_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = MemoryMetadataStore(tmp_path / "processed", {"test_a": _metadata("test_a", 4)})
    requested: list[tuple[str, ...]] = []

    def build(slide_ids: tuple[str, ...]) -> tuple[SlidePreprocessingResult, ...]:
        requested.append(slide_ids)
        store.publish("test_b", slide_index=9)
        return (_result(store, "test_b", 9),)

    prepared = _ensure(store, mode="prepare", builder=build)

    assert requested == [("test_b",)]
    assert prepared["status"] == "complete"
    assert prepared["cache_result"] == "prepared"
    assert prepared["requested_mode"] == "prepare"
    assert prepared["built"] == 1
    assert prepared["reused"] == 1
    assert prepared["slides"] == 2
    assert prepared["training_dependency"] is False
    assert prepared["existing_development_slide_artifacts_modified"] is False
    assert prepared["development_index_modified"] is False
    assert prepared["development_slide_artifacts_modified"] is False
    assert prepared["shared_index_extended"] is False
    assert prepared["started_at"]
    assert prepared["completed_at"]
    assert not store.index_path.exists()
    manifest_path = Path(prepared["manifest"])
    index_path = Path(prepared["index"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["index_fingerprint"] == index["fingerprint"]
    assert index["test_slides"] == ["test_a", "test_b"]
    assert [item["slide_id"] for item in index["slides"]] == ["test_a", "test_b"]
    assert all(item["metadata_fingerprint"] for item in index["slides"])
    loaded_entries = test_artifacts.load_test_artifact_entries(
        index_path,
        expected_fingerprint=prepared["index_fingerprint"],
    )
    assert tuple(loaded_entries) == ("test_a", "test_b")
    assert index["fingerprint"] == fingerprint(
        {key: value for key, value in index.items() if key != "fingerprint"}
    )

    before_manifest = manifest_path.read_bytes()
    before_index = index_path.read_bytes()
    store.fail_on_load = True

    def unexpected_write(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("warm manifest path must not write")

    monkeypatch.setattr(test_artifacts, "atomic_json_dump", unexpected_write)
    reused = _ensure(
        store,
        builder=lambda _slide_ids: (_ for _ in ()).throw(AssertionError("must not build")),
    )

    assert reused["cache_result"] == "reused"
    assert reused["requested_mode"] == "auto"
    assert reused["built"] == 0
    assert reused["reused"] == 2
    assert manifest_path.read_bytes() == before_manifest
    assert index_path.read_bytes() == before_index

    required = _ensure(store, mode="require")

    assert required["cache_result"] == "reused"
    assert required["requested_mode"] == "require"
    assert manifest_path.read_bytes() == before_manifest
    assert index_path.read_bytes() == before_index


def test_require_is_strictly_read_only_when_artifacts_are_absent(tmp_path: Path) -> None:
    store = MemoryMetadataStore(tmp_path / "processed")

    with pytest.raises(PreprocessingError, match="prepare-test"):
        _ensure(store, mode="require")

    assert not store.root.exists()


@pytest.mark.parametrize("mode", ["auto", "prepare", "require"])
def test_all_modes_reject_incompatible_complete_index_without_modifying_it(
    tmp_path: Path, mode: test_artifacts.TestArtifactMode
) -> None:
    store = MemoryMetadataStore(
        tmp_path / "processed",
        {"test_a": _metadata("test_a", 4), "test_b": _metadata("test_b", 9)},
    )
    prepared = _ensure(store)
    manifest_path = Path(prepared["manifest"])
    index_path = Path(prepared["index"])
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["slides"].reverse()
    index_path.write_text(json.dumps(index), encoding="utf-8")
    before_manifest = manifest_path.read_bytes()
    before_index = index_path.read_bytes()

    with pytest.raises(PreprocessingError, match="index is incompatible"):
        _ensure(store, mode=mode)

    assert manifest_path.read_bytes() == before_manifest
    assert index_path.read_bytes() == before_index


def test_failed_preparation_is_published_and_can_resume(tmp_path: Path) -> None:
    store = MemoryMetadataStore(tmp_path / "processed")

    def fail(_slide_ids: tuple[str, ...]) -> tuple[SlidePreprocessingResult, ...]:
        raise RuntimeError("builder failed")

    with pytest.raises(RuntimeError, match="builder failed"):
        _ensure(store, builder=fail)

    evaluation_fingerprint = test_artifacts.test_artifact_fingerprint(
        preprocessing_fingerprint="preprocessing-fingerprint",
        split_fingerprint="split-fingerprint",
        test_slide_ids=("test_a", "test_b"),
    )
    manifest_path = store.root / "test_evaluations" / evaluation_fingerprint / "manifest.json"
    failed = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert "builder failed" in failed["error"]

    def build(slide_ids: tuple[str, ...]) -> tuple[SlidePreprocessingResult, ...]:
        assert slide_ids == ("test_a", "test_b")
        store.publish("test_a", slide_index=4)
        store.publish("test_b", slide_index=9)
        return (_result(store, "test_a", 4), _result(store, "test_b", 9))

    prepared = _ensure(store, builder=build)

    assert prepared["cache_result"] == "prepared"
    assert prepared["built"] == 2
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["status"] == "complete"


def test_complete_manifest_is_immutable_and_skips_later_failing_builder(tmp_path: Path) -> None:
    store = MemoryMetadataStore(
        tmp_path / "processed",
        {"test_a": _metadata("test_a", 4), "test_b": _metadata("test_b", 9)},
    )
    prepared = _ensure(store)
    manifest_path = Path(prepared["manifest"])
    before = manifest_path.read_bytes()

    def fail(_slide_ids: tuple[str, ...]) -> tuple[SlidePreprocessingResult, ...]:
        raise AssertionError("completed preparation must not invoke a builder")

    result = _ensure(store, mode="prepare", builder=fail)

    assert result["cache_result"] == "reused"
    assert manifest_path.read_bytes() == before


def test_concurrent_callers_build_once_and_waiter_rechecks_after_lock(tmp_path: Path) -> None:
    store = MemoryMetadataStore(tmp_path / "processed")
    builder_entered = threading.Event()
    release_builder = threading.Event()
    callback_lock = threading.Lock()
    callback_count = 0

    def build(slide_ids: tuple[str, ...]) -> tuple[SlidePreprocessingResult, ...]:
        nonlocal callback_count
        with callback_lock:
            callback_count += 1
        assert slide_ids == ("test_a", "test_b")
        builder_entered.set()
        assert release_builder.wait(timeout=2)
        store.publish("test_a", slide_index=4)
        store.publish("test_b", slide_index=9)
        return (_result(store, "test_a", 4), _result(store, "test_b", 9))

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(_ensure, store, builder=build)
        assert builder_entered.wait(timeout=2)
        second = executor.submit(_ensure, store, builder=build)
        release_builder.set()
        results = (first.result(timeout=3), second.result(timeout=3))

    assert callback_count == 1
    assert {result["cache_result"] for result in results} == {"prepared", "reused"}


def test_different_split_fingerprints_have_isolated_manifests(tmp_path: Path) -> None:
    store = MemoryMetadataStore(
        tmp_path / "processed",
        {"test_a": _metadata("test_a", 4), "test_b": _metadata("test_b", 9)},
    )

    first = _ensure(store, split="split-one")
    second = _ensure(store, split="split-two")

    assert first["fingerprint"] != second["fingerprint"]
    assert Path(first["manifest"]).parent != Path(second["manifest"]).parent
    assert first["cache_result"] == second["cache_result"] == "prepared"


@pytest.mark.parametrize(
    "slide_ids, match",
    [
        (("test_a", "test_a"), "must be unique"),
        (("../test_a",), "unsafe"),
        ((), "requires test slides"),
    ],
)
def test_invalid_slide_id_sets_fail_before_writing(
    tmp_path: Path, slide_ids: tuple[str, ...], match: str
) -> None:
    store = MemoryMetadataStore(tmp_path / "processed")

    with pytest.raises(PreprocessingError, match=match):
        test_artifacts.ensure_test_artifacts(
            store=store,
            preprocessing_fingerprint="preprocessing-fingerprint",
            split_fingerprint="split-fingerprint",
            test_slide_ids=slide_ids,
        )

    assert not store.root.exists()


def test_unknown_mode_fails_before_writing(tmp_path: Path) -> None:
    store = MemoryMetadataStore(tmp_path / "processed")

    with pytest.raises(ValueError, match="unknown test-artifact mode"):
        test_artifacts.ensure_test_artifacts(
            store=store,
            preprocessing_fingerprint="preprocessing-fingerprint",
            split_fingerprint="split-fingerprint",
            test_slide_ids=("test_a",),
            mode="invalid",  # type: ignore[arg-type]
        )

    assert not store.root.exists()
