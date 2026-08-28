from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from wsi_lab.errors import PreprocessingError
from wsi_lab.preprocessing.atlaspatch import (
    ATLASPATCH_MODEL_FILENAME,
    ATLASPATCH_MODEL_REPOSITORY,
    ATLASPATCH_SAM_CONFIG_FILENAME,
    PINNED_SAM2_COMMIT,
    AtlasPatchOptions,
    AtlasPatchResult,
)
from wsi_lab.preprocessing.tissue import load_or_create_tissue_mask
from wsi_lab.wsi.reader import OverviewTransform


def _overview(transform: OverviewTransform) -> np.ndarray:
    return np.full((transform.canvas_height, transform.canvas_width, 3), 255, dtype=np.uint8)


def test_existing_binary_mask_is_read_only_and_letterboxed(tmp_path: Path) -> None:
    slide = tmp_path / "slide.png"
    Image.new("RGB", (80, 40), "white").save(slide)
    source = tmp_path / "slide_tissue.tif"
    values = np.zeros((20, 40), dtype=np.uint8)
    values[:, :20] = 255
    tifffile.imwrite(source, values)
    before = source.read_bytes()
    transform = OverviewTransform.fit((80, 40), (32, 32))

    mask = load_or_create_tissue_mask(
        slide,
        _overview(transform),
        transform,
        existing_path=source,
        source="existing",
    )

    assert mask.values.shape == (32, 32)
    assert not mask.values[: transform.content_y0].any()
    assert mask.source == "existing"
    assert source.read_bytes() == before


def test_existing_mask_is_normalized_into_workspace_cache(tmp_path: Path) -> None:
    slide = tmp_path / "slide.png"
    Image.new("RGB", (80, 40), "white").save(slide)
    source = tmp_path / "slide_tissue.png"
    native = np.zeros((20, 40), dtype=np.uint8)
    native[5:15, 10:30] = 255
    Image.fromarray(native).save(source)
    source_before = source.read_bytes()
    transform = OverviewTransform.fit((80, 40), (32, 32))
    cached = tmp_path / "workspace" / "masks" / "slide.npy"

    mask = load_or_create_tissue_mask(
        slide,
        _overview(transform),
        transform,
        existing_path=source,
        generated_path=cached,
        source="existing",
    )

    cached_values = np.load(cached, allow_pickle=False)
    manifest = json.loads(cached.with_name("slide.npy.manifest.json").read_text(encoding="utf-8"))
    assert cached_values.dtype == np.bool_
    assert cached_values.shape == (32, 32)
    assert np.array_equal(cached_values, mask.values)
    assert not cached_values[: transform.content_y0].any()
    assert mask.source == "existing"
    assert mask.source_path == source.resolve()
    assert source.read_bytes() == source_before
    assert manifest["status"] == "complete"
    assert manifest["cache_kind"] == "normalized-existing"
    assert manifest["source_path"] == str(source.resolve())
    assert manifest["normalized_mask"]["dtype"] == "bool"


def test_existing_multiclass_mask_is_rejected_not_thresholded(tmp_path: Path) -> None:
    slide = tmp_path / "slide.png"
    Image.new("RGB", (12, 12), "white").save(slide)
    source = tmp_path / "mask.png"
    Image.fromarray(np.asarray([[0, 1, 2]] * 12, dtype=np.uint8)).save(source)
    transform = OverviewTransform.fit((12, 12), (12, 12))

    with pytest.raises(PreprocessingError, match="not binary"):
        load_or_create_tissue_mask(
            slide,
            _overview(transform),
            transform,
            existing_path=source,
            source="existing",
        )


def test_existing_anisotropic_mask_is_rejected(tmp_path: Path) -> None:
    slide = tmp_path / "slide.png"
    Image.new("RGB", (80, 40), "white").save(slide)
    source = tmp_path / "mask.png"
    Image.fromarray(np.zeros((10, 40), dtype=np.uint8)).save(source)
    transform = OverviewTransform.fit((80, 40), (32, 32))

    with pytest.raises(PreprocessingError, match="incompatible"):
        load_or_create_tissue_mask(
            slide,
            _overview(transform),
            transform,
            existing_path=source,
            source="existing",
        )


def test_generated_mask_is_atomically_cached_with_provenance(tmp_path: Path, monkeypatch) -> None:
    slide = tmp_path / "slide.png"
    Image.new("RGB", (20, 10), "white").save(slide)
    transform = OverviewTransform.fit((20, 10), (16, 16))
    generated = tmp_path / "workspace" / "slide.npy"

    class FakeRunner:
        options = AtlasPatchOptions(executable="not-invoked")

        def __init__(self) -> None:
            self.calls = 0
            self.snapshot = "test-snapshot"
            self.checkpoint_sha256 = "a" * 64
            self.started = threading.Event()
            self.release = threading.Event()

        def run(self, slide_path, output_dir, *, mpp=None):
            del slide_path, mpp
            self.calls += 1
            self.started.set()
            assert self.release.wait(timeout=2)
            raw = Path(output_dir) / "slide_tissue.png"
            values = np.zeros((10, 20), dtype=np.uint8)
            values[:, :10] = 255
            Image.fromarray(values).save(raw)
            return AtlasPatchResult(
                mask_path=raw,
                command=("atlaspatch", "detect-tissue"),
                reproduction_command="atlaspatch detect-tissue",
                returncode=0,
                stdout="ok",
                stderr="",
                atlaspatch_version="1.1.0",
                provenance={
                    "test_provenance": True,
                    "segmentation_model": {
                        "implementation": "sam2",
                        "repository": ATLASPATCH_MODEL_REPOSITORY,
                        "filename": ATLASPATCH_MODEL_FILENAME,
                        "requested_revision": "main",
                        "resolved_snapshot": self.snapshot,
                        "checkpoint": {
                            "repository": ATLASPATCH_MODEL_REPOSITORY,
                            "filename": ATLASPATCH_MODEL_FILENAME,
                            "requested_revision": "main",
                            "resolved_snapshot": self.snapshot,
                            "cache_path": "/cache/model.pth",
                            "size": 10,
                            "sha256": self.checkpoint_sha256,
                        },
                        "sam2": {"pinned_commit": PINNED_SAM2_COMMIT},
                        "sam2_config": {
                            "filename": ATLASPATCH_SAM_CONFIG_FILENAME,
                            "installed_path": "/site-packages/atlas_patch/configs/sam2.1_hiera_t.yaml",
                            "size": 20,
                            "sha256": "b" * 64,
                        },
                    },
                },
            )

    runner = FakeRunner()
    monkeypatch.setattr(
        "wsi_lab.preprocessing.tissue.resolve_atlaspatch_checkpoint_identity",
        lambda: {
            "repository": ATLASPATCH_MODEL_REPOSITORY,
            "filename": ATLASPATCH_MODEL_FILENAME,
            "requested_revision": "main",
            "resolved_snapshot": runner.snapshot,
            "cache_path": "/cache/model.pth",
            "size": 10,
            "sha256": runner.checkpoint_sha256,
        },
    )
    monkeypatch.setattr(
        "wsi_lab.preprocessing.tissue.installed_sam2_provenance",
        lambda: {"pinned_commit": PINNED_SAM2_COMMIT},
    )
    monkeypatch.setattr(
        "wsi_lab.preprocessing.tissue.resolve_atlaspatch_sam_config_identity",
        lambda: {
            "filename": ATLASPATCH_SAM_CONFIG_FILENAME,
            "installed_path": "/site-packages/atlas_patch/configs/sam2.1_hiera_t.yaml",
            "size": 20,
            "sha256": "b" * 64,
        },
    )

    def load_mask():
        return load_or_create_tissue_mask(
            slide,
            _overview(transform),
            transform,
            generated_path=generated,
            source="atlaspatch",
            atlaspatch_runner=runner,  # type: ignore[arg-type]
            mpp=(0.25, 0.25),
            concurrent_writes=True,
        )

    second_started = threading.Event()
    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(load_mask)
        assert runner.started.wait(timeout=2)

        def load_second():
            second_started.set()
            return load_mask()

        second_future = executor.submit(load_second)
        assert second_started.wait(timeout=2)
        time.sleep(0.05)
        runner.release.set()
        first = first_future.result(timeout=2)
        second = second_future.result(timeout=2)

    assert {first.source, second.source} == {"atlaspatch", "atlaspatch-cache"}
    assert np.array_equal(first.values, second.values)
    assert runner.calls == 1
    assert generated.is_file()
    manifest = generated.with_name("slide.npy.manifest.json")
    assert manifest.is_file()
    text = manifest.read_text(encoding="utf-8")
    assert '"resolved_snapshot": "test-snapshot"' in text
    assert '"sam2_commit": "2b90b9f5ceec907a1c18123530e92e794ad901a4"' in text
    assert f'"sam2_config_filename": "{ATLASPATCH_SAM_CONFIG_FILENAME}"' in text
    assert (
        '"sam2_config_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"'
        in text
    )

    runner.snapshot = "updated-snapshot"
    runner.checkpoint_sha256 = "d" * 64
    rebuilt = load_or_create_tissue_mask(
        slide,
        _overview(transform),
        transform,
        generated_path=generated,
        source="atlaspatch",
        atlaspatch_runner=runner,  # type: ignore[arg-type]
        mpp=(0.25, 0.25),
    )

    assert rebuilt.source == "atlaspatch"
    assert runner.calls == 2
    assert '"resolved_snapshot": "updated-snapshot"' in manifest.read_text(encoding="utf-8")
