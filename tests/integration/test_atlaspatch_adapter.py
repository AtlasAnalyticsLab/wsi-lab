from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest
from PIL import Image

from wsi_lab.data.base import EmbeddingLocation
from wsi_lab.data.embeddings.hdf5 import AtlasPatchHDF5Reader
from wsi_lab.preprocessing.atlaspatch import (
    ATLASPATCH_MODEL_FILENAME,
    ATLASPATCH_MODEL_REPOSITORY,
    ATLASPATCH_SAM_CONFIG_FILENAME,
    PINNED_SAM2_COMMIT,
    AtlasPatchOptions,
    AtlasPatchRunner,
    ensure_mpp_override_csv,
    resolve_atlaspatch_checkpoint_identity,
)


def test_atlaspatch_hdf5_adapter_preserves_requested_source_row_order(tmp_path: Path) -> None:
    path = tmp_path / "slide.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "coords", data=np.asarray([[0, 0, 0, 0, 0], [512, 0, 0, 0, 0]], dtype=np.int32)
        )
        handle.create_dataset("passports", data=np.asarray([b"a", b"b"], dtype="S1"))
        handle.create_dataset(
            "features/uni_v2", data=np.asarray([[1, 2], [3, 4]], dtype=np.float32)
        )
        handle.attrs.update(
            patch_size=256,
            patch_size_level0=512,
            level0_width=1024,
            level0_height=512,
            level0_magnification=40.0,
            target_magnification=20.0,
            overlap=0,
            mpp=0.25,
        )
    location = EmbeddingLocation(
        slide_id="slide",
        magnification=20,
        path=path,
        encoder="uni_v2",
        feature_key="features/uni_v2",
    )

    batch = AtlasPatchHDF5Reader().read(location, [1, 0, 1])

    assert batch.source_indices.tolist() == [1, 0, 1]
    assert batch.features.tolist() == [[3, 4], [1, 2], [3, 4]]
    assert batch.coordinates[:, :2].tolist() == [[512, 0], [0, 0], [512, 0]]


def test_detect_tissue_uses_exact_argv_and_records_provenance(tmp_path: Path, monkeypatch) -> None:
    slide = tmp_path / "slide with spaces.tif"
    slide.write_bytes(b"wsi")
    mpp_csv = ensure_mpp_override_csv(tmp_path / "mpp.csv", slide, 0.243)
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        del kwargs
        calls.append(list(argv))
        if argv[1:] == ["--version"]:
            return subprocess.CompletedProcess(argv, 0, stdout="atlaspatch 1.1.0\n", stderr="")
        output = Path(argv[argv.index("--output") + 1])
        output.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray([[0, 255], [255, 0]], dtype=np.uint8)).save(
            output / "slide_tissue.png"
        )
        return subprocess.CompletedProcess(argv, 0, stdout="generated\n", stderr="diagnostic\n")

    monkeypatch.setattr(
        "wsi_lab.preprocessing.atlaspatch.shutil.which", lambda _: "/bin/atlaspatch"
    )
    monkeypatch.setattr("wsi_lab.preprocessing.atlaspatch.subprocess.run", fake_run)
    monkeypatch.setattr(
        "wsi_lab.preprocessing.atlaspatch.resolve_atlaspatch_checkpoint_identity",
        lambda: {
            "repository": ATLASPATCH_MODEL_REPOSITORY,
            "filename": ATLASPATCH_MODEL_FILENAME,
            "requested_revision": "main",
            "resolved_snapshot": "snapshot123",
            "cache_path": "/cache/model.pth",
            "size": 10,
            "sha256": "checkpoint-sha256",
        },
    )
    monkeypatch.setattr(
        "wsi_lab.preprocessing.atlaspatch.installed_sam2_provenance",
        lambda: {
            "distribution": "SAM-2",
            "version": "1.0",
            "pinned_commit": PINNED_SAM2_COMMIT,
            "requested_revision": PINNED_SAM2_COMMIT,
            "resolved_commit": PINNED_SAM2_COMMIT,
        },
    )
    monkeypatch.setattr(
        "wsi_lab.preprocessing.atlaspatch.resolve_atlaspatch_sam_config_identity",
        lambda: {
            "filename": ATLASPATCH_SAM_CONFIG_FILENAME,
            "installed_path": "/site-packages/atlas_patch/configs/sam2.1_hiera_t.yaml",
            "size": 20,
            "sha256": "c" * 64,
        },
    )
    options = AtlasPatchOptions(
        executable="atlaspatch",
        device="cuda:3",
        segmentation_batch_size=7,
        mpp_csv=mpp_csv,
    )

    result = AtlasPatchRunner(options).run(slide, tmp_path / "output", mpp=(0.243, 0.243))

    assert calls[0] == [
        "atlaspatch",
        "detect-tissue",
        str(slide.resolve()),
        "--output",
        str((tmp_path / "output").resolve()),
        "--device",
        "cuda:3",
        "--seg-batch-size",
        "7",
        "--mpp-csv",
        str(mpp_csv),
    ]
    if len(calls) > 1:
        assert calls[1] == ["atlaspatch", "--version"]
    assert result.provenance["runner"] == "argv_subprocess"
    model = result.provenance["segmentation_model"]
    assert model["repository"] == "AtlasAnalyticsLab/AtlasPatch"
    assert model["filename"] == "model.pth"
    assert model["resolved_snapshot"] == "snapshot123"
    assert model["checkpoint"]["sha256"] == "checkpoint-sha256"
    assert model["sam2"]["pinned_commit"] == PINNED_SAM2_COMMIT
    assert model["sam2_config"]["filename"] == ATLASPATCH_SAM_CONFIG_FILENAME
    assert model["sam2_config"]["sha256"] == "c" * 64
    assert "generated" in result.provenance["stdout"]


def test_mpp_csv_is_content_stable(tmp_path: Path) -> None:
    slide = tmp_path / "slide.tif"
    slide.write_bytes(b"x")
    path = ensure_mpp_override_csv(tmp_path / "mpp" / "slide.csv", slide, 0.25)
    first_mtime = path.stat().st_mtime_ns
    again = ensure_mpp_override_csv(path, slide, 0.25)

    assert again == path
    assert path.read_text(encoding="utf-8").splitlines()[0] == "wsi,mpp"
    assert path.stat().st_mtime_ns == first_mtime


def test_checkpoint_identity_records_huggingface_snapshot_and_content(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint = (
        tmp_path
        / "models--AtlasAnalyticsLab--AtlasPatch"
        / "snapshots"
        / "resolved123"
        / "model.pth"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"atlaspatch-checkpoint")
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda **_: str(checkpoint))
    identity = resolve_atlaspatch_checkpoint_identity()

    assert identity["repository"] == ATLASPATCH_MODEL_REPOSITORY
    assert identity["filename"] == ATLASPATCH_MODEL_FILENAME
    assert identity["requested_revision"] == "main"
    assert identity["resolved_snapshot"] == "resolved123"
    assert identity["size"] == len(b"atlaspatch-checkpoint")
    assert identity["sha256"] == hashlib.sha256(b"atlaspatch-checkpoint").hexdigest()


def test_atlaspatch_prefers_binary_mask_over_rgb_visualization(tmp_path: Path) -> None:
    output = tmp_path / "atlas-output" / "visualization"
    output.mkdir(parents=True)
    binary = output / "slide_mask_bw.png"
    overlay = output / "slide_mask.png"
    Image.fromarray(np.asarray([[0, 255]], dtype=np.uint8)).save(binary)
    Image.new("RGB", (2, 1), "red").save(overlay)

    assert AtlasPatchRunner._discover_mask(tmp_path / "atlas-output") == binary


@pytest.mark.real
@pytest.mark.gpu
@pytest.mark.atlaspatch
@pytest.mark.slow
def test_real_atlaspatch_detect_tissue_single_slide_in_temporary_workspace(
    tmp_path: Path, camelyon16_root: Path
) -> None:
    """Explicit opt-in integration run; never writes into CAMELYON16."""

    if os.environ.get("WSI_LAB_RUN_REAL_ATLASPATCH") != "1":
        pytest.skip("set WSI_LAB_RUN_REAL_ATLASPATCH=1 for the bounded real AtlasPatch run")
    from wsi_lab.data.camelyon16 import Camelyon16Adapter
    from wsi_lab.data.embeddings.hdf5 import HDF5EmbeddingProvider

    adapter = Camelyon16Adapter(camelyon16_root)
    slide_id = "normal_031"
    record = adapter.get_slide(slide_id)
    metadata = HDF5EmbeddingProvider(adapter).metadata(slide_id, 2.5)
    assert metadata.mpp is not None
    mpp_csv = ensure_mpp_override_csv(
        tmp_path / "mpp" / f"{slide_id}.csv",
        record.slide_path,
        metadata.mpp,
    )
    result = AtlasPatchRunner(
        AtlasPatchOptions(
            executable="atlaspatch",
            device="cuda:0",
            segmentation_batch_size=1,
            timeout_seconds=7200,
            mpp_csv=mpp_csv,
        )
    ).run(record.slide_path, tmp_path / "atlaspatch-output", mpp=(metadata.mpp, metadata.mpp))

    assert result.returncode == 0
    assert result.mask_path.is_file()
    assert result.mask_path.is_relative_to(tmp_path)
    assert result.atlaspatch_version is not None
    assert result.provenance["raw_mask"]["sha256"]
    model = result.provenance["segmentation_model"]
    assert model["repository"] == ATLASPATCH_MODEL_REPOSITORY
    assert model["filename"] == ATLASPATCH_MODEL_FILENAME
    assert model["resolved_snapshot"]
    assert model["checkpoint"]["sha256"]
    assert model["sam2"]["pinned_commit"] == PINNED_SAM2_COMMIT
    assert model["sam2_config"]["filename"] == ATLASPATCH_SAM_CONFIG_FILENAME
    assert model["sam2_config"]["sha256"]
    assert str(camelyon16_root.resolve()) not in str(result.mask_path)
