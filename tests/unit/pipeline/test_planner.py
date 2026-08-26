from __future__ import annotations

from pathlib import Path

import torch

from rl4xs_wsi.config.models import AppConfig, DatasetConfig, WorkspaceConfig
from rl4xs_wsi.pipeline.planner import artifact_paths, build_plan, stage_fingerprints
from rl4xs_wsi.pipeline.stages import Stage, StageAction
from rl4xs_wsi.rl.checkpoints import save_ppo_checkpoint
from rl4xs_wsi.rl.recurrent_ppo import PPOConfig
from rl4xs_wsi.utils.atomic import atomic_json_dump


def test_dry_plan_is_non_mutating(tmp_path: Path) -> None:
    workspace = tmp_path / "does-not-exist"
    config = AppConfig(
        dataset=DatasetConfig(root=tmp_path / "dataset"),
        workspace=WorkspaceConfig(root=workspace),
    )
    plan = build_plan(config)
    assert not workspace.exists()
    assert plan.stages[0].action is StageAction.BUILD
    assert plan.stages[-1].action is StageAction.TRAIN


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        dataset=DatasetConfig(root=tmp_path / "dataset"),
        workspace=WorkspaceConfig(root=tmp_path / "workspace"),
    )


def test_complete_manifest_skips_only_when_required_output_exists(tmp_path: Path) -> None:
    config = _config(tmp_path)
    fingerprints = stage_fingerprints(config)
    manifest = artifact_paths(config, fingerprints)[Stage.SPLIT]
    manifest.parent.mkdir(parents=True)
    atomic_json_dump(
        manifest,
        {"status": "complete", "fingerprint": fingerprints[Stage.SPLIT]},
    )
    split = manifest.parent / "camelyon16_split.json"
    split.write_text("{}")
    assert build_plan(config, (Stage.SPLIT,)).stages[0].action is StageAction.SKIP
    split.unlink()
    stale = build_plan(config, (Stage.SPLIT,)).stages[0]
    assert stale.action is StageAction.REBUILD
    assert "required outputs" in stale.reason


def test_incomplete_training_checkpoint_resumes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    fingerprints = stage_fingerprints(config)
    manifest = artifact_paths(config, fingerprints)[Stage.BELIEF]
    manifest.parent.mkdir(parents=True)
    atomic_json_dump(
        manifest,
        {"status": "incomplete", "fingerprint": fingerprints[Stage.BELIEF]},
    )
    (manifest.parent / "checkpoint.pt").write_bytes(b"checkpoint")
    decision = build_plan(config, (Stage.BELIEF,)).stages[0]
    assert decision.action is StageAction.RESUME


def test_complete_preprocessing_manifest_requires_nonempty_slide_artifacts(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    fingerprints = stage_fingerprints(config)
    manifest = artifact_paths(config, fingerprints)[Stage.PREPROCESSING]
    manifest.parent.mkdir(parents=True)
    atomic_json_dump(
        manifest,
        {"status": "complete", "fingerprint": fingerprints[Stage.PREPROCESSING]},
    )
    (manifest.parent / "slides").mkdir()
    (manifest.parent / "slides.json").write_text("{}")
    (manifest.parent / "cell_offsets.json").write_text("{}")

    empty = build_plan(config, (Stage.PREPROCESSING,)).stages[0]
    assert empty.action is StageAction.REBUILD
    assert "no complete processed-slide" in empty.reason

    slide = manifest.parent / "slides" / "normal_001"
    slide.mkdir()
    (slide / "metadata.json").write_text("{}")
    assert build_plan(config, (Stage.PREPROCESSING,)).stages[0].action is StageAction.SKIP


def _write_ppo_manifest(config: AppConfig, status: str) -> Path:
    fingerprints = stage_fingerprints(config)
    manifest = artifact_paths(config, fingerprints)[Stage.PPO]
    manifest.parent.mkdir(parents=True)
    atomic_json_dump(
        manifest,
        {"status": status, "fingerprint": fingerprints[Stage.PPO]},
    )
    return manifest


def _write_valid_ppo_checkpoint(path: Path, *, global_step: int = 8) -> Path:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    return save_ppo_checkpoint(
        path,
        model,
        optimizer,
        PPOConfig(device="cpu"),
        global_step=global_step,
        update=1,
    )


def test_complete_ppo_manifest_requires_valid_complete_checkpoint_payload(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    manifest = _write_ppo_manifest(config, "complete")
    checkpoint_directory = manifest.parent / "checkpoints"
    checkpoint_directory.mkdir()

    missing = build_plan(config, (Stage.PPO,)).stages[0]
    assert missing.action is StageAction.TRAIN
    assert "no valid complete PPO checkpoint candidates" in missing.reason

    corrupt = checkpoint_directory / "checkpoint_000000000008.pt"
    corrupt.write_bytes(b"not a torch checkpoint")
    invalid = build_plan(config, (Stage.PPO,)).stages[0]
    assert invalid.action is StageAction.TRAIN
    assert "no valid complete PPO checkpoint candidates" in invalid.reason

    _write_valid_ppo_checkpoint(corrupt)
    complete = build_plan(config, (Stage.PPO,)).stages[0]
    assert complete.action is StageAction.SKIP


def test_incomplete_ppo_manifest_resumes_from_nested_checkpoint(tmp_path: Path) -> None:
    config = _config(tmp_path)
    manifest = _write_ppo_manifest(config, "incomplete")
    checkpoint_directory = manifest.parent / "checkpoints"
    checkpoint_directory.mkdir()
    checkpoint = checkpoint_directory / "checkpoint_000000000008.pt"
    _write_valid_ppo_checkpoint(checkpoint)

    decision = build_plan(config, (Stage.PPO,)).stages[0]

    assert decision.action is StageAction.RESUME
    assert checkpoint.name in decision.reason


def test_incomplete_ppo_manifest_does_not_resume_empty_or_legacy_checkpoint(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    manifest = _write_ppo_manifest(config, "incomplete")
    checkpoint_directory = manifest.parent / "checkpoints"
    checkpoint_directory.mkdir()
    (checkpoint_directory / "checkpoint_000000000008.pt").touch()
    (manifest.parent / "checkpoint.pt").write_bytes(b"legacy")

    decision = build_plan(config, (Stage.PPO,)).stages[0]

    assert decision.action is StageAction.TRAIN


def test_incomplete_ppo_manifest_does_not_resume_corrupt_checkpoint(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    manifest = _write_ppo_manifest(config, "incomplete")
    checkpoint_directory = manifest.parent / "checkpoints"
    checkpoint_directory.mkdir()
    (checkpoint_directory / "checkpoint_000000000008.pt").write_bytes(b"corrupt")

    decision = build_plan(config, (Stage.PPO,)).stages[0]

    assert decision.action is StageAction.TRAIN


def test_ppo_checkpoint_does_not_override_manifest_fingerprint_mismatch(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    manifest = _write_ppo_manifest(config, "incomplete")
    atomic_json_dump(
        manifest,
        {"status": "incomplete", "fingerprint": "stale-fingerprint"},
    )
    checkpoint_directory = manifest.parent / "checkpoints"
    checkpoint_directory.mkdir()
    (checkpoint_directory / "checkpoint_000000000008.pt").write_bytes(b"checkpoint")

    decision = build_plan(config, (Stage.PPO,)).stages[0]

    assert decision.action is StageAction.TRAIN
    assert "fingerprint changed" in decision.reason
