from __future__ import annotations

from pathlib import Path

import pytest
import torch

from wsi_lab.config.models import AppConfig, DatasetConfig, WorkspaceConfig
from wsi_lab.pipeline import planner
from wsi_lab.pipeline.planner import artifact_paths, build_plan, stage_fingerprints
from wsi_lab.pipeline.stages import Stage, StageAction
from wsi_lab.rl.checkpoints import save_ppo_checkpoint
from wsi_lab.rl.recurrent_ppo import PPOConfig
from wsi_lab.utils.atomic import atomic_json_dump


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


def test_training_reporting_contract_invalidates_only_ppo_algorithm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    current = stage_fingerprints(config)

    monkeypatch.setitem(planner._ALGORITHM_VERSION, Stage.PPO, 2)
    legacy = stage_fingerprints(config)

    assert current[Stage.PPO] != legacy[Stage.PPO]
    for stage in Stage:
        if stage is not Stage.PPO:
            assert current[stage] == legacy[stage]


@pytest.mark.parametrize("field", ("geometric_augmentation", "photometric_augmentation"))
def test_runtime_augmentation_invalidates_only_ppo_fingerprint(tmp_path: Path, field: str) -> None:
    config = _config(tmp_path)
    before = stage_fingerprints(config)

    setattr(config.environment, field, True)
    after = stage_fingerprints(config)

    assert before[Stage.PPO] != after[Stage.PPO]
    for stage in Stage:
        if stage is not Stage.PPO:
            assert before[stage] == after[stage]


def test_tracking_configuration_does_not_change_artifact_fingerprints(tmp_path: Path) -> None:
    config = _config(tmp_path)
    before = stage_fingerprints(config)

    config.logging.tensorboard = False
    config.logging.wandb = True
    config.logging.wandb_project = "another-project"
    config.logging.wandb_entity = "research-team"
    config.logging.wandb_mode = "offline"
    config.logging.wandb_tags = ("ablation",)

    assert stage_fingerprints(config) == before


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


def test_generic_tissue_sidecar_changes_only_tissue_dependent_fingerprints(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "generic"
    (dataset_root / "images").mkdir(parents=True)
    (dataset_root / "masks").mkdir()
    (dataset_root / "images" / "case-001.svs").touch()
    tissue_mask = dataset_root / "masks" / "case-001.tif"
    tissue_mask.write_bytes(b"first-mask")
    (dataset_root / "metadata.csv").write_text(
        "slide_id,image_path,label,label_name,patient_id,center,official_cohort,"
        "tissue_mask_path\n"
        "case-001,images/case-001.svs,0,negative,patient-001,site-a,development,"
        "masks/case-001.tif\n",
        encoding="utf-8",
    )
    config = AppConfig(
        dataset=DatasetConfig(name="generic", root=dataset_root),
        workspace=WorkspaceConfig(root=tmp_path / "workspace"),
    )

    before = stage_fingerprints(config)
    tissue_mask.write_bytes(b"second-mask-with-a-different-size")
    after = stage_fingerprints(config)

    assert before[Stage.DATASET] == after[Stage.DATASET]
    assert before[Stage.BELIEF_DATA] == after[Stage.BELIEF_DATA]
    assert before[Stage.SPLIT] == after[Stage.SPLIT]
    assert before[Stage.BELIEF] == after[Stage.BELIEF]
    assert before[Stage.TISSUE] != after[Stage.TISSUE]
    assert before[Stage.PREPROCESSING] != after[Stage.PREPROCESSING]
    assert before[Stage.PPO] != after[Stage.PPO]


def test_tcga_native_inventory_change_invalidates_persisted_split(tmp_path: Path) -> None:
    dataset_root = tmp_path / "tcga-nsclc"
    for cohort, slide_id in (
        ("TCGA-LUAD", "TCGA-AA-0001-01Z-00-DX1.first"),
        ("TCGA-LUSC", "TCGA-BB-0002-01Z-00-DX1.second"),
    ):
        images = dataset_root / cohort / "images"
        images.mkdir(parents=True)
        (images / f"{slide_id}.svs").touch()
    config = AppConfig(
        dataset=DatasetConfig(name="tcga_nsclc", root=dataset_root),
        workspace=WorkspaceConfig(root=tmp_path / "workspace"),
    )

    before = stage_fingerprints(config)
    added = dataset_root / "TCGA-LUAD" / "images" / "TCGA-CC-0003-01Z-00-DX1.third.svs"
    added.touch()
    after = stage_fingerprints(config)

    assert before[Stage.SPLIT] != after[Stage.SPLIT]


def _combined_camelyon_config(tmp_path: Path) -> tuple[AppConfig, Path, Path]:
    dataset_root = tmp_path / "datasets"
    camelyon16 = dataset_root / "CAMELYON16"
    camelyon17 = dataset_root / "CAMELYON17"
    (camelyon16 / "evaluation").mkdir(parents=True)
    (camelyon16 / "images").mkdir()
    reference = camelyon16 / "evaluation/reference.csv"
    reference.write_text(
        "image,type,class,center\nnormal_001.tif,normal,negative,0\n",
        encoding="utf-8",
    )
    (camelyon16 / "images/normal_001.tif").touch()
    (camelyon17 / "images").mkdir(parents=True)
    stages = camelyon17 / "stages.csv"
    stages.write_text(
        "patient,stage,center\npatient_000_node_0.tif,negative,0\n",
        encoding="utf-8",
    )
    (camelyon17 / "images/patient_000_node_0.tif").touch()
    return (
        AppConfig(
            dataset=DatasetConfig(name="camelyon16_17", root=dataset_root),
            workspace=WorkspaceConfig(root=tmp_path / "workspace"),
        ),
        reference,
        stages,
    )


@pytest.mark.parametrize("metadata_index", (1, 2))
def test_combined_metadata_tables_participate_in_fingerprints(
    tmp_path: Path, metadata_index: int
) -> None:
    config, reference, stages = _combined_camelyon_config(tmp_path)
    before = stage_fingerprints(config)
    metadata = (config, reference, stages)[metadata_index]
    metadata.write_text(
        metadata.read_text(encoding="utf-8").replace(",0\n", ",1\n"),
        encoding="utf-8",
    )
    after = stage_fingerprints(config)

    assert before[Stage.DATASET] != after[Stage.DATASET]
    assert before[Stage.BELIEF_DATA] != after[Stage.BELIEF_DATA]
    assert before[Stage.SPLIT] != after[Stage.SPLIT]


def test_combined_only_config_fields_do_not_invalidate_existing_camelyon17(
    tmp_path: Path,
) -> None:
    root = tmp_path / "CAMELYON17"
    (root / "images").mkdir(parents=True)
    (root / "images/patient_000_node_0.tif").touch()
    (root / "stages.csv").write_text(
        "patient,stage,center\npatient_000_node_0.tif,negative,0\n",
        encoding="utf-8",
    )
    config = AppConfig(
        dataset=DatasetConfig(name="camelyon17", root=root),
        workspace=WorkspaceConfig(root=tmp_path / "workspace"),
    )
    before = stage_fingerprints(config)

    config.dataset.camelyon16_root = tmp_path / "irrelevant-camelyon16"
    config.dataset.camelyon17_root = tmp_path / "irrelevant-camelyon17"
    config.dataset.camelyon17_labels = tmp_path / "irrelevant-stages.csv"
    after = stage_fingerprints(config)

    assert before == after
