"""Command-line entry point for the shared idempotent pipeline."""

from __future__ import annotations

import argparse
import functools
import json
import logging
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from wsi_lab.config import AppConfig, load_config
from wsi_lab.data.base import DatasetAdapter
from wsi_lab.data.factory import create_dataset_adapter
from wsi_lab.data.splits import (
    DatasetSplit,
    SplitName,
    create_dataset_split,
    validate_dataset_split,
)
from wsi_lab.errors import PreprocessingError, TestSplitAccessError, WSILabError
from wsi_lab.pipeline import PipelineRunner, Stage, StageAction
from wsi_lab.pipeline.planner import artifact_paths, stage_fingerprints
from wsi_lab.utils.atomic import atomic_json_dump, atomic_write_text
from wsi_lab.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def _adapter(config: AppConfig) -> DatasetAdapter:
    return create_dataset_adapter(config.dataset, config.embeddings)


def _split_path(config: AppConfig) -> Path:
    return config.workspace.root / "splits" / f"{config.dataset.name}_split.json"


def _dataset_stage(
    config: AppConfig, artifact_dir: Path, fingerprint_value: str, action: StageAction
) -> dict[str, object]:
    del action
    from wsi_lab.data.embeddings.hdf5 import AtlasPatchHDF5Reader
    from wsi_lab.data.validation import DatasetValidator, ValidationOptions, report_has_errors
    from wsi_lab.wsi.reader import OpenSlideReader

    adapter = _adapter(config)
    report_path = artifact_dir / "embedding_validation.json"
    reader = OpenSlideReader(handle_cache_size=config.environment.slide_handle_cache_size)
    try:
        report = DatasetValidator(
            adapter,
            reader=AtlasPatchHDF5Reader(),
            slide_inspector=reader,
            options=ValidationOptions(
                feature_chunk_rows=config.embeddings.validation_chunk_rows,
                mapping_tolerance=config.embeddings.mapping_tolerance,
                check_feature_finiteness=True,
                check_wsi_readability=True,
            ),
        ).validate(report_path=report_path, raise_on_error=True)
    finally:
        reader.close()
    warnings = sum(issue.severity == "warning" for issue in report.issues)
    return {
        "input_fingerprint": fingerprint_value,
        "validation_report": str(report_path),
        "checked_files": report.checked_files,
        "feature_dimension": report.feature_dimension,
        "warnings": warnings,
        "has_errors": report_has_errors(report),
    }


def _belief_data_stage(
    config: AppConfig, artifact_dir: Path, fingerprint_value: str, action: StageAction
) -> dict[str, object]:
    """Validate train/validation embeddings without opening guarded test data or WSIs."""

    del action
    from wsi_lab.data.embeddings.hdf5 import AtlasPatchHDF5Reader
    from wsi_lab.data.validation import DatasetValidator, ValidationOptions, report_has_errors

    adapter = _adapter(config)
    split = _load_verified_split(config)
    development_ids = (*split.ids("train"), *split.ids("val"))
    report_path = artifact_dir / "belief_embedding_validation.json"
    report = DatasetValidator(
        adapter,
        reader=AtlasPatchHDF5Reader(),
        slide_inspector=None,
        options=ValidationOptions(
            feature_chunk_rows=config.embeddings.validation_chunk_rows,
            mapping_tolerance=config.embeddings.mapping_tolerance,
            check_feature_finiteness=True,
            check_wsi_existence=False,
            check_wsi_readability=False,
        ),
    ).validate(
        report_path=report_path,
        raise_on_error=True,
        resume=True,
        slide_ids=development_ids,
    )
    warnings = sum(issue.severity == "warning" for issue in report.issues)
    return {
        "input_fingerprint": fingerprint_value,
        "validation_report": str(report_path),
        "checked_files": report.checked_files,
        "development_slides": len(development_ids),
        "feature_dimension": report.feature_dimension,
        "warnings": warnings,
        "has_errors": report_has_errors(report),
        "official_test_opened": False,
    }


def _split_stage(
    config: AppConfig, artifact_dir: Path, fingerprint_value: str, action: StageAction
) -> dict[str, object]:
    del action
    adapter = _adapter(config)
    split = create_dataset_split(
        adapter.iter_slides(),
        dataset=adapter.name,
        validation_fraction=config.split.validation_fraction,
        test_fraction=config.split.test_fraction,
        seed=config.split.seed,
        stratify_by_center=config.split.stratify_by_center,
        prefer_patient_level=True,
    )
    split_path = artifact_dir / f"{adapter.name}_split.json"
    split.save(split_path)
    split_names: tuple[SplitName, ...] = ("train", "val", "test")
    return {
        "input_fingerprint": fingerprint_value,
        "split_path": str(split_path),
        "split_fingerprint": split.split_fingerprint,
        "strategy": split.strategy,
        "counts": {name: len(split.ids(name)) for name in split_names},
        "unassigned": len(split.unassigned),
        "limitation": (
            "No reliable patient IDs are available; consecutive-section leakage cannot be guaranteed"
            if not any(record.patient_id for record in adapter.iter_slides())
            else None
        ),
    }


def _load_verified_split(config: AppConfig) -> DatasetSplit:
    split = DatasetSplit.load(_split_path(config))
    validate_dataset_split(split, expected_records=tuple(_adapter(config).iter_slides()))
    return split


def _belief_expected_dependencies(config: AppConfig, embedding_dimension: int) -> dict[str, str]:
    fingerprints = stage_fingerprints(config)
    split = _load_verified_split(config)
    return {
        "belief_stage": fingerprints[Stage.BELIEF],
        "split": split.split_fingerprint,
        "embedding_dimension": str(embedding_dimension),
        "embedding_encoder": config.embeddings.encoder,
    }


def _active_embedding_dimension(config: AppConfig) -> int:
    from wsi_lab.data.embeddings.hdf5 import AtlasPatchHDF5Reader

    split = _load_verified_split(config)
    train_ids = split.ids("train")
    if not train_ids:
        raise ValueError("persisted training split is empty")
    adapter = _adapter(config)
    reader = AtlasPatchHDF5Reader()
    dimensions = {
        reader.inspect(adapter.embedding_location(train_ids[0], magnification)).feature_dim
        for magnification in adapter.magnifications
    }
    if len(dimensions) != 1:
        raise ValueError(f"active embedding scales have mixed dimensions: {sorted(dimensions)}")
    return dimensions.pop()


def _load_active_belief(config: AppConfig, *, device: str):
    from wsi_lab.belief.trainer import load_calibrated_belief

    embedding_dimension = _active_embedding_dimension(config)
    fingerprints = stage_fingerprints(config)
    paths = artifact_paths(config, fingerprints)
    checkpoint = paths[Stage.BELIEF].parent / "calibrated_checkpoint.pt"
    return load_calibrated_belief(
        checkpoint,
        expected_dependencies=_belief_expected_dependencies(config, embedding_dimension),
        device=device,
    )


def _preprocessing_store(config: AppConfig):
    from wsi_lab.preprocessing.artifacts import ArtifactStore

    fingerprints = stage_fingerprints(config)
    return ArtifactStore(artifact_paths(config, fingerprints)[Stage.PREPROCESSING].parent)


def _make_production_environment(
    config: AppConfig,
    *,
    split_name: SplitName,
    evaluation: bool,
    allow_test: bool = False,
    render_mode: str | None = None,
):
    from wsi_lab.belief.accumulator import BeliefAccumulator
    from wsi_lab.envs.wsi_env import WSIEnv
    from wsi_lab.preprocessing.pipeline import ProcessedEpisodeSource

    _ensure_test_access(split_name, allow_test)
    split = _load_verified_split(config)
    source = ProcessedEpisodeSource.from_config(
        config,
        _preprocessing_store(config),
        split,
        allow_test=allow_test,
    )
    model, temperature, _ = _load_active_belief(config, device=config.rl.belief_device)
    belief = BeliefAccumulator(model, temperature=temperature)
    return WSIEnv(
        source,
        belief,
        split=split_name,
        evaluation=evaluation,
        overview_size=config.environment.overview_size,
        viewport_size=config.environment.viewport_size,
        start_position=config.environment.start_position,
        geometric_augmentation=(
            config.environment.geometric_augmentation if not evaluation else False
        ),
        photometric_augmentation=(
            config.environment.photometric_augmentation if not evaluation else False
        ),
        max_episode_steps=config.environment.max_episode_steps,
        min_diagnostic_evidence=config.environment.min_diagnostic_evidence,
        max_visit_count=config.environment.max_visit_count,
        reward_type=config.environment.reward_type,
        step_cost=config.environment.step_cost,
        belief_reward_beta=config.environment.belief_reward_beta,
        render_mode=render_mode,
    )


def _preprocessing_stage(
    config: AppConfig, artifact_dir: Path, fingerprint_value: str, action: StageAction
) -> dict[str, object]:
    from wsi_lab.data.embeddings.hdf5 import HDF5EmbeddingProvider
    from wsi_lab.preprocessing.artifacts import ArtifactStore
    from wsi_lab.preprocessing.pipeline import PreprocessingOptions, Preprocessor
    from wsi_lab.wsi.reader import OpenSlideReader

    fingerprints = stage_fingerprints(config)
    paths = artifact_paths(config, fingerprints)
    adapter = _adapter(config)
    reader = OpenSlideReader(
        handle_cache_size=config.environment.slide_handle_cache_size,
        viewport_cache_size=config.environment.viewport_cache_size,
        persistent_cache_dir=config.environment.persistent_rgb_cache,
    )
    processor = Preprocessor(
        adapter,
        HDF5EmbeddingProvider(adapter),
        reader,
        ArtifactStore(artifact_dir),
        options=PreprocessingOptions.from_config(
            config,
            tissue_cache_root=paths[Stage.TISSUE].parent,
        ),
    )
    started = time.monotonic()
    split = _load_verified_split(config)
    development_ids = (*split.ids("train"), *split.ids("val"))
    try:
        results = processor.preprocess(
            development_ids,
            overwrite=action is StageAction.REBUILD,
        )
    finally:
        reader.close()
    return {
        "input_fingerprint": fingerprint_value,
        "slides": len(results),
        "scope": "development-only",
        "official_test_opened": False,
        "built": sum(not result.reused for result in results),
        "reused": sum(result.reused for result in results),
        "duration_seconds": time.monotonic() - started,
        "index": str(artifact_dir / "slides.json"),
    }


def _tissue_stage(
    config: AppConfig, artifact_dir: Path, fingerprint_value: str, action: StageAction
) -> dict[str, object]:
    from wsi_lab.data.embeddings.hdf5 import HDF5EmbeddingProvider
    from wsi_lab.preprocessing.artifacts import ArtifactStore
    from wsi_lab.preprocessing.pipeline import PreprocessingOptions, Preprocessor
    from wsi_lab.wsi.reader import OpenSlideReader

    adapter = _adapter(config)
    reader = OpenSlideReader(
        handle_cache_size=config.environment.slide_handle_cache_size,
        viewport_cache_size=0,
    )
    processor = Preprocessor(
        adapter,
        HDF5EmbeddingProvider(adapter),
        reader,
        ArtifactStore(artifact_dir / "_context"),
        options=PreprocessingOptions.from_config(config, tissue_cache_root=artifact_dir),
    )
    split = _load_verified_split(config)
    development_ids = (*split.ids("train"), *split.ids("val"))
    entries: list[dict[str, object]] = []
    started = time.monotonic()
    try:
        for slide_id in development_ids:
            tissue = processor.prepare_tissue_mask(slide_id)
            entries.append(
                {
                    "slide_id": slide_id,
                    "source": tissue.source,
                    "source_path": str(tissue.source_path) if tissue.source_path else None,
                    "source_page": tissue.source_page,
                    "shape": list(tissue.values.shape),
                    "tissue_pixels": int(tissue.values.sum()),
                    "provenance": dict(tissue.provenance),
                }
            )
    finally:
        reader.close()
    index = artifact_dir / "index.json"
    atomic_json_dump(
        index,
        {
            "status": "complete",
            "fingerprint": fingerprint_value,
            "scope": "development-only",
            "official_test_opened": False,
            "slides": entries,
        },
    )
    return {
        "input_fingerprint": fingerprint_value,
        "index": str(index),
        "slides": len(entries),
        "existing": sum(entry["source"] == "existing" for entry in entries),
        "atlaspatch": sum(str(entry["source"]).startswith("atlaspatch") for entry in entries),
        "duration_seconds": time.monotonic() - started,
        "scope": "development-only",
        "official_test_opened": False,
        "action": action.value,
    }


def _belief_records(config: AppConfig, split_name: SplitName):
    from wsi_lab.belief.providers import HDF5FeatureMatrix
    from wsi_lab.belief.sampler import SlideEmbeddingRecord
    from wsi_lab.data.embeddings.hdf5 import AtlasPatchHDF5Reader

    split = _load_verified_split(config)
    adapter = _adapter(config)
    reader = AtlasPatchHDF5Reader()
    records = []
    observed_dimensions: set[int] = set()
    for slide_id in split.ids(split_name):
        slide = adapter.get_slide(slide_id)
        if slide.label not in {0, 1}:
            raise ValueError(f"split {split_name!r} contains unlabeled slide {slide_id!r}")
        matrices = {}
        for magnification in adapter.magnifications:
            location = adapter.embedding_location(slide_id, magnification)
            metadata = reader.inspect(location)
            observed_dimensions.add(metadata.feature_dim)
            matrices[magnification] = HDF5FeatureMatrix(
                location.path,
                location.feature_key,
                shape=(metadata.num_instances, metadata.feature_dim),
            )
        records.append(
            SlideEmbeddingRecord(
                slide_id=slide_id,
                label=slide.label,
                patient_id=slide.patient_id,
                features_by_magnification=matrices,
                metadata={
                    "center": slide.center,
                    "tumor_class": slide.tumor_class,
                    "split": "validation" if split_name == "val" else split_name,
                    "official_cohort": slide.official_cohort,
                },
            )
        )
    if len(observed_dimensions) != 1:
        raise ValueError(f"Mixed belief feature dimensions: {sorted(observed_dimensions)}")
    return tuple(records), observed_dimensions.pop()


def _numeric_metrics(prefix: str, value: Any):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _numeric_metrics(f"{prefix}/{key}" if prefix else str(key), item)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, float(value)


def _belief_stage(
    config: AppConfig, artifact_dir: Path, fingerprint_value: str, action: StageAction
) -> dict[str, object]:
    from torch.utils.tensorboard import SummaryWriter

    from wsi_lab.belief.model import GatedAttentionMIL
    from wsi_lab.belief.trainer import BeliefTrainer, build_belief_loaders

    train_records, train_dimension = _belief_records(config, "train")
    validation_records, validation_dimension = _belief_records(config, "val")
    if train_dimension != validation_dimension:
        raise ValueError(
            f"Training dimension {train_dimension} != validation dimension {validation_dimension}"
        )
    split = _load_verified_split(config)
    model = GatedAttentionMIL.from_config(train_dimension, config.belief)
    atomic_write_text(
        artifact_dir / "config.yaml",
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True),
    )
    loaders = build_belief_loaders(train_records, validation_records, config.belief)
    tensorboard = (
        SummaryWriter(log_dir=str(artifact_dir / "tensorboard"))
        if config.logging.tensorboard
        else None
    )
    wandb_run = None
    if config.logging.wandb:
        import wandb

        wandb_run = wandb.init(
            project=config.logging.wandb_project,
            config=config.model_dump(mode="json"),
            dir=str(artifact_dir),
            resume="allow",
        )

    def record(event: str, payload: Mapping[str, Any]) -> None:
        step = int(payload.get("global_step", payload.get("epoch", 0)))
        metrics = dict(_numeric_metrics(event, payload))
        if tensorboard is not None:
            for name, value in metrics.items():
                tensorboard.add_scalar(name, value, step)
            tensorboard.flush()
        if wandb_run is not None:
            wandb_run.log(metrics, step=step)
        LOGGER.info("%s %s", event, json.dumps(payload, default=str))

    trainer = BeliefTrainer(
        model,
        config.belief,
        artifact_dir,
        dependencies={
            "belief_stage": fingerprint_value,
            "split": split.split_fingerprint,
            "embedding_dimension": str(train_dimension),
            "embedding_encoder": config.embeddings.encoder,
        },
        metric_callback=record,
        manifest_name="training_state.json",
    )
    started = time.monotonic()
    try:
        result = trainer.fit(
            loaders.train,
            loaders.validation,
            resume=action is StageAction.RESUME or config.belief.resume,
        )
    finally:
        if tensorboard is not None:
            tensorboard.close()
        if wandb_run is not None:
            wandb_run.finish()
        for record_item in (*train_records, *validation_records):
            for matrix in record_item.features_by_magnification.values():
                close = getattr(matrix, "close", None)
                if callable(close):
                    close()
    return {
        "training_duration_seconds": time.monotonic() - started,
        "best_epoch": result.best_epoch,
        "epochs_completed": result.epochs_completed,
        "global_step": result.global_step,
        "stopped_early": result.stopped_early,
        "temperature": result.temperature,
        "checkpoint": str(result.checkpoint_path),
        "calibrated_checkpoint": str(result.calibrated_checkpoint_path),
        "validation": result.validation.to_dict(),
        "dependencies": trainer.dependencies,
        "architecture": model.architecture_config(),
    }


def _ppo_dependencies(config: AppConfig) -> dict[str, str]:
    fingerprints = stage_fingerprints(config)
    return {
        "dataset_fingerprint": fingerprints[Stage.DATASET],
        "preprocessing_fingerprint": fingerprints[Stage.PREPROCESSING],
        "belief_fingerprint": fingerprints[Stage.BELIEF],
        "split_fingerprint": _load_verified_split(config).split_fingerprint,
    }


def _ppo_stage(
    config: AppConfig, artifact_dir: Path, fingerprint_value: str, action: StageAction
) -> dict[str, object]:
    from torch.utils.tensorboard import SummaryWriter

    from wsi_lab.rl.best_checkpoint import BestValidationCheckpointTracker
    from wsi_lab.rl.evaluation import evaluate_policy
    from wsi_lab.rl.networks import RecurrentActorCritic
    from wsi_lab.rl.trainer import PPOTrainer, make_vector_env

    factories = [
        functools.partial(
            _make_production_environment,
            config,
            split_name="train",
            evaluation=False,
            allow_test=False,
        )
        for _ in range(config.rl.num_envs)
    ]
    vector = make_vector_env(factories, backend=config.rl.vector_backend.value)
    writer = (
        SummaryWriter(log_dir=str(artifact_dir / "tensorboard"))
        if config.logging.tensorboard
        else None
    )
    model = RecurrentActorCritic()
    dependencies = _ppo_dependencies(config)
    checkpoint_dir = artifact_dir / "checkpoints"
    best_tracker = BestValidationCheckpointTracker(
        checkpoint_dir,
        resume_existing=action is StageAction.RESUME,
    )
    trainer = PPOTrainer(
        vector,
        model,
        config.rl,
        writer=writer,
        dependencies=dependencies,
    )
    atomic_write_text(
        artifact_dir / "config.yaml",
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True),
    )

    def validate(current: PPOTrainer) -> None:
        environment = _make_production_environment(
            config,
            split_name="val",
            evaluation=True,
            allow_test=False,
        )
        try:
            evaluation = evaluate_policy(
                current.model,
                environment,
                episodes=len(_load_verified_split(config).ids("val")),
                device=current.device,
                seed=config.rl.seed,
            )
        finally:
            environment.close()
        decision = best_tracker.consider(
            evaluation.metrics,
            model=current.model,
            optimizer=current.optimizer,
            config=current.config,
            global_step=current.global_step,
            update=current.update_number,
            dependencies=current.dependencies,
        )
        validation_result = evaluation.to_dict()
        validation_result["checkpoint_tracking"] = decision.to_dict()
        atomic_json_dump(
            artifact_dir / f"validation_{current.global_step:012d}.json",
            validation_result,
        )
        if writer is not None:
            for name, value in evaluation.metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    writer.add_scalar(f"validation/{name}", value, current.global_step)
            writer.add_scalar(
                "validation/is_best_checkpoint",
                float(decision.improved),
                current.global_step,
            )
            writer.add_scalar(
                "validation/best_selection_value",
                decision.best.metric_value,
                current.global_step,
            )

    started = time.monotonic()
    try:
        summary = trainer.train(
            checkpoint_dir=checkpoint_dir,
            resume_latest=action is StageAction.RESUME,
            validation_callback=validate,
        )
    finally:
        trainer.close()
        if writer is not None:
            writer.close()
    return {
        "input_fingerprint": fingerprint_value,
        "duration_seconds": time.monotonic() - started,
        "global_step": summary.global_step,
        "updates": summary.updates,
        "completed_episodes": summary.completed_episodes,
        "dependencies": dependencies,
        "last_metrics": summary.last_metrics.to_dict()
        if summary.last_metrics is not None
        else None,
        "checkpoint_directory": str(checkpoint_dir),
        "best_validation": best_tracker.best.to_dict() if best_tracker.best is not None else None,
    }


def _configured_runner(config: AppConfig) -> PipelineRunner:
    runner = PipelineRunner(config)
    runner.register(Stage.DATASET, _dataset_stage)
    runner.register(Stage.BELIEF_DATA, _belief_data_stage)
    runner.register(Stage.SPLIT, _split_stage)
    runner.register(Stage.TISSUE, _tissue_stage)
    runner.register(Stage.PREPROCESSING, _preprocessing_stage)
    runner.register(Stage.BELIEF, _belief_stage)
    runner.register(Stage.PPO, _ppo_stage)
    return runner


def _ensure_test_access(split: str, allow_test: bool) -> None:
    if split == "test" and not allow_test:
        raise TestSplitAccessError("Test split access requires --split test --allow-test")


def _prepare_authorized_test_preprocessing(config: AppConfig) -> dict[str, Any]:
    """Build guarded-test navigation artifacts for an authorized evaluation.

    The normal preprocessing stage owns only development artifacts.  This
    separate, resumable manifest is reached solely by an explicit test request
    with ``--allow-test`` and requires development preprocessing to be complete.
    Its output is never a dependency of belief or PPO training.
    """

    from wsi_lab.data.embeddings.hdf5 import HDF5EmbeddingProvider
    from wsi_lab.preprocessing.artifacts import ArtifactStore
    from wsi_lab.preprocessing.pipeline import PreprocessingOptions, Preprocessor
    from wsi_lab.utils.fingerprints import fingerprint
    from wsi_lab.wsi.reader import OpenSlideReader

    fingerprints = stage_fingerprints(config)
    paths = artifact_paths(config, fingerprints)
    development_manifest_path = paths[Stage.PREPROCESSING]
    try:
        development_manifest = json.loads(development_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreprocessingError(
            "authorized test evaluation requires completed development preprocessing; "
            f"run `wsi-lab preprocess --config ...` first ({exc})"
        ) from exc
    metadata = development_manifest.get("metadata", {})
    if (
        development_manifest.get("status") != "complete"
        or development_manifest.get("fingerprint") != fingerprints[Stage.PREPROCESSING]
        or not isinstance(metadata, Mapping)
        or metadata.get("scope") != "development-only"
        or metadata.get("official_test_opened") is not False
    ):
        raise PreprocessingError(
            "authorized test evaluation requires a compatible, development-only "
            "preprocessing manifest"
        )

    adapter = _adapter(config)
    records = tuple(adapter.iter_slides())
    split = _load_verified_split(config)
    development_ids = (*split.ids("train"), *split.ids("val"))
    test_ids_in_adapter_order = tuple(record.slide_id for record in records)
    authorized_ids = set(split.ids("test"))
    test_ids = tuple(
        slide_id for slide_id in test_ids_in_adapter_order if slide_id in authorized_ids
    )
    if len(test_ids) != len(authorized_ids):
        missing = sorted(authorized_ids.difference(test_ids))
        raise PreprocessingError(f"persisted test split contains unknown slides: {missing[:8]}")

    store = ArtifactStore(development_manifest_path.parent)
    offsets = store.load_offsets()
    missing_development = set(development_ids).difference(offsets)
    if missing_development:
        raise PreprocessingError(
            "development preprocessing offset table is incomplete; missing slides: "
            f"{sorted(missing_development)[:8]}"
        )
    evaluation_fingerprint = fingerprint(
        {
            "scope": "authorized-official-test-evaluation",
            "preprocessing": fingerprints[Stage.PREPROCESSING],
            "split": split.split_fingerprint,
            "test_slides": test_ids,
        }
    )
    manifest_path = store.root / "test_evaluation_preprocessing.json"
    started_at = datetime.now(UTC)
    atomic_json_dump(
        manifest_path,
        {
            "status": "incomplete",
            "fingerprint": evaluation_fingerprint,
            "scope": "authorized-official-test-evaluation",
            "started_at": started_at.isoformat(),
            "slides": list(test_ids),
        },
    )

    reader = OpenSlideReader(
        handle_cache_size=config.environment.slide_handle_cache_size,
        viewport_cache_size=config.environment.viewport_cache_size,
        persistent_cache_dir=config.environment.persistent_rgb_cache,
    )
    processor = Preprocessor(
        adapter,
        HDF5EmbeddingProvider(adapter),
        reader,
        store,
        options=PreprocessingOptions.from_config(
            config,
            tissue_cache_root=paths[Stage.TISSUE].parent,
        ),
    )
    try:
        results = processor.preprocess(test_ids, merge_index=True)
    except BaseException as exc:
        atomic_json_dump(
            manifest_path,
            {
                "status": "failed",
                "fingerprint": evaluation_fingerprint,
                "scope": "authorized-official-test-evaluation",
                "started_at": started_at.isoformat(),
                "failed_at": datetime.now(UTC).isoformat(),
                "slides": list(test_ids),
                "error": repr(exc),
            },
        )
        raise
    finally:
        reader.close()
    completed_at = datetime.now(UTC)
    value: dict[str, Any] = {
        "status": "complete",
        "fingerprint": evaluation_fingerprint,
        "scope": "authorized-official-test-evaluation",
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "slides": len(results),
        "built": sum(not result.reused for result in results),
        "reused": sum(result.reused for result in results),
        "development_slide_artifacts_modified": False,
        "shared_index_extended": True,
        "training_dependency": False,
    }
    atomic_json_dump(manifest_path, value)
    return {"manifest": str(manifest_path), **value}


def _evaluate_belief_command(
    config: AppConfig, split_name: SplitName, allow_test: bool
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    from wsi_lab.belief.dataset import ValidationBagDataset, collate_belief_bags
    from wsi_lab.belief.evaluation import evaluate_belief
    from wsi_lab.belief.trainer import load_calibrated_belief

    _ensure_test_access(split_name, allow_test)
    records, embedding_dimension = _belief_records(config, split_name)
    fingerprints = stage_fingerprints(config)
    paths = artifact_paths(config, fingerprints)
    checkpoint = paths[Stage.BELIEF].parent / "calibrated_checkpoint.pt"
    verified_split = _load_verified_split(config)
    model, temperature, _ = load_calibrated_belief(
        checkpoint,
        expected_dependencies={
            "belief_stage": fingerprints[Stage.BELIEF],
            "split": verified_split.split_fingerprint,
            "embedding_dimension": str(embedding_dimension),
            "embedding_encoder": config.embeddings.encoder,
        },
        device=config.belief.device,
    )
    dataset = ValidationBagDataset.from_config(records, config.belief)
    loader = DataLoader(
        dataset,
        batch_size=config.belief.batch_size,
        shuffle=False,
        num_workers=config.belief.num_workers,
        collate_fn=collate_belief_bags,
        pin_memory=str(config.belief.device).startswith("cuda"),
    )
    evaluation = evaluate_belief(
        model,
        loader,
        temperature=temperature,
        device=config.belief.device,
        ece_bins=config.belief.ece_bins,
    )
    output = paths[Stage.BELIEF].parent / f"evaluation_{split_name}.json"
    value = evaluation.to_dict(include_predictions=False)
    atomic_json_dump(output, value)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"output": str(output), **value}


def _load_ppo_model(config: AppConfig, checkpoint_path: str | Path | None = None):
    from wsi_lab.rl.checkpoints import checkpoint_candidates, load_ppo_checkpoint
    from wsi_lab.rl.networks import RecurrentActorCritic
    from wsi_lab.rl.recurrent_ppo import PPOConfig
    from wsi_lab.rl.trainer import RESUME_CONFIG_FIELDS, resolve_device

    fingerprints = stage_fingerprints(config)
    directory = artifact_paths(config, fingerprints)[Stage.PPO].parent / "checkpoints"
    candidates = (
        [Path(checkpoint_path).expanduser().resolve()]
        if checkpoint_path is not None
        else checkpoint_candidates(directory)
    )
    if not candidates:
        raise FileNotFoundError(f"no PPO checkpoint exists under {directory}")
    expected_config = PPOConfig.from_object(config.rl).to_dict()
    device = resolve_device(config.rl.device)
    failures: list[str] = []
    for candidate in candidates:
        model = RecurrentActorCritic()
        try:
            loaded = load_ppo_checkpoint(
                candidate,
                model,
                map_location=device,
                restore_rng=False,
                expected_config=expected_config,
                compatibility_fields=RESUME_CONFIG_FIELDS,
                expected_dependencies=_ppo_dependencies(config),
            )
        except (KeyError, RuntimeError, ValueError) as exc:
            failures.append(f"{candidate.name}: {exc}")
            continue
        return model, loaded
    raise ValueError("no compatible PPO checkpoint found: " + "; ".join(failures))


def _evaluate_rl_command(
    config: AppConfig,
    split_name: SplitName,
    allow_test: bool,
    *,
    policy_name: str = "ppo",
    fixed_budget: int = 128,
    confidence_threshold: float = 0.9,
    checkpoint_path: str | Path | None = None,
    checkpoint_selection: str = "latest",
) -> dict[str, Any]:
    _ensure_test_access(split_name, allow_test)
    from wsi_lab.rl.evaluation import evaluate_baseline, evaluate_policy

    test_preprocessing = (
        _prepare_authorized_test_preprocessing(config) if split_name == "test" else None
    )
    environment = _make_production_environment(
        config,
        split_name=split_name,
        evaluation=True,
        allow_test=allow_test,
    )
    split = _load_verified_split(config)
    checkpoint: str | None = None
    checkpoint_global_step: int | None = None
    checkpoint_dependencies: dict[str, str] = {}
    checkpoint_best_validation: dict[str, Any] | None = None
    try:
        if policy_name == "ppo":
            from wsi_lab.rl.trainer import resolve_device

            model, loaded = _load_ppo_model(config, checkpoint_path=checkpoint_path)
            checkpoint = str(loaded.path)
            checkpoint_global_step = loaded.global_step
            checkpoint_dependencies = dict(loaded.dependencies)
            embedded_best = loaded.extra.get("best_validation")
            if isinstance(embedded_best, Mapping):
                checkpoint_best_validation = dict(embedded_best)
            evaluation = evaluate_policy(
                model,
                environment,
                episodes=len(split.ids(split_name)),
                device=resolve_device(config.rl.device),
                seed=config.rl.seed,
            )
        else:
            from wsi_lab.baselines import (
                BaselinePolicy,
                ConfidenceThresholdPolicy,
                FixedBudgetPolicy,
                RandomPolicy,
                RasterPolicy,
            )

            policies: dict[str, Callable[[], BaselinePolicy]] = {
                "random": lambda: RandomPolicy(seed=config.rl.seed),
                "raster": RasterPolicy,
                "fixed-budget": lambda: FixedBudgetPolicy(
                    fixed_budget,
                    navigation_policy=RandomPolicy(seed=config.rl.seed),
                ),
                "confidence": lambda: ConfidenceThresholdPolicy(
                    confidence_threshold,
                    navigation_policy=RandomPolicy(seed=config.rl.seed),
                ),
            }
            try:
                baseline = policies[policy_name]()
            except KeyError as exc:
                raise ValueError(f"unknown RL evaluation policy: {policy_name}") from exc
            evaluation = evaluate_baseline(
                baseline,
                environment,
                episodes=len(split.ids(split_name)),
                seed=config.rl.seed,
            )
    finally:
        environment.close()
    fingerprints = stage_fingerprints(config)
    output = (
        artifact_paths(config, fingerprints)[Stage.PPO].parent
        / f"evaluation_{policy_name}_{split_name}.json"
    )
    value = evaluation.to_dict()
    if policy_name == "ppo":
        value["checkpoint_selection"] = checkpoint_selection
        value["checkpoint_global_step"] = checkpoint_global_step
        value["checkpoint_path"] = checkpoint
        value["checkpoint_provenance"] = {
            "path": checkpoint,
            "global_step": checkpoint_global_step,
            "selection": checkpoint_selection,
            "dependencies": checkpoint_dependencies,
            "best_validation": checkpoint_best_validation,
        }
    atomic_json_dump(output, value)
    result = {
        "output": str(output),
        "policy": policy_name,
        "checkpoint": checkpoint,
        **value,
    }
    if test_preprocessing is not None:
        result["test_preprocessing"] = test_preprocessing
    return result


def _automatic_final_test_evaluation(config: AppConfig) -> dict[str, Any]:
    """Run the guarded final test evaluation, preferring the validation winner."""

    fingerprints = stage_fingerprints(config)
    run_dir = artifact_paths(config, fingerprints)[Stage.PPO].parent
    best_checkpoint = run_dir / "checkpoints" / "best_validation.pt"
    if best_checkpoint.is_file():
        checkpoint_path: Path | None = best_checkpoint
        selection = "best_validation"
    else:
        checkpoint_path = None
        selection = "latest_final_fallback"
        LOGGER.warning(
            "No best-validation checkpoint exists; final test evaluation will use "
            "the latest compatible checkpoint"
        )
    result = _evaluate_rl_command(
        config,
        "test",
        allow_test=True,
        checkpoint_path=checkpoint_path,
        checkpoint_selection=selection,
    )
    if config.logging.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
        try:
            global_step = int(result.get("checkpoint_global_step") or 0)
            metrics = result.get("metrics")
            if isinstance(metrics, Mapping):
                for name, value in metrics.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        writer.add_scalar(f"test/{name}", value, global_step)
        finally:
            writer.close()
    return result


def _manual_play_command(
    config: AppConfig,
    split_name: SplitName,
    allow_test: bool,
    *,
    slide_id: str | None,
    seed: int | None,
) -> None:
    """Launch manual control over the same production environment used by RL."""

    from wsi_lab.manual.ui import run_manual_play

    _ensure_test_access(split_name, allow_test)
    if split_name == "test":
        _prepare_authorized_test_preprocessing(config)
    split = _load_verified_split(config)
    slide_ids = split.ids(split_name)
    if slide_id is not None and slide_id not in slide_ids:
        raise ValueError(f"slide {slide_id!r} is not in split {split_name!r}")
    environment = _make_production_environment(
        config,
        split_name=split_name,
        evaluation=True,
        allow_test=allow_test,
    )
    try:
        run_manual_play(
            environment,
            slide_ids=slide_ids,
            initial_slide_id=slide_id,
            seed=config.rl.seed if seed is None else seed,
        )
    finally:
        environment.close()


def _render_trajectory_command(
    config: AppConfig,
    trajectory_path: Path,
    output_path: Path | None,
    *,
    allow_test: bool,
) -> dict[str, Any]:
    from wsi_lab.envs.rendering import render_trajectory_figure
    from wsi_lab.wsi.reader import OverviewTransform

    source = trajectory_path.expanduser().resolve()
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("trajectory document must be a JSON object")
    slide_id = str(raw.get("slide_id", ""))
    points = raw.get("trajectory")
    if not slide_id or not isinstance(points, list):
        raise ValueError("trajectory JSON requires slide_id and a trajectory list")
    assignment = _load_verified_split(config).assignment(slide_id)
    _ensure_test_access(assignment.split, allow_test)
    store = _preprocessing_store(config)
    metadata = store.load_metadata(slide_id)
    transform = OverviewTransform.from_dict(metadata["overview_transform"])
    destination = output_path or source.with_suffix(".png")
    rendered = render_trajectory_figure(
        store.load_overview(slide_id, mmap=False),
        transform,
        points,
        destination,
    )
    return {"output": str(rendered), "slide_id": slide_id, "points": len(points)}


def _benchmark_command(config: AppConfig, steps: int) -> dict[str, Any]:
    from wsi_lab.benchmarks.environment import benchmark_environment

    factory = functools.partial(
        _make_production_environment,
        config,
        split_name="val",
        evaluation=True,
        allow_test=False,
    )
    result = benchmark_environment(
        factory, backend="openslide-hdf5", steps=steps, seed=config.rl.seed
    )
    fingerprints = stage_fingerprints(config)
    output = (
        config.workspace.root
        / "benchmarks"
        / f"environment_{fingerprints[Stage.PREPROCESSING]}.json"
    )
    atomic_json_dump(output, result.to_dict())
    return {"output": str(output), **result.to_dict()}


def _smoke_test() -> dict[str, Any]:
    import numpy as np
    from gymnasium.utils.env_checker import check_env

    from wsi_lab.envs import make_env

    environment = make_env(overview_size=32, viewport_size=32, max_episode_steps=12)
    check_env(environment)
    observation, info = environment.reset(seed=1)
    steps = 0
    terminated = False
    while not terminated:
        valid = np.flatnonzero(info["action_mask"])
        action = int(valid[0])
        observation, _, terminated, truncated, info = environment.step(action)
        if truncated:
            raise RuntimeError("Synthetic environment unexpectedly truncated")
        steps += 1
    environment.close()
    return {
        "environment_checker": "passed",
        "episode_steps": steps,
        "observation_keys": sorted(observation),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wsi-lab", description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "preprocess", "train-belief", "train-rl", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--dry-run", action="store_true")
        if name in {"train-rl", "run"}:
            command.add_argument(
                "--final-test-eval",
                action=argparse.BooleanOptionalAction,
                default=True,
                help=(
                    "run one guarded official-test evaluation after PPO training (default: enabled)"
                ),
            )
    for name in ("eval-belief", "eval-rl"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--split", choices=("val", "test"), default="val")
        command.add_argument("--allow-test", action="store_true")
        if name == "eval-rl":
            command.add_argument(
                "--policy",
                choices=("ppo", "random", "raster", "fixed-budget", "confidence"),
                default="ppo",
            )
            command.add_argument("--fixed-budget", type=int, default=128)
            command.add_argument("--confidence-threshold", type=float, default=0.9)
    render = commands.add_parser("render-trajectory")
    render.add_argument("--config", required=True, type=Path)
    render.add_argument("--trajectory", required=True, type=Path)
    render.add_argument("--output", type=Path)
    render.add_argument("--allow-test", action="store_true")
    manual = commands.add_parser("manual-play")
    manual.add_argument("--config", required=True, type=Path)
    manual.add_argument("--split", choices=("train", "val", "test"), default="val")
    manual.add_argument("--slide-id")
    manual.add_argument("--seed", type=int)
    manual.add_argument("--allow-test", action="store_true")
    benchmark = commands.add_parser("benchmark-env")
    benchmark.add_argument("--config", required=True, type=Path)
    benchmark.add_argument("--steps", type=int, default=100)
    smoke = commands.add_parser("smoke-test")
    smoke.add_argument("--config", type=Path)
    return parser


def _dispatch(arguments: argparse.Namespace) -> int:
    if arguments.command == "smoke-test":
        if arguments.config is not None:
            load_config(arguments.config)
        print(json.dumps(_smoke_test(), indent=2))
        return 0
    config = load_config(arguments.config)
    if arguments.command == "eval-belief":
        print(
            json.dumps(
                _evaluate_belief_command(config, arguments.split, arguments.allow_test), indent=2
            )
        )
        return 0
    if arguments.command == "eval-rl":
        print(
            json.dumps(
                _evaluate_rl_command(
                    config,
                    arguments.split,
                    arguments.allow_test,
                    policy_name=arguments.policy,
                    fixed_budget=arguments.fixed_budget,
                    confidence_threshold=arguments.confidence_threshold,
                ),
                indent=2,
            )
        )
        return 0
    if arguments.command == "manual-play":
        _manual_play_command(
            config,
            arguments.split,
            arguments.allow_test,
            slide_id=arguments.slide_id,
            seed=arguments.seed,
        )
        return 0
    if arguments.command == "render-trajectory":
        print(
            json.dumps(
                _render_trajectory_command(
                    config,
                    arguments.trajectory,
                    arguments.output,
                    allow_test=arguments.allow_test,
                ),
                indent=2,
            )
        )
        return 0
    if arguments.command == "benchmark-env":
        print(json.dumps(_benchmark_command(config, arguments.steps), indent=2))
        return 0
    runner = _configured_runner(config)
    stages: tuple[Stage, ...]
    if arguments.command == "validate":
        stages = (Stage.DATASET, Stage.SPLIT)
    elif arguments.command == "train-belief":
        stages = (Stage.SPLIT, Stage.BELIEF_DATA, Stage.BELIEF)
    elif arguments.command == "preprocess":
        stages = (Stage.SPLIT, Stage.BELIEF_DATA, Stage.TISSUE, Stage.PREPROCESSING)
    elif arguments.command == "train-rl":
        stages = (Stage.PPO,)
    elif arguments.command == "run":
        stages = (
            Stage.SPLIT,
            Stage.BELIEF_DATA,
            Stage.TISSUE,
            Stage.PREPROCESSING,
            Stage.BELIEF,
            Stage.PPO,
        )
    else:
        raise AssertionError(arguments.command)
    dry_run = bool(getattr(arguments, "dry_run", False))
    plan = runner.run(stages, dry_run=dry_run)
    print(plan.format())
    if (
        arguments.command in {"train-rl", "run"}
        and bool(getattr(arguments, "final_test_eval", True))
        and not dry_run
    ):
        final_test = _automatic_final_test_evaluation(config)
        print("Final test evaluation:")
        print(json.dumps(final_test, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    configure_logging(arguments.verbose)
    try:
        return _dispatch(arguments)
    except (WSILabError, ValueError, OSError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
