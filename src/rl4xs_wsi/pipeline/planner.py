"""Non-mutating, reason-bearing pipeline planning."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from rl4xs_wsi.config.models import AppConfig
from rl4xs_wsi.constants import SCHEMA_VERSION
from rl4xs_wsi.pipeline.stages import STAGE_ORDER, Stage, StageAction
from rl4xs_wsi.utils.fingerprints import fingerprint

_ALGORITHM_VERSION = {
    Stage.DATASET: 1,
    Stage.BELIEF_DATA: 1,
    Stage.SPLIT: 1,
    Stage.TISSUE: 1,
    Stage.PREPROCESSING: 2,
    Stage.BELIEF: 2,
    Stage.PPO: 1,
}


@dataclass(frozen=True, slots=True)
class StagePlan:
    stage: Stage
    action: StageAction
    reason: str
    artifact: Path
    fingerprint: str


@dataclass(frozen=True, slots=True)
class PipelinePlan:
    stages: tuple[StagePlan, ...]

    def for_stage(self, stage: Stage) -> StagePlan:
        return next(item for item in self.stages if item.stage is stage)

    def format(self) -> str:
        rows = ["Pipeline stage decisions:"]
        rows.extend(
            f"  {item.action.value:<7} {item.stage.value:<14} {item.reason} [{item.artifact}]"
            for item in self.stages
        )
        return "\n".join(rows)


def _file_signature(path: Path) -> dict[str, object]:
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path), "missing": True}
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _directory_signature(path: Path, suffixes: tuple[str, ...]) -> list[dict[str, object]]:
    if not path.is_dir():
        return [{"path": str(path), "missing": True}]
    result: list[dict[str, object]] = []
    for item in sorted(path.iterdir(), key=lambda value: value.name):
        if item.is_file() and item.suffix.casefold() in suffixes:
            result.append(_file_signature(item))
    return result


def _embedding_signature(
    config: AppConfig, *, prefixes: tuple[str, ...] | None = None
) -> dict[str, object]:
    embeddings: dict[str, object] = {}
    for magnification in config.embeddings.magnifications:
        token = str(int(magnification)) if float(magnification).is_integer() else str(magnification)
        relative = config.embeddings.directory_template.format(
            magnification=token,
            encoder=config.embeddings.encoder,
            slide_id="",
        )
        entries = _directory_signature(config.embeddings.root / relative, (".h5", ".hdf5"))
        if prefixes is not None:
            entries = [
                entry
                for entry in entries
                if Path(str(entry.get("path", ""))).stem.startswith(prefixes)
                or entry.get("missing") is True
            ]
        embeddings[token] = entries
    return embeddings


def _dataset_signature(config: AppConfig) -> dict[str, object]:
    return {
        "labels": _file_signature(config.dataset.labels),
        "slides": _directory_signature(
            config.dataset.slides_dir, (".tif", ".tiff", ".svs", ".ndpi", ".mrxs")
        ),
        "embeddings": _embedding_signature(config),
    }


def _belief_input_signature(config: AppConfig) -> dict[str, object]:
    return {
        "labels": _file_signature(config.dataset.labels),
        "embeddings": _embedding_signature(config, prefixes=("normal_", "tumor_")),
    }


def _versioned(stage: Stage, value: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": _ALGORITHM_VERSION[stage],
        **value,
    }


def stage_fingerprints(config: AppConfig) -> dict[Stage, str]:
    inputs = _dataset_signature(config)
    dataset = fingerprint(
        _versioned(
            Stage.DATASET,
            {
                "dataset": config.dataset,
                "embeddings": config.embeddings,
                "validation_chunk_rows": config.embeddings.validation_chunk_rows,
                "inputs": inputs,
            },
        )
    )
    belief_inputs = _belief_input_signature(config)
    belief_data = fingerprint(
        _versioned(
            Stage.BELIEF_DATA,
            {
                "embedding_inputs": belief_inputs,
                "encoder": config.embeddings.encoder,
                "scales": config.embeddings.magnifications,
                "feature_key": config.embeddings.feature_key,
                "coordinate_key": config.embeddings.coordinate_key,
                "validation_chunk_rows": config.embeddings.validation_chunk_rows,
            },
        )
    )
    split = fingerprint(
        _versioned(Stage.SPLIT, {"labels": belief_inputs["labels"], "split": config.split})
    )
    tissue = fingerprint(
        _versioned(
            Stage.TISSUE,
            {
                "slides": inputs["slides"],
                "tissue": config.tissue,
                "existing_masks": _directory_signature(
                    config.dataset.tissue_masks_dir, (".tif", ".tiff", ".png", ".npy")
                ),
            },
        )
    )
    preprocessing = fingerprint(
        _versioned(
            Stage.PREPROCESSING,
            {
                "dataset": dataset,
                "tissue": tissue,
                "environment_spatial": {
                    "overview_size": config.environment.overview_size,
                    "viewport_size": config.environment.viewport_size,
                },
                "mapping_tolerance": config.embeddings.mapping_tolerance,
            },
        )
    )
    belief = fingerprint(
        _versioned(
            Stage.BELIEF,
            {
                "belief_data": belief_data,
                "split": split,
                "embedding_scales": config.embeddings.magnifications,
                "feature_key": config.embeddings.feature_key,
                "belief": config.belief,
            },
        )
    )
    ppo = fingerprint(
        _versioned(
            Stage.PPO,
            {
                "dataset": dataset,
                "split": split,
                "preprocessing": preprocessing,
                "belief": belief,
                "environment": config.environment,
                "rl": config.rl,
            },
        )
    )
    return {
        Stage.DATASET: dataset,
        Stage.BELIEF_DATA: belief_data,
        Stage.SPLIT: split,
        Stage.TISSUE: tissue,
        Stage.PREPROCESSING: preprocessing,
        Stage.BELIEF: belief,
        Stage.PPO: ppo,
    }


def artifact_paths(config: AppConfig, fingerprints: dict[Stage, str]) -> dict[Stage, Path]:
    root = config.workspace.root
    return {
        Stage.DATASET: root / "manifests" / "dataset_manifest.json",
        Stage.BELIEF_DATA: root / "manifests" / "belief_embedding_manifest.json",
        Stage.SPLIT: root / "splits" / "manifest.json",
        Stage.TISSUE: root / "tissue" / "camelyon16" / fingerprints[Stage.TISSUE] / "manifest.json",
        Stage.PREPROCESSING: root
        / "processed"
        / "camelyon16"
        / fingerprints[Stage.PREPROCESSING]
        / "manifest.json",
        Stage.BELIEF: root / "models" / "belief" / fingerprints[Stage.BELIEF] / "manifest.json",
        Stage.PPO: root / "runs" / "rl" / fingerprints[Stage.PPO] / "manifest.json",
    }


def _ppo_checkpoint_candidates(run_directory: Path) -> tuple[Path, ...]:
    """Discover structurally complete, readable PPO checkpoint payloads."""

    checkpoint_directory = run_directory / "checkpoints"
    if not checkpoint_directory.is_dir():
        return ()
    # Keep torch-backed inspection local so planning unrelated stages does not
    # import PyTorch.  The inspector maps tensor storage to ``meta`` and does not
    # mutate model, optimizer, RNG, or filesystem state.
    from rl4xs_wsi.rl.checkpoints import inspect_ppo_checkpoint

    candidates: list[Path] = []
    for candidate in checkpoint_directory.glob("checkpoint_*.pt"):
        try:
            if not candidate.is_file() or candidate.stat().st_size <= 0:
                continue
            inspect_ppo_checkpoint(candidate)
            candidates.append(candidate)
        except (OSError, ValueError):
            continue
    # PPOTrainer uses zero-padded global steps. Full run compatibility is still
    # deferred to PPOTrainer.resume_latest_compatible, which has the active
    # configuration, dependency fingerprints, model, and optimizer.
    return tuple(sorted(candidates, key=lambda candidate: candidate.name, reverse=True))


def _inspect_manifest(
    stage: Stage, path: Path, expected_fingerprint: str
) -> tuple[StageAction, str]:
    missing = StageAction.TRAIN if stage in (Stage.BELIEF, Stage.PPO) else StageAction.BUILD
    stale = StageAction.TRAIN if stage in (Stage.BELIEF, Stage.PPO) else StageAction.REBUILD
    if not path.is_file():
        return missing, "artifact is absent"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return stale, f"manifest is unreadable: {exc}"
    actual = raw.get("fingerprint")
    if actual != expected_fingerprint:
        return stale, f"fingerprint changed ({actual or 'missing'} -> {expected_fingerprint})"
    status = raw.get("status")
    if status == "complete":
        if stage is Stage.PPO:
            candidates = _ppo_checkpoint_candidates(path.parent)
            if not candidates:
                return (
                    stale,
                    "manifest is complete but no valid complete PPO checkpoint candidates exist",
                )
            return StageAction.SKIP, "complete compatible artifact exists"
        required: tuple[Path, ...]
        if stage is Stage.DATASET:
            required = (path.parent / "embedding_validation.json",)
        elif stage is Stage.BELIEF_DATA:
            required = (path.parent / "belief_embedding_validation.json",)
        elif stage is Stage.SPLIT:
            required = (path.parent / "camelyon16_split.json",)
        elif stage is Stage.TISSUE:
            required = (path.parent / "index.json",)
        elif stage is Stage.PREPROCESSING:
            required = (
                path.parent / "slides",
                path.parent / "slides.json",
                path.parent / "cell_offsets.json",
            )
        elif stage is Stage.BELIEF:
            required = (
                path.parent / "checkpoint.pt",
                path.parent / "calibrated_checkpoint.pt",
                path.parent / "temperature.json",
                path.parent / "metrics.json",
            )
        else:
            required = ()
        absent = [str(item) for item in required if not item.exists()]
        if absent:
            return stale, f"manifest is complete but required outputs are missing: {absent}"
        if stage is Stage.PREPROCESSING and not any(
            item.is_dir() and (item / "metadata.json").is_file()
            for item in (path.parent / "slides").iterdir()
        ):
            return stale, "manifest is complete but no complete processed-slide artifacts exist"
        return StageAction.SKIP, "complete compatible artifact exists"
    if status in {"incomplete", "failed"}:
        if stage is Stage.BELIEF and (path.parent / "checkpoint.pt").is_file():
            return StageAction.RESUME, "compatible incomplete checkpoint exists"
        if stage is Stage.PPO:
            candidates = _ppo_checkpoint_candidates(path.parent)
            if candidates:
                return (
                    StageAction.RESUME,
                    f"PPO checkpoint candidate exists: {candidates[0].name}",
                )
    return stale, f"manifest status is {status!r}"


def build_plan(config: AppConfig, stages: Iterable[Stage] | None = None) -> PipelinePlan:
    """Inspect artifacts only; never create directories or files."""
    selected = tuple(stages) if stages is not None else STAGE_ORDER
    fingerprints = stage_fingerprints(config)
    paths = artifact_paths(config, fingerprints)
    plans: list[StagePlan] = []
    for stage in selected:
        action, reason = _inspect_manifest(stage, paths[stage], fingerprints[stage])
        plans.append(StagePlan(stage, action, reason, paths[stage], fingerprints[stage]))
    return PipelinePlan(tuple(plans))
