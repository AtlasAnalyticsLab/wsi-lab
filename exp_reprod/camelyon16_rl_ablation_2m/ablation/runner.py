#!/usr/bin/env python3
"""Training-only runner for the published CAMELYON16 RL ablation study.

This is intentionally not a replacement training implementation.  It wires
ablation-specific environment/model contracts and reporting hooks into WSI-Lab's
existing recurrent PPO trainer, rollout buffer, optimizer, and checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
REPRODUCTION_DIR = PACKAGE_DIR.parent
DEFAULT_SOURCE_ROOT = REPRODUCTION_DIR.parents[1]
SOURCE_ROOT = (
    Path(os.environ.get("WSI_LAB_SOURCE_ROOT", str(DEFAULT_SOURCE_ROOT))).expanduser().resolve()
)
SOURCE_PACKAGE_ROOT = SOURCE_ROOT / "src"
if str(SOURCE_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_PACKAGE_ROOT))

from wsi_lab.config.loader import load_config  # noqa: E402
from wsi_lab.rl.checkpoints import save_ppo_checkpoint  # noqa: E402
from wsi_lab.rl.trainer import (  # noqa: E402
    PPOTrainer,
    RolloutCollection,
    make_vector_env,
    seed_everything,
)
from wsi_lab.utils.atomic import atomic_json_dump, atomic_write_text  # noqa: E402
from wsi_lab.utils.fingerprints import fingerprint  # noqa: E402

from .analysis_cache import AnalysisCache  # noqa: E402
from .contract import (  # noqa: E402
    MONITORING_CONTRACT,
    RUNNER_SCHEMA_VERSION,
    SMOKE_OVERRIDES,
    SOURCE_REVISION,
    TRAINING_PROGRESS_SCHEMA_VERSION,
    assert_protocol_config,
    run_contract_payload,
)
from .monitoring import build_training_monitor  # noqa: E402
from .provenance import SourceCompatibilityError, verify_source_tree  # noqa: E402
from .spec import load_study  # noqa: E402
from .telemetry import (  # noqa: E402
    EpisodeTelemetryWrapper,
    TrainingTelemetry,
)
from .variants import (  # noqa: E402
    RANDOM_START_IMPLEMENTATION,
    AblationActorCritic,
    VariantSpec,
    candidate_ids,
    get_variant_spec,
    make_variant_environment,
    variant_dependencies,
)
from .verification import VerificationError, verify_completed_run  # noqa: E402


class RunnerError(RuntimeError):
    """Fail-closed ablation-study contract error."""


class GracefulInterruption(RuntimeError):
    """Raised at a rollout boundary after a process termination signal."""


_STOP_SIGNAL: int | None = None


def _request_stop(signum: int, _frame: Any) -> None:
    global _STOP_SIGNAL
    _STOP_SIGNAL = int(signum)


def _assert_source_pin() -> None:
    """Require the exact recorded core source while allowing this added folder.

    The reproduction package was added after the recorded implementation, so
    its checkout revision may differ. Compare the core Python files by content
    instead; this also works without Git history.
    """

    source = load_study()["source_tree"]
    try:
        verify_source_tree(
            SOURCE_ROOT,
            expected_sha256=str(source["sha256"]),
            expected_files=int(source["python_files"]),
        )
    except (KeyError, TypeError, SourceCompatibilityError) as exc:
        raise RunnerError(
            "the selected src/wsi_lab tree does not match the recorded ablation source; "
            "use the source root described in README.md"
        ) from exc


def _resolved_config(config_path: str | Path, mode: str) -> Any:
    config = load_config(config_path)
    if mode not in {"main", "smoke"}:
        raise RunnerError("mode must be 'main' or 'smoke'")
    if config.dataset.name != "camelyon16":
        raise RunnerError("the ablation study is CAMELYON16-only")
    if str(config.environment.reward_type.value) != "default":
        raise RunnerError("the study requires the default navigation reward")
    if int(config.environment.max_episode_steps) != 500:
        raise RunnerError("the study requires a 500-navigation-action horizon")
    if str(config.rl.vector_backend.value) != "sync":
        raise RunnerError("the study requires the synchronous vector backend")
    if mode == "smoke":
        for name, value in SMOKE_OVERRIDES.items():
            setattr(config.rl, name, value)
    elif int(config.rl.total_timesteps) != 2_000_000:
        raise RunnerError("main runs require exactly 2,000,000 requested timesteps")
    return config


def describe_run(
    config_path: str | Path,
    *,
    variant: str,
    run_id: str,
    mode: str,
) -> dict[str, Any]:
    """Resolve stable output paths without creating files or loading models."""

    config = _resolved_config(config_path, mode)
    spec = get_variant_spec(variant)
    try:
        assert_protocol_config(
            config,
            variant=spec.id,
            run_id=run_id,
            mode=mode,
        )
    except ValueError as exc:
        raise RunnerError(str(exc)) from exc
    run_fingerprint = fingerprint(run_contract_payload(config, spec, run_id=run_id, mode=mode))
    artifact_dir = (Path(config.workspace.root) / "rl_ablation" / run_fingerprint).resolve()
    return {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "run_id": str(run_id),
        "mode": str(mode),
        "variant": spec.id,
        "start": str(config.environment.start_position),
        "seed": int(config.rl.seed),
        "run_fingerprint": run_fingerprint,
        "workspace": str(Path(config.workspace.root).resolve()),
        "artifact_dir": str(artifact_dir),
        "checkpoint_dir": str(artifact_dir / "checkpoints"),
        # Verification treats this atomic manifest as the authoritative run
        # completion record; avoid maintaining a second, divergent result.
        "result_path": str(artifact_dir / "run_manifest.json"),
        "requested_timesteps": int(config.rl.total_timesteps),
        "expected_final_step": int(
            np.ceil(
                int(config.rl.total_timesteps)
                / (int(config.rl.num_envs) * int(config.rl.rollout_steps))
            )
            * (int(config.rl.num_envs) * int(config.rl.rollout_steps))
        ),
    }


def _analysis_cache_root() -> Path:
    work_root = os.environ.get("WSI_LAB_ABLATION_WORK_ROOT")
    if not work_root:
        raise RunnerError("WSI_LAB_ABLATION_WORK_ROOT is required")
    expected = (Path(work_root).expanduser().resolve() / "analysis_cache").resolve()
    override = os.environ.get("WSI_LAB_ABLATION_ANALYSIS_CACHE")
    if override and Path(override).expanduser().resolve() != expected:
        raise RunnerError("analysis cache must be WORK_ROOT/analysis_cache")
    return expected


def _production_source(config: Any) -> tuple[Any, Any]:
    """Build a training-only source without loading the belief checkpoint."""

    from wsi_lab.cli import _load_verified_split, _preprocessing_store
    from wsi_lab.preprocessing.pipeline import ProcessedEpisodeSource

    split = _load_verified_split(config)
    source = ProcessedEpisodeSource.from_config(
        config,
        _preprocessing_store(config),
        split,
        allow_test=False,
    )
    return source, split


def optimized_random_start_preflight(
    config_path: Path,
    *,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate vectorized coarse candidates for every training slide.

    The no-mask branch intentionally never indexes ``tissue_valid``.  The
    production grid itself is the geometry-valid domain; padded edge cells are
    valid WSI observations because the reader supplies white padding.
    """

    mode = str(row.get("mode", "main"))
    config = _resolved_config(config_path, mode)
    variant_id = str(row.get("variant", row.get("variant_id", "")))
    spec = get_variant_spec(variant_id)
    try:
        assert_protocol_config(
            config,
            variant=spec.id,
            run_id=str(row.get("run_id", "")),
            mode=mode,
        )
    except ValueError as exc:
        raise RunnerError(str(exc)) from exc
    if str(row.get("start", row.get("start_position", ""))) != "random":
        return {
            "enabled": False,
            "implementation": RANDOM_START_IMPLEMENTATION,
            "candidate_domain": "not_required",
            "candidate_equivalence": "not_required",
            "checks": 0,
        }

    tissue_only = spec.tissue_action_masking
    source, split = _production_source(config)
    checks = 0
    try:
        train_ids = tuple(split.ids("train"))
        if not train_ids:
            raise RunnerError("persisted split contains no training slides")
        for slide_id in train_ids:
            slide = source.load_slide(slide_id)
            arrays = getattr(slide.cells, "_arrays", None)
            if not isinstance(arrays, tuple) or not arrays:
                raise RunnerError(f"slide {slide_id} does not expose mmap navigation grids")
            coarse = arrays[0]
            names = set(coarse.dtype.names or ())
            required = {"cell_id", "magnification"}
            if tissue_only:
                required.add("tissue_valid")
            if not required.issubset(names):
                raise RunnerError(f"slide {slide_id} coarse grid lacks fields {sorted(required)}")
            optimized = candidate_ids(slide, tissue_only=tissue_only)
            if tissue_only:
                expected_rows = coarse[coarse["tissue_valid"].astype(np.bool_)]
            else:
                expected_rows = coarse
            expected = tuple(map(int, expected_rows["cell_id"]))
            if optimized != expected or tuple(sorted(optimized)) != optimized:
                raise RunnerError(f"optimized random candidates differ on {slide_id}")
            checks += 1
    finally:
        source.close()
    return {
        "enabled": True,
        "implementation": RANDOM_START_IMPLEMENTATION,
        "candidate_domain": (
            "coarse_tissue_valid" if tissue_only else "coarse_geometry_valid_no_mask"
        ),
        "candidate_equivalence": "passed",
        "checks": checks,
        "tissue_mask_accessed": bool(tissue_only),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RunnerError(f"could not hash prerequisite artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _checkpoint_dependencies(
    config: Any,
    spec: VariantSpec,
    run_fingerprint: str,
    cache_root: Path,
) -> dict[str, str]:
    from wsi_lab.cli import _ppo_dependencies

    dependencies = {str(key): str(value) for key, value in _ppo_dependencies(config).items()}
    if not spec.belief_enabled:
        dependencies.pop("belief_fingerprint", None)
    dependencies.update(variant_dependencies(spec))
    dependencies.update(
        {
            "ablation_source_revision": SOURCE_REVISION,
            "ablation_run_fingerprint": run_fingerprint,
            "monitoring_contract_sha256": fingerprint(MONITORING_CONTRACT),
            "analysis_cache_manifest_sha256": _sha256_file(cache_root / "manifest.json"),
            "analysis_monitor_cohort_sha256": _sha256_file(cache_root / "monitor_slides.json"),
        }
    )
    return dependencies


class AblationPPOTrainer(PPOTrainer):
    """Thin telemetry hook around the shared rollout collector."""

    def __init__(self, *args: Any, telemetry: TrainingTelemetry, **kwargs: Any) -> None:
        self.ablation_telemetry = telemetry
        super().__init__(*args, **kwargs)

    def collect_rollout(self) -> RolloutCollection:
        collection = super().collect_rollout()
        self.ablation_telemetry.record_rollout(
            self.global_step,
            collection.buffer,
            collection.completed_episodes,
            update=self.update_number + 1,
        )
        return collection


def _write_interruption_checkpoint(
    trainer: PPOTrainer,
    checkpoint_dir: Path,
    *,
    last_metrics: Mapping[str, Any] | None,
) -> Path:
    return save_ppo_checkpoint(
        checkpoint_dir / f"checkpoint_{trainer.global_step:012d}.pt",
        trainer.model,
        trainer.optimizer,
        trainer.config,
        global_step=trainer.global_step,
        update=trainer.update_number,
        dependencies=trainer.dependencies,
        extra={
            "completed_episode_count": trainer.completed_episode_count,
            "interrupted": True,
            "signal": _STOP_SIGNAL,
            "last_metrics": None if last_metrics is None else dict(last_metrics),
        },
    )


def _load_monitor_cohort(cache_root: Path, *, mode: str) -> tuple[dict[str, Any], ...]:
    path = cache_root / "monitor_slides.json"
    if not path.is_file():
        raise RunnerError(f"training monitor cohort is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        slides = tuple(dict(item) for item in payload["slides"])
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RunnerError(f"invalid training monitor cohort {path}: {exc}") from exc
    expected = 30
    if len(slides) != expected:
        raise RunnerError(f"monitor cohort has {len(slides)} slides, expected {expected}")
    if any(str(item.get("split", "train")) != "train" for item in slides):
        raise RunnerError("monitor cohort contains a non-training slide")
    if mode == "main":
        return slides
    positives = [item for item in slides if int(item["label"]) == 1]
    negatives = [item for item in slides if int(item["label"]) == 0]
    if not positives or not negatives:
        raise RunnerError("smoke monitoring requires one positive and one benign training slide")
    return (positives[0], negatives[0])


def _analysis_cache_ready(cache_root: Path, cohort: Sequence[Mapping[str, Any]]) -> None:
    manifest = cache_root / "manifest.json"
    if not manifest.is_file():
        raise RunnerError(f"analysis cache completion manifest is missing: {manifest}")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"invalid analysis cache manifest: {exc}") from exc
    if payload.get("status") != "complete":
        raise RunnerError("analysis cache is not complete")
    cache = AnalysisCache(cache_root / "slides")
    missing = [
        str(item["slide_id"]) for item in cohort if cache.load(str(item["slide_id"])) is None
    ]
    if missing:
        raise RunnerError(f"analysis cache lacks monitor slides: {missing[:8]}")


def _environment_factory(
    config: Any,
    spec: VariantSpec,
    analysis_provider: Callable[[str], Any],
):
    def factory() -> EpisodeTelemetryWrapper:
        environment = make_variant_environment(
            config,
            spec,
            split_name="train",
            evaluation=False,
            allow_test=False,
        )
        return EpisodeTelemetryWrapper(
            environment,
            analysis_provider=analysis_provider,
        )

    return factory


def _resolved_config_yaml(config: Any) -> str:
    return yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True)


def _load_training_progress(
    path: Path,
    *,
    run_fingerprint: str,
    variant: str,
    restored_step: int,
    restored_update: int,
) -> Mapping[str, Any] | None:
    """Load metrics only when the durable update matches restored PPO state."""

    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"could not read durable training progress: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RunnerError("durable training progress is not a JSON object")
    identity = {
        "schema_version": TRAINING_PROGRESS_SCHEMA_VERSION,
        "status": "update_complete",
        "run_fingerprint": run_fingerprint,
        "variant": variant,
    }
    mismatches = [name for name, expected in identity.items() if payload.get(name) != expected]
    if mismatches:
        raise RunnerError(f"durable training progress identity mismatch: {sorted(mismatches)}")
    step = payload.get("global_step")
    updates = payload.get("updates")
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or step <= 0
        or isinstance(updates, bool)
        or not isinstance(updates, int)
        or updates <= 0
    ):
        raise RunnerError("durable training progress has invalid counters")
    metrics = payload.get("last_metrics")
    if not isinstance(metrics, Mapping) or not metrics:
        raise RunnerError("durable training progress has no update metrics")

    def require_finite(value: Any) -> None:
        if isinstance(value, Mapping):
            for item in value.values():
                require_finite(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                require_finite(item)
        elif isinstance(value, float) and not math.isfinite(value):
            raise RunnerError("durable training progress contains non-finite metrics")

    require_finite(metrics)
    if step != int(restored_step):
        return None
    if updates != int(restored_update):
        raise RunnerError("durable training progress update does not match restored PPO state")
    return dict(metrics)


def run_training(config_path: Path, *, variant: str, run_id: str, mode: str) -> dict[str, Any]:
    """Run recurrent PPO locally without validation or test evaluation."""

    _assert_source_pin()
    description = describe_run(
        config_path,
        variant=variant,
        run_id=run_id,
        mode=mode,
    )
    config = _resolved_config(config_path, mode)
    spec = get_variant_spec(variant)
    artifact_dir = Path(description["artifact_dir"])
    checkpoint_dir = Path(description["checkpoint_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_dir / "run_manifest.json"
    result_path = Path(description["result_path"])
    progress_path = artifact_dir / "training_progress.json"

    existing: Mapping[str, Any] | None = None
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            existing = loaded if isinstance(loaded, Mapping) else None
        except (OSError, json.JSONDecodeError) as exc:
            raise RunnerError(f"could not read existing run manifest: {exc}") from exc
        if existing is None or existing.get("run_fingerprint") != description["run_fingerprint"]:
            raise RunnerError("workspace contains a different ablation run fingerprint")
        # A duplicate task invocation is a no-op only after every generated
        # artifact has passed the same completion checks used by the launcher.
        # Reopening a valid final checkpoint would otherwise overwrite its PPO
        # metrics with an empty (zero-update) summary.
        if existing.get("status") == "complete":
            try:
                verify_completed_run(description, manifest_path=manifest_path)
            except VerificationError as exc:
                raise RunnerError(
                    "the existing completed run failed artifact verification"
                ) from exc
            return dict(existing)

    cache_root = _analysis_cache_root()
    cohort = _load_monitor_cohort(cache_root, mode=mode)
    _analysis_cache_ready(cache_root, cohort)
    dependencies = _checkpoint_dependencies(
        config,
        spec,
        str(description["run_fingerprint"]),
        cache_root,
    )
    started = time.time()
    manifest = {
        **description,
        "status": "running",
        "source_revision": SOURCE_REVISION,
        "split": "train",
        "validation_enabled": False,
        "test_access": False,
        "dependencies": dependencies,
        "started_at_unix": started,
        "resume_count": int(existing.get("resume_count", -1)) + 1 if existing else 0,
        # Preserve the last completed PPO update across an interrupted run,
        # including the edge case where that checkpoint already reached the
        # requested environment-step target.
        "last_metrics": existing.get("last_metrics") if existing else None,
    }
    atomic_json_dump(manifest_path, manifest)
    atomic_write_text(artifact_dir / "resolved_config.yaml", _resolved_config_yaml(config))
    atomic_json_dump(artifact_dir / "variant_contract.json", asdict(spec))

    from torch.utils.tensorboard import SummaryWriter

    writer = (
        SummaryWriter(log_dir=str(artifact_dir / "tensorboard"))
        if config.logging.tensorboard
        else None
    )
    telemetry = TrainingTelemetry(
        artifact_dir / "telemetry",
        writer,
        spec.id,
        trailing_window_episodes=256,
    )
    analysis_cache = AnalysisCache(cache_root / "slides")

    # SyncVectorEnv keeps all workers in this process, so share the small,
    # immutable train-only reporting sidecars instead of reopening a compressed
    # NPZ at every episode reset.  This is telemetry-only and cannot affect the
    # environment's transition or reward logic.
    @cache
    def analysis_provider(slide_id: str) -> Any:
        return analysis_cache.load(slide_id)

    factories = [
        _environment_factory(config, spec, analysis_provider)
        for _ in range(int(config.rl.num_envs))
    ]
    vector = make_vector_env(factories, backend="sync")
    # PPOTrainer seeds after receiving the model, so seed once here as well to
    # make parameter initialization a deterministic function of the run seed.
    seed_everything(int(config.rl.seed))
    model = AblationActorCritic(spec)
    trainer = AblationPPOTrainer(
        vector,
        model,
        config.rl,
        writer=writer,
        dependencies=dependencies,
        telemetry=telemetry,
    )

    resumed_from = trainer.resume_latest_compatible(checkpoint_dir)
    telemetry.reconcile(trainer.global_step, trainer.completed_episode_count)
    manifest["resumed_from"] = str(resumed_from) if resumed_from is not None else None
    manifest["resumed_global_step"] = trainer.global_step
    atomic_json_dump(manifest_path, manifest)

    monitor = build_training_monitor(
        output_dir=artifact_dir,
        cohort=cohort,
        variant_id=spec.id,
        mode=mode,
        config=config,
        analysis_cache=analysis_provider,
    )
    monitor.reconcile(trainer.global_step)

    try:
        last_update_metrics = _load_training_progress(
            progress_path,
            run_fingerprint=str(description["run_fingerprint"]),
            variant=spec.id,
            restored_step=trainer.global_step,
            restored_update=trainer.update_number,
        )
    except BaseException:
        trainer.close()
        if writer is not None:
            writer.close()
        raise
    if (
        resumed_from is not None
        and int(trainer.global_step) == int(description["expected_final_step"])
        and last_update_metrics is None
    ):
        trainer.close()
        if writer is not None:
            writer.close()
        raise RunnerError(
            "the final checkpoint has no matching durable training progress; "
            "the generated run cannot be completed safely"
        )

    def after_update(current: PPOTrainer, _metrics: Any) -> None:
        nonlocal last_update_metrics
        last_update_metrics = _metrics.to_dict()
        atomic_json_dump(
            progress_path,
            {
                "schema_version": TRAINING_PROGRESS_SCHEMA_VERSION,
                "status": "update_complete",
                "run_fingerprint": description["run_fingerprint"],
                "variant": spec.id,
                "global_step": int(current.global_step),
                "updates": int(current.update_number),
                "last_metrics": dict(last_update_metrics),
            },
        )
        if _STOP_SIGNAL is not None:
            _write_interruption_checkpoint(
                current,
                checkpoint_dir,
                last_metrics=last_update_metrics,
            )
            raise GracefulInterruption(f"received process signal {_STOP_SIGNAL}")
        monitor.maybe_run(current.model, current.global_step)

    status = "failed"
    summary: Any | None = None
    error: str | None = None
    try:
        # Recover a milestone that was checkpointed before its monitor callback
        # completed, then train with no validation callback at all.
        monitor.maybe_run(trainer.model, trainer.global_step)
        summary = trainer.train(
            checkpoint_dir=checkpoint_dir,
            resume_latest=False,
            update_callback=after_update,
            validation_callback=None,
        )
        monitor.finalize(summary.global_step, trainer.model)
        telemetry.finalize(summary.global_step)
        status = "complete"
    except GracefulInterruption as exc:
        status = "interrupted"
        error = str(exc)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        global_step = trainer.global_step
        completed_episodes = trainer.completed_episode_count
        trainer.close()
        if writer is not None:
            writer.close()
        if summary is not None and summary.last_metrics is not None:
            final_metrics: Mapping[str, Any] | None = summary.last_metrics.to_dict()
        elif last_update_metrics is not None:
            final_metrics = last_update_metrics
        elif existing is not None and isinstance(existing.get("last_metrics"), Mapping):
            final_metrics = dict(existing["last_metrics"])
        else:
            final_metrics = None
        result = {
            **description,
            "status": status,
            "source_revision": SOURCE_REVISION,
            "split": "train",
            "validation_enabled": False,
            "test_access": False,
            "global_step": int(global_step),
            "updates": int(trainer.update_number),
            "completed_episodes": int(completed_episodes),
            "duration_seconds": time.time() - started,
            "dependencies": dependencies,
            "error": error,
            "last_metrics": final_metrics,
        }
        atomic_json_dump(result_path, result)
        atomic_json_dump(manifest_path, {**manifest, **result})
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--variant",
        required=True,
        choices=tuple(
            sorted(
                get_variant_spec(key).id
                for key in (
                    "v1_full",
                    "v2_viewport_only",
                    "v3_no_tissue_mask",
                    "v4_no_diagnosis",
                    "v5_no_belief",
                    "v6_combined",
                )
            )
        ),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", required=True, choices=("main", "smoke"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    signal.signal(signal.SIGTERM, _request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _request_stop)
    result = run_training(
        arguments.config,
        variant=arguments.variant,
        run_id=arguments.run_id,
        mode=arguments.mode,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result.get("status") == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
