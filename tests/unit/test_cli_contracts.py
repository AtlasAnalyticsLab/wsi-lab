from __future__ import annotations

import inspect
import json
import sys
from argparse import Namespace
from hashlib import sha256
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import wsi_lab
from wsi_lab import cli
from wsi_lab.config.models import AppConfig, DatasetConfig, WorkspaceConfig
from wsi_lab.data.base import SlideRecord
from wsi_lab.data.splits import DatasetSplit, SplitAssignment
from wsi_lab.errors import PreprocessingError, TestSplitAccessError
from wsi_lab.pipeline.planner import (
    PipelinePlan,
    StagePlan,
    artifact_paths,
    stage_fingerprints,
)
from wsi_lab.pipeline.runner import PipelineRunner
from wsi_lab.pipeline.stages import Stage, StageAction
from wsi_lab.schemas import ValidationReport
from wsi_lab.utils.atomic import atomic_json_dump


class _FormattedPlan:
    def format(self) -> str:
        return "test plan"


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        dataset=DatasetConfig(root=tmp_path / "dataset"),
        workspace=WorkspaceConfig(root=tmp_path / "workspace"),
    )


def test_train_belief_dry_run_selects_only_split_development_validation_and_belief(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    calls: list[tuple[tuple[Stage, ...], bool]] = []

    original_run = PipelineRunner.run

    def run(
        runner: PipelineRunner,
        stages: tuple[Stage, ...],
        *,
        dry_run: bool,
    ) -> PipelinePlan:
        calls.append((stages, dry_run))
        return original_run(runner, stages, dry_run=dry_run)

    monkeypatch.setattr(cli, "load_config", lambda _: config)
    monkeypatch.setattr(PipelineRunner, "run", run)

    result = cli._dispatch(
        Namespace(command="train-belief", config=tmp_path / "config.yaml", dry_run=True)
    )

    assert result == 0
    assert calls == [((Stage.SPLIT, Stage.BELIEF_DATA, Stage.BELIEF), True)]
    assert not config.workspace.root.exists()


def test_configured_runner_registers_an_implementation_for_every_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registered: dict[Stage, object] = {}

    class RegisteringRunner:
        def register(self, stage: Stage, function: object) -> None:
            registered[stage] = function

    runner = RegisteringRunner()
    monkeypatch.setattr(cli, "PipelineRunner", lambda _: runner)

    assert cli._configured_runner(_config(tmp_path)) is runner
    assert set(registered) == set(Stage)


def test_split_stage_uses_adapter_specific_split_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    records = tuple(
        SlideRecord(
            slide_id=f"slide_{index}",
            slide_path=tmp_path / f"slide_{index}.tif",
            label=index % 2,
            label_name="negative" if index % 2 == 0 else "positive",
            official_cohort="development",
            patient_id=f"patient_{index}",
        )
        for index in range(3)
    )
    expected = DatasetSplit(
        dataset="custom_combined",
        seed=7,
        validation_fraction=0.25,
        test_fraction=0.15,
        strategy="adapter_specific",
        assignments=tuple(
            SplitAssignment(record.slide_id, int(record.label), split, record.patient_id)
            for record, split in zip(records, ("train", "val", "test"), strict=True)
        ),
        schema_version=2,
    )
    captured: dict[str, object] = {}

    class Adapter:
        name = "custom_combined"

        def iter_slides(self):
            return iter(records)

        def create_split(self, **kwargs):
            captured.update(kwargs)
            return expected

    monkeypatch.setattr(cli, "_adapter", lambda _: Adapter())
    config.split.seed = 7
    config.split.validation_fraction = 0.25
    config.split.test_fraction = 0.15

    result = cli._split_stage(config, tmp_path / "split", "fingerprint", StageAction.BUILD)

    assert captured == {
        "validation_fraction": 0.25,
        "test_fraction": 0.15,
        "seed": 7,
        "stratify_by_center": True,
        "prefer_patient_level": True,
    }
    assert result["strategy"] == "adapter_specific"
    assert result["counts"] == {"train": 1, "val": 1, "test": 1}
    assert DatasetSplit.load(tmp_path / "split/custom_combined_split.json") == expected


@pytest.mark.parametrize("command", ("train-rl", "run"))
def test_rl_pipeline_commands_enable_final_test_evaluation_by_default(command: str) -> None:
    parser = cli.build_parser()

    enabled = parser.parse_args((command, "--config", "config.yaml"))
    disabled = parser.parse_args((command, "--config", "config.yaml", "--no-final-test-eval"))

    assert enabled.final_test_eval is True
    assert disabled.final_test_eval is False


def test_eval_rl_infers_incomplete_run_from_saved_config_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / ("a" * 64)
    (run_dir / "checkpoints").mkdir(parents=True)
    config_path = run_dir / "config.yaml"
    config_path.write_text("workspace: {}\n", encoding="utf-8")
    atomic_json_dump(
        run_dir / "manifest.json",
        {
            "artifact_type": "ppo",
            "fingerprint": run_dir.name,
            "status": "incomplete",
        },
    )
    captured: dict[str, Any] = {}

    def evaluate(*args, **kwargs) -> dict[str, str]:
        captured.update(args=args, kwargs=kwargs)
        return {"status": "mocked"}

    monkeypatch.setattr(cli, "load_config", lambda _: object())
    monkeypatch.setattr(cli, "_evaluate_rl_command", evaluate)
    arguments = cli.build_parser().parse_args(
        (
            "eval-rl",
            "--config",
            str(config_path),
            "--split",
            "test",
            "--allow-test",
            "--test-artifacts",
            "require",
        )
    )

    assert cli._dispatch(arguments) == 0
    assert captured["kwargs"]["run_directory"] == run_dir.resolve()
    assert captured["kwargs"]["test_artifact_mode"] == "require"


def test_eval_rl_explicit_run_directory_overrides_saved_config_parent(tmp_path: Path) -> None:
    parser = cli.build_parser()
    explicit = tmp_path / "chosen-run"

    arguments = parser.parse_args(
        ("eval-rl", "--config", "config.yaml", "--run-dir", str(explicit))
    )

    assert arguments.run_dir == explicit
    assert arguments.test_artifacts == "auto"


def test_prepare_test_parser_requires_explicit_test_authorization() -> None:
    arguments = cli.build_parser().parse_args(
        ("prepare-test", "--config", "config.yaml", "--allow-test")
    )

    assert arguments.command == "prepare-test"
    assert arguments.allow_test is True


def test_prepare_test_dispatch_uses_explicit_prepare_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(cli, "load_config", lambda _: "loaded-config")

    def prepare(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return {"status": "complete"}

    monkeypatch.setattr(cli, "_prepare_authorized_test_preprocessing", prepare)
    arguments = cli.build_parser().parse_args(
        ("prepare-test", "--config", "config.yaml", "--allow-test")
    )

    assert cli._dispatch(arguments) == 0
    assert captured == {
        "args": ("loaded-config",),
        "kwargs": {"mode": "prepare"},
    }


def test_prepare_test_dispatch_rejects_missing_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "load_config", lambda _: object())
    monkeypatch.setattr(
        cli,
        "_prepare_authorized_test_preprocessing",
        lambda *_args, **_kwargs: pytest.fail("preparation ran before the test guard"),
    )
    arguments = cli.build_parser().parse_args(("prepare-test", "--config", "config.yaml"))

    with pytest.raises(TestSplitAccessError, match="--allow-test"):
        cli._dispatch(arguments)


def test_eval_baselines_parser_defaults_match_comparison_protocol() -> None:
    arguments = cli.build_parser().parse_args(("eval-baselines", "--config", "config.yaml"))

    assert tuple(arguments.navigators) == ("random", "raster")
    assert tuple(arguments.thresholds) == (0.5, 0.9)
    assert arguments.confidence_min_evidence == 32
    assert tuple(arguments.step_budgets) == (1, 4, 8, 16, 32, 64, 128, 256, 500)
    assert arguments.confidence_max_total_actions == 500
    assert arguments.random_seed is None
    assert arguments.output is None
    assert arguments.test_artifacts == "auto"


def test_eval_baselines_parser_accepts_explicit_protocol_and_output(tmp_path: Path) -> None:
    output = tmp_path / "comparison.json"
    arguments = cli.build_parser().parse_args(
        (
            "eval-baselines",
            "--config",
            "config.yaml",
            "--split",
            "test",
            "--allow-test",
            "--test-artifacts",
            "require",
            "--navigators",
            "raster",
            "--thresholds",
            "0.75",
            "--confidence-min-evidence",
            "8",
            "--step-budgets",
            "2",
            "16",
            "--confidence-max-total-actions",
            "300",
            "--random-seed",
            "17",
            "--output",
            str(output),
        )
    )

    assert arguments.split == "test"
    assert arguments.allow_test is True
    assert arguments.test_artifacts == "require"
    assert arguments.navigators == ["raster"]
    assert arguments.thresholds == [0.75]
    assert arguments.confidence_min_evidence == 8
    assert arguments.step_budgets == [2, 16]
    assert arguments.confidence_max_total_actions == 300
    assert arguments.random_seed == 17
    assert arguments.output == output


def test_evaluation_context_resolves_shared_dependencies_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    split = object()
    calls = {"fingerprints": 0, "paths": 0, "split": 0, "dimension": 0}

    def fingerprints(_config):
        calls["fingerprints"] += 1
        return {Stage.PREPROCESSING: "preprocessing"}

    def paths(_config, _fingerprints):
        calls["paths"] += 1
        return {Stage.PREPROCESSING: tmp_path / "processed/manifest.json"}

    def load_split(_config):
        calls["split"] += 1
        return split

    def dimension(_config, *, split):
        calls["dimension"] += 1
        assert split is not None
        return 1536

    monkeypatch.setattr(cli, "stage_fingerprints", fingerprints)
    monkeypatch.setattr(cli, "artifact_paths", paths)
    monkeypatch.setattr(cli, "_load_verified_split", load_split)
    monkeypatch.setattr(cli, "_active_embedding_dimension", dimension)
    context = cli._EvaluationContext(config)

    assert context.fingerprints is context.fingerprints
    assert context.paths is context.paths
    assert context.split is context.split
    assert context.embedding_dimension == context.embedding_dimension == 1536
    assert calls == {"fingerprints": 1, "paths": 1, "split": 1, "dimension": 1}


def test_belief_data_stage_passes_only_development_ids_and_disables_wsi_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    records = (
        SlideRecord(
            slide_id="normal_001",
            slide_path=tmp_path / "normal_001.tif",
            label=0,
            label_name="negative",
            official_cohort="development",
        ),
        SlideRecord(
            slide_id="tumor_001",
            slide_path=tmp_path / "tumor_001.tif",
            label=1,
            label_name="macro",
            official_cohort="development",
        ),
        SlideRecord(
            slide_id="test_001",
            slide_path=tmp_path / "must-not-open-test_001.tif",
            label=1,
            label_name="micro",
            official_cohort="test",
        ),
    )
    captured: dict[str, Any] = {}

    class Adapter:
        def iter_slides(self):
            return iter(records)

    class Reader:
        pass

    class Validator:
        def __init__(self, adapter, *, reader, slide_inspector, options) -> None:
            captured.update(
                adapter=adapter,
                reader=reader,
                slide_inspector=slide_inspector,
                options=options,
            )

        def validate(self, **kwargs) -> ValidationReport:
            captured["validate"] = kwargs
            return ValidationReport(
                fingerprint="development-only",
                complete=True,
                checked_files=8,
                feature_dimension=1536,
            )

    import wsi_lab.data.embeddings.hdf5 as hdf5_module
    import wsi_lab.data.validation as validation_module

    monkeypatch.setattr(cli, "_adapter", lambda _: Adapter())
    monkeypatch.setattr(
        cli,
        "_load_verified_split",
        lambda _: SimpleNamespace(
            ids=lambda name: {
                "train": ("normal_001",),
                "val": ("tumor_001",),
                "test": ("test_001",),
            }[name]
        ),
    )
    monkeypatch.setattr(hdf5_module, "AtlasPatchHDF5Reader", Reader)
    monkeypatch.setattr(validation_module, "DatasetValidator", Validator)

    result = cli._belief_data_stage(
        config,
        tmp_path / "belief-data",
        "fingerprint",
        StageAction.BUILD,
    )

    options = captured["options"]
    validate = captured["validate"]
    assert captured["slide_inspector"] is None
    assert options.check_wsi_existence is False
    assert options.check_wsi_readability is False
    assert validate["slide_ids"] == ("normal_001", "tumor_001")
    assert validate["resume"] is True
    assert validate["raise_on_error"] is True
    assert "test_001" not in validate["slide_ids"]
    assert result["development_slides"] == 2
    assert result["official_test_opened"] is False


def _fake_module(monkeypatch: pytest.MonkeyPatch, name: str, **members: object) -> ModuleType:
    module = ModuleType(name)
    for member_name, value in members.items():
        setattr(module, member_name, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.mark.parametrize(("evaluation", "expected"), ((False, True), (True, False)))
def test_production_environment_enables_augmentation_only_for_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evaluation: bool,
    expected: bool,
) -> None:
    config = _config(tmp_path)
    config.environment.geometric_augmentation = True
    config.environment.photometric_augmentation = True
    split = object()
    source = object()

    class EpisodeSource:
        @classmethod
        def from_config(cls, *args, **kwargs):
            del cls, args, kwargs
            return source

    class Accumulator:
        def __init__(self, model, *, temperature) -> None:
            del model, temperature

    _fake_module(
        monkeypatch,
        "wsi_lab.preprocessing.pipeline",
        ProcessedEpisodeSource=EpisodeSource,
    )
    _fake_module(
        monkeypatch,
        "wsi_lab.belief.accumulator",
        BeliefAccumulator=Accumulator,
    )
    _fake_module(
        monkeypatch,
        "wsi_lab.envs.wsi_env",
        WSIEnv=lambda episode_source, belief, **kwargs: {
            "source": episode_source,
            "belief": belief,
            **kwargs,
        },
    )
    monkeypatch.setattr(cli, "_load_verified_split", lambda _: split)
    monkeypatch.setattr(cli, "_preprocessing_store", lambda _: object())
    monkeypatch.setattr(cli, "_load_active_belief", lambda *_args, **_kwargs: (object(), 1.0, {}))

    environment = cli._make_production_environment(
        config,
        split_name="train" if not evaluation else "val",
        evaluation=evaluation,
        max_episode_steps_override=701,
        min_diagnostic_evidence_override=0,
    )

    assert environment["source"] is source
    assert environment["geometric_augmentation"] is expected
    assert environment["photometric_augmentation"] is expected
    assert environment["max_episode_steps"] == 701
    assert environment["min_diagnostic_evidence"] == 0
    assert environment["lightweight_evaluation"] is False


def test_test_environment_receives_immutable_artifact_expectations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    split = object()
    source = object()
    expectations = {"test_001": {"input_fingerprint": "prepared-generation"}}
    captured: dict[str, Any] = {}

    class EpisodeSource:
        @classmethod
        def from_config(cls, *args, **kwargs):
            captured.update(args=args, kwargs=kwargs)
            return source

    class Accumulator:
        def __init__(self, model, *, temperature) -> None:
            del model, temperature

    _fake_module(
        monkeypatch,
        "wsi_lab.preprocessing.pipeline",
        ProcessedEpisodeSource=EpisodeSource,
    )
    _fake_module(
        monkeypatch,
        "wsi_lab.belief.accumulator",
        BeliefAccumulator=Accumulator,
    )
    _fake_module(
        monkeypatch,
        "wsi_lab.envs.wsi_env",
        WSIEnv=lambda episode_source, belief, **kwargs: {
            "source": episode_source,
            "belief": belief,
            **kwargs,
        },
    )
    monkeypatch.setattr(cli, "_preprocessing_store", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        cli,
        "_load_active_belief",
        lambda *_args, **_kwargs: (object(), 1.0, {}),
    )
    context = SimpleNamespace(split=split, test_artifact_entries=expectations)

    environment = cli._make_production_environment(
        config,
        split_name="test",
        evaluation=True,
        allow_test=True,
        context=context,  # type: ignore[arg-type]
    )

    assert environment["source"] is source
    assert captured["kwargs"]["allow_test"] is True
    assert captured["kwargs"]["artifact_expectations"] is expectations


@pytest.mark.parametrize(
    ("action", "expected_resume"),
    [
        (StageAction.TRAIN, False),
        (StageAction.REBUILD, False),
        (StageAction.RESUME, True),
    ],
)
def test_ppo_stage_resumes_only_for_explicit_resume_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action: StageAction,
    expected_resume: bool,
) -> None:
    config = _config(tmp_path)
    config.logging.tensorboard = False
    captured: dict[str, Any] = {}

    class Model:
        pass

    class Trainer:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def train(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                global_step=8,
                updates=1,
                last_metrics=None,
                completed_episodes=3,
            )

        def close(self) -> None:
            pass

    class Tracker:
        def __init__(self, *_args, **_kwargs) -> None:
            self.best = None

    _fake_module(
        monkeypatch,
        "wsi_lab.rl.best_checkpoint",
        BestValidationCheckpointTracker=Tracker,
    )
    _fake_module(
        monkeypatch,
        "wsi_lab.rl.evaluation",
        evaluate_baseline=object(),
        evaluate_policy=object(),
    )
    _fake_module(
        monkeypatch,
        "wsi_lab.rl.networks",
        RecurrentActorCritic=Model,
    )
    _fake_module(
        monkeypatch,
        "wsi_lab.rl.trainer",
        PPOTrainer=Trainer,
        make_vector_env=lambda factories, *, backend: (factories, backend),
    )
    monkeypatch.setattr(cli, "_ppo_dependencies", lambda _: {"belief": "current"})

    cli._ppo_stage(config, tmp_path / "ppo", "fingerprint", action)

    assert captured["resume_latest"] is expected_resume


@pytest.mark.parametrize(
    ("action", "expected_resume_existing"),
    ((StageAction.TRAIN, False), (StageAction.RESUME, True)),
)
def test_ppo_validation_tracks_and_records_the_best_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action: StageAction,
    expected_resume_existing: bool,
) -> None:
    config = _config(tmp_path)
    config.logging.tensorboard = False
    captured: dict[str, Any] = {}
    best_record = SimpleNamespace(
        metric_value=0.82,
        to_dict=lambda: {"metric_name": "auroc", "metric_value": 0.82},
    )
    decision = SimpleNamespace(
        improved=True,
        best=best_record,
        to_dict=lambda: {"improved": True, "best": best_record.to_dict()},
    )

    class Tracker:
        def __init__(self, checkpoint_dir: Path, *, resume_existing: bool) -> None:
            captured["checkpoint_dir"] = checkpoint_dir
            captured["resume_existing"] = resume_existing
            self.best = None

        def consider(self, metrics, **kwargs):
            captured["metrics"] = metrics
            captured["consider"] = kwargs
            self.best = best_record
            return decision

    class Model:
        pass

    class Trainer:
        def __init__(self, _vector, model, trainer_config, **kwargs) -> None:
            self.model = model
            self.config = trainer_config
            self.optimizer = object()
            self.dependencies = kwargs["dependencies"]
            self.global_step = 100_352
            self.update_number = 98
            self.device = "cpu"

        def train(self, **kwargs):
            kwargs["validation_callback"](self)
            return SimpleNamespace(
                global_step=self.global_step,
                updates=self.update_number,
                last_metrics=None,
                completed_episodes=12,
            )

        def close(self) -> None:
            pass

    class Environment:
        def close(self) -> None:
            captured["environment_closed"] = True

    evaluation = SimpleNamespace(
        metrics={"accuracy": 0.75, "auroc": 0.82},
        to_dict=lambda: {"episodes": [], "metrics": {"accuracy": 0.75, "auroc": 0.82}},
    )
    _fake_module(
        monkeypatch,
        "wsi_lab.rl.best_checkpoint",
        BestValidationCheckpointTracker=Tracker,
    )
    _fake_module(
        monkeypatch,
        "wsi_lab.rl.evaluation",
        evaluate_baseline=object(),
        evaluate_policy=lambda *_args, **_kw: evaluation,
    )
    _fake_module(monkeypatch, "wsi_lab.rl.networks", RecurrentActorCritic=Model)
    _fake_module(
        monkeypatch,
        "wsi_lab.rl.trainer",
        PPOTrainer=Trainer,
        make_vector_env=lambda factories, *, backend: (factories, backend),
    )
    monkeypatch.setattr(cli, "_ppo_dependencies", lambda _: {"belief": "current"})
    monkeypatch.setattr(cli, "_make_production_environment", lambda *_args, **_kw: Environment())
    monkeypatch.setattr(
        cli,
        "_load_verified_split",
        lambda _: SimpleNamespace(ids=lambda split_name: ("slide-a", "slide-b")),
    )

    result = cli._ppo_stage(config, tmp_path / "ppo", "fingerprint", action)

    assert captured["metrics"] == evaluation.metrics
    assert captured["resume_existing"] is expected_resume_existing
    assert captured["consider"]["global_step"] == 100_352
    assert captured["consider"]["update"] == 98
    assert captured["environment_closed"] is True
    validation = json.loads(
        (tmp_path / "ppo" / "validation_000000100352.json").read_text(encoding="utf-8")
    )
    assert validation["checkpoint_tracking"]["improved"] is True
    assert result["completed_episodes"] == 12
    assert result["best_validation"]["metric_value"] == pytest.approx(0.82)


def test_ppo_checkpoint_load_resolves_auto_before_passing_map_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    config.rl.device = "auto"
    run_dir = tmp_path / "incomplete-run"
    checkpoint = run_dir / "checkpoints" / "checkpoint_000000000008.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    atomic_json_dump(
        run_dir / "manifest.json",
        {"artifact_type": "ppo", "fingerprint": "incomplete-run", "status": "incomplete"},
    )
    loaded = SimpleNamespace(path=checkpoint)
    captured: dict[str, Any] = {}

    def load(candidate, model, **kwargs):
        captured.update(candidate=candidate, model=model, **kwargs)
        return loaded

    class ConvertedConfig:
        def to_dict(self) -> dict[str, object]:
            return {"device": "auto"}

    class PPOConfig:
        @classmethod
        def from_object(cls, value):
            assert value is config.rl
            return ConvertedConfig()

    model = object()

    def candidates(directory: Path) -> list[Path]:
        captured["checkpoint_directory"] = directory
        return [checkpoint]

    _fake_module(
        monkeypatch,
        "wsi_lab.rl.checkpoints",
        checkpoint_candidates=candidates,
        load_ppo_checkpoint=load,
    )
    _fake_module(
        monkeypatch,
        "wsi_lab.rl.networks",
        RecurrentActorCritic=lambda: model,
    )
    _fake_module(
        monkeypatch,
        "wsi_lab.rl.recurrent_ppo",
        PPOConfig=PPOConfig,
    )
    _fake_module(
        monkeypatch,
        "wsi_lab.rl.trainer",
        RESUME_CONFIG_FIELDS=(),
        resolve_device=lambda requested: f"resolved:{requested}",
    )
    monkeypatch.setattr(cli, "_ppo_dependencies", lambda _: {})

    observed_model, observed_loaded = cli._load_ppo_model(config, run_directory=run_dir)

    assert observed_model is model
    assert observed_loaded is loaded
    assert captured["checkpoint_directory"] == checkpoint.parent.resolve()
    assert captured["map_location"] == "resolved:auto"


def test_automatic_final_test_evaluation_requests_best_and_latest_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    config.logging.tensorboard = False
    run_dir = tmp_path / "run"
    captured: dict[str, Any] = {}

    monkeypatch.setattr(cli, "stage_fingerprints", lambda _: {Stage.PPO: "ppo"})
    monkeypatch.setattr(
        cli,
        "artifact_paths",
        lambda *_: {Stage.PPO: run_dir / "manifest.json"},
    )

    def evaluate(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return {
            "evaluations": {
                "best_validation": {"status": "complete", "metrics": {"accuracy": 1.0}},
                "latest": {"status": "complete", "metrics": {"accuracy": 0.5}},
            }
        }

    monkeypatch.setattr(cli, "_evaluate_rl_command", evaluate)

    result = cli._automatic_final_test_evaluation(config)

    assert set(result["evaluations"]) == {"best_validation", "latest"}
    assert captured["args"][1:] == ("test",)
    assert captured["kwargs"]["allow_test"] is True
    assert captured["kwargs"]["run_directory"] == run_dir
    assert "checkpoint_selection" not in captured["kwargs"]
    assert "checkpoint_path" not in captured["kwargs"]


def test_automatic_final_test_evaluation_logs_numeric_test_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch.utils.tensorboard as tensorboard_module

    config = _config(tmp_path)
    run_dir = tmp_path / "run"
    best = run_dir / "checkpoints" / "best_validation.pt"
    best.parent.mkdir(parents=True)
    best.touch()
    captured: dict[str, Any] = {}

    class Writer:
        def __init__(self, *, log_dir: str) -> None:
            captured["log_dir"] = log_dir
            captured["events"] = []

        def add_scalar(self, name: str, value: float, step: int) -> None:
            captured["events"].append((name, float(value), int(step)))

        def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(cli, "stage_fingerprints", lambda _: {Stage.PPO: "ppo"})
    monkeypatch.setattr(
        cli,
        "artifact_paths",
        lambda *_: {Stage.PPO: run_dir / "manifest.json"},
    )
    monkeypatch.setattr(
        cli,
        "_evaluate_rl_command",
        lambda *_args, **_kwargs: {
            "evaluations": {
                "best_validation": {
                    "status": "complete",
                    "checkpoint_global_step": 1234,
                    "metrics": {
                        "accuracy": 0.75,
                        "auroc": 0.81,
                        "has_both_classes": True,
                        "magnification_usage": {"20": 9},
                    },
                    "ppo_patches_belief_metrics": {
                        "accuracy": 0.8,
                        "auroc": 0.84,
                    },
                },
                "latest": {
                    "status": "complete",
                    "checkpoint_global_step": 5678,
                    "metrics": {"accuracy": 0.7, "auroc": 0.79},
                    "ppo_patches_belief_metrics": {
                        "accuracy": 0.72,
                        "auroc": 0.8,
                    },
                },
            },
        },
    )
    monkeypatch.setattr(tensorboard_module, "SummaryWriter", Writer)

    cli._automatic_final_test_evaluation(config)

    assert captured["log_dir"] == str(run_dir / "tensorboard")
    assert captured["events"] == [
        ("test/best_validation/accuracy", 0.75, 1234),
        ("test/best_validation/auroc", 0.81, 1234),
        ("test/best_validation/ppo_patches_belief/accuracy", 0.8, 1234),
        ("test/best_validation/ppo_patches_belief/auroc", 0.84, 1234),
        ("test/latest/accuracy", 0.7, 5678),
        ("test/latest/auroc", 0.79, 5678),
        ("test/latest/ppo_patches_belief/accuracy", 0.72, 5678),
        ("test/latest/ppo_patches_belief/auroc", 0.8, 5678),
    ]
    assert captured["closed"] is True


def test_automatic_final_test_evaluation_routes_metrics_to_shared_tracking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    config.logging.tensorboard = False
    run_dir = tmp_path / "run"
    fingerprint = "ppo-fingerprint"
    captured: dict[str, Any] = {"events": []}

    class Tracking:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            captured["closed"] = True

        def add_scalars(self, metrics, step) -> None:
            captured["events"].append((metrics, step))

    def create_tracking(config_arg, directory, spec):
        captured.update(config=config_arg, directory=directory, spec=spec)
        return Tracking()

    monkeypatch.setattr(cli, "stage_fingerprints", lambda _: {Stage.PPO: fingerprint})
    monkeypatch.setattr(
        cli,
        "artifact_paths",
        lambda *_: {Stage.PPO: run_dir / "manifest.json"},
    )
    monkeypatch.setattr(cli, "create_tracking", create_tracking)
    monkeypatch.setattr(
        cli,
        "_evaluate_rl_command",
        lambda *_args, **_kwargs: {
            "evaluations": {
                "best_validation": {
                    "status": "complete",
                    "checkpoint_global_step": 1234,
                    "metrics": {"accuracy": 0.75, "has_both_classes": True},
                    "ppo_patches_belief_metrics": {"accuracy": 0.8},
                },
                "latest": {"status": "unavailable"},
            }
        },
    )

    cli._automatic_final_test_evaluation(config)

    assert captured["config"] is config
    assert captured["directory"] == run_dir
    assert captured["spec"].job_type == "final-test"
    assert captured["spec"].fingerprint == fingerprint
    assert captured["spec"].group == f"ppo-{fingerprint}"
    assert captured["events"] == [
        (
            {
                "test/best_validation/accuracy": 0.75,
                "test/best_validation/ppo_patches_belief/accuracy": 0.8,
            },
            1234,
        )
    ]
    assert captured["closed"] is True


def test_ppo_test_evaluation_reports_best_and_latest_in_one_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    best = (checkpoint_dir / "best_validation.pt").resolve()
    latest = (checkpoint_dir / "checkpoint_000000000016.pt").resolve()
    best.touch()
    latest.touch()
    calls: list[tuple[str, Path | None]] = []
    contexts: list[cli._EvaluationContext] = []
    preprocessing_calls = 0

    def preprocess(_config: AppConfig, **_kwargs: Any) -> dict[str, str]:
        nonlocal preprocessing_calls
        preprocessing_calls += 1
        contexts.append(_kwargs["context"])
        return {"status": "complete"}

    def evaluate(*_args, **kwargs) -> dict[str, Any]:
        selection = str(kwargs["checkpoint_selection"])
        selected_path = kwargs["checkpoint_path"]
        calls.append((selection, selected_path))
        contexts.append(kwargs["context"])
        path = best if selection == "best_validation" else latest
        step = 8 if selection == "best_validation" else 16
        return {
            "episodes": [{"slide_id": f"{selection}-slide"}],
            "metrics": {"accuracy": 0.8 if selection == "best_validation" else 0.7},
            "ppo_patches_belief_metrics": {
                "accuracy": 0.9 if selection == "best_validation" else 0.85
            },
            "checkpoint_selection": selection,
            "checkpoint_global_step": step,
            "checkpoint_path": str(path),
            "checkpoint_provenance": {
                "path": str(path),
                "global_step": step,
                "selection": selection,
                "dependencies": {},
                "best_validation": None,
            },
        }

    monkeypatch.setattr(cli, "_prepare_authorized_test_preprocessing", preprocess)
    monkeypatch.setattr(
        cli,
        "_load_verified_split",
        lambda _: SimpleNamespace(ids=lambda _split_name: ("slide-a", "slide-b")),
    )
    monkeypatch.setattr(cli, "_evaluate_ppo_checkpoint", evaluate)

    result = cli._evaluate_rl_command(
        config,
        "test",
        allow_test=True,
        run_directory=run_dir,
    )

    assert preprocessing_calls == 1
    assert calls == [("latest", None), ("best_validation", best)]
    assert len(contexts) == 3
    assert all(context is contexts[0] for context in contexts)
    assert result["schema_version"] == 2
    assert result["split"] == "test"
    assert result["evaluations"]["best_validation"]["status"] == "complete"
    assert result["evaluations"]["best_validation"]["metrics"]["accuracy"] == 0.8
    assert result["evaluations"]["best_validation"]["ppo_patches_belief_metrics"]["accuracy"] == 0.9
    assert result["evaluations"]["latest"]["status"] == "complete"
    assert result["evaluations"]["latest"]["metrics"]["accuracy"] == 0.7
    assert result["evaluations"]["latest"]["ppo_patches_belief_metrics"]["accuracy"] == 0.85

    persisted = json.loads((run_dir / "evaluation_ppo_test.json").read_text(encoding="utf-8"))
    assert persisted == {
        "schema_version": 2,
        "policy": "ppo",
        "split": "test",
        "evaluations": result["evaluations"],
        "test_preprocessing": {"status": "complete"},
    }


def test_ppo_validation_evaluation_remains_latest_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    run_dir = tmp_path / "run"
    calls: list[dict[str, Any]] = []

    def evaluate(*_args, **kwargs) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "episodes": [],
            "metrics": {"accuracy": 0.75},
            "checkpoint_selection": kwargs["checkpoint_selection"],
            "checkpoint_global_step": 16,
            "checkpoint_path": str(run_dir / "checkpoints/checkpoint_000000000016.pt"),
            "checkpoint_provenance": {},
        }

    monkeypatch.setattr(
        cli,
        "_load_verified_split",
        lambda _: SimpleNamespace(ids=lambda _split_name: ("slide-a",)),
    )
    monkeypatch.setattr(cli, "_evaluate_ppo_checkpoint", evaluate)

    result = cli._evaluate_rl_command(
        config,
        "val",
        allow_test=False,
        run_directory=run_dir,
    )

    assert len(calls) == 1
    assert calls[0]["checkpoint_path"] is None
    assert calls[0]["checkpoint_selection"] == "latest"
    assert "evaluations" not in result
    assert result["metrics"]["accuracy"] == 0.75


@pytest.mark.parametrize("best_exists", (False, True))
def test_unavailable_best_checkpoint_does_not_suppress_latest_test_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    best_exists: bool,
) -> None:
    config = _config(tmp_path)
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    best = (checkpoint_dir / "best_validation.pt").resolve()
    if best_exists:
        best.write_bytes(b"corrupt")

    def evaluate(*_args, **kwargs) -> dict[str, Any]:
        selection = str(kwargs["checkpoint_selection"])
        if selection == "best_validation":
            raise ValueError("corrupt best checkpoint")
        return {
            "episodes": [],
            "metrics": {"accuracy": 0.625},
            "checkpoint_selection": "latest",
            "checkpoint_global_step": 24,
            "checkpoint_path": str(checkpoint_dir / "checkpoint_000000000024.pt"),
            "checkpoint_provenance": {},
        }

    monkeypatch.setattr(
        cli,
        "_prepare_authorized_test_preprocessing",
        lambda *_args, **_kwargs: {"status": "complete"},
    )
    monkeypatch.setattr(
        cli,
        "_load_verified_split",
        lambda _: SimpleNamespace(ids=lambda _split_name: ("slide-a",)),
    )
    monkeypatch.setattr(cli, "_evaluate_ppo_checkpoint", evaluate)

    result = cli._evaluate_rl_command(
        config,
        "test",
        allow_test=True,
        run_directory=run_dir,
    )

    best_result = result["evaluations"]["best_validation"]
    assert best_result["status"] == "unavailable"
    assert best_result["reason"] == (
        "checkpoint_unavailable" if best_exists else "checkpoint_not_found"
    )
    assert result["evaluations"]["latest"]["status"] == "complete"
    assert result["evaluations"]["latest"]["checkpoint_global_step"] == 24


def test_ppo_evaluation_persists_resolved_checkpoint_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import wsi_lab.rl.evaluation as evaluation_module

    config = _config(tmp_path)
    config.rl.device = "cpu"
    config.belief.ece_bins = 7
    run_dir = tmp_path / "run"
    checkpoint = (run_dir / "checkpoints" / "best_validation.pt").resolve()
    loaded = SimpleNamespace(
        path=checkpoint,
        global_step=100_352,
        dependencies={"belief_fingerprint": "belief-v1"},
        extra={
            "best_validation": {
                "metric_name": "auroc",
                "metric_value": 0.82,
                "global_step": 100_352,
            }
        },
    )

    class Environment:
        def close(self) -> None:
            pass

    evaluation = SimpleNamespace(
        episodes=("episode",),
        to_dict=lambda: {
            "episodes": [],
            "metrics": {"accuracy": 0.75, "auroc": 0.82},
        },
    )
    aggregate_calls: list[tuple[object, int]] = []

    def aggregate_belief(episodes, *, ece_bins):
        aggregate_calls.append((episodes, ece_bins))
        return {"accuracy": 0.625, "auroc": 0.73}

    monkeypatch.setattr(cli, "_make_production_environment", lambda *_args, **_kw: Environment())
    monkeypatch.setattr(
        cli,
        "_prepare_authorized_test_preprocessing",
        lambda *_args, **_kwargs: {"status": "complete"},
    )
    monkeypatch.setattr(
        cli,
        "_load_verified_split",
        lambda _: SimpleNamespace(ids=lambda split_name: ("slide-a", "slide-b")),
    )
    monkeypatch.setattr(cli, "_load_ppo_model", lambda *_args, **_kw: (object(), loaded))
    monkeypatch.setattr(evaluation_module, "evaluate_policy", lambda *_args, **_kw: evaluation)
    monkeypatch.setattr(
        evaluation_module,
        "aggregate_ppo_patches_belief_metrics",
        aggregate_belief,
    )
    monkeypatch.setattr(cli, "stage_fingerprints", lambda _: {Stage.PPO: "ppo"})
    monkeypatch.setattr(
        cli,
        "artifact_paths",
        lambda *_: {Stage.PPO: run_dir / "manifest.json"},
    )

    result = cli._evaluate_rl_command(
        config,
        "test",
        allow_test=True,
        checkpoint_path=checkpoint,
        checkpoint_selection="best_validation",
    )

    persisted = json.loads((run_dir / "evaluation_ppo_test.json").read_text(encoding="utf-8"))
    expected = {
        "path": str(checkpoint),
        "global_step": 100_352,
        "selection": "best_validation",
        "dependencies": {"belief_fingerprint": "belief-v1"},
        "best_validation": loaded.extra["best_validation"],
    }
    assert persisted["checkpoint_path"] == str(checkpoint)
    assert persisted["checkpoint_provenance"] == expected
    assert persisted["ppo_patches_belief_metrics"] == {"accuracy": 0.625, "auroc": 0.73}
    assert aggregate_calls == [(evaluation.episodes, 7)]
    assert result["checkpoint"] == str(checkpoint)
    assert result["checkpoint_provenance"] == expected


def test_belief_evaluation_rejects_test_before_loading_records_or_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch_utils_data = _fake_module(monkeypatch, "torch.utils.data", DataLoader=object)
    torch_utils = _fake_module(monkeypatch, "torch.utils", __path__=[], data=torch_utils_data)
    _fake_module(monkeypatch, "torch", __path__=[], utils=torch_utils)
    belief_package = _fake_module(monkeypatch, "wsi_lab.belief", __path__=[])
    monkeypatch.setattr(wsi_lab, "belief", belief_package, raising=False)
    _fake_module(
        monkeypatch,
        "wsi_lab.belief.dataset",
        ValidationBagDataset=object,
        collate_belief_bags=object(),
    )
    _fake_module(monkeypatch, "wsi_lab.belief.evaluation", evaluate_belief=object())
    _fake_module(
        monkeypatch,
        "wsi_lab.belief.trainer",
        load_calibrated_belief=object(),
    )
    monkeypatch.setattr(
        cli,
        "_belief_records",
        lambda *_: pytest.fail("test embeddings were accessed before the guard"),
    )

    with pytest.raises(TestSplitAccessError, match="--allow-test"):
        cli._evaluate_belief_command(object(), "test", allow_test=False)  # type: ignore[arg-type]


def test_rl_evaluation_rejects_test_before_loading_a_ppo_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rl_package = _fake_module(monkeypatch, "wsi_lab.rl", __path__=[])
    monkeypatch.setattr(wsi_lab, "rl", rl_package, raising=False)
    _fake_module(monkeypatch, "wsi_lab.rl.evaluation", evaluate_policy=object())
    monkeypatch.setattr(
        cli,
        "_load_ppo_model",
        lambda *_: pytest.fail("a PPO checkpoint was accessed before the guard"),
    )

    with pytest.raises(TestSplitAccessError, match="--allow-test"):
        cli._evaluate_rl_command(object(), "test", allow_test=False)  # type: ignore[arg-type]


def test_authorized_rl_test_evaluation_routes_through_separate_preprocessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rl_package = _fake_module(monkeypatch, "wsi_lab.rl", __path__=[])
    monkeypatch.setattr(wsi_lab, "rl", rl_package, raising=False)
    _fake_module(
        monkeypatch,
        "wsi_lab.rl.evaluation",
        evaluate_baseline=object(),
        evaluate_policy=object(),
    )

    def prepare(*_args, **kwargs):
        assert kwargs["mode"] == "require"
        assert isinstance(kwargs["context"], cli._EvaluationContext)
        raise RuntimeError("authorized test preprocessing")

    monkeypatch.setattr(cli, "_prepare_authorized_test_preprocessing", prepare)
    monkeypatch.setattr(
        cli,
        "_make_production_environment",
        lambda *_args, **_kwargs: pytest.fail("environment opened before test preprocessing"),
    )

    with pytest.raises(RuntimeError, match="authorized test preprocessing"):
        cli._evaluate_rl_command(
            object(),
            "test",
            allow_test=True,  # type: ignore[arg-type]
            test_artifact_mode="require",
        )


def test_baseline_sweep_rejects_test_before_importing_the_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "wsi_lab.baselines.sweep", raising=False)

    with pytest.raises(TestSplitAccessError, match="--allow-test"):
        cli._evaluate_baselines_command(
            object(),  # type: ignore[arg-type]
            "test",
            allow_test=False,
        )


def test_baseline_sweep_wires_protocol_environment_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    config.rl.seed = 23
    belief_dir = tmp_path / "belief"
    belief_dir.mkdir()
    checkpoint_bytes = b"calibrated-checkpoint"
    (belief_dir / "calibrated_checkpoint.pt").write_bytes(checkpoint_bytes)
    atomic_json_dump(
        belief_dir / "manifest.json",
        {
            "metadata": {
                "architecture": {"embedding_dim": 1536, "hidden_dim": 512},
                "dependencies": {"belief_stage": "belief-fingerprint"},
                "temperature": 1.1,
            }
        },
    )
    atomic_json_dump(belief_dir / "temperature.json", {"temperature": 1.25})
    captured: dict[str, Any] = {}

    class Protocol:
        def __init__(self, **kwargs) -> None:
            captured["protocol"] = kwargs
            for name, value in kwargs.items():
                setattr(self, name, value)

    def evaluate(environment_factory, navigator_names, episodes, protocol):
        captured["navigator_names"] = navigator_names
        captured["episodes"] = episodes
        captured["protocol_object"] = protocol
        captured["environment"] = environment_factory()
        return {
            "schema_version": 2,
            "protocol": {"source": "core"},
            "navigators": {"random": {}, "raster": {}},
        }

    _fake_module(
        monkeypatch,
        "wsi_lab.baselines.sweep",
        SweepProtocol=Protocol,
        evaluate_navigation_sweep=evaluate,
    )
    monkeypatch.setattr(
        cli,
        "_prepare_authorized_test_preprocessing",
        lambda *_args, **_kwargs: {
            "status": "complete",
            "fingerprint": "authorized-test-preprocessing",
            "training_dependency": False,
        },
    )
    monkeypatch.setattr(
        cli,
        "_load_verified_split",
        lambda _: SimpleNamespace(
            ids=lambda name: ("test-a", "test-b") if name == "test" else (),
            assignment=lambda slide_id: SimpleNamespace(label={"test-a": 0, "test-b": 1}[slide_id]),
            split_fingerprint="verified-split",
        ),
    )

    def make_environment(*args, **kwargs):
        captured["environment_args"] = args
        captured["environment_kwargs"] = kwargs
        return object()

    monkeypatch.setattr(cli, "_make_production_environment", make_environment)
    monkeypatch.setattr(
        cli,
        "stage_fingerprints",
        lambda _: {
            Stage.DATASET: "dataset-fingerprint",
            Stage.PREPROCESSING: "preprocessing-fingerprint",
            Stage.BELIEF: "belief-fingerprint",
        },
    )
    monkeypatch.setattr(
        cli,
        "artifact_paths",
        lambda *_: {Stage.BELIEF: belief_dir / "manifest.json"},
    )

    result = cli._evaluate_baselines_command(
        config,
        "test",
        allow_test=True,
    )

    assert captured["navigator_names"] == ("random", "raster")
    assert captured["episodes"] == 2
    assert captured["protocol"] == {
        "confidence_thresholds": (0.5, 0.9),
        "step_budgets": (1, 4, 8, 16, 32, 64, 128, 256, 500),
        "confidence_min_evidence": 32,
        "confidence_max_total_actions": 500,
        "random_seed": 23,
        "ece_bins": 15,
    }
    assert captured["environment_args"] == (config,)
    assert isinstance(captured["environment_kwargs"].pop("context"), cli._EvaluationContext)
    assert captured["environment_kwargs"] == {
        "split_name": "test",
        "evaluation": True,
        "allow_test": True,
        "max_episode_steps_override": 501,
        "min_diagnostic_evidence_override": 0,
        "lightweight_evaluation_override": True,
    }
    assert result["evaluation_type"] == "navigation_baseline_sweep"
    assert result["split"] == "test"
    assert result["provenance"]["belief_fingerprint"] == "belief-fingerprint"
    assert result["provenance"]["split_fingerprint"] == "verified-split"
    assert result["provenance"]["test_preprocessing"]["training_dependency"] is False
    assert (
        result["provenance"]["authorized_test_preprocessing_fingerprint"]
        == "authorized-test-preprocessing"
    )
    assert result["provenance"]["split_counts"] == {
        "total": 2,
        "positive": 1,
        "negative": 1,
        "unique_slide_count": 2,
    }
    assert result["provenance"]["belief_checkpoint"] == {
        "path": str(belief_dir / "calibrated_checkpoint.pt"),
        "sha256": sha256(checkpoint_bytes).hexdigest(),
        "temperature": 1.25,
        "dependencies": {"belief_stage": "belief-fingerprint"},
        "architecture": {"embedding_dim": 1536, "hidden_dim": 512},
        "manifest": str(belief_dir / "manifest.json"),
        "temperature_file": str(belief_dir / "temperature.json"),
    }
    assert result["provenance"]["environment"] == {
        "configured": {
            "max_episode_steps": 500,
            "min_diagnostic_evidence": 32,
            "lightweight_evaluation": False,
        },
        "effective": {
            "max_episode_steps": 501,
            "min_diagnostic_evidence": 0,
            "lightweight_evaluation": True,
        },
    }
    output = belief_dir / "evaluation_test_navigation_baseline_sweep.json"
    assert result["output"] == str(output)
    assert json.loads(output.read_text(encoding="utf-8")) == {
        key: value for key, value in result.items() if key != "output"
    }


def test_eval_baselines_dispatch_forwards_explicit_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(cli, "load_config", lambda _: "loaded-config")

    def evaluate(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return {"status": "mocked"}

    monkeypatch.setattr(cli, "_evaluate_baselines_command", evaluate)
    output = tmp_path / "sweep.json"
    arguments = cli.build_parser().parse_args(
        (
            "eval-baselines",
            "--config",
            "config.yaml",
            "--navigators",
            "raster",
            "--thresholds",
            "0.8",
            "--confidence-min-evidence",
            "12",
            "--step-budgets",
            "4",
            "64",
            "--confidence-max-total-actions",
            "250",
            "--random-seed",
            "19",
            "--test-artifacts",
            "require",
            "--output",
            str(output),
        )
    )

    assert cli._dispatch(arguments) == 0
    assert captured["args"] == ("loaded-config", "val", False)
    assert captured["kwargs"] == {
        "navigator_names": ("raster",),
        "confidence_thresholds": (0.8,),
        "confidence_min_evidence": 12,
        "step_budgets": (4, 64),
        "confidence_max_total_actions": 250,
        "random_seed": 19,
        "output_path": output,
        "test_artifact_mode": "require",
    }


def test_authorized_test_preprocessing_requires_dev_manifest_and_processes_only_test_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from wsi_lab.preprocessing.artifacts import ArtifactStore, SlideOffset

    config = _config(tmp_path)
    fingerprints = stage_fingerprints(config)
    paths = artifact_paths(config, fingerprints)
    store = ArtifactStore(paths[Stage.PREPROCESSING].parent)
    atomic_json_dump(
        paths[Stage.PREPROCESSING],
        {
            "status": "complete",
            "fingerprint": fingerprints[Stage.PREPROCESSING],
            "metadata": {
                "scope": "development-only",
                "official_test_opened": False,
            },
        },
    )
    store.write_offsets(
        {
            "normal_001": SlideOffset(
                slide_id="normal_001",
                slide_index=0,
                offset=0,
                num_cells=10,
                level0_width=900,
                level0_height=400,
            )
        },
        input_fingerprint="development-prefix",
    )
    records = (
        SlideRecord(
            slide_id="normal_001",
            slide_path=tmp_path / "normal_001.tif",
            label=0,
            label_name="negative",
            official_cohort="development",
        ),
        SlideRecord(
            slide_id="test_001",
            slide_path=tmp_path / "test_001.tif",
            label=1,
            label_name="macro",
            official_cohort="test",
        ),
    )

    class Adapter:
        def iter_slides(self):
            return iter(records)

    class Reader:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def close(self) -> None:
            return None

    calls: list[tuple[tuple[str, ...], bool]] = []

    class Processor:
        def __init__(self, adapter, embeddings, reader, artifacts, *, options) -> None:
            assert artifacts.root == store.root

        def preprocess(self, slide_ids, *, write_index=True):
            calls.append((tuple(slide_ids), write_index))
            return (SimpleNamespace(reused=False),)

    class Options:
        @classmethod
        def from_config(cls, config, *, tissue_cache_root, concurrent_writes):
            assert concurrent_writes is True
            return cls()

    split = SimpleNamespace(
        split_fingerprint="split-fingerprint",
        ids=lambda name: ("test_001",) if name == "test" else (),
    )
    import wsi_lab.data.embeddings.hdf5 as hdf5_module
    import wsi_lab.preprocessing.pipeline as preprocessing_module
    import wsi_lab.preprocessing.test_artifacts as test_artifacts_module
    import wsi_lab.wsi.reader as reader_module

    manager_calls: list[dict[str, Any]] = []

    def ensure_test_artifacts(**kwargs):
        manager_calls.append(kwargs)
        if kwargs["mode"] == "require":
            assert kwargs["builder"] is None
        else:
            kwargs["builder"](tuple(kwargs["test_slide_ids"]))
        return {
            "status": "complete",
            "scope": "authorized-official-test-evaluation",
            "slides": len(kwargs["test_slide_ids"]),
            "training_dependency": False,
        }

    monkeypatch.setattr(cli, "_adapter", lambda _: Adapter())
    monkeypatch.setattr(cli, "_load_verified_split", lambda _: split)
    monkeypatch.setattr(hdf5_module, "HDF5EmbeddingProvider", lambda _: object())
    monkeypatch.setattr(preprocessing_module, "Preprocessor", Processor)
    monkeypatch.setattr(preprocessing_module, "PreprocessingOptions", Options)
    monkeypatch.setattr(test_artifacts_module, "ensure_test_artifacts", ensure_test_artifacts)
    monkeypatch.setattr(reader_module, "OpenSlideReader", Reader)

    result = cli._prepare_authorized_test_preprocessing(config)

    assert calls == [(("test_001",), False)]
    assert tuple(manager_calls[0]["test_slide_ids"]) == ("test_001",)
    assert manager_calls[0]["mode"] == "auto"
    assert result["scope"] == "authorized-official-test-evaluation"
    assert result["slides"] == 1
    assert result["training_dependency"] is False

    manifest_before = paths[Stage.PREPROCESSING].read_bytes()
    offsets_before = store.offsets_path.read_bytes()
    required = cli._prepare_authorized_test_preprocessing(config, mode="require")

    assert required["status"] == "complete"
    assert calls == [(("test_001",), False)]
    assert manager_calls[-1]["mode"] == "require"
    assert paths[Stage.PREPROCESSING].read_bytes() == manifest_before
    assert store.offsets_path.read_bytes() == offsets_before
    assert not store.index_path.exists()


def test_authorized_test_preprocessing_fails_closed_without_dev_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cli,
        "_adapter",
        lambda *_: pytest.fail("test cohort discovered before development manifest guard"),
    )

    with pytest.raises(PreprocessingError, match="development preprocessing"):
        cli._prepare_authorized_test_preprocessing(_config(tmp_path))


def test_trajectory_rendering_rejects_test_before_processed_artifact_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    envs_package = _fake_module(monkeypatch, "wsi_lab.envs", __path__=[])
    wsi_package = _fake_module(monkeypatch, "wsi_lab.wsi", __path__=[])
    monkeypatch.setattr(wsi_lab, "envs", envs_package, raising=False)
    monkeypatch.setattr(wsi_lab, "wsi", wsi_package, raising=False)
    _fake_module(
        monkeypatch,
        "wsi_lab.envs.rendering",
        render_trajectory_figure=object(),
    )
    _fake_module(monkeypatch, "wsi_lab.wsi.reader", OverviewTransform=object)
    trajectory = tmp_path / "trajectory.json"
    trajectory.write_text(
        json.dumps({"slide_id": "test_001", "trajectory": [{"x": 0, "y": 0}]}),
        encoding="utf-8",
    )
    split = SimpleNamespace(
        assignment=lambda slide_id: SimpleNamespace(slide_id=slide_id, split="test")
    )
    monkeypatch.setattr(cli, "_load_verified_split", lambda _: split)
    monkeypatch.setattr(
        cli,
        "_preprocessing_store",
        lambda *_: pytest.fail("processed test artifacts were accessed before the guard"),
    )

    with pytest.raises(TestSplitAccessError, match="--allow-test"):
        cli._render_trajectory_command(
            object(),
            trajectory,
            None,
            allow_test=False,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("argv", "expected_stages"),
    [
        (
            ("validate", "--config", "config.yaml", "--dry-run"),
            (Stage.DATASET, Stage.SPLIT),
        ),
        (
            ("preprocess", "--config", "config.yaml", "--dry-run"),
            (Stage.SPLIT, Stage.BELIEF_DATA, Stage.TISSUE, Stage.PREPROCESSING),
        ),
        (
            ("train-belief", "--config", "config.yaml", "--dry-run"),
            (Stage.SPLIT, Stage.BELIEF_DATA, Stage.BELIEF),
        ),
        (("train-rl", "--config", "config.yaml", "--dry-run"), (Stage.PPO,)),
        (
            ("run", "--config", "config.yaml", "--dry-run"),
            (
                Stage.SPLIT,
                Stage.BELIEF_DATA,
                Stage.TISSUE,
                Stage.PREPROCESSING,
                Stage.BELIEF,
                Stage.PPO,
            ),
        ),
    ],
)
def test_every_pipeline_command_has_an_explicit_dispatch_path(
    argv: tuple[str, ...],
    expected_stages: tuple[Stage, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[Stage, ...], bool]] = []

    class Runner:
        def run(self, stages: tuple[Stage, ...], *, dry_run: bool) -> _FormattedPlan:
            calls.append((stages, dry_run))
            return _FormattedPlan()

    monkeypatch.setattr(cli, "load_config", lambda _: object())
    monkeypatch.setattr(cli, "_configured_runner", lambda _: Runner())
    arguments = cli.build_parser().parse_args(argv)

    assert cli._dispatch(arguments) == 0
    assert calls == [(expected_stages, True)]


@pytest.mark.parametrize(
    ("extra_arguments", "expected_evaluations"),
    [
        ((), 1),
        (("--final-test-eval",), 1),
        (("--no-final-test-eval",), 0),
    ],
)
def test_train_rl_runs_default_on_final_test_only_after_pipeline_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    extra_arguments: tuple[str, ...],
    expected_evaluations: int,
) -> None:
    events: list[str] = []

    class Runner:
        def run(self, stages: tuple[Stage, ...], *, dry_run: bool) -> _FormattedPlan:
            assert stages == (Stage.PPO,)
            assert not dry_run
            events.append("pipeline-complete")
            return _FormattedPlan()

    def evaluate(config: object) -> dict[str, object]:
        del config
        assert events == ["pipeline-complete"]
        events.append("test-evaluated")
        return {"metrics": {"accuracy": 0.75, "auroc": 0.8}}

    monkeypatch.setattr(cli, "load_config", lambda _: object())
    monkeypatch.setattr(cli, "_configured_runner", lambda _: Runner())
    monkeypatch.setattr(cli, "_automatic_final_test_evaluation", evaluate)
    arguments = cli.build_parser().parse_args(
        ("train-rl", "--config", str(tmp_path / "config.yaml"), *extra_arguments)
    )

    assert cli._dispatch(arguments) == 0
    assert events.count("test-evaluated") == expected_evaluations
    output = capsys.readouterr().out
    assert ("Final test evaluation:" in output) is bool(expected_evaluations)


def test_train_rl_dry_run_never_opens_the_test_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Runner:
        def run(self, stages: tuple[Stage, ...], *, dry_run: bool) -> _FormattedPlan:
            assert stages == (Stage.PPO,)
            assert dry_run
            return _FormattedPlan()

    monkeypatch.setattr(cli, "load_config", lambda _: object())
    monkeypatch.setattr(cli, "_configured_runner", lambda _: Runner())
    monkeypatch.setattr(
        cli,
        "_automatic_final_test_evaluation",
        lambda _: pytest.fail("dry-run accessed the test split"),
    )

    arguments = cli.build_parser().parse_args(("train-rl", "--config", "config.yaml", "--dry-run"))
    assert cli._dispatch(arguments) == 0


@pytest.mark.parametrize(
    ("argv", "helper"),
    [
        (("eval-belief", "--config", "config.yaml"), "_evaluate_belief_command"),
        (("eval-rl", "--config", "config.yaml"), "_evaluate_rl_command"),
        (("eval-baselines", "--config", "config.yaml"), "_evaluate_baselines_command"),
        (
            (
                "render-trajectory",
                "--config",
                "config.yaml",
                "--trajectory",
                "trajectory.json",
            ),
            "_render_trajectory_command",
        ),
        (("benchmark-env", "--config", "config.yaml"), "_benchmark_command"),
        (("smoke-test",), "_smoke_test"),
    ],
)
def test_every_standalone_command_has_an_explicit_dispatch_path(
    argv: tuple[str, ...], helper: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def invoke(*args, **kwargs):
        nonlocal called
        called = True
        return {"status": "mocked"}

    monkeypatch.setattr(cli, "load_config", lambda _: object())
    monkeypatch.setattr(cli, helper, invoke)
    arguments = cli.build_parser().parse_args(argv)

    assert cli._dispatch(arguments) == 0
    assert called
    assert "command integration is unavailable" not in inspect.getsource(cli._dispatch).casefold()


def test_pipeline_runner_owns_manifest_while_belief_training_owns_sibling_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    manifest_path = tmp_path / "belief" / "manifest.json"
    plan = PipelinePlan(
        (
            StagePlan(
                stage=Stage.BELIEF,
                action=StageAction.TRAIN,
                reason="artifact is absent",
                artifact=manifest_path,
                fingerprint="belief-fingerprint",
            ),
        )
    )
    observed_statuses: list[str] = []

    def train_belief(config, artifact_dir, fingerprint, action):
        del config, fingerprint, action
        manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
        observed_statuses.append(manifest["status"])
        atomic_json_dump(
            artifact_dir / "training_state.json",
            {"status": "complete", "owner": "belief-trainer"},
        )
        return {"checkpoint": "checkpoint.pt"}

    runner = PipelineRunner(config)
    runner.register(Stage.BELIEF, train_belief)
    monkeypatch.setattr(runner, "plan", lambda stages=None: plan)

    runner.run((Stage.BELIEF,))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    training_state = json.loads(
        (manifest_path.parent / "training_state.json").read_text(encoding="utf-8")
    )
    assert observed_statuses == ["incomplete"]
    assert manifest["status"] == "complete"
    assert manifest["metadata"]["checkpoint"] == "checkpoint.pt"
    assert training_state == {"owner": "belief-trainer", "status": "complete"}


def test_pipeline_runner_marks_its_manifest_failed_when_stage_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    manifest_path = tmp_path / "belief" / "manifest.json"
    plan = PipelinePlan(
        (
            StagePlan(
                stage=Stage.BELIEF,
                action=StageAction.TRAIN,
                reason="artifact is absent",
                artifact=manifest_path,
                fingerprint="belief-fingerprint",
            ),
        )
    )

    def fail(*args, **kwargs):
        raise RuntimeError("training failed")

    runner = PipelineRunner(config)
    runner.register(Stage.BELIEF, fail)
    monkeypatch.setattr(runner, "plan", lambda stages=None: plan)

    with pytest.raises(RuntimeError, match="training failed"):
        runner.run((Stage.BELIEF,))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert "training failed" in manifest["metadata"]["error"]
