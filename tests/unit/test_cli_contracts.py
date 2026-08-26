from __future__ import annotations

import inspect
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import rl4xs_wsi
from rl4xs_wsi import cli
from rl4xs_wsi.config.models import AppConfig, DatasetConfig, WorkspaceConfig
from rl4xs_wsi.data.base import SlideRecord
from rl4xs_wsi.errors import PreprocessingError, TestSplitAccessError
from rl4xs_wsi.pipeline.planner import (
    PipelinePlan,
    StagePlan,
    artifact_paths,
    stage_fingerprints,
)
from rl4xs_wsi.pipeline.runner import PipelineRunner
from rl4xs_wsi.pipeline.stages import Stage, StageAction
from rl4xs_wsi.schemas import ValidationReport
from rl4xs_wsi.utils.atomic import atomic_json_dump


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

    import rl4xs_wsi.data.embeddings.hdf5 as hdf5_module
    import rl4xs_wsi.data.validation as validation_module

    monkeypatch.setattr(cli, "_adapter", lambda _: Adapter())
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
            return SimpleNamespace(global_step=8, updates=1, last_metrics=None)

        def close(self) -> None:
            pass

    _fake_module(monkeypatch, "rl4xs_wsi.rl.evaluation", evaluate_policy=object())
    _fake_module(
        monkeypatch,
        "rl4xs_wsi.rl.networks",
        RecurrentActorCritic=Model,
    )
    _fake_module(
        monkeypatch,
        "rl4xs_wsi.rl.trainer",
        PPOTrainer=Trainer,
        make_vector_env=lambda factories, *, backend: (factories, backend),
    )
    monkeypatch.setattr(cli, "_ppo_dependencies", lambda _: {"belief": "current"})

    cli._ppo_stage(config, tmp_path / "ppo", "fingerprint", action)

    assert captured["resume_latest"] is expected_resume


def test_ppo_checkpoint_load_resolves_auto_before_passing_map_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    config.rl.device = "auto"
    checkpoint = tmp_path / "checkpoint_000000000008.pt"
    checkpoint.touch()
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
    _fake_module(
        monkeypatch,
        "rl4xs_wsi.rl.checkpoints",
        checkpoint_candidates=lambda _: [checkpoint],
        load_ppo_checkpoint=load,
    )
    _fake_module(
        monkeypatch,
        "rl4xs_wsi.rl.networks",
        RecurrentActorCritic=lambda: model,
    )
    _fake_module(
        monkeypatch,
        "rl4xs_wsi.rl.recurrent_ppo",
        PPOConfig=PPOConfig,
    )
    _fake_module(
        monkeypatch,
        "rl4xs_wsi.rl.trainer",
        RESUME_CONFIG_FIELDS=(),
        resolve_device=lambda requested: f"resolved:{requested}",
    )
    monkeypatch.setattr(cli, "stage_fingerprints", lambda _: {Stage.PPO: "ppo"})
    monkeypatch.setattr(
        cli,
        "artifact_paths",
        lambda *_: {Stage.PPO: tmp_path / "run" / "manifest.json"},
    )
    monkeypatch.setattr(cli, "_ppo_dependencies", lambda _: {})

    observed_model, observed_loaded = cli._load_ppo_model(config)

    assert observed_model is model
    assert observed_loaded is loaded
    assert captured["map_location"] == "resolved:auto"


def test_belief_evaluation_rejects_test_before_loading_records_or_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch_utils_data = _fake_module(monkeypatch, "torch.utils.data", DataLoader=object)
    torch_utils = _fake_module(monkeypatch, "torch.utils", __path__=[], data=torch_utils_data)
    _fake_module(monkeypatch, "torch", __path__=[], utils=torch_utils)
    belief_package = _fake_module(monkeypatch, "rl4xs_wsi.belief", __path__=[])
    monkeypatch.setattr(rl4xs_wsi, "belief", belief_package, raising=False)
    _fake_module(
        monkeypatch,
        "rl4xs_wsi.belief.dataset",
        ValidationBagDataset=object,
        collate_belief_bags=object(),
    )
    _fake_module(monkeypatch, "rl4xs_wsi.belief.evaluation", evaluate_belief=object())
    _fake_module(
        monkeypatch,
        "rl4xs_wsi.belief.trainer",
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
    rl_package = _fake_module(monkeypatch, "rl4xs_wsi.rl", __path__=[])
    monkeypatch.setattr(rl4xs_wsi, "rl", rl_package, raising=False)
    _fake_module(monkeypatch, "rl4xs_wsi.rl.evaluation", evaluate_policy=object())
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
    rl_package = _fake_module(monkeypatch, "rl4xs_wsi.rl", __path__=[])
    monkeypatch.setattr(rl4xs_wsi, "rl", rl_package, raising=False)
    _fake_module(
        monkeypatch,
        "rl4xs_wsi.rl.evaluation",
        evaluate_baseline=object(),
        evaluate_policy=object(),
    )
    monkeypatch.setattr(
        cli,
        "_prepare_authorized_test_preprocessing",
        lambda *_: (_ for _ in ()).throw(RuntimeError("authorized test preprocessing")),
    )
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
        )


def test_authorized_test_preprocessing_requires_dev_manifest_and_processes_only_test_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rl4xs_wsi.preprocessing.artifacts import ArtifactStore, SlideOffset

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

        def preprocess(self, slide_ids, *, merge_index=False):
            calls.append((tuple(slide_ids), merge_index))
            return (SimpleNamespace(reused=False),)

    class Options:
        @classmethod
        def from_config(cls, config, *, tissue_cache_root):
            return cls()

    split = SimpleNamespace(
        split_fingerprint="split-fingerprint",
        ids=lambda name: ("test_001",) if name == "test" else (),
    )
    import rl4xs_wsi.data.embeddings.hdf5 as hdf5_module
    import rl4xs_wsi.preprocessing.pipeline as preprocessing_module
    import rl4xs_wsi.wsi.reader as reader_module

    monkeypatch.setattr(cli, "_adapter", lambda _: Adapter())
    monkeypatch.setattr(cli, "_load_verified_split", lambda _: split)
    monkeypatch.setattr(hdf5_module, "HDF5EmbeddingProvider", lambda _: object())
    monkeypatch.setattr(preprocessing_module, "Preprocessor", Processor)
    monkeypatch.setattr(preprocessing_module, "PreprocessingOptions", Options)
    monkeypatch.setattr(reader_module, "OpenSlideReader", Reader)

    result = cli._prepare_authorized_test_preprocessing(config)

    assert calls == [(("test_001",), True)]
    assert result["scope"] == "authorized-official-test-evaluation"
    assert result["slides"] == 1
    assert result["training_dependency"] is False
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"


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
    envs_package = _fake_module(monkeypatch, "rl4xs_wsi.envs", __path__=[])
    wsi_package = _fake_module(monkeypatch, "rl4xs_wsi.wsi", __path__=[])
    monkeypatch.setattr(rl4xs_wsi, "envs", envs_package, raising=False)
    monkeypatch.setattr(rl4xs_wsi, "wsi", wsi_package, raising=False)
    _fake_module(
        monkeypatch,
        "rl4xs_wsi.envs.rendering",
        render_trajectory_figure=object(),
    )
    _fake_module(monkeypatch, "rl4xs_wsi.wsi.reader", OverviewTransform=object)
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
    ("argv", "helper"),
    [
        (("eval-belief", "--config", "config.yaml"), "_evaluate_belief_command"),
        (("eval-rl", "--config", "config.yaml"), "_evaluate_rl_command"),
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
