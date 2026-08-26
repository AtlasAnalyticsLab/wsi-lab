from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import wsi_lab.rl.best_checkpoint as best_checkpoint_module
from wsi_lab.pipeline.planner import _ppo_checkpoint_candidates
from wsi_lab.rl.best_checkpoint import (
    BEST_ARCHIVE_DIRECTORY,
    BEST_CHECKPOINT_FILENAME,
    BEST_METADATA_FILENAME,
    BestValidationCheckpointTracker,
)
from wsi_lab.rl.checkpoints import checkpoint_candidates, save_ppo_checkpoint
from wsi_lab.rl.recurrent_ppo import PPOConfig


def _model_and_optimizer() -> tuple[torch.nn.Linear, torch.optim.Adam]:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    # Materialize Adam's moments so the test verifies a non-empty optimizer state.
    optimizer.zero_grad()
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    return model, optimizer


def _consider(
    tracker: BestValidationCheckpointTracker,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics: dict[str, object],
    *,
    step: int,
):
    return tracker.consider(
        metrics,
        model=model,
        optimizer=optimizer,
        config=PPOConfig(device="cpu"),
        global_step=step,
        update=step // 8,
        dependencies={"belief_fingerprint": "belief-v1"},
    )


def _load(path: Path) -> dict[str, object]:
    return torch.load(path, map_location="cpu", weights_only=False)


def test_first_validation_saves_complete_current_training_state(tmp_path: Path) -> None:
    model, optimizer = _model_and_optimizer()
    expected_weight = model.weight.detach().clone()
    tracker = BestValidationCheckpointTracker(tmp_path)

    decision = _consider(
        tracker,
        model,
        optimizer,
        {"auroc": 0.71, "accuracy": 0.63, "mean_return": -2.0},
        step=16,
    )

    assert decision.improved
    assert decision.best.metric_name == "auroc"
    assert decision.best.metric_value == pytest.approx(0.71)
    assert decision.best.global_step == 16
    assert decision.checkpoint_path.name == BEST_CHECKPOINT_FILENAME
    assert decision.metadata_path.name == BEST_METADATA_FILENAME
    payload = _load(decision.checkpoint_path)
    assert payload["global_step"] == 16
    assert payload["update"] == 2
    assert payload["dependencies"] == {"belief_fingerprint": "belief-v1"}
    assert payload["config"] == PPOConfig(device="cpu").to_dict()
    assert payload["optimizer"]["state"]
    assert torch.equal(payload["model"]["weight"], expected_weight)
    assert payload["extra"]["best_validation"]["selection_metric"] == "auroc"
    metadata = json.loads(decision.metadata_path.read_text(encoding="utf-8"))
    assert metadata["metric_name"] == "auroc"
    assert metadata["metric_value"] == pytest.approx(0.71)
    assert metadata["selection_value"] == pytest.approx(0.71)
    assert metadata["global_step"] == 16
    assert decision.to_dict()["best"]["metric_name"] == "auroc"


def test_higher_primary_metric_wins_and_exact_tie_keeps_earliest(tmp_path: Path) -> None:
    model, optimizer = _model_and_optimizer()
    tracker = BestValidationCheckpointTracker(tmp_path)
    with torch.no_grad():
        model.weight.fill_(1.0)
    first = _consider(tracker, model, optimizer, {"auroc": 0.75, "accuracy": 0.8}, step=8)
    with torch.no_grad():
        model.weight.fill_(2.0)

    tie = _consider(tracker, model, optimizer, {"auroc": 0.75, "accuracy": 0.99}, step=16)
    lower = _consider(tracker, model, optimizer, {"auroc": 0.74, "accuracy": 1.0}, step=24)

    assert first.improved
    assert not tie.improved
    assert not lower.improved
    assert tie.best.global_step == 8
    assert lower.best.global_step == 8
    assert torch.equal(
        _load(tracker.checkpoint_path)["model"]["weight"],
        torch.ones_like(model.weight),
    )

    better = _consider(tracker, model, optimizer, {"auroc": 0.76, "accuracy": 0.5}, step=32)
    assert better.improved
    assert better.best.global_step == 32
    assert torch.equal(_load(tracker.checkpoint_path)["model"]["weight"], model.weight)


def test_accuracy_fallback_persists_across_resume_until_auroc_is_available(
    tmp_path: Path,
) -> None:
    model, optimizer = _model_and_optimizer()
    tracker = BestValidationCheckpointTracker(tmp_path)
    first = _consider(tracker, model, optimizer, {"auroc": None, "accuracy": 0.8}, step=8)
    assert first.best.metric_name == "accuracy"

    resumed = BestValidationCheckpointTracker(tmp_path)
    assert resumed.best is not None
    assert resumed.best.global_step == 8
    assert not _consider(
        resumed, model, optimizer, {"auroc": float("nan"), "accuracy": 0.8}, step=16
    ).improved
    assert _consider(resumed, model, optimizer, {"auroc": None, "accuracy": 0.81}, step=24).improved

    # Any finite AUROC establishes the primary objective. Subsequent
    # accuracy-only passes cannot be compared to and replace that objective.
    primary = _consider(resumed, model, optimizer, {"auroc": 0.55, "accuracy": 0.4}, step=32)
    assert primary.improved
    assert primary.best.metric_name == "auroc"
    assert not _consider(
        resumed, model, optimizer, {"auroc": None, "accuracy": 1.0}, step=40
    ).improved


def test_embedded_record_recovers_interrupted_metadata_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, optimizer = _model_and_optimizer()
    tracker = BestValidationCheckpointTracker(tmp_path)
    original_dump = best_checkpoint_module.atomic_json_dump

    def fail_dump(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated metadata write interruption")

    monkeypatch.setattr(best_checkpoint_module, "atomic_json_dump", fail_dump)
    with pytest.raises(OSError, match="interruption"):
        _consider(tracker, model, optimizer, {"auroc": 0.65, "accuracy": 0.6}, step=8)
    assert tracker.checkpoint_path.is_file()
    assert not tracker.metadata_path.exists()

    monkeypatch.setattr(best_checkpoint_module, "atomic_json_dump", original_dump)
    recovered = BestValidationCheckpointTracker(tmp_path)
    assert recovered.best is not None
    assert recovered.best.global_step == 8
    assert recovered.metadata_path.is_file()


def test_best_checkpoint_is_never_a_latest_resume_candidate(tmp_path: Path) -> None:
    model, optimizer = _model_and_optimizer()
    run_directory = tmp_path / "run"
    checkpoint_directory = run_directory / "checkpoints"
    tracker = BestValidationCheckpointTracker(checkpoint_directory)
    _consider(tracker, model, optimizer, {"auroc": 0.7, "accuracy": 0.6}, step=16)
    periodic = save_ppo_checkpoint(
        checkpoint_directory / "checkpoint_000000000008.pt",
        model,
        optimizer,
        PPOConfig(device="cpu"),
        global_step=8,
        update=1,
    )

    assert checkpoint_candidates(checkpoint_directory) == [periodic]
    assert tracker.checkpoint_path not in checkpoint_candidates(checkpoint_directory)
    assert _ppo_checkpoint_candidates(run_directory) == (periodic,)


def test_fresh_tracker_quarantines_crash_window_best_and_selects_from_scratch(
    tmp_path: Path,
) -> None:
    model, optimizer = _model_and_optimizer()
    run_directory = tmp_path / "run"
    checkpoint_directory = run_directory / "checkpoints"
    interrupted = BestValidationCheckpointTracker(checkpoint_directory)
    _consider(
        interrupted,
        model,
        optimizer,
        {"auroc": 0.9, "accuracy": 0.8},
        step=16,
    )
    assert checkpoint_candidates(checkpoint_directory) == []

    fresh = BestValidationCheckpointTracker(checkpoint_directory, resume_existing=False)

    assert fresh.best is None
    assert fresh.quarantine_directory is not None
    assert fresh.quarantine_directory.parent == checkpoint_directory / BEST_ARCHIVE_DIRECTORY
    assert (fresh.quarantine_directory / BEST_CHECKPOINT_FILENAME).is_file()
    assert (fresh.quarantine_directory / BEST_METADATA_FILENAME).is_file()
    assert not fresh.checkpoint_path.exists()
    assert not fresh.metadata_path.exists()
    assert checkpoint_candidates(checkpoint_directory) == []
    assert _ppo_checkpoint_candidates(run_directory) == ()

    # The new attempt is independent: its first candidate wins even though its
    # score is below the abandoned attempt's validation result.
    selected = _consider(
        fresh,
        model,
        optimizer,
        {"auroc": 0.6, "accuracy": 0.7},
        step=8,
    )
    assert selected.improved
    assert selected.best.metric_value == pytest.approx(0.6)
    assert (fresh.quarantine_directory / BEST_CHECKPOINT_FILENAME).is_file()


@pytest.mark.parametrize("orphan_name", (BEST_CHECKPOINT_FILENAME, BEST_METADATA_FILENAME))
def test_fresh_tracker_quarantines_single_orphan(tmp_path: Path, orphan_name: str) -> None:
    tmp_path.mkdir(exist_ok=True)
    orphan = tmp_path / orphan_name
    orphan.write_text("stale", encoding="utf-8")

    fresh = BestValidationCheckpointTracker(tmp_path, resume_existing=False)

    assert fresh.best is None
    assert fresh.quarantine_directory is not None
    assert not orphan.exists()
    assert (fresh.quarantine_directory / orphan_name).read_text(encoding="utf-8") == "stale"
