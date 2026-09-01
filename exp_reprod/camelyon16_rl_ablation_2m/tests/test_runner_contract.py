"""Training-runner wiring checks that do not open CAMELYON16 data."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ablation import analysis_cache, launcher, monitoring, runner, verification
from ablation.contract import run_contract_payload
from ablation.spec import RunSpec, main_runs, smoke_runs


def _write_config(
    path: Path,
    *,
    mode: str,
) -> tuple[Path, RunSpec]:
    row = main_runs()[0] if mode == "main" else smoke_runs()[0]
    config = launcher._config_for(
        row,
        ppo_device="cpu",
        belief_inference_device="cpu",
        preparation_device="cpu",
        tissue_device="cpu",
    )
    path.write_text(launcher._config_text(config), encoding="utf-8")
    return path, row


def _training_progress_payload(*, global_step: int = 1_024) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "update_complete",
        "run_fingerprint": "a" * 64,
        "variant": "v1_full",
        "global_step": global_step,
        "updates": 4,
        "last_metrics": {"synthetic_placeholder": 1},
    }


def _completion_telemetry() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes = [
        {
            "schema_version": verification.TELEMETRY_SCHEMA_VERSION,
            "variant_id": "v1_full",
            "episode_id": update,
            "completed_episode_count": update,
            "update": update,
            "global_step": update * 8,
            "completion_global_step": (update - 1) * 8 + 2,
            "episode_length": 1,
            "observed_views": 1,
            "action_histogram": {"pan_up": 1},
            "magnification_histogram": {"synthetic_level": 1},
        }
        for update in range(1, 4)
    ]
    rollouts = [
        {
            "schema_version": verification.TELEMETRY_SCHEMA_VERSION,
            "variant_id": "v1_full",
            "global_step": update * 8,
            "update": update,
            "transitions": 8,
            "completed_episodes": 1,
            "action_counts": {"pan_up": 8},
            "action_fractions": {"pan_up": 1.0},
            "cumulative": {"episodes": update},
        }
        for update in range(1, 4)
    ]
    return episodes, rollouts


def _write_completion_telemetry(
    artifact_dir: Path,
    *,
    episodes: list[dict[str, Any]],
    rollouts: list[dict[str, Any]],
    state_overrides: dict[str, Any] | None = None,
) -> None:
    telemetry_dir = artifact_dir / "telemetry"
    telemetry_dir.mkdir(parents=True)
    episodes_path = telemetry_dir / "training_episodes.jsonl"
    rollouts_path = telemetry_dir / "training_rollouts.jsonl"
    episodes_path.write_text(
        "".join(f"{json.dumps(record)}\n" for record in episodes),
        encoding="utf-8",
    )
    rollouts_path.write_text(
        "".join(f"{json.dumps(record)}\n" for record in rollouts),
        encoding="utf-8",
    )
    aggregate = verification.aggregate_episode_records(episodes)
    state = {
        "schema_version": verification.TELEMETRY_SCHEMA_VERSION,
        "status": "complete",
        "variant_id": "v1_full",
        "global_step": 24,
        "episodes": 3,
        "last_episode_id": 3,
        "episode_jsonl": str(episodes_path.resolve()),
        "rollout_jsonl": str(rollouts_path.resolve()),
        "cumulative": aggregate,
        "window": aggregate,
        **(state_overrides or {}),
    }
    runner.atomic_json_dump(telemetry_dir / "telemetry_state.json", state)


def _write_smoke_monitoring_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, list[Path], list[Path]]:
    study = {
        "smoke": {"requested_timesteps": 8},
        "monitoring": {
            "milestones": [8],
            "cohort_size": 2,
            "positive_slides": 1,
            "cohort_seed": 7,
        },
    }
    monkeypatch.setattr(verification, "load_study", lambda: study)
    workspace = tmp_path / "generated_workspace"
    artifact_dir = workspace / "rl_ablation" / "synthetic_run"
    cache_root = workspace.parent / "analysis_cache"
    cache_root.mkdir(parents=True)
    records = (
        {
            "slide_id": "synthetic_positive",
            "label": 1,
            "split": "train",
            "annotation_available": True,
            "annotation_exhaustive": True,
        },
        {
            "slide_id": "synthetic_negative",
            "label": 0,
            "split": "train",
            "annotation_available": False,
            "annotation_exhaustive": False,
        },
    )
    entries = {str(record["slide_id"]): dict(record) for record in records}
    runner.atomic_json_dump(
        cache_root / "manifest.json",
        {
            "status": "complete",
            "scope": "training-only-analysis",
            "train_only": True,
            "entries": entries,
        },
    )
    full_cohort = analysis_cache.select_monitor_cohort(
        records,
        cohort_size=2,
        positive_slides=1,
        seed=7,
    )
    runner.atomic_json_dump(
        cache_root / "monitor_slides.json",
        {
            "schema_version": 1,
            "selection": {
                "split": "train",
                "cohort_size": 2,
                "positive_slides": 1,
                "seed": 7,
                "positive_annotation_requirement": "available_and_exhaustive",
            },
            "slides": list(full_cohort),
        },
    )
    positive = next(item for item in full_cohort if item["label"] == 1)
    negative = next(item for item in full_cohort if item["label"] == 0)
    selected = (positive, negative)
    output_dir = artifact_dir / "monitoring" / "step_000000000008"
    trajectory_paths: list[Path] = []
    overlay_paths: list[Path] = []
    slide_summaries: list[dict[str, Any]] = []
    for cohort_record in selected:
        slide_id = str(cohort_record["slide_id"])
        slide_root = output_dir / slide_id
        trajectory_path = slide_root / "trajectory.json.gz"
        overlay_path = slide_root / "overlay.png"
        episode_summary = {
            "slide_id": slide_id,
            "label": int(cohort_record["label"]),
            "episode_length": 1,
            "terminated": True,
            "truncated": False,
        }
        monitoring.write_trajectory_gzip(
            trajectory_path,
            {
                "schema_version": verification.MONITORING_SCHEMA_VERSION,
                "variant_id": "v1_full",
                "start_strategy": "center",
                "nominal_step": 8,
                "actual_step": 8,
                "slide_id": slide_id,
                "label": int(cohort_record["label"]),
                "reset_seed": monitoring.monitor_reset_seed(7, slide_id),
                "cohort_record": dict(cohort_record),
                "episode_summary": episode_summary,
                "trajectory": [
                    {
                        "step": 0,
                        "action": None,
                        "canonical_bounds": [0, 0, 1, 1],
                        "terminated": False,
                        "truncated": False,
                    },
                    {
                        "step": 1,
                        "action": 0,
                        "canonical_bounds": [0, 0, 1, 1],
                        "terminated": True,
                        "truncated": False,
                    },
                ],
            },
        )
        from PIL import Image

        slide_root.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (1, 1)).save(overlay_path)
        trajectory_paths.append(trajectory_path)
        overlay_paths.append(overlay_path)
        slide_summaries.append(
            {
                **episode_summary,
                "trajectory": str(trajectory_path.resolve()),
                "overlay": str(overlay_path.resolve()),
            }
        )
    summary_path = output_dir / "summary.json"
    runner.atomic_json_dump(
        summary_path,
        {
            "schema_version": verification.MONITORING_SCHEMA_VERSION,
            "variant_id": "v1_full",
            "nominal_step": 8,
            "actual_step": 8,
            "slides": slide_summaries,
            "aggregate": verification.aggregate_episode_records(slide_summaries),
        },
    )
    runner.atomic_json_dump(
        artifact_dir / "monitoring" / "manifest.json",
        {
            "schema_version": verification.MONITORING_SCHEMA_VERSION,
            "status": "complete",
            "variant_id": "v1_full",
            "expected_milestones": [8],
            "cohort_sha256": monitoring.cohort_sha256(selected),
            "completed": [
                {
                    "nominal_step": 8,
                    "actual_step": 8,
                    "slides": 2,
                    "output_dir": str(output_dir.resolve()),
                    "summary": str(summary_path.resolve()),
                }
            ],
        },
    )
    return artifact_dir, workspace, trajectory_paths, overlay_paths


def test_completion_telemetry_accepts_exact_batch_and_episode_contract(
    tmp_path: Path,
) -> None:
    episodes, rollouts = _completion_telemetry()
    artifact_dir = tmp_path / "artifact"
    _write_completion_telemetry(
        artifact_dir,
        episodes=episodes,
        rollouts=rollouts,
    )

    observed, aggregate = verification._verify_telemetry(
        artifact_dir,
        variant="v1_full",
        final_step=24,
        batch_size=8,
        num_envs=2,
        updates=3,
        completed_episodes=3,
        max_episode_steps=500,
    )

    assert observed == episodes
    assert aggregate["episodes"] == 3


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("missing_boundary", "has 2 rows, expected 3"),
        ("off_grid_step", "exact batch grid"),
        ("skipped_update", "update number is not consecutive"),
        ("wrong_transition_count", "transition count does not equal batch size"),
        ("wrong_completed_count", "completed-episode count mismatch"),
        ("wrong_cumulative_count", "cumulative episode count mismatch"),
    ),
)
def test_rollout_telemetry_rejects_completion_integrity_breaks(
    corruption: str,
    message: str,
) -> None:
    episodes, rollouts = _completion_telemetry()
    if corruption == "missing_boundary":
        rollouts.pop(1)
    elif corruption == "off_grid_step":
        rollouts[1]["global_step"] = 24
    elif corruption == "skipped_update":
        rollouts[1]["update"] = 3
    elif corruption == "wrong_transition_count":
        rollouts[1]["transitions"] = 7
    elif corruption == "wrong_completed_count":
        rollouts[1]["completed_episodes"] = 0
    elif corruption == "wrong_cumulative_count":
        rollouts[1]["cumulative"] = {"episodes": 1}
    else:  # pragma: no cover - keeps the table fail-closed when extended
        raise AssertionError(f"unknown synthetic corruption: {corruption}")

    with pytest.raises(verification.VerificationError, match=message):
        verification._verify_rollout_records(
            rollouts,
            variant="v1_full",
            final_step=24,
            batch_size=8,
            updates=3,
            episodes=episodes,
            location="synthetic-rollouts",
        )


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("nonconsecutive_id", "episode IDs are not consecutive"),
        ("wrong_rollout_step", "rollout step does not match its update"),
        ("wrong_completion_step", "completion step is inconsistent"),
    ),
)
def test_episode_telemetry_rejects_completion_integrity_breaks(
    corruption: str,
    message: str,
) -> None:
    episodes, _ = _completion_telemetry()
    if corruption == "nonconsecutive_id":
        episodes[1]["episode_id"] = 3
        episodes[1]["completed_episode_count"] = 3
    elif corruption == "wrong_rollout_step":
        episodes[1]["global_step"] = 24
    elif corruption == "wrong_completion_step":
        episodes[1]["completion_global_step"] = 8
    else:  # pragma: no cover - keeps the table fail-closed when extended
        raise AssertionError(f"unknown synthetic corruption: {corruption}")

    with pytest.raises(verification.VerificationError, match=message):
        verification._verify_episode_records(
            episodes,
            variant="v1_full",
            final_step=24,
            batch_size=8,
            num_envs=2,
            updates=3,
            completed_episodes=3,
            max_episode_steps=500,
            location="synthetic-episodes",
        )


@pytest.mark.parametrize(
    ("state_overrides", "message"),
    (
        ({"global_step": 16}, "telemetry completion mismatch"),
        ({"episodes": 2}, "telemetry completion mismatch"),
        ({"last_episode_id": 2}, "telemetry completion mismatch"),
        ({"cumulative": {"episodes": 2}}, "telemetry completion mismatch"),
        ({"window": {"episodes": 2}}, "trailing-window aggregate mismatch"),
    ),
)
def test_telemetry_state_rejects_manifest_counter_disagreement(
    state_overrides: dict[str, Any],
    message: str,
    tmp_path: Path,
) -> None:
    episodes, rollouts = _completion_telemetry()
    artifact_dir = tmp_path / "artifact"
    _write_completion_telemetry(
        artifact_dir,
        episodes=episodes,
        rollouts=rollouts,
        state_overrides=state_overrides,
    )

    with pytest.raises(verification.VerificationError, match=message):
        verification._verify_telemetry(
            artifact_dir,
            variant="v1_full",
            final_step=24,
            batch_size=8,
            num_envs=2,
            updates=3,
            completed_episodes=3,
            max_episode_steps=500,
        )


def _checkpoint_payload() -> dict[str, Any]:
    return {
        "global_step": 24,
        "update": 3,
        "config": {"synthetic": "config"},
        "dependencies": {"synthetic": "dependency"},
        "extra": {"final": True, "completed_episode_count": 3},
    }


def test_checkpoint_accepts_exact_completion_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "checkpoint_000000000024.pt"
    checkpoint.write_text("synthetic checkpoint placeholder", encoding="utf-8")
    payload = _checkpoint_payload()
    monkeypatch.setattr(
        "wsi_lab.rl.checkpoints.inspect_ppo_checkpoint",
        lambda _path: SimpleNamespace(**payload),
    )

    observed = verification._verify_checkpoint(
        checkpoint_dir,
        final_step=24,
        updates=3,
        completed_episodes=3,
        ppo_config=payload["config"],
        dependencies=payload["dependencies"],
    )

    assert observed == checkpoint


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("global_step", 16, "wrong global step"),
        ("update", 2, "update count does not match"),
        ("config", {"synthetic": "other"}, "PPO configuration does not match"),
        (
            "dependencies",
            {"synthetic": "other"},
            "dependencies do not match",
        ),
        ("extra.final", False, "not marked final"),
        (
            "extra.completed_episode_count",
            2,
            "completed-episode count does not match",
        ),
    ),
)
def test_checkpoint_rejects_completion_contract_disagreement(
    field: str,
    replacement: Any,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "checkpoint_000000000024.pt").write_text(
        "synthetic checkpoint placeholder",
        encoding="utf-8",
    )
    payload = deepcopy(_checkpoint_payload())
    if field.startswith("extra."):
        payload["extra"][field.removeprefix("extra.")] = replacement
    else:
        payload[field] = replacement
    monkeypatch.setattr(
        "wsi_lab.rl.checkpoints.inspect_ppo_checkpoint",
        lambda _path: SimpleNamespace(**payload),
    )

    with pytest.raises(verification.VerificationError, match=message):
        verification._verify_checkpoint(
            checkpoint_dir,
            final_step=24,
            updates=3,
            completed_episodes=3,
            ppo_config={"synthetic": "config"},
            dependencies={"synthetic": "dependency"},
        )


def test_exact_smoke_monitor_artifacts_are_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir, workspace, _, _ = _write_smoke_monitoring_fixture(tmp_path, monkeypatch)

    observed = verification._verify_monitoring(
        artifact_dir,
        workspace,
        mode="smoke",
        variant="v1_full",
        start="center",
        final_step=8,
    )

    assert observed == {"trajectories": 2, "overlays": 2}


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("empty_trajectory", "trajectory records are missing"),
        ("truncated_png", "could not read monitoring overlay"),
    ),
)
def test_monitoring_rejects_empty_or_nondecodable_artifacts(
    corruption: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir, workspace, trajectories, overlays = _write_smoke_monitoring_fixture(
        tmp_path,
        monkeypatch,
    )
    if corruption == "empty_trajectory":
        document = verification._read_trajectory(trajectories[0])
        document["trajectory"] = []
        monitoring.write_trajectory_gzip(trajectories[0], document)
    elif corruption == "truncated_png":
        overlays[0].write_bytes(b"\x89PNG\r\n\x1a\n")
    else:  # pragma: no cover - keeps the table fail-closed when extended
        raise AssertionError(f"unknown synthetic corruption: {corruption}")

    with pytest.raises(verification.VerificationError, match=message):
        verification._verify_monitoring(
            artifact_dir,
            workspace,
            mode="smoke",
            variant="v1_full",
            start="center",
            final_step=8,
        )


def test_runner_accepts_generated_main_and_smoke_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAMELYON16_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("WSI_LAB_ABLATION_WORKSPACE", str(tmp_path / "workspace"))
    runner._assert_source_pin()
    for mode in ("main", "smoke"):
        config_path, row = _write_config(tmp_path / f"{mode}.yaml", mode=mode)
        description = runner.describe_run(
            config_path,
            variant=row.variant,
            run_id=row.run_id,
            mode=mode,
        )
        assert description["run_id"] == row.run_id
        assert description["mode"] == mode
        assert description["variant"] == row.variant
        assert description["start"] == row.start
        assert description["seed"] == row.seed
        assert description["requested_timesteps"] == (2_000_000 if mode == "main" else 1_024)
        assert Path(description["artifact_dir"]).is_relative_to(tmp_path / "workspace")


@pytest.mark.parametrize("mode", ("main", "smoke"))
def test_protocol_accepts_canonical_configs_with_all_documented_device_differences(
    mode: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAMELYON16_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("WSI_LAB_ABLATION_WORKSPACE", str(tmp_path / "workspace"))
    row = main_runs()[0] if mode == "main" else smoke_runs()[0]
    config = launcher._config_for(
        row,
        ppo_device="cuda:1",
        belief_inference_device="cuda:2",
        preparation_device="cuda:3",
        tissue_device="cuda:4",
    )
    config_path = tmp_path / f"{mode}.yaml"
    config_path.write_text(launcher._config_text(config), encoding="utf-8")

    description = runner.describe_run(
        config_path,
        variant=row.variant,
        run_id=row.run_id,
        mode=mode,
    )
    resolved = runner._resolved_config(config_path, mode)

    assert description["run_id"] == row.run_id
    assert str(resolved.rl.device) == "cuda:1"
    assert str(resolved.rl.belief_device) == "cuda:2"
    assert str(resolved.belief.device) == "cuda:3"
    assert str(resolved.tissue.atlaspatch.device) == "cuda:4"


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("rl", "learning_rate"), 0.0005),
        (("belief", "learning_rate"), 0.0002),
        (("environment", "step_cost"), 0.002),
        (("tissue", "threshold"), 0.02),
        (("embeddings", "mapping_tolerance"), 1),
    ),
)
def test_describe_run_rejects_every_sampled_behavior_mutation(
    path: tuple[str, str],
    replacement: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAMELYON16_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("WSI_LAB_ABLATION_WORKSPACE", str(tmp_path / "workspace"))
    row = main_runs()[0]
    config = launcher._config_for(
        row,
        ppo_device="cpu",
        belief_inference_device="cpu",
        preparation_device="cpu",
        tissue_device="cpu",
    )
    config[path[0]][path[1]] = replacement
    config_path = tmp_path / "mutated.yaml"
    config_path.write_text(launcher._config_text(config), encoding="utf-8")

    with pytest.raises(
        runner.RunnerError,
        match="differs from the committed ablation protocol",
    ):
        runner.describe_run(
            config_path,
            variant=row.variant,
            run_id=row.run_id,
            mode=row.mode,
        )


def test_run_fingerprint_is_invariant_to_absolute_data_and_workspace_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = main_runs()[0]
    config = launcher._config_for(
        row,
        ppo_device="cpu",
        belief_inference_device="cpu",
        preparation_device="cpu",
        tissue_device="cpu",
    )
    config_path = tmp_path / "canonical.yaml"
    config_path.write_text(launcher._config_text(config), encoding="utf-8")

    descriptions: list[dict[str, Any]] = []
    for suffix in ("first", "second"):
        monkeypatch.setenv("CAMELYON16_ROOT", str(tmp_path / suffix / "data"))
        monkeypatch.setenv(
            "WSI_LAB_ABLATION_WORKSPACE",
            str(tmp_path / suffix / "workspace"),
        )
        descriptions.append(
            runner.describe_run(
                config_path,
                variant=row.variant,
                run_id=row.run_id,
                mode=row.mode,
            )
        )

    assert descriptions[0]["run_fingerprint"] == descriptions[1]["run_fingerprint"]
    assert descriptions[0]["workspace"] != descriptions[1]["workspace"]
    assert descriptions[0]["artifact_dir"] != descriptions[1]["artifact_dir"]


def test_run_fingerprint_changes_for_behavior_and_canonical_run_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAMELYON16_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("WSI_LAB_ABLATION_WORKSPACE", str(tmp_path / "workspace"))
    row = main_runs()[0]
    other_row = main_runs()[1]
    config_path, _ = _write_config(tmp_path / "canonical.yaml", mode="main")
    canonical_config = runner._resolved_config(config_path, "main")
    changed_config = deepcopy(canonical_config)
    changed_config.rl.learning_rate = float(canonical_config.rl.learning_rate) * 2
    spec = runner.get_variant_spec(row.variant)

    canonical = runner.fingerprint(
        run_contract_payload(
            canonical_config,
            spec,
            run_id=row.run_id,
            mode=row.mode,
        )
    )
    behavior_changed = runner.fingerprint(
        run_contract_payload(
            changed_config,
            spec,
            run_id=row.run_id,
            mode=row.mode,
        )
    )
    identity_changed = runner.fingerprint(
        run_contract_payload(
            canonical_config,
            spec,
            run_id=other_row.run_id,
            mode=row.mode,
        )
    )

    assert len({canonical, behavior_changed, identity_changed}) == 3


def test_belief_free_variants_drop_belief_checkpoint_dependency() -> None:
    for variant in ("v5_no_belief", "v6_combined"):
        dependencies = runner.variant_dependencies(variant)
        assert dependencies["ablation_variant_id"] == variant


def test_runner_restores_metrics_from_matching_durable_training_progress(
    tmp_path: Path,
) -> None:
    path = tmp_path / "training_progress.json"
    payload = _training_progress_payload()
    runner.atomic_json_dump(path, payload)

    restored = runner._load_training_progress(
        path,
        run_fingerprint=str(payload["run_fingerprint"]),
        variant=str(payload["variant"]),
        restored_step=int(payload["global_step"]),
        restored_update=int(payload["updates"]),
    )

    assert restored == payload["last_metrics"]


def test_runner_ignores_valid_progress_from_a_different_restored_step(
    tmp_path: Path,
) -> None:
    path = tmp_path / "training_progress.json"
    payload = _training_progress_payload(global_step=1_024)
    runner.atomic_json_dump(path, payload)

    restored = runner._load_training_progress(
        path,
        run_fingerprint=str(payload["run_fingerprint"]),
        variant=str(payload["variant"]),
        restored_step=2_048,
        restored_update=8,
    )

    assert restored is None


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("schema_version", 2),
        ("status", "running"),
        ("run_fingerprint", "b" * 64),
        ("variant", "v2_viewport_only"),
    ),
)
def test_runner_rejects_mismatched_training_progress_identity(
    field: str,
    replacement: Any,
    tmp_path: Path,
) -> None:
    path = tmp_path / "training_progress.json"
    payload = _training_progress_payload()
    payload[field] = replacement
    runner.atomic_json_dump(path, payload)

    with pytest.raises(runner.RunnerError, match="training progress identity mismatch"):
        runner._load_training_progress(
            path,
            run_fingerprint="a" * 64,
            variant="v1_full",
            restored_step=1_024,
            restored_update=4,
        )


@pytest.mark.parametrize(
    ("content", "message"),
    (
        ("not-json\n", "could not read durable training progress"),
        ("[]\n", "training progress is not a JSON object"),
    ),
)
def test_runner_rejects_malformed_training_progress(
    content: str,
    message: str,
    tmp_path: Path,
) -> None:
    path = tmp_path / "training_progress.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(runner.RunnerError, match=message):
        runner._load_training_progress(
            path,
            run_fingerprint="a" * 64,
            variant="v1_full",
            restored_step=1_024,
            restored_update=4,
        )


def test_runner_rejects_progress_update_that_disagrees_with_checkpoint(
    tmp_path: Path,
) -> None:
    path = tmp_path / "training_progress.json"
    payload = _training_progress_payload()
    runner.atomic_json_dump(path, payload)

    with pytest.raises(runner.RunnerError, match="update does not match"):
        runner._load_training_progress(
            path,
            run_fingerprint=str(payload["run_fingerprint"]),
            variant=str(payload["variant"]),
            restored_step=int(payload["global_step"]),
            restored_update=int(payload["updates"]) + 1,
        )


def test_completed_run_verifier_requires_durable_training_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAMELYON16_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("WSI_LAB_ABLATION_WORKSPACE", str(tmp_path / "workspace"))
    config_path, row = _write_config(tmp_path / "generated.yaml", mode="main")
    expected = runner.describe_run(
        config_path,
        variant=row.variant,
        run_id=row.run_id,
        mode=row.mode,
    )
    artifact_dir = Path(expected["artifact_dir"])
    artifact_dir.mkdir(parents=True)
    resolved = runner._resolved_config(config_path, row.mode)
    (artifact_dir / "resolved_config.yaml").write_text(
        runner._resolved_config_yaml(resolved),
        encoding="utf-8",
    )
    spec = runner.get_variant_spec(row.variant)
    runner.atomic_json_dump(artifact_dir / "variant_contract.json", runner.asdict(spec))
    last_metrics = _training_progress_payload()["last_metrics"]
    cache_root = Path(expected["workspace"]).parent / "analysis_cache"
    cache_root.mkdir()
    cache_manifest = cache_root / "manifest.json"
    monitor_cohort = cache_root / "monitor_slides.json"
    runner.atomic_json_dump(cache_manifest, {})
    runner.atomic_json_dump(monitor_cohort, {})
    dependencies = {
        **runner.variant_dependencies(spec),
        "ablation_run_fingerprint": expected["run_fingerprint"],
        "ablation_source_revision": runner.SOURCE_REVISION,
        "analysis_cache_manifest_sha256": verification._sha256_file(cache_manifest),
        "analysis_monitor_cohort_sha256": verification._sha256_file(monitor_cohort),
    }
    batch_size = int(resolved.rl.num_envs) * int(resolved.rl.rollout_steps)
    runner.atomic_json_dump(
        Path(expected["result_path"]),
        {
            **expected,
            "status": "complete",
            "source_revision": runner.SOURCE_REVISION,
            "split": "train",
            "validation_enabled": False,
            "test_access": False,
            "global_step": expected["expected_final_step"],
            "dependencies": dependencies,
            "last_metrics": last_metrics,
            "updates": expected["expected_final_step"] // batch_size,
            "completed_episodes": 0,
            "duration_seconds": 0,
        },
    )
    progress_path = artifact_dir / "training_progress.json"
    assert not progress_path.exists()

    def must_not_reach_later_artifacts(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("verification continued after finding no training progress")

    monkeypatch.setattr(verification, "_verify_checkpoint", must_not_reach_later_artifacts)
    monkeypatch.setattr(verification, "_verify_telemetry", must_not_reach_later_artifacts)
    monkeypatch.setattr(verification, "_verify_monitoring", must_not_reach_later_artifacts)

    with pytest.raises(verification.VerificationError, match=r"training_progress\.json"):
        verification.verify_completed_run(expected)


def test_verifier_accepts_exact_durable_training_progress(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    payload = _training_progress_payload()
    path = artifact_dir / "training_progress.json"
    runner.atomic_json_dump(path, payload)

    verified = verification._verify_training_progress(
        artifact_dir,
        run_fingerprint=str(payload["run_fingerprint"]),
        variant=str(payload["variant"]),
        final_step=int(payload["global_step"]),
        updates=int(payload["updates"]),
        last_metrics=payload["last_metrics"],
    )

    assert verified == path


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("run_fingerprint", "b" * 64),
        ("variant", "v2_viewport_only"),
        ("global_step", 2_048),
        ("updates", 5),
        ("last_metrics", {"synthetic_placeholder": 2}),
    ),
)
def test_verifier_rejects_training_progress_that_differs_from_final_manifest(
    field: str,
    replacement: Any,
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    payload = _training_progress_payload()
    payload[field] = replacement
    runner.atomic_json_dump(artifact_dir / "training_progress.json", payload)

    with pytest.raises(
        verification.VerificationError,
        match="durable training progress mismatch",
    ):
        verification._verify_training_progress(
            artifact_dir,
            run_fingerprint="a" * 64,
            variant="v1_full",
            final_step=1_024,
            updates=4,
            last_metrics={"synthetic_placeholder": 1},
        )


def test_run_contract_fingerprint_survives_resolved_config_persistence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAMELYON16_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("WSI_LAB_ABLATION_WORKSPACE", str(tmp_path / "workspace"))
    config_path, row = _write_config(tmp_path / "generated.yaml", mode="main")
    generated = runner._resolved_config(config_path, "main")
    persisted_path = tmp_path / "resolved_config.yaml"
    persisted_path.write_text(runner._resolved_config_yaml(generated), encoding="utf-8")
    persisted = runner.load_config(persisted_path)
    spec = runner.get_variant_spec(row.variant)

    generated_fingerprint = runner.fingerprint(
        run_contract_payload(
            generated,
            spec,
            run_id=row.run_id,
            mode=row.mode,
        )
    )
    persisted_fingerprint = runner.fingerprint(
        run_contract_payload(
            persisted,
            spec,
            run_id=row.run_id,
            mode=row.mode,
        )
    )

    assert persisted_fingerprint == generated_fingerprint
    assert (
        runner.describe_run(
            config_path,
            variant=row.variant,
            run_id=row.run_id,
            mode=row.mode,
        )["run_fingerprint"]
        == generated_fingerprint
    )


def test_verifier_rejects_a_missing_run_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAMELYON16_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("WSI_LAB_ABLATION_WORKSPACE", str(tmp_path / "workspace"))
    config_path, row = _write_config(tmp_path / "generated.yaml", mode="main")
    expected = runner.describe_run(
        config_path,
        variant=row.variant,
        run_id=row.run_id,
        mode=row.mode,
    )
    manifest_path = Path(expected["result_path"])
    assert not manifest_path.exists()

    with pytest.raises(verification.VerificationError, match="could not read JSON object"):
        verification.verify_completed_run(expected, manifest_path=manifest_path)


def test_existing_complete_run_is_verified_before_runner_returns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAMELYON16_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("WSI_LAB_ABLATION_WORKSPACE", str(tmp_path / "workspace"))
    config_path, row = _write_config(tmp_path / "generated.yaml", mode="main")
    expected = runner.describe_run(
        config_path,
        variant=row.variant,
        run_id=row.run_id,
        mode=row.mode,
    )
    manifest_path = Path(expected["result_path"])
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "run_fingerprint": expected["run_fingerprint"],
            }
        ),
        encoding="utf-8",
    )
    observed: dict[str, Any] = {}

    def reject_existing(expected_run: dict[str, Any], *, manifest_path: Path) -> None:
        observed.update(expected=expected_run, manifest_path=manifest_path)
        raise runner.VerificationError("synthetic verification rejection")

    monkeypatch.setattr(runner, "_assert_source_pin", lambda: None)
    monkeypatch.setattr(runner, "verify_completed_run", reject_existing)
    monkeypatch.setattr(
        runner,
        "_analysis_cache_root",
        lambda: pytest.fail("verified-complete path must not initialize training"),
    )

    with pytest.raises(
        runner.RunnerError,
        match="existing completed run failed artifact verification",
    ):
        runner.run_training(
            config_path,
            variant=row.variant,
            run_id=row.run_id,
            mode=row.mode,
        )

    assert observed == {"expected": expected, "manifest_path": manifest_path}


@pytest.mark.parametrize("status", ("interrupted", "failed", "running", None))
def test_runner_main_returns_nonzero_for_every_noncomplete_result(
    status: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, Any] = {}

    def fake_run_training(config_path: Path, **kwargs: Any) -> dict[str, Any]:
        captured.update(config_path=config_path, **kwargs)
        return {"status": status, "run_id": "c16-v1-full-center-s1"}

    monkeypatch.setattr(runner, "run_training", fake_run_training)
    monkeypatch.setattr(runner.signal, "signal", lambda *_args: None)
    config_path = tmp_path / "unused.yaml"

    returncode = runner.main(
        [
            "--config",
            str(config_path),
            "--variant",
            "v1_full",
            "--run-id",
            "c16-v1-full-center-s1",
            "--mode",
            "main",
        ]
    )

    assert returncode == 3
    assert captured == {
        "config_path": config_path,
        "variant": "v1_full",
        "run_id": "c16-v1-full-center-s1",
        "mode": "main",
    }
    assert json.loads(capsys.readouterr().out)["status"] == status


def test_runner_main_returns_zero_only_for_complete_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "run_training",
        lambda *_args, **_kwargs: {
            "status": "complete",
            "run_id": "c16-v1-full-center-s1",
        },
    )
    monkeypatch.setattr(runner.signal, "signal", lambda *_args: None)

    returncode = runner.main(
        [
            "--config",
            str(tmp_path / "unused.yaml"),
            "--variant",
            "v1_full",
            "--run-id",
            "c16-v1-full-center-s1",
            "--mode",
            "main",
        ]
    )

    assert returncode == 0
