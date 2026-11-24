"""Shared immutable protocol payloads used for run fingerprinting."""

from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from .analysis_cache import SCHEMA_VERSION as ANALYSIS_SCHEMA_VERSION
from .spec import load_study
from .telemetry import SCHEMA_VERSION as TELEMETRY_SCHEMA_VERSION
from .variants import VariantSpec

SOURCE_REVISION = str(load_study()["source_revision"])
REPRODUCTION_DIR = Path(__file__).resolve().parent.parent
BASE_CONFIG_PATH = REPRODUCTION_DIR / "base_config.yaml"
RUNNER_SCHEMA_VERSION = 4
TRAINING_PROGRESS_SCHEMA_VERSION = 1
MONITORING_CONTRACT = {
    "schema_version": 2,
    "split": "train",
    "cohort_seed": 20_260_831,
    "main_cohort_size": 30,
    "main_positive_slides": 15,
    "main_milestones": (100_000, 1_000_000, 2_000_000),
    "smoke_cohort_size": 2,
    "smoke_positive_slides": 1,
    "smoke_milestones": (1_024,),
    "deterministic_policy": True,
    "random_start_seed": 20_260_831,
}
SMOKE_OVERRIDES = {
    "num_envs": 2,
    "rollout_steps": 64,
    "sequence_length": 16,
    "num_minibatches": 2,
    "total_timesteps": 1_024,
    "checkpoint_interval": 1_024,
    "validation_interval": 2_048,
}


def _behavior_config(config: Any) -> dict[str, Any]:
    """Remove machine locations while retaining every behavior-setting field."""

    value = config.model_dump(mode="json")
    for name in (
        "root",
        "camelyon16_root",
        "camelyon17_root",
        "camelyon17_labels",
        "slides_dir",
        "labels",
        "tissue_masks_dir",
        "tumor_masks_dir",
        "annotations_dir",
        "luad_dir",
        "lusc_dir",
    ):
        value["dataset"][name] = f"<DATA:{name}>"
    value["embeddings"]["root"] = "<DATA:embeddings>"
    value["workspace"]["root"] = "<WORKSPACE>"
    if value["environment"].get("persistent_rgb_cache") is not None:
        value["environment"]["persistent_rgb_cache"] = "<WORKSPACE:rgb-cache>"
    return value


def _canonical_row(run_id: str, mode: str) -> Any:
    from .spec import main_runs, smoke_runs

    rows = main_runs() if mode == "main" else smoke_runs() if mode == "smoke" else ()
    matches = [row for row in rows if row.run_id == str(run_id)]
    if len(matches) != 1:
        raise ValueError(f"run_id is not a canonical {mode} study cell: {run_id}")
    return matches[0]


def assert_protocol_config(
    config: Any,
    *,
    variant: str,
    run_id: str,
    mode: str,
) -> None:
    """Require the committed protocol, allowing only paths and documented devices."""

    row = _canonical_row(run_id, mode)
    if row.variant != str(variant):
        raise ValueError("run variant does not match its canonical matrix cell")
    try:
        raw = yaml.safe_load(BASE_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"could not read committed base protocol: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("committed base protocol is not a mapping")
    expected_raw = copy.deepcopy(raw)
    expected_raw["dataset"]["root"] = str(config.dataset.root)
    expected_raw["workspace"]["root"] = str(config.workspace.root)
    expected_raw["environment"]["start_position"] = row.start
    expected_raw["rl"]["seed"] = row.seed
    for name, value in SMOKE_OVERRIDES.items() if mode == "smoke" else ():
        expected_raw["rl"][name] = value
    expected_raw["rl"]["device"] = str(config.rl.device)
    expected_raw["rl"]["belief_device"] = str(config.rl.belief_device)
    expected_raw["belief"]["device"] = str(config.belief.device)
    expected_raw["tissue"]["atlaspatch"]["device"] = str(config.tissue.atlaspatch.device)
    try:
        from wsi_lab.config.loader import _resolve_paths
        from wsi_lab.config.models import AppConfig

        expected = _resolve_paths(AppConfig.model_validate(expected_raw), BASE_CONFIG_PATH)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"could not construct committed protocol configuration: {exc}") from exc
    actual_payload = config.model_dump(mode="json")
    expected_payload = expected.model_dump(mode="json")
    if actual_payload != expected_payload:
        differing_sections = sorted(
            name
            for name in set(actual_payload) | set(expected_payload)
            if actual_payload.get(name) != expected_payload.get(name)
        )
        raise ValueError(
            "resolved configuration differs from the committed ablation protocol in "
            f"sections: {differing_sections}"
        )


def run_contract_payload(
    config: Any,
    spec: VariantSpec,
    *,
    run_id: str,
    mode: str,
) -> dict[str, Any]:
    """Return the complete path-independent behavior fingerprint payload."""

    study = load_study()
    return {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "source_revision": SOURCE_REVISION,
        "run_id": str(run_id),
        "mode": str(mode),
        "variant": asdict(spec),
        "config": _behavior_config(config),
        "monitoring": MONITORING_CONTRACT,
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "telemetry_schema_version": TELEMETRY_SCHEMA_VERSION,
        "training_progress_schema_version": TRAINING_PROGRESS_SCHEMA_VERSION,
        "atlaspatch": study["atlaspatch"],
        "software_reference": study["software_reference"],
        "software_reproduction": study["software_reproduction"],
    }


__all__ = [
    "MONITORING_CONTRACT",
    "RUNNER_SCHEMA_VERSION",
    "SMOKE_OVERRIDES",
    "SOURCE_REVISION",
    "TRAINING_PROGRESS_SCHEMA_VERSION",
    "assert_protocol_config",
    "run_contract_payload",
]
