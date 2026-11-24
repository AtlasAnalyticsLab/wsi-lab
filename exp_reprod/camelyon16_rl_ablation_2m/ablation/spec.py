"""Declarative run-matrix helpers for the portable ablation study."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPRODUCTION_DIR = Path(__file__).resolve().parent.parent
STUDY_PATH = REPRODUCTION_DIR / "study.json"


class StudyError(RuntimeError):
    """The committed study definition is inconsistent."""


@dataclass(frozen=True, slots=True)
class RunSpec:
    """One deterministic cell in the main or smoke matrix."""

    mode: str
    index: int
    variant: str
    start: str
    seed: int

    @property
    def run_id(self) -> str:
        prefix = "c16" if self.mode == "main" else "c16-smoke"
        return f"{prefix}-{self.variant.replace('_', '-')}-{self.start}-s{self.seed}"

    @property
    def config_name(self) -> str:
        return f"{self.index:02d}_{self.run_id}.yaml"


def load_study(path: str | Path = STUDY_PATH) -> dict[str, Any]:
    """Load and validate the public, result-free study definition."""

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StudyError(f"could not load study definition: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise StudyError("study.json must be an object with schema_version 1")
    validate_study(value)
    return value


def main_runs(study: Mapping[str, Any] | None = None) -> tuple[RunSpec, ...]:
    """Return seed-major Cartesian order used by the recorded experiment."""

    value = load_study() if study is None else study
    variants = tuple(str(item["id"]) for item in value["variants"])
    starts = tuple(str(item) for item in value["starts"])
    seeds = tuple(int(item) for item in value["seeds"])
    rows: list[RunSpec] = []
    for seed_index, seed in enumerate(seeds):
        for variant_index, variant in enumerate(variants):
            for start_index, start in enumerate(starts):
                index = seed_index * len(variants) * len(starts)
                index += variant_index * len(starts) + start_index
                rows.append(RunSpec("main", index, variant, start, seed))
    return tuple(rows)


def smoke_runs(study: Mapping[str, Any] | None = None) -> tuple[RunSpec, ...]:
    """Return the six-cell smoke matrix, one representative per variant."""

    value = load_study() if study is None else study
    return tuple(
        RunSpec(
            "smoke",
            index,
            str(item["variant"]),
            str(item["start"]),
            int(item["seed"]),
        )
        for index, item in enumerate(value["smoke"]["mapping"])
    )


def select_runs(
    rows: Iterable[RunSpec],
    *,
    variants: Iterable[str] | None = None,
    starts: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
) -> tuple[RunSpec, ...]:
    """Filter rows without changing their canonical order or indices."""

    variant_set = None if variants is None else frozenset(variants)
    start_set = None if starts is None else frozenset(starts)
    seed_set = None if seeds is None else frozenset(int(seed) for seed in seeds)
    return tuple(
        row
        for row in rows
        if (variant_set is None or row.variant in variant_set)
        and (start_set is None or row.start in start_set)
        and (seed_set is None or row.seed in seed_set)
    )


def validate_study(study: Mapping[str, Any]) -> None:
    """Fail closed if any publication-critical protocol value drifts."""

    revision = study.get("source_revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise StudyError("source_revision must be a full 40-character source identifier")
    source_tree = study.get("source_tree", {})
    if source_tree.get("algorithm") != "sha256sum-lines-v1":
        raise StudyError("source-tree checksum algorithm changed")
    if source_tree.get("python_files") != 86:
        raise StudyError("recorded source-tree file count changed")
    tree_sha256 = source_tree.get("sha256")
    if not isinstance(tree_sha256, str) or len(tree_sha256) != 64:
        raise StudyError("source-tree SHA-256 is invalid")

    expected_variants = (
        "v1_full",
        "v2_viewport_only",
        "v3_no_tissue_mask",
        "v4_no_diagnosis",
        "v5_no_belief",
        "v6_combined",
    )
    variants = tuple(str(item["id"]) for item in study.get("variants", ()))
    if variants != expected_variants:
        raise StudyError(f"variant order changed: {variants}")
    if tuple(study.get("starts", ())) != ("center", "random"):
        raise StudyError("starts must be center, random")
    if tuple(study.get("seeds", ())) != (1, 2, 3):
        raise StudyError("PPO seeds must be 1, 2, 3")
    main = study.get("main", {})
    if main.get("requested_timesteps") != 2_000_000:
        raise StudyError("main runs must request 2,000,000 timesteps")
    if main.get("checkpoint_interval") != 100_000:
        raise StudyError("main checkpoint interval must be 100,000")
    environment = study.get("environment", {})
    if environment != {
        "max_episode_steps": 500,
        "reward_type": "default",
        "vector_backend": "sync",
    }:
        raise StudyError("environment contract changed")
    evaluation = study.get("evaluation", {})
    if evaluation != {
        "validation_enabled": False,
        "official_test_access": False,
    }:
        raise StudyError("validation and official-test access must remain disabled")

    rows = main_runs(study)
    if len(rows) != 36 or main.get("run_count") != 36:
        raise StudyError("main matrix must contain 36 runs")
    if tuple(row.index for row in rows) != tuple(range(36)):
        raise StudyError("main matrix indices must be contiguous 0..35")
    cells = Counter((row.variant, row.start) for row in rows)
    if len(cells) != 12 or set(cells.values()) != {3}:
        raise StudyError("every variant/start cell must contain exactly three seeds")
    if len({row.run_id for row in rows}) != len(rows):
        raise StudyError("main run IDs must be unique")

    smoke = study.get("smoke", {})
    expected_smoke = (
        ("v1_full", "center", 1),
        ("v2_viewport_only", "random", 1),
        ("v3_no_tissue_mask", "random", 1),
        ("v4_no_diagnosis", "center", 1),
        ("v5_no_belief", "center", 1),
        ("v6_combined", "random", 1),
    )
    observed_smoke = tuple((row.variant, row.start, row.seed) for row in smoke_runs(study))
    if observed_smoke != expected_smoke:
        raise StudyError("smoke mapping changed")
    expected_smoke_shape = {
        "requested_timesteps": 1_024,
        "num_envs": 2,
        "rollout_steps": 64,
        "sequence_length": 16,
        "num_minibatches": 2,
    }
    if any(smoke.get(name) != expected for name, expected in expected_smoke_shape.items()):
        raise StudyError("smoke PPO geometry changed")


__all__ = [
    "STUDY_PATH",
    "RunSpec",
    "StudyError",
    "load_study",
    "main_runs",
    "select_runs",
    "smoke_runs",
    "validate_study",
]
