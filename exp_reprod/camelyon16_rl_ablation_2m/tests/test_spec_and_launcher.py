"""Portable launcher and declarative protocol tests."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml
from ablation import launcher
from ablation.provenance import source_tree_digest
from ablation.spec import load_study, main_runs, select_runs, smoke_runs


def _atlaspatch_wrapper_constants() -> dict[str, object]:
    wrapper = launcher.REPRODUCTION_DIR / "bin" / "atlaspatch-pinned"
    tree = ast.parse(wrapper.read_text(encoding="utf-8"), filename=str(wrapper))
    constants: dict[str, object] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
    return constants


def _load_atlaspatch_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> ModuleType:
    path = launcher.REPRODUCTION_DIR / "bin" / "atlaspatch-pinned"
    name = "synthetic_atlaspatch_pinned_wrapper"
    loader = SourceFileLoader(name, str(path))
    specification = importlib.util.spec_from_loader(name, loader)
    assert specification is not None
    module = importlib.util.module_from_spec(specification)
    monkeypatch.setitem(sys.modules, name, module)
    loader.exec_module(module)
    return module


def test_canonical_camelyon16_inventory_has_all_399_exact_ids() -> None:
    development, official_test = launcher._canonical_inventory()
    expected_development = frozenset(
        [
            *(f"normal_{index:03d}" for index in range(1, 161) if index != 86),
            *(f"tumor_{index:03d}" for index in range(1, 112)),
        ]
    )
    expected_official_test = frozenset(
        f"test_{index:03d}" for index in range(1, 131) if index != 49
    )

    assert development == expected_development
    assert official_test == expected_official_test
    assert len(development) == 270
    assert len(official_test) == 129
    assert development.isdisjoint(official_test)
    assert len(development | official_test) == 399


def test_reference_rows_rejects_missing_required_columns(tmp_path: Path) -> None:
    reference = tmp_path / "reference.csv"
    reference.write_text(
        "image,type,class\nnormal_001.tif,normal,negative\n",
        encoding="utf-8",
    )

    with pytest.raises(launcher.LauncherError, match="missing required columns"):
        launcher._reference_rows(reference)


@pytest.mark.parametrize(
    "rows",
    (
        "normal_001.tif,normal,negative,0\nNORMAL_001.TIF,normal,negative,0\n",
        ",normal,negative,0\n",
    ),
    ids=("duplicate-casefolded-id", "missing-id"),
)
def test_reference_rows_rejects_missing_or_duplicate_slide_ids(
    rows: str,
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.csv"
    reference.write_text(f"image,type,class,center\n{rows}", encoding="utf-8")

    with pytest.raises(launcher.LauncherError, match="missing or duplicate slide ID"):
        launcher._reference_rows(reference)


@pytest.mark.parametrize(
    ("slide_id", "expected"),
    (
        ("normal_100", ("normal", "negative", "0")),
        ("normal_101", ("normal", "negative", "1")),
        ("normal_144", ("normal", "negative", "0")),
        ("normal_145", ("normal", "negative", "1")),
        ("tumor_070", ("tumor", "tumor", "0")),
        ("tumor_071", ("tumor", "tumor", "1")),
        ("tumor_111", ("tumor", "tumor", "0")),
    ),
)
def test_canonical_development_metadata_center_boundaries_and_special_cases(
    slide_id: str,
    expected: tuple[str, str, str],
) -> None:
    assert launcher._expected_development_metadata(slide_id) == expected


def test_exact_result_free_matrix() -> None:
    study = load_study()
    rows = main_runs(study)
    assert len(rows) == 36
    assert [row.index for row in rows] == list(range(36))
    assert len({row.run_id for row in rows}) == 36
    for seed_index, seed in enumerate((1, 2, 3)):
        tier = rows[seed_index * 12 : (seed_index + 1) * 12]
        assert {row.seed for row in tier} == {seed}
        assert {(row.variant, row.start) for row in tier} == {
            (variant["id"], start) for variant in study["variants"] for start in study["starts"]
        }
    assert [(row.variant, row.start, row.seed) for row in smoke_runs(study)] == [
        ("v1_full", "center", 1),
        ("v2_viewport_only", "random", 1),
        ("v3_no_tissue_mask", "random", 1),
        ("v4_no_diagnosis", "center", 1),
        ("v5_no_belief", "center", 1),
        ("v6_combined", "random", 1),
    ]


def test_filter_preserves_canonical_indices_and_order() -> None:
    selected = select_runs(
        main_runs(), variants=("v3_no_tissue_mask",), starts=("random",), seeds=(2, 3)
    )
    assert [(row.index, row.seed) for row in selected] == [(17, 2), (29, 3)]


def test_base_config_contains_protocol_not_machine_paths() -> None:
    value = yaml.safe_load(launcher.BASE_CONFIG_PATH.read_text(encoding="utf-8"))
    assert value["dataset"]["root"] == "${CAMELYON16_ROOT}"
    assert value["workspace"]["root"] == "${WSI_LAB_ABLATION_WORKSPACE}"
    assert value["environment"]["max_episode_steps"] == 500
    assert value["rl"]["total_timesteps"] == 2_000_000
    assert value["rl"]["num_envs"] == 8
    assert value["rl"]["rollout_steps"] == 128
    assert value["rl"]["sequence_length"] == 32
    assert value["rl"]["num_minibatches"] == 4
    assert value["logging"]["wandb"] is False


def test_base_config_uses_executable_pinned_atlaspatch_wrapper() -> None:
    value = yaml.safe_load(launcher.BASE_CONFIG_PATH.read_text(encoding="utf-8"))
    wrapper = launcher.REPRODUCTION_DIR / "bin" / "atlaspatch-pinned"

    assert value["tissue"]["source"] == "atlaspatch"
    assert value["tissue"]["atlaspatch"]["executable"] == "atlaspatch-pinned"
    assert wrapper.is_file()
    assert wrapper.stat().st_mode & 0o111


@pytest.mark.parametrize("filename", ("environment.yml", "environment.cpu.yml"))
def test_environment_recipe_matches_portable_software_contract(filename: str) -> None:
    study = load_study()
    software = study["software_reproduction"]
    value = yaml.safe_load((launcher.REPRODUCTION_DIR / filename).read_text(encoding="utf-8"))
    dependencies = value["dependencies"]
    conda_specs = [item for item in dependencies if isinstance(item, str)]
    pip_specs = next(item["pip"] for item in dependencies if isinstance(item, dict))

    assert f"python={software['python']}" in conda_specs
    for name in ("numpy", "openslide-python"):
        assert f"{name}={software['distributions'][name]}" in conda_specs
    assert any(
        item.startswith(f"pytorch::pytorch={software['distributions']['torch']}=")
        for item in conda_specs
    )
    assert any(
        item.startswith(f"pytorch::torchvision={software['distributions']['torchvision']}=")
        for item in conda_specs
    )

    conda_distributions = {"numpy", "openslide-python", "torch", "torchvision"}
    for name, version in software["distributions"].items():
        if name in conda_distributions or name == "SAM-2":
            continue
        assert f"{name}=={version}".casefold() in {item.casefold() for item in pip_specs}
    sam2_commit = study["atlaspatch"]["sam2_commit"]
    assert any(
        item.startswith("SAM-2 @ git+") and item.endswith(f"@{sam2_commit}") for item in pip_specs
    )


def test_launcher_keeps_wrapper_path_and_huggingface_cache_in_scoped_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inherited_bin = tmp_path / "inherited-bin"
    monkeypatch.setenv("PATH", str(inherited_bin))
    work_root = tmp_path / "work"

    environment = launcher._environment(
        source_root=tmp_path / "source",
        data_root=tmp_path / "data",
        work_root=work_root,
    )

    path_entries = environment["PATH"].split(launcher.os.pathsep)
    wrapper_bin = Path(path_entries[0]).resolve()
    huggingface_home = Path(environment["HF_HOME"]).resolve()
    assert wrapper_bin == (launcher.REPRODUCTION_DIR / "bin").resolve()
    assert wrapper_bin.is_relative_to(launcher.REPRODUCTION_DIR)
    assert path_entries[1] == str(inherited_bin)
    assert huggingface_home == (work_root / "external_cache" / "huggingface").resolve()
    assert huggingface_home.is_relative_to(work_root.resolve())


def test_study_atlaspatch_model_constants_match_wrapper_source_without_importing_it() -> None:
    atlaspatch = load_study()["atlaspatch"]
    constants = _atlaspatch_wrapper_constants()

    assert {
        "REPOSITORY": atlaspatch["model_repository"],
        "FILENAME": atlaspatch["model_filename"],
        "SNAPSHOT": atlaspatch["model_snapshot"],
        "SHA256": atlaspatch["model_sha256"],
        "SIZE": atlaspatch["model_size"],
    }.items() <= constants.items()
    assert len(str(constants["SNAPSHOT"])) == 40
    assert len(str(constants["SHA256"])) == 64


def test_atlaspatch_download_guard_pins_revision_and_checks_tiny_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _load_atlaspatch_wrapper(monkeypatch)
    snapshot = "synthetic-snapshot"
    payload = b"synthetic pinned model payload"
    checkpoint = tmp_path / "models--synthetic" / "snapshots" / snapshot / str(wrapper.FILENAME)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(payload)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_download(*args: object, **kwargs: object) -> str:
        calls.append((args, dict(kwargs)))
        return str(checkpoint)

    atlas_patch = ModuleType("atlas_patch")
    services = ModuleType("atlas_patch.services")
    segmentation = ModuleType("atlas_patch.services.segmentation")
    segmentation.hf_hub_download = fake_download
    services.segmentation = segmentation
    atlas_patch.services = services
    monkeypatch.setitem(sys.modules, "atlas_patch", atlas_patch)
    monkeypatch.setitem(sys.modules, "atlas_patch.services", services)
    monkeypatch.setitem(sys.modules, "atlas_patch.services.segmentation", segmentation)
    monkeypatch.setattr(wrapper, "SNAPSHOT", snapshot)
    monkeypatch.setattr(wrapper, "SIZE", len(payload))
    monkeypatch.setattr(wrapper, "SHA256", hashlib.sha256(payload).hexdigest())

    wrapper._install_download_guard()
    guarded_download = segmentation.hf_hub_download
    observed = guarded_download(wrapper.REPOSITORY, wrapper.FILENAME)

    assert observed == str(checkpoint.resolve())
    assert calls == [
        (
            (wrapper.REPOSITORY, wrapper.FILENAME),
            {"revision": snapshot},
        )
    ]
    assert (checkpoint.parents[2] / "refs" / "main").read_text(encoding="utf-8") == snapshot

    with pytest.raises(RuntimeError, match="different tissue-model revision"):
        guarded_download(
            wrapper.REPOSITORY,
            wrapper.FILENAME,
            revision="different-snapshot",
        )
    checkpoint.write_bytes(b"corrupted")
    with pytest.raises(RuntimeError, match="checksum does not match"):
        guarded_download(wrapper.REPOSITORY, wrapper.FILENAME)


def test_source_tree_checksum_matches_recorded_implementation() -> None:
    study = load_study()
    observed, count = source_tree_digest(launcher.DEFAULT_SOURCE_ROOT)
    assert count == study["source_tree"]["python_files"]
    assert observed == study["source_tree"]["sha256"]


def test_dry_run_needs_no_dataset_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    work_root = tmp_path / "new-workspace"
    returncode = launcher.main(
        [
            "run",
            "--data-root",
            str(tmp_path / "missing-data"),
            "--work-root",
            str(work_root),
            "--device",
            "cpu",
            "--variants",
            "v6_combined",
            "--starts",
            "center",
            "--seeds",
            "1",
            "--dry-run",
        ]
    )
    output = capsys.readouterr().out
    assert returncode == 0
    assert "c16-v6-combined-center-s1" in output
    assert "no files were written" in output
    assert not work_root.exists()


def test_work_root_inside_publication_package_is_rejected(
    capsys: pytest.CaptureFixture[str],
) -> None:
    returncode = launcher.main(
        [
            "run",
            "--data-root",
            "unused",
            "--work-root",
            str(launcher.REPRODUCTION_DIR / "generated"),
            "--dry-run",
        ]
    )
    assert returncode == 2
    assert "outside the committed reproduction package" in capsys.readouterr().err


def test_run_config_reuses_preparation_fingerprint_fields() -> None:
    preparation = launcher._config_for(
        launcher._preparation_row(),
        ppo_device="cpu",
        belief_inference_device="cpu",
        preparation_device="cpu",
        tissue_device="cpu",
    )
    run = launcher._config_for(
        main_runs()[1],
        ppo_device="cuda:1",
        belief_inference_device="cpu",
        template=copy.deepcopy(preparation),
    )
    assert run["belief"] == preparation["belief"]
    assert run["tissue"] == preparation["tissue"]
    assert run["rl"]["device"] == "cuda:1"
    assert run["environment"]["start_position"] == "random"


def test_exact_generated_preparation_config_is_accepted_with_portable_devices() -> None:
    generated = launcher._config_for(
        launcher._preparation_row(),
        ppo_device="cpu",
        belief_inference_device="cuda:1",
        preparation_device="cuda:2",
        tissue_device="cuda:3",
    )

    validated = launcher._validated_preparation_template(copy.deepcopy(generated))

    assert validated == generated


def test_mutated_preparation_config_outside_device_fields_is_rejected() -> None:
    generated = launcher._config_for(
        launcher._preparation_row(),
        ppo_device="cpu",
        belief_inference_device="cpu",
        preparation_device="cpu",
        tissue_device="cpu",
    )
    generated["environment"]["max_episode_steps"] = 499

    with pytest.raises(launcher.LauncherError, match="preparation config"):
        launcher._validated_preparation_template(generated)


def test_zero_exit_training_fails_when_post_run_verification_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = main_runs()[0]
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(returncode=0 if len(calls) == 1 else 9)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)

    run_id, returncode = launcher._run_one(row, tmp_path / "run.yaml", {})

    assert run_id == row.run_id
    assert returncode == 9
    assert len(calls) == 2
    assert calls[0][0:3] == [launcher.sys.executable, "-m", "ablation.runner"]
    assert calls[1][0:3] == [launcher.sys.executable, "-m", "ablation.verification"]
    assert calls[0][-6:] == [
        "--variant",
        row.variant,
        "--run-id",
        row.run_id,
        "--mode",
        row.mode,
    ]
    assert calls[1][-6:] == calls[0][-6:]
