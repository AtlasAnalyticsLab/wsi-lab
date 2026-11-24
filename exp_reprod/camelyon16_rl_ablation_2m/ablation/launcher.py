"""Portable command-line orchestration for the CAMELYON16 ablation study."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

from .provenance import (
    RuntimeCompatibilityError,
    SourceCompatibilityError,
    verify_runtime_environment,
    verify_source_tree,
)
from .spec import RunSpec, load_study, main_runs, select_runs, smoke_runs

REPRODUCTION_DIR = Path(__file__).resolve().parent.parent
BASE_CONFIG_PATH = REPRODUCTION_DIR / "base_config.yaml"
DEFAULT_SOURCE_ROOT = Path(
    os.environ.get("WSI_LAB_SOURCE_ROOT", REPRODUCTION_DIR.parents[1] / "wsi-lab")
).expanduser().resolve()
DEFAULT_WORK_ROOT = DEFAULT_SOURCE_ROOT / "workspace" / "exp_reprod" / "camelyon16_rl_ablation_2m"


class LauncherError(RuntimeError):
    """A portable-launch or reproducibility invariant was violated."""


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _require_external_work_root(path: str | Path) -> Path:
    root = _resolved(path)
    if root == REPRODUCTION_DIR or root.is_relative_to(REPRODUCTION_DIR):
        raise LauncherError("the work root must be outside the committed reproduction package")
    repository_root = DEFAULT_SOURCE_ROOT
    ignored_workspace = repository_root / "workspace"
    if root.is_relative_to(repository_root) and not root.is_relative_to(ignored_workspace):
        raise LauncherError(
            "a work root inside this repository must be beneath its ignored workspace/ directory"
        )
    return root


def _source_compatibility(source_root: Path) -> None:
    source = load_study()["source_tree"]
    try:
        verify_source_tree(
            source_root,
            expected_sha256=str(source["sha256"]),
            expected_files=int(source["python_files"]),
        )
    except (KeyError, TypeError, SourceCompatibilityError) as exc:
        raise LauncherError(
            "the selected source root does not contain the exact WSI-Lab "
            "implementation recorded in study.json; re-extract the unmodified "
            "full-source submission ZIP"
        ) from exc


def _runtime_compatibility() -> None:
    try:
        verify_runtime_environment(load_study())
    except RuntimeCompatibilityError as exc:
        raise LauncherError(str(exc)) from exc


def _canonical_inventory() -> tuple[frozenset[str], frozenset[str]]:
    development = {
        *(f"normal_{index:03d}" for index in range(1, 161) if index != 86),
        *(f"tumor_{index:03d}" for index in range(1, 112)),
    }
    official_test = {f"test_{index:03d}" for index in range(1, 131) if index != 49}
    return frozenset(development), frozenset(official_test)


def _expected_development_metadata(slide_id: str) -> tuple[str, str, str]:
    prefix, raw_index = slide_id.split("_", maxsplit=1)
    index = int(raw_index)
    if prefix == "normal":
        center = "0" if index <= 100 or index == 144 else "1"
        return "normal", "negative", center
    center = "0" if index <= 70 or index == 111 else "1"
    return "tumor", "tumor", center


def _reference_rows(path: Path) -> dict[str, dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"image", "type", "class", "center"}
            if not required.issubset(reader.fieldnames or ()):
                raise LauncherError("CAMELYON16 reference.csv is missing required columns")
            rows: dict[str, dict[str, str]] = {}
            for row in reader:
                image = str(row.get("image", "")).strip()
                slide_id = Path(image).stem.casefold()
                if not slide_id or slide_id in rows:
                    raise LauncherError(
                        "CAMELYON16 reference.csv has a missing or duplicate slide ID"
                    )
                rows[slide_id] = {
                    name: str(row.get(name, "")).strip().casefold() for name in required
                }
    except OSError as exc:
        raise LauncherError(f"could not read CAMELYON16 reference.csv: {exc}") from exc
    return rows


def _dataset_requirements(data_root: Path) -> None:
    required_files = (data_root / "evaluation" / "reference.csv",)
    required_directories = (
        data_root / "images",
        data_root / "Embeddings",
        data_root / "masks",
        data_root / "annotations",
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    missing.extend(str(path) for path in required_directories if not path.is_dir())
    if missing:
        rendered = "\n  - ".join(missing)
        raise LauncherError(f"CAMELYON16 prerequisites are missing:\n  - {rendered}")
    rows = _reference_rows(required_files[0])
    development, official_test = _canonical_inventory()
    expected_inventory = development | official_test
    if set(rows) != expected_inventory:
        absent = sorted(expected_inventory - set(rows))
        unexpected = sorted(set(rows) - expected_inventory)
        raise LauncherError(
            "CAMELYON16 reference.csv does not match the canonical 399-slide inventory "
            f"(missing={absent[:8]}, unexpected={unexpected[:8]})"
        )
    metadata_errors: list[str] = []
    for slide_id in sorted(development):
        row = rows[slide_id]
        expected_type, expected_class, expected_center = _expected_development_metadata(slide_id)
        if row["image"] != f"{slide_id}.tif":
            metadata_errors.append(f"{slide_id}: image")
        if row["type"] != expected_type:
            metadata_errors.append(f"{slide_id}: type")
        if expected_class == "negative":
            if row["class"] != expected_class:
                metadata_errors.append(f"{slide_id}: class")
        elif row["class"] not in {"micro", "macro"}:
            metadata_errors.append(f"{slide_id}: class")
        if row["center"] != expected_center:
            metadata_errors.append(f"{slide_id}: center")
    if metadata_errors:
        raise LauncherError(
            "CAMELYON16 development metadata differs from the canonical split inputs: "
            f"{metadata_errors[:8]}"
        )
    missing_inputs: list[Path] = []
    for slide_id in sorted(development):
        image_name = rows[slide_id]["image"]
        image_path = data_root / "images" / image_name
        if not image_path.is_file():
            missing_inputs.append(image_path)
        for magnification in ("2.5", "5", "10", "20"):
            embedding = (
                data_root
                / "Embeddings"
                / f"UNI_v2_{magnification}x_256"
                / "patches"
                / f"{slide_id}.h5"
            )
            if not embedding.is_file():
                missing_inputs.append(embedding)
        if slide_id.startswith("tumor_"):
            mask = data_root / "masks" / f"{slide_id}_mask.tif"
            annotation = data_root / "annotations" / f"{slide_id}.xml"
            if not mask.is_file():
                missing_inputs.append(mask)
            if not annotation.is_file():
                missing_inputs.append(annotation)
    if missing_inputs:
        rendered = "\n  - ".join(str(path) for path in missing_inputs[:12])
        remaining = len(missing_inputs) - min(len(missing_inputs), 12)
        suffix = f"\n  - ... and {remaining} more" if remaining else ""
        raise LauncherError("canonical development inputs are missing:\n  - " + rendered + suffix)


def _environment(
    *,
    source_root: Path,
    data_root: Path,
    work_root: Path,
) -> dict[str, str]:
    environment = os.environ.copy()
    external_cache = work_root / "external_cache" / "huggingface"
    package_path = str(REPRODUCTION_DIR)
    source_path = str(source_root / "src")
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (source_path, package_path, existing) if item
    )
    environment["PATH"] = os.pathsep.join(
        item for item in (str(REPRODUCTION_DIR / "bin"), environment.get("PATH")) if item
    )
    environment.update(
        {
            "CAMELYON16_ROOT": str(data_root),
            "WSI_LAB_SOURCE_ROOT": str(source_root),
            "WSI_LAB_ABLATION_WORK_ROOT": str(work_root),
            "WSI_LAB_ABLATION_WORKSPACE": str(work_root / "shared_workspace"),
            "WSI_LAB_ABLATION_ANALYSIS_CACHE": str(work_root / "analysis_cache"),
            "HF_HOME": str(external_cache),
            "HF_HUB_CACHE": str(external_cache / "hub"),
            "HUGGINGFACE_HUB_CACHE": str(external_cache / "hub"),
            "HF_XET_CACHE": str(external_cache / "xet"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return environment


def _load_base_config() -> dict[str, Any]:
    try:
        value = yaml.safe_load(BASE_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise LauncherError(f"could not load base_config.yaml: {exc}") from exc
    if not isinstance(value, dict):
        raise LauncherError("base_config.yaml must contain a mapping")
    return value


def _config_for(
    row: RunSpec,
    *,
    ppo_device: str,
    belief_inference_device: str,
    preparation_device: str | None = None,
    tissue_device: str | None = None,
    template: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = copy.deepcopy(dict(template)) if template is not None else _load_base_config()
    config["environment"]["start_position"] = row.start
    config["rl"]["seed"] = row.seed
    config["rl"]["device"] = ppo_device
    config["rl"]["belief_device"] = belief_inference_device
    if preparation_device is not None:
        config["belief"]["device"] = preparation_device
    if tissue_device is not None:
        config["tissue"]["atlaspatch"]["device"] = tissue_device
    if row.mode == "smoke":
        smoke = load_study()["smoke"]
        for key in (
            "num_envs",
            "rollout_steps",
            "sequence_length",
            "num_minibatches",
        ):
            config["rl"][key] = int(smoke[key])
        config["rl"]["total_timesteps"] = int(smoke["requested_timesteps"])
        config["rl"]["checkpoint_interval"] = int(smoke["requested_timesteps"])
        config["rl"]["validation_interval"] = int(smoke["requested_timesteps"]) * 2
    return config


def _config_text(config: Mapping[str, Any]) -> str:
    return yaml.safe_dump(
        dict(config),
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=False,
    )


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _config_path(work_root: Path, row: RunSpec) -> Path:
    return work_root / "generated_configs" / row.mode / row.config_name


def _write_config(
    work_root: Path,
    row: RunSpec,
    *,
    device: str,
    belief_inference_device: str,
    template: Mapping[str, Any],
) -> Path:
    path = _config_path(work_root, row)
    _atomic_text(
        path,
        _config_text(
            _config_for(
                row,
                ppo_device=device,
                belief_inference_device=belief_inference_device,
                template=template,
            )
        ),
    )
    return path


def _preparation_row() -> RunSpec:
    return RunSpec("main", 0, "v1_full", "center", 1)


def _preparation_config_path(work_root: Path) -> Path:
    return work_root / "generated_configs" / "preparation.yaml"


def _write_preparation_config(
    work_root: Path,
    *,
    device: str,
    belief_inference_device: str,
    tissue_device: str,
) -> Path:
    path = _preparation_config_path(work_root)
    config = _config_for(
        _preparation_row(),
        ppo_device=device,
        belief_inference_device=belief_inference_device,
        preparation_device=device,
        tissue_device=tissue_device,
    )
    _atomic_text(path, _config_text(config))
    return path


def _required_string(
    value: Mapping[str, Any],
    path: tuple[str, ...],
) -> str:
    current: Any = value
    for name in path:
        if not isinstance(current, Mapping) or name not in current:
            raise LauncherError(f"preparation config is missing {'.'.join(path)}")
        current = current[name]
    if not isinstance(current, str) or not current:
        raise LauncherError(f"preparation config field {'.'.join(path)} must be a non-empty string")
    return current


def _validated_preparation_template(value: Any) -> dict[str, Any]:
    """Allow only the four documented device choices to differ from base config."""

    if not isinstance(value, Mapping):
        raise LauncherError("preparation config must contain a mapping")
    expected = _config_for(
        _preparation_row(),
        ppo_device=_required_string(value, ("rl", "device")),
        belief_inference_device=_required_string(value, ("rl", "belief_device")),
        preparation_device=_required_string(value, ("belief", "device")),
        tissue_device=_required_string(value, ("tissue", "atlaspatch", "device")),
    )
    if dict(value) != expected:
        raise LauncherError(
            "preparation config differs from the committed protocol outside "
            "the documented device fields; rerun the prepare command"
        )
    return expected


def _print_command(command: Sequence[str]) -> None:
    print(f"$ {shlex.join(command)}", flush=True)


def _run_checked(command: Sequence[str], *, environment: Mapping[str, str]) -> None:
    _print_command(command)
    try:
        subprocess.run(list(command), env=dict(environment), check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LauncherError(f"command failed: {shlex.join(command)}") from exc


def _prepare_commands(config_path: Path, work_root: Path) -> tuple[tuple[str, ...], ...]:
    python = sys.executable
    cache = work_root / "analysis_cache"
    return (
        ("atlaspatch-pinned", "--prepare-model"),
        (python, "-m", "wsi_lab", "preprocess", "--config", str(config_path)),
        (python, "-m", "wsi_lab", "train-belief", "--config", str(config_path)),
        (
            python,
            "-m",
            "ablation.analysis_cache",
            "build",
            "--config",
            str(config_path),
            "--output",
            str(cache),
            "--cohort-size",
            "30",
            "--positive-slides",
            "15",
            "--cohort-seed",
            "20260831",
        ),
        (
            python,
            "-m",
            "ablation.analysis_cache",
            "verify",
            "--config",
            str(config_path),
            "--output",
            str(cache),
        ),
    )


def _print_rows(rows: Sequence[RunSpec]) -> None:
    print("index  mode   variant                   start   seed  run_id")
    for row in rows:
        print(
            f"{row.index:>5}  {row.mode:<5}  {row.variant:<24}  "
            f"{row.start:<6}  {row.seed:>4}  {row.run_id}"
        )
    print(f"runs: {len(rows)}")


def _doctor(arguments: argparse.Namespace) -> int:
    source_root = _resolved(arguments.source_root)
    data_root = _resolved(arguments.data_root)
    work_root = _require_external_work_root(arguments.work_root)
    _source_compatibility(source_root)
    _dataset_requirements(data_root)
    environment = _environment(
        source_root=source_root,
        data_root=data_root,
        work_root=work_root,
    )
    _runtime_compatibility()
    if shutil.which("atlaspatch-pinned", path=environment["PATH"]) is None:
        raise LauncherError("the pinned AtlasPatch wrapper is not available on PATH")
    probe = (
        "import openslide, torch, wsi_lab; "
        "print(f'wsi_lab={wsi_lab.__file__}'); "
        "print(f'torch={torch.__version__}'); "
        "print(f'cuda_available={torch.cuda.is_available()}'); "
        "print(f'openslide={openslide.__version__}')"
    )
    _run_checked((sys.executable, "-c", probe), environment=environment)
    print("doctor: source, dataset layout, and runtime imports passed")
    return 0


def _prepare(arguments: argparse.Namespace) -> int:
    source_root = _resolved(arguments.source_root)
    data_root = _resolved(arguments.data_root)
    work_root = _require_external_work_root(arguments.work_root)
    tissue_device = arguments.tissue_device or arguments.device
    prospective = _preparation_config_path(work_root)
    commands = _prepare_commands(prospective, work_root)
    if arguments.dry_run:
        print(f"work_root={work_root}")
        print("dry-run: no files will be written and no data will be opened")
        for command in commands:
            _print_command(command)
        return 0
    _source_compatibility(source_root)
    _dataset_requirements(data_root)
    _runtime_compatibility()
    environment = _environment(
        source_root=source_root,
        data_root=data_root,
        work_root=work_root,
    )
    config_path = _write_preparation_config(
        work_root,
        device=arguments.device,
        belief_inference_device=arguments.belief_inference_device,
        tissue_device=tissue_device,
    )
    for command in _prepare_commands(config_path, work_root):
        _run_checked(command, environment=environment)
    print("prepare: shared training prerequisites and analysis cache are complete")
    return 0


def _run_command(row: RunSpec, config_path: Path) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "ablation.runner",
        "--config",
        str(config_path),
        "--variant",
        row.variant,
        "--run-id",
        row.run_id,
        "--mode",
        row.mode,
    )


def _verification_command(row: RunSpec, config_path: Path) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "ablation.verification",
        "--config",
        str(config_path),
        "--variant",
        row.variant,
        "--run-id",
        row.run_id,
        "--mode",
        row.mode,
    )


def _run_one(
    row: RunSpec,
    config_path: Path,
    environment: Mapping[str, str],
) -> tuple[str, int]:
    command = _run_command(row, config_path)
    _print_command(command)
    try:
        completed = subprocess.run(list(command), env=dict(environment), check=False)
    except OSError:
        return row.run_id, 127
    if completed.returncode:
        return row.run_id, int(completed.returncode)
    verification = _verification_command(row, config_path)
    _print_command(verification)
    try:
        completed = subprocess.run(list(verification), env=dict(environment), check=False)
    except OSError:
        return row.run_id, 127
    return row.run_id, int(completed.returncode)


def _execute(arguments: argparse.Namespace, *, mode: str) -> int:
    source_root = _resolved(arguments.source_root)
    data_root = _resolved(arguments.data_root)
    work_root = _require_external_work_root(arguments.work_root)
    canonical = main_runs() if mode == "main" else smoke_runs()
    rows = select_runs(
        canonical,
        variants=arguments.variants,
        starts=arguments.starts,
        seeds=arguments.seeds,
    )
    if not rows:
        raise LauncherError("the requested filters select no runs")
    if arguments.dry_run:
        print(f"work_root={work_root}")
        print(f"jobs={arguments.jobs}")
        _print_rows(rows)
        for row in rows:
            _print_command(_run_command(row, _config_path(work_root, row)))
        print("dry-run: no files were written and no training was started")
        return 0

    _source_compatibility(source_root)
    _dataset_requirements(data_root)
    _runtime_compatibility()
    preparation_config = _preparation_config_path(work_root)
    if not preparation_config.is_file():
        raise LauncherError("preparation config is missing; run the prepare command first")
    try:
        preparation_template = yaml.safe_load(preparation_config.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise LauncherError(f"could not read preparation config: {exc}") from exc
    preparation_template = _validated_preparation_template(preparation_template)
    environment = _environment(
        source_root=source_root,
        data_root=data_root,
        work_root=work_root,
    )
    verify = _prepare_commands(preparation_config, work_root)[-1]
    _run_checked(verify, environment=environment)

    configs = {
        row: _write_config(
            work_root,
            row,
            device=arguments.device,
            belief_inference_device=arguments.belief_inference_device,
            template=preparation_template,
        )
        for row in rows
    }
    failures: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=arguments.jobs) as executor:
        futures = {executor.submit(_run_one, row, configs[row], environment): row for row in rows}
        for future in as_completed(futures):
            run_id, returncode = future.result()
            if returncode:
                failures[run_id] = returncode
            print(f"finished {run_id}: exit={returncode}", flush=True)
    if failures:
        rendered = ", ".join(f"{name}={code}" for name, code in sorted(failures.items()))
        raise LauncherError(f"{len(failures)} run(s) failed: {rendered}")
    print(f"{mode}: all {len(rows)} selected runs completed")
    return 0


def _status(arguments: argparse.Namespace) -> int:
    work_root = _require_external_work_root(arguments.work_root)
    manifests = sorted((work_root / "shared_workspace" / "rl_ablation").glob("*/run_manifest.json"))
    counts: Counter[str] = Counter()
    for path in manifests:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            counts["unreadable"] += 1
            print(f"unreadable  {path.parent.name}")
            continue
        status = str(value.get("status", "unknown"))
        counts[status] += 1
        print(
            f"{status:<11} {value.get('run_id', 'unknown')!s:<42} "
            f"step={value.get('global_step', 0)}"
        )
    print(f"discovered={len(manifests)} status={dict(sorted(counts.items()))}")
    return 0


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="extracted source archive root providing the recorded src/wsi_lab implementation",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)


def _device_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="primary compute device (for example cuda:0 or cpu)",
    )
    parser.add_argument(
        "--belief-inference-device",
        default="cpu",
        help="device used by the frozen belief model during PPO",
    )


def _execution_arguments(parser: argparse.ArgumentParser) -> None:
    _common_arguments(parser)
    _device_arguments(parser)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=(
            "v1_full",
            "v2_viewport_only",
            "v3_no_tissue_mask",
            "v4_no_diagnosis",
            "v5_no_belief",
            "v6_combined",
        ),
    )
    parser.add_argument("--starts", nargs="+", choices=("center", "random"))
    parser.add_argument("--seeds", nargs="+", type=int, choices=(1, 2, 3))
    parser.add_argument("--dry-run", action="store_true")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="print the complete main run matrix")

    doctor = commands.add_parser("doctor", help="check source, data layout, and imports")
    _common_arguments(doctor)

    prepare = commands.add_parser("prepare", help="build the shared train/validation prerequisites")
    _common_arguments(prepare)
    _device_arguments(prepare)
    prepare.add_argument(
        "--tissue-device",
        default=None,
        help="AtlasPatch device; defaults to --device",
    )
    prepare.add_argument("--dry-run", action="store_true")

    smoke = commands.add_parser("smoke", help="run one short job per variant")
    _execution_arguments(smoke)
    main = commands.add_parser("run", help="run the complete 36-job study")
    _execution_arguments(main)

    status = commands.add_parser("status", help="inspect user-generated run manifests")
    status.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        load_study()
        if getattr(arguments, "jobs", 1) < 1:
            raise LauncherError("--jobs must be at least 1")
        if arguments.command == "list":
            _print_rows(main_runs())
            return 0
        if arguments.command == "doctor":
            return _doctor(arguments)
        if arguments.command == "prepare":
            return _prepare(arguments)
        if arguments.command == "smoke":
            return _execute(arguments, mode="smoke")
        if arguments.command == "run":
            return _execute(arguments, mode="main")
        if arguments.command == "status":
            return _status(arguments)
        raise LauncherError(f"unsupported command: {arguments.command}")
    except LauncherError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


__all__ = ["LauncherError", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
