"""YAML loading with strict validation and deterministic path semantics."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from wsi_lab.config.models import AppConfig
from wsi_lab.errors import ConfigurationError

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_environment(value: Any, location: str = "config") -> Any:
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in os.environ:
                raise ConfigurationError(f"Missing environment variable {name!r} at {location}")
            return os.environ[name]

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [
            _expand_environment(item, f"{location}[{index}]") for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        return {key: _expand_environment(item, f"{location}.{key}") for key, item in value.items()}
    return value


def _absolute(path: Path, base: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (base / path).resolve()


def _resolve_paths(config: AppConfig, config_path: Path) -> AppConfig:
    base = config_path.parent.resolve()
    config.dataset.root = _absolute(config.dataset.root, base)
    dataset_root = config.dataset.root
    config.dataset.camelyon16_root = _absolute(config.dataset.camelyon16_root, dataset_root)
    config.dataset.camelyon17_root = _absolute(config.dataset.camelyon17_root, dataset_root)
    config.dataset.slides_dir = _absolute(config.dataset.slides_dir, dataset_root)
    labels_root = (
        config.dataset.camelyon16_root if config.dataset.name == "camelyon16_17" else dataset_root
    )
    config.dataset.labels = _absolute(config.dataset.labels, labels_root)
    config.dataset.camelyon17_labels = _absolute(
        config.dataset.camelyon17_labels, config.dataset.camelyon17_root
    )
    config.dataset.tissue_masks_dir = _absolute(config.dataset.tissue_masks_dir, dataset_root)
    config.dataset.tumor_masks_dir = _absolute(config.dataset.tumor_masks_dir, dataset_root)
    config.dataset.annotations_dir = _absolute(config.dataset.annotations_dir, dataset_root)
    config.dataset.luad_dir = _absolute(config.dataset.luad_dir, dataset_root)
    config.dataset.lusc_dir = _absolute(config.dataset.lusc_dir, dataset_root)
    config.embeddings.root = _absolute(config.embeddings.root, dataset_root)
    config.workspace.root = _absolute(config.workspace.root, base)
    if config.environment.persistent_rgb_cache is not None:
        config.environment.persistent_rgb_cache = _absolute(
            config.environment.persistent_rgb_cache, config.workspace.root
        )
    excluded = {"pean", "bench_embeddings"}
    for label, path in (
        ("dataset root", config.dataset.root),
        ("CAMELYON16 root", config.dataset.camelyon16_root),
        ("CAMELYON17 root", config.dataset.camelyon17_root),
        ("embedding root", config.embeddings.root),
    ):
        if any(part.casefold() in excluded for part in path.parts):
            raise ConfigurationError(f"{label} uses an excluded PEAN/bench_embeddings path: {path}")
    return config


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {config_path}")
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigurationError(f"Configuration root must be a mapping: {config_path}")
    try:
        expanded = _expand_environment(loaded)
        return _resolve_paths(AppConfig.model_validate(expanded), config_path)
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid configuration {config_path}:\n{exc}") from exc
