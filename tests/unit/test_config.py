from __future__ import annotations

from pathlib import Path

import pytest

from rl4xs_wsi.config import load_config
from rl4xs_wsi.errors import ConfigurationError


def test_config_expands_environment_and_resolves_paths(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    monkeypatch.setenv("TEST_CAMELYON", str(dataset))
    path = tmp_path / "config.yaml"
    path.write_text("dataset:\n  root: ${TEST_CAMELYON}\nworkspace:\n  root: output\n")
    config = load_config(path)
    assert config.dataset.root == dataset
    assert config.dataset.slides_dir == dataset / "images"
    assert config.workspace.root == tmp_path / "output"


def test_config_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("dataset:\n  root: /tmp/data\n  surprise: true\n")
    with pytest.raises(ConfigurationError, match="surprise"):
        load_config(path)


@pytest.mark.parametrize("field", ("checkpoint", "model_identifier", "model_revision"))
def test_config_rejects_unsupported_atlaspatch_model_overrides(tmp_path: Path, field: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"dataset:\n  root: /tmp/data\ntissue:\n  atlaspatch:\n    {field}: unsupported\n"
    )
    with pytest.raises(ConfigurationError, match=field):
        load_config(path)


def test_config_requires_environment_variable(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("dataset:\n  root: ${VARIABLE_THAT_DOES_NOT_EXIST}\n")
    with pytest.raises(ConfigurationError, match="VARIABLE_THAT_DOES_NOT_EXIST"):
        load_config(path)


@pytest.mark.parametrize(
    "body",
    (
        "belief:\n  epochs: '100'\n",
        "logging:\n  tensorboard: 'true'\n",
        "environment:\n  step_cost: .inf\n",
    ),
)
def test_config_rejects_coerced_and_nonfinite_scalars(tmp_path: Path, body: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(f"dataset:\n  root: /tmp/data\n{body}")
    with pytest.raises(ConfigurationError):
        load_config(path)


@pytest.mark.parametrize(
    "values, message",
    (("[]", "empty"), ("[0]", "positive"), ("[1, 1]", "unique"), ("[501]", "max_bag_size")),
)
def test_config_validates_belief_validation_bag_sizes(
    tmp_path: Path, values: str, message: str
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "dataset:\n  root: /tmp/data\nbelief:\n  validation_bag_sizes: " + values + "\n"
    )
    with pytest.raises(ConfigurationError, match=message):
        load_config(path)


@pytest.mark.parametrize(
    "body, message",
    (
        ("embeddings:\n  root: ../bench_embeddings\n", "bench_embeddings"),
        ("embeddings:\n  directory_template: ../../outside/{magnification}\n", "traversal"),
        ("embeddings:\n  directory_template: PEAN/{magnification}\n", "PEAN"),
    ),
)
def test_config_rejects_excluded_or_escaping_embedding_sources(
    tmp_path: Path, body: str, message: str
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(f"dataset:\n  root: /tmp/data\n{body}")
    with pytest.raises(ConfigurationError, match=message):
        load_config(path)
