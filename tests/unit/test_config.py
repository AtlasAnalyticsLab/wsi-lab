from __future__ import annotations

from pathlib import Path

import pytest

from wsi_lab.config import load_config
from wsi_lab.config.models import TissueSource
from wsi_lab.errors import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def test_config_defaults_to_atlaspatch_tissue_source(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("dataset:\n  root: /tmp/data\n")

    config = load_config(path)

    assert config.tissue.source is TissueSource.ATLASPATCH


@pytest.mark.parametrize("source", tuple(TissueSource))
def test_config_accepts_each_tissue_source(tmp_path: Path, source: TissueSource) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(f"dataset:\n  root: /tmp/data\ntissue:\n  source: {source.value}\n")

    config = load_config(path)

    assert config.tissue.source is source


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


@pytest.mark.parametrize(
    ("filename", "environment_name", "dataset_name"),
    (
        ("camelyon17.example.yaml", "CAMELYON17_ROOT", "camelyon17"),
        ("tcga_nsclc.example.yaml", "TCGA_NSCLC_ROOT", "tcga_nsclc"),
        ("generic.example.yaml", "WSI_DATASET_ROOT", "generic"),
    ),
)
def test_multi_dataset_examples_are_strict_and_portable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    environment_name: str,
    dataset_name: str,
) -> None:
    dataset_root = tmp_path / dataset_name
    dataset_root.mkdir()
    monkeypatch.setenv(environment_name, str(dataset_root))

    config = load_config(PROJECT_ROOT / "configs" / filename)

    assert config.dataset.name == dataset_name
    assert config.dataset.root == dataset_root.resolve()
    assert config.embeddings.root == (dataset_root / "Embeddings").resolve()
    assert config.tissue.source is TissueSource.ATLASPATCH
    assert not str(config.dataset.root).startswith(str(PROJECT_ROOT))
