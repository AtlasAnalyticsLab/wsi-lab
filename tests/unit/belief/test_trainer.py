from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wsi_lab.belief.model import GatedAttentionMIL  # noqa: E402
from wsi_lab.belief.sampler import SlideEmbeddingRecord  # noqa: E402
from wsi_lab.belief.trainer import (  # noqa: E402
    BeliefTrainer,
    build_belief_loaders,
    load_calibrated_belief,
)
from wsi_lab.errors import TestSplitAccessError  # noqa: E402


def _slides(prefix: str, *, split: str | None = None) -> list[SlideEmbeddingRecord]:
    result = []
    for index, label in enumerate((0, 1)):
        rng = np.random.default_rng(index + 13)
        result.append(
            SlideEmbeddingRecord(
                f"{prefix}_{index}",
                label,
                {
                    scale: rng.normal(loc=label, size=(4, 3)).astype(np.float32)
                    for scale in (2.5, 5.0, 10.0, 20.0)
                },
                metadata={"split": split} if split is not None else {},
            )
        )
    return result


def test_one_synthetic_optimizer_step_finalizes_atomic_artifact(tmp_path) -> None:
    config = {
        "seed": 2,
        "max_bag_size": 4,
        "hidden_dim": 6,
        "attention_dim": 4,
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "batch_size": 2,
        "epochs": 1,
        # Two slides and batch size two means exactly one optimizer step.
        "bags_per_slide_per_epoch": 1,
        "validation_bags_per_slide": 1,
        "validation_bag_sizes": (1, 4),
        "early_stopping_patience": 1,
        "device": "cpu",
        "num_workers": 0,
        "ece_bins": 3,
        "resume": True,
    }
    loaders = build_belief_loaders(_slides("train"), _slides("validation"), config)
    model = GatedAttentionMIL.from_config(3, config)
    dependencies = {
        "dataset": "synthetic-dataset",
        "embeddings": "synthetic-embeddings",
        "split": "synthetic-split",
        "preprocessing": "synthetic-preprocessing",
    }
    events = []
    trainer = BeliefTrainer(
        model,
        config,
        tmp_path / "belief",
        dependencies=dependencies,
        metric_callback=lambda event, payload: events.append((event, dict(payload))),
    )
    result = trainer.fit(loaders.train, loaders.validation)
    assert result.global_step == 1
    assert result.checkpoint_path.exists()
    assert result.calibrated_checkpoint_path.exists()
    assert (result.artifact_dir / "temperature.json").exists()
    manifest = json.loads((result.artifact_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    loaded, temperature, checkpoint = load_calibrated_belief(
        result.calibrated_checkpoint_path,
        expected_dependencies=dependencies,
    )
    assert loaded.embedding_dim == 3
    assert temperature == result.temperature
    assert "validation" in checkpoint
    assert [event for event, _payload in events] == ["belief/epoch", "belief/complete"]
    assert events[-1][1]["global_step"] == result.global_step


def test_loaders_reject_test_or_wrong_partition_records() -> None:
    config = {
        "max_bag_size": 4,
        "batch_size": 2,
        "bags_per_slide_per_epoch": 1,
        "validation_bags_per_slide": 1,
        "validation_bag_sizes": (1, 4),
        "device": "cpu",
        "num_workers": 0,
    }
    with pytest.raises(TestSplitAccessError, match="test slide"):
        build_belief_loaders(_slides("test", split="official_test"), _slides("v"), config)
    with pytest.raises(TestSplitAccessError, match="provided to the belief train"):
        build_belief_loaders(
            _slides("wrong", split="validation"),
            _slides("v", split="validation"),
            config,
        )
    loaders = build_belief_loaders(
        _slides("train", split="train"),
        _slides("validation", split="validation"),
        config,
    )
    assert len(loaders.train.dataset) == 2
