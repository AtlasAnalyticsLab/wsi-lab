from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from wsi_lab.belief.sampler import (
    PartialBagSampler,
    PartialBagSamplingConfig,
    SlideEmbeddingRecord,
)


def _slide(counts: tuple[int, int, int, int] = (20, 30, 40, 50)) -> SlideEmbeddingRecord:
    return SlideEmbeddingRecord(
        slide_id="synthetic",
        label=1,
        features_by_magnification={
            scale: np.arange(count * 3, dtype=np.float32).reshape(count, 3)
            for scale, count in zip((2.5, 5.0, 10.0, 20.0), counts, strict=True)
            if count
        },
    )


def test_selection_is_unique_capacity_capped_and_reproducible() -> None:
    slide = _slide((2, 3, 4, 1))
    sampler = PartialBagSampler(PartialBagSamplingConfig(max_bag_size=20))
    first = sampler.sample(slide, np.random.default_rng(9), bag_size=19)
    second = sampler.sample(slide, np.random.default_rng(9), bag_size=19)
    assert first.requested_size == 19
    assert first.actual_size == 10
    for scale, indices in first.indices_by_magnification.items():
        assert len(indices) == len(set(indices.tolist()))
        assert np.array_equal(indices, second.indices_by_magnification[scale])


def test_missing_scale_instances_are_redistributed() -> None:
    slide = _slide((0, 1, 0, 8))
    selection = PartialBagSampler().sample(slide, np.random.default_rng(3), bag_size=7)
    assert selection.actual_size == 7
    assert set(selection.indices_by_magnification) <= {5.0, 20.0}


def test_bags_are_mixed_scale_whenever_k_allows() -> None:
    slide = _slide()
    sampler = PartialBagSampler()
    for seed in range(50):
        selection = sampler.sample(slide, np.random.default_rng(seed), bag_size=8)
        assert len(selection.indices_by_magnification) >= 2


def test_bag_size_band_mass_matches_specification() -> None:
    sampler = PartialBagSampler()
    rng = np.random.default_rng(123)
    sizes = np.asarray([sampler.sample_bag_size(rng) for _ in range(20_000)])
    mass = np.asarray(
        [
            np.mean(sizes <= 32),
            np.mean((sizes > 32) & (sizes <= 128)),
            np.mean(sizes > 128),
        ]
    )
    assert np.allclose(mass, (0.5, 0.3, 0.2), atol=0.015)


def test_record_rejects_cross_scale_dimension_mismatch() -> None:
    try:
        SlideEmbeddingRecord(
            "bad",
            0,
            {2.5: np.zeros((2, 3)), 5.0: np.zeros((3, 4))},
        )
    except ValueError as error:
        assert "dimensions are inconsistent" in str(error)
    else:
        raise AssertionError("Expected mismatched dimensions to fail")
