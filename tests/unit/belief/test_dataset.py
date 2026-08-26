from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wsi_lab.belief.dataset import (  # noqa: E402
    FullBagDataset,
    PartialBagDataset,
    ValidationBagDataset,
    collate_belief_bags,
)
from wsi_lab.belief.sampler import (  # noqa: E402
    PartialBagSampler,
    PartialBagSamplingConfig,
    SlideEmbeddingRecord,
)


def _records() -> list[SlideEmbeddingRecord]:
    records = []
    for slide_index in range(2):
        records.append(
            SlideEmbeddingRecord(
                f"slide_{slide_index}",
                slide_index,
                {
                    scale: np.random.default_rng(slide_index + int(scale))
                    .normal(size=(12, 5))
                    .astype(np.float32)
                    for scale in (2.5, 5.0, 10.0, 20.0)
                },
            )
        )
    return records


def test_training_dataset_changes_by_epoch_but_repeats_exactly() -> None:
    sampler = PartialBagSampler(PartialBagSamplingConfig(max_bag_size=16))
    dataset = PartialBagDataset(_records(), sampler=sampler, bags_per_slide_per_epoch=2, seed=4)
    epoch_zero = dataset[0]
    repeated = dataset[0]
    assert torch.equal(epoch_zero.features, repeated.features)
    dataset.set_epoch(1)
    epoch_one = dataset[0]
    assert not torch.equal(epoch_zero.features, epoch_one.features)


def test_training_dataset_groups_each_slides_bags_for_hdf5_locality() -> None:
    dataset = PartialBagDataset(_records(), bags_per_slide_per_epoch=2, seed=17)
    slide_ids = [dataset[index].slide_id for index in range(len(dataset))]

    assert slide_ids[0] == slide_ids[1]
    assert slide_ids[2] == slide_ids[3]
    assert slide_ids[0] != slide_ids[2]


def test_validation_dataset_and_collation_are_deterministic() -> None:
    dataset = ValidationBagDataset(
        _records(), bag_sizes=(1, 7), bags_per_slide=2, max_bag_size=10, seed=11
    )
    first = dataset[1]
    second = dataset[1]
    assert torch.equal(first.features, second.features)
    assert torch.equal(first.magnifications, second.magnifications)
    batch = collate_belief_bags([dataset[0], dataset[2]])
    assert batch.features.shape == (2, 7, 5)
    assert batch.mask.sum(dim=1).tolist() == [1, 7]
    assert batch.requested_sizes.tolist() == [1, 7]
    assert batch.slide_ids == ("slide_0", "slide_0")


def test_full_bag_dataset_contains_every_scale_instance() -> None:
    dataset = FullBagDataset(_records(), seed=5)
    bag = dataset[0]
    assert bag.bag_size == 48
    assert set(bag.magnifications.tolist()) == {2.5, 5.0, 10.0, 20.0}


def test_hdf5_provider_opens_existing_embeddings_lazily(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    from wsi_lab.belief.providers import HDF5FeatureMatrix

    path = tmp_path / "embedding.h5"
    expected = np.arange(30, dtype=np.float32).reshape(10, 3)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("features/uni_v2", data=expected)
    provider = HDF5FeatureMatrix(path, "features/uni_v2")
    assert provider.shape == (10, 3)
    assert np.array_equal(provider[np.asarray([7, 2, 5])], expected[[2, 5, 7]])
    provider.close()
