from __future__ import annotations

import numpy as np

from wsi_lab.constants import LEVEL0_FOOTPRINTS, NAVIGATION_MAGNIFICATIONS
from wsi_lab.testing.synthetic import SyntheticEpisodeSource


def test_synthetic_slide_exposes_snapped_roi_without_rebasing_global_cells() -> None:
    coarse_footprint = LEVEL0_FOOTPRINTS[NAVIGATION_MAGNIFICATIONS[0]]
    source = SyntheticEpisodeSource(overview_size=32, level0_size=2 * coarse_footprint)

    slide = source.load_slide("synthetic_cancer")

    assert slide.navigation_roi_level0 == (0, 0, 2 * coarse_footprint, 2 * coarse_footprint)
    assert slide.cells[0].bounds == (0, 0, coarse_footprint, coarse_footprint)
    assert slide.cells[1].bounds == (
        coarse_footprint,
        0,
        2 * coarse_footprint,
        coarse_footprint,
    )


def test_synthetic_roi_does_not_change_embedding_lookup_identity() -> None:
    source = SyntheticEpisodeSource(overview_size=32)
    slide = source.load_slide("synthetic_cancer")
    embedded = slide.cells[20]
    feature = source.read_embedding(slide.slide_id, embedded)

    assert embedded.magnification == 2.5
    assert embedded.embedding_index == 0
    assert embedded.bounds == (0, 0, 4096, 4096)
    assert np.allclose(feature, np.full(8, 0.1275, dtype=np.float32))
