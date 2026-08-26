"""Dense geometry, sparse embedding mapping, tissue masks, and artifacts."""

from wsi_lab.preprocessing.artifacts import ArtifactStore, SlideOffset
from wsi_lab.preprocessing.atlaspatch import (
    AtlasPatchOptions,
    AtlasPatchRunner,
    ensure_mpp_override_csv,
)
from wsi_lab.preprocessing.geometry import (
    GRID_DTYPE,
    GeometryBuilder,
    GridLevel,
    NavigationGeometry,
)
from wsi_lab.preprocessing.mapping import EmbeddingMapping, map_embeddings_to_grid
from wsi_lab.preprocessing.pipeline import (
    NavigationCellMapping,
    PreprocessingOptions,
    Preprocessor,
    ProcessedEpisodeSource,
    SlidePreprocessingResult,
)
from wsi_lab.preprocessing.tissue import (
    TissueMask,
    apply_tissue_to_geometry,
    choose_start_cell,
    load_or_create_tissue_mask,
    navigation_roi_level0,
    tissue_bbox_level0,
)

__all__ = [
    "GRID_DTYPE",
    "ArtifactStore",
    "AtlasPatchOptions",
    "AtlasPatchRunner",
    "EmbeddingMapping",
    "GeometryBuilder",
    "GridLevel",
    "NavigationCellMapping",
    "NavigationGeometry",
    "PreprocessingOptions",
    "Preprocessor",
    "ProcessedEpisodeSource",
    "SlideOffset",
    "SlidePreprocessingResult",
    "TissueMask",
    "apply_tissue_to_geometry",
    "choose_start_cell",
    "ensure_mpp_override_csv",
    "load_or_create_tissue_mask",
    "map_embeddings_to_grid",
    "navigation_roi_level0",
    "tissue_bbox_level0",
]
