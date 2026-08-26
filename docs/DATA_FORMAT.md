# Data and artifact formats

## Dataset layout policy

Keep supported datasets in their native directory structures. The adapters are
read-only: they discover images, labels, optional sidecars, and embeddings in
place. They do not reorganize large WSIs or require physical `train/`, `val/`,
and `test/` folders. Generated split membership is a versioned artifact under
`workspace/splits/`, separate from the source dataset.

Use one of these values for `dataset.name`:

```text
camelyon16
camelyon17
tcga_nsclc
generic
```

Dataset roots may be supplied through environment-variable expressions in the
YAML configuration. For example, a portable configuration can use
`${DATASET_ROOT}`; users then set `DATASET_ROOT` to their own dataset root.
Never put machine-specific absolute paths into a shared configuration.

All adapters preserve deterministic discovery order, reject duplicate slide
IDs, and resolve embeddings without moving them. Missing required images or
embeddings are validation errors rather than silent exclusions.

## Native CAMELYON16 adapter

The default native layout is:

```text
CAMELYON16/
├── images/
│   ├── normal_001.tif
│   ├── tumor_001.tif
│   └── test_001.tif
├── evaluation/
│   └── reference.csv
├── Embeddings/
│   ├── UNI_v2_2.5x_256/patches/<slide_id>.h5
│   ├── UNI_v2_5x_256/patches/<slide_id>.h5
│   ├── UNI_v2_10x_256/patches/<slide_id>.h5
│   └── UNI_v2_20x_256/patches/<slide_id>.h5
├── background_tissue/                 # optional existing tissue masks
│   └── <slide_id>_tissue.tif
├── masks/                             # optional tumor masks, evaluation only
│   └── <slide_id>_mask.tif
└── annotations/                       # optional XML annotations, evaluation only
    └── <slide_id>.xml
```

`evaluation/reference.csv` must contain the columns `image`, `type`, `class`,
and `center`. Slide discovery is driven by this table, not by filename globbing.
`type=normal` maps to 0 and `type=tumor` maps to 1. `class` retains the source
tumor subtype, including `micro` and `macro`.

Every labeled `test_*` record is preserved as official test regardless of its
class. The remaining 270 development slides receive a deterministic,
label-and-center-stratified 80/20 train/validation split by default (216 train,
54 validation with seed 1). The configured `test_fraction` is not used when a
labeled official test cohort exists. CAMELYON16 exposes no reliable patient
identifier in this layout, so the split is slide-level. Consecutive tissue
sections can introduce patient/tissue leakage that the package cannot detect;
this limitation is recorded in split metadata.

The folder names above are configurable, but preserving the native tree is
recommended. The dataset root is the `CAMELYON16/` directory, not `images/` or
one of the embedding directories.

## Native CAMELYON17 adapter

The default native layout is:

```text
CAMELYON17/
├── images/
│   ├── patient_000_node_0.tif
│   ├── ...
│   └── patient_199_node_4.tif
├── stages.csv
├── Embeddings/
│   ├── UNI_v2_2.5x_256/patches/<slide_id>.h5
│   ├── UNI_v2_5x_256/patches/<slide_id>.h5
│   ├── UNI_v2_10x_256/patches/<slide_id>.h5
│   └── UNI_v2_20x_256/patches/<slide_id>.h5
├── background_tissue/                 # optional existing tissue masks
│   └── <slide_id>_tissue.tif
├── masks/                             # optional tumor masks, evaluation only
│   └── <slide_id>_mask.tif
└── annotations/                       # optional XML annotations, evaluation only
    └── <slide_id>.xml
```

`stages.csv` must contain `patient`, `stage`, and `center`. Native patient-level
`.zip` summary rows are ignored; slide-level rows such as
`patient_000_node_0.tif` are authoritative. Slide stages map as follows:

```text
negative     0
itc          1
micro        1
macro        1
```

The adapter derives `patient_id` from `patient_NNN_node_N` and reads `center`
from the slide's stage row. It does not infer slide labels from node or patient
numbers. In the native release, stage rows label patients 000–099. Discovered
images without a stage row, normally patients 100–199, are retained with
`label=None` and `official_cohort=test_unlabeled`. A missing slide-level stage
row for patient 000–099 is an error, so damaged development metadata cannot be
silently reclassified as blind test data.

The blind official-test records are written to the split manifest as
`unassigned` with reason `missing_label`; they cannot enter training,
calibration, reward computation, checkpoint selection, or metric evaluation.
Because no *labeled* official test partition is available, the package creates
an internal train/validation/test split from labeled patients. All five nodes
from a patient remain together. The default fractions are 60/20/20 before
per-stratum rounding, stratified by patient-level label and center when the
strata support it. A mixed positive/negative patient is stratified by its
maximum slide label while every node retains its own label.

The dataset root is the `CAMELYON17/` directory, not `images/`, `masks/`, or
`Embeddings/`.

## Native TCGA-NSCLC adapter

Keep LUAD and LUSC as native class cohorts:

```text
TCGA-NSCLC/
├── TCGA-LUAD/
│   ├── images/
│   │   └── TCGA-<TSS>-<participant>-<sample-and-slide-fields>.svs
│   └── Embeddings/
│       ├── UNI_v2_2.5x_256/patches/<slide_id>.h5
│       ├── UNI_v2_5x_256/patches/<slide_id>.h5
│       ├── UNI_v2_10x_256/patches/<slide_id>.h5
│       └── UNI_v2_20x_256/patches/<slide_id>.h5
└── TCGA-LUSC/
    ├── images/
    │   └── TCGA-<TSS>-<participant>-<sample-and-slide-fields>.svs
    └── Embeddings/
        ├── UNI_v2_2.5x_256/patches/<slide_id>.h5
        ├── UNI_v2_5x_256/patches/<slide_id>.h5
        ├── UNI_v2_10x_256/patches/<slide_id>.h5
        └── UNI_v2_20x_256/patches/<slide_id>.h5
```

Only `.svs` images are discovered. LUAD maps to label 0 and LUSC maps to label
1. These directories represent diagnostic classes, not existing experimental
splits. The adapter extracts the patient ID from the first three TCGA barcode
components, for example `TCGA-AB-1234`, and uses the tissue-source-site segment
as `center`. It rejects malformed or duplicate barcodes and a patient appearing
in both disease cohorts.

TCGA-NSCLC has no official split in this layout. The package generates a
deterministic train/validation/test split using the configured fractions
(60/20/20 by default), grouped by patient and stratified by disease class and
tissue-source site when possible. Multiple slides belonging to one participant
can never cross partitions. Missing scale embeddings fail dataset validation;
slides are not silently dropped.

The configured `embeddings.root` is interpreted beneath *each* disease cohort,
so its default value remains the relative path `Embeddings`. The dataset root
is `TCGA-NSCLC/`, not either cohort directory.

## Generic manifest adapter

Use `dataset.name: generic` for a binary WSI dataset without a dedicated
adapter. The generic adapter never infers labels, patients, centers, cohorts,
or sidecars from filenames. Its default and recommended root contract is:

```text
CUSTOM_DATASET/
├── metadata.csv
├── images/
│   ├── case-001.svs
│   ├── case-002.svs
│   └── blind-001.svs
├── masks/                             # optional; names are not inferred
│   ├── tissue/<slide_id>.tif
│   └── tumor/<slide_id>.tif
├── annotations/                       # optional; names are not inferred
│   └── <slide_id>.xml
└── Embeddings/
    ├── UNI_v2_2.5x_256/patches/<slide_id>.h5
    ├── UNI_v2_5x_256/patches/<slide_id>.h5
    ├── UNI_v2_10x_256/patches/<slide_id>.h5
    └── UNI_v2_20x_256/patches/<slide_id>.h5
```

`metadata.csv` must have this exact required header set (column order is not
significant):

```text
slide_id,image_path,label,label_name,patient_id,center,official_cohort
```

It may additionally have these optional path columns:

```text
tissue_mask_path,tumor_mask_path,annotation_path
```

A complete example is:

```csv
slide_id,image_path,label,label_name,patient_id,center,official_cohort,tissue_mask_path,tumor_mask_path,annotation_path
case-001,images/case-001.svs,0,negative,patient-001,site-a,development,masks/tissue/case-001.tif,,
case-002,images/case-002.svs,1,positive,patient-002,site-b,test,,masks/tumor/case-002.tif,annotations/case-002.xml
blind-001,images/blind-001.svs,,unlabeled,patient-003,site-b,test_unlabeled,,,
```

The row contract is strict:

- `slide_id` is unique, begins with an ASCII letter or digit, and otherwise
  contains only letters, digits, `.`, `_`, or `-`. It cannot contain a path
  separator.
- `image_path` is required. It and every nonblank optional path must be relative
  to `dataset.root`, use forward slashes, remain below that root, and contain no
  `..` traversal. Absolute Unix and Windows paths are rejected.
- `label` is exactly `0` or `1`. It may be blank only when
  `official_cohort=test_unlabeled`.
- `label_name` is descriptive metadata. Keep it meaningful for labeled rows;
  `unlabeled` is recommended for blind rows.
- `patient_id` and `center` columns are required but their values may be blank.
  Supply a stable patient ID on every related development slide to enable
  leakage-safe grouping. If even one development row lacks a patient ID, the
  split manager must fall back to slide-level grouping for that dataset.
- `official_cohort` is one of `development`, `test`, or `test_unlabeled`.
  This field describes the source cohort; it does not encode train or validation
  membership.
- Optional path cells may be blank, and the optional columns may be omitted.

When labeled `official_cohort=test` rows exist, they are preserved as test and
only development records are divided into train/validation. When no labeled
official test exists, the package creates train/validation/test membership from
the labeled development records. `test_unlabeled` rows are always recorded as
unassigned and excluded from learning and metrics. Patient grouping is used
when every development row supplies `patient_id`; otherwise grouping is
slide-level. Split membership is persisted and never implemented by moving
files.

Image and sidecar folder names are not inferred by this adapter and may differ
from the recommended tree, provided each CSV path follows the root-relative
rules. Embedding paths remain convention-based:

```text
<embeddings.root>/<rendered directory_template>/<slide_id>.h5
```

The defaults render to the four `Embeddings/UNI_v2_*_256/patches/` directories
shown above. The generic adapter changes discovery metadata only; it does not
add a new embedding format. The canonical HDF5 contract below still applies.

## Canonical embeddings

Only these v1 paths are supported:

```text
Embeddings/UNI_v2_2.5x_256/patches/{slide_id}.h5
Embeddings/UNI_v2_5x_256/patches/{slide_id}.h5
Embeddings/UNI_v2_10x_256/patches/{slide_id}.h5
Embeddings/UNI_v2_20x_256/patches/{slide_id}.h5
```

Each file must provide:

```text
coords            int32 [N,5]
passports         bytes [N]
features/uni_v2   float32 [N,D]
```

`coords[:,0:2]` are unsorted level-0 top-left origins. The remaining columns carry extraction width, height, and native pyramid level. `D` is inferred and checked across every file and scale. HDF5 `wsi_path` is provenance-only; configured paths are authoritative.

Exact duplicate coordinates are reported and collapse to the lowest source row only if passport and feature bytes match. Conflicting duplicates are fatal. Boundary overhang is valid when an origin is inside the WSI. Off-grid or outside origins are fatal.

The embeddings contain tissue-filtered instances only. No background vectors are fabricated, and their absence cannot define tissue or geometric validity. `bench_embeddings` is not used because its backfilled hierarchy changes the sparse-evidence problem.

Each per-scale preprocessing summary reports `tissue_overlapping_cells` and
`tissue_overlap_fraction`. Embeddings and masks can originate from different
AtlasPatch runs, so this comparison remains diagnostic; navigation validity is
derived from the mask alone and never from whether an HDF5 row exists.

## Processed grids

Preprocessing normalizes the selected mask to a cached binary mask for every
slide. It finds the level-0 box enclosing the union of all nonzero tissue,
including disconnected islands, and expands all four boundaries outward to the
global 0.625× grid with its origin fixed at `(0, 0)`. The persisted
`navigation_roi_level0` is `[x0, y0, x1, y1]`; a nominal maximum may extend
beyond the WSI at its right or bottom edge. This box describes the navigation
envelope; preprocessing retains the canonical full-slide rows and global IDs.
It does not rebase or independently crop a level, preserving exact
parent/child geometry and alignment with existing HDF5 coordinates.

Each scale is stored as a memory-mappable structured NumPy array with stable
int64 cell identifiers, a deterministic dataset-order `slide_index`, global
row/column, nominal and clipped level-0 bounds, parent/four children,
directional neighbors, tissue fraction/status, a cached `tissue_valid` flag,
and a nullable source HDF5 row (`-1` sentinel). A directional neighbor is the
nearest tissue-valid cell farther along the same row or column, so it can skip
one or more background cells. A missing neighbor masks that pan direction.
Zoom-in remains tied to the four aligned child slots, but only tissue-valid
children are actionable. Small per-slide metadata maps the numeric slide
identifier to `slide_id` and records the tissue ROI, mask/letterbox transforms,
and fingerprints.

`tissue_valid` means that at least one binary-mask pixel overlaps the cell; it
is independent of `tissue.threshold`. The older thresholded `tissue_status`
field is retained for compatibility and diagnostics.

The tissue-valid graph and embedding mapping are computed independently. A
tissue-valid cell with `embedding_index == -1` remains navigable and provides
raw RGB; it simply adds no evidence to ABMIL. Conversely, an embedding cannot
make a non-tissue cell navigable. Existing HDF5 origins are mapped using their
global grid coordinates, not ROI-local coordinates.

## Spatial observation frame

The environment treats `navigation_roi_level0` as the common level-0 extent of
the overview RGB, current-location channel, and scaled-visit channel. It crops
the persisted WSI overview to this extent and projects global cell bounds and
visit counts into the same ROI-relative raster before applying the configured
observation resize. This remains one bounding box when tissue has multiple
disconnected regions; empty background gaps inside the box are retained in the
overview.

The last two scalar observations are the current cell center normalized between
the nominal ROI edges, not the full WSI edges. Mapping in either direction
explicitly adds or removes the ROI's global level-0 origin. That observation
transform is not an embedding transform: `embedding_index`, HDF5 coordinates,
global cell bounds, grid origins, and quadtree IDs remain exactly as persisted.
The separate RGB viewport is likewise read from the current cell's original
global bounds.

Dataset-global IDs reserve a fixed block of `2^40` values for each
deterministic adapter `slide_index`; a slide's first ID is therefore
`slide_index * 2^40`. The persisted `cell_offsets.json` may be sparse when a
bounded subset is preprocessed. Its ranges are sorted and validated as unique,
positive, non-overlapping int64 ranges, and each slide's actual cell count must
fit inside one block. This keeps subset preprocessing deterministic without
reading unrequested embedding files.

## Workspace

```text
workspace/
  manifests/dataset_manifest.json
  splits/<dataset_name>_split.json
  tissue/<dataset_name>/<fingerprint>/
  processed/<dataset_name>/<fingerprint>/slides/<slide_id>/
  models/belief/<fingerprint>/
  runs/rl/<fingerprint>/
```

Large embeddings remain referenced in place. Manifests use path/size/mtime and structured metadata fingerprints rather than hashing terabytes of WSI pixels.

Generated AtlasPatch tissue manifests are the exception for model provenance:
they include the fixed Hugging Face repository/filename, exact resolved
snapshot, checkpoint size and SHA256, and pinned/resolved SAM2 commit. That
record also includes the installed default SAM2 configuration filename, size,
and SHA256. The path-independent checkpoint/config identity participates in
the normalized mask fingerprint and is compared with the currently cached
Hugging Face `main` snapshot before reuse. AtlasPatch's RGB mask visualization
is diagnostic only; the binary `*_mask_bw.png` is the normalized mask source.
