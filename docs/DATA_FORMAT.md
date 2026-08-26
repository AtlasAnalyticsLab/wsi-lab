# Data and artifact formats

## CAMELYON16

Slide discovery is driven by `evaluation/reference.csv`, not filename globbing. `type=normal` maps to 0; `type=tumor` maps to 1. Both `micro` and `macro` tumor classes are binary cancer. Every `test_*` record is preserved as official test regardless of label.

CAMELYON16 exposes no reliable patient identifier in this layout. The default split is therefore slide-level. Consecutive tissue sections can introduce patient/tissue leakage that the package cannot detect; this limitation is recorded in split metadata.

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
`tissue_overlap_fraction`. This comparison uses the selected validated tissue
mask and is diagnostic only: embeddings and masks can originate from different
AtlasPatch runs.

## Processed grids

Each scale is stored as a memory-mappable structured NumPy array with stable int64 cell identifiers, a deterministic dataset-order `slide_index`, magnification, row/column, nominal and clipped level-0 bounds, parent/four children/four neighbors, tissue fraction/status, and a nullable source HDF5 row (`-1` sentinel). Small per-slide metadata maps the numeric slide identifier to `slide_id` and records the letterbox transform and fingerprints.

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
  splits/camelyon16_split.json
  tissue/camelyon16/<fingerprint>/
  processed/camelyon16/<fingerprint>/slides/<slide_id>/
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
