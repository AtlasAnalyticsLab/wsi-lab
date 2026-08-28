# WSI Lab

`wsi-lab` is a research package for sequential examination of whole-slide images. It builds a dense six-level navigation hierarchy, attaches sparse tissue-filtered UNI v2 evidence, trains a calibrated gated-attention MIL belief model, and supplies a local recurrent-PPO implementation for `WSILab-v0`.

The package has native-layout adapters for CAMELYON16, CAMELYON17, their
combined binary-classification cohort, and TCGA-NSCLC (LUAD and LUSC), plus a
manifest adapter for other binary WSI datasets. All adapters use the existing
AtlasPatch/UNI v2 HDF5 schema. PEAN is intentionally unsupported.

## Install

Create the dedicated environment; do not reuse another WSI environment:

```bash
git clone <repository-url>
cd wsi-lab
conda env create --file environment.yml
conda activate wsi-lab
```

See [the installation guide](docs/INSTALLATION.md) for verification, CPU setup, AtlasPatch/SAM2 requirements, and troubleshooting.

## Configure

```bash
export CAMELYON16_ROOT=/path/to/CAMELYON16
wsi-lab validate --config configs/camelyon16.yaml
```

The default split preserves every `test_*` slide as official test and creates a deterministic 216/54 train/validation split over development slides (seed 1, stratified by binary label and center). Test access is guarded; evaluation defaults to validation.

AtlasPatch is the default tissue-mask source. Preprocessing caches one binary
mask per slide; tumor masks and annotations are never used to construct this
mask or the navigation graph. Set `tissue.source` explicitly to `auto` to
prefer a compatible dataset sidecar or to `existing` to require one.

## Keep datasets in their native layouts

Do not move, copy, rename, or manually arrange WSIs into `train/`, `val/`, and
`test/` directories. Dataset storage and experimental split membership are
separate. A dataset adapter reads the native files in place, normalizes slide
metadata, and the split manager writes deterministic membership to the ignored
`workspace/` directory.

Select the matching adapter with `dataset.name`:

- `camelyon16` reads `images/`, `evaluation/reference.csv`, and the dataset-level
  `Embeddings/` directory. The labeled `test_*` cases remain the official test
  partition; development slides are split at slide level because this layout
  has no reliable patient identifier.
- `camelyon17` reads `images/`, `stages.csv`, and the dataset-level
  `Embeddings/` directory. Labeled patients are split as patient groups. Images
  without stage rows are retained as blind official-test records and are never
  used for training, reward computation, or metrics.
- `camelyon16_17` reads both native datasets beneath one parent without copying
  them. It preserves each source's independently generated split and unions the
  corresponding train, validation, and labeled-test partitions. CAMELYON16
  normal and CAMELYON17 negative slides map to benign (`0`); tumor, ITC, micro,
  and macro slides map to cancerous (`1`). CAMELYON17's blind official-test
  slides remain unassigned.
- `tcga_nsclc` reads the native `TCGA-LUAD/` and `TCGA-LUSC/` cohorts, each with
  its own `images/` and `Embeddings/` directories. LUAD/LUSC are class cohorts,
  not data splits. The package generates train/validation/test membership while
  keeping every slide from one TCGA patient in one partition.
- `generic` reads an explicit root-relative `metadata.csv`. Use this for an
  otherwise unsupported binary WSI dataset; it never guesses labels, patient
  membership, cohorts, or image paths from filenames.

Portable starting configurations are supplied as
`configs/camelyon16.example.yaml`, `configs/camelyon17.example.yaml`,
`configs/camelyon16_17.example.yaml`, `configs/tcga_nsclc.example.yaml`, and
`configs/generic.example.yaml`. Set the environment variable named at the top
of the selected file; do not replace it with a personal absolute path in a
configuration that will be committed.

For a generic dataset, use this minimum structure:

```text
CUSTOM_DATASET/
├── metadata.csv
├── images/
│   └── <slide image files>
└── Embeddings/
    ├── UNI_v2_2.5x_256/patches/<slide_id>.h5
    ├── UNI_v2_5x_256/patches/<slide_id>.h5
    ├── UNI_v2_10x_256/patches/<slide_id>.h5
    └── UNI_v2_20x_256/patches/<slide_id>.h5
```

Its required CSV columns are:

```text
slide_id,image_path,label,label_name,patient_id,center,official_cohort
```

Every `image_path` is relative to `CUSTOM_DATASET/`; optional tissue-mask,
tumor-mask, and annotation paths may be added in dedicated columns. Use
`official_cohort=development`, `test`, or `test_unlabeled`. A label is `0` or
`1` and may be blank only for `test_unlabeled`. Populate `patient_id` for every
related slide so the generated split can prevent patient leakage.

See [the data-format guide](docs/DATA_FORMAT.md) for the exact native directory
trees, split policies, and generic `metadata.csv` contract.

## Commands

```bash
wsi-lab validate --config configs/camelyon16.yaml
wsi-lab preprocess --config configs/camelyon16.yaml
wsi-lab train-belief --config configs/camelyon16.yaml
wsi-lab eval-belief --config configs/camelyon16.yaml --split val
wsi-lab train-rl --config configs/camelyon16.yaml
wsi-lab eval-rl --config configs/camelyon16.yaml --split val
wsi-lab eval-rl --config configs/camelyon16.yaml --split test --allow-test
wsi-lab manual-play --config configs/camelyon16.yaml --split val
wsi-lab render-trajectory --config configs/camelyon16.yaml --trajectory trajectory.json
wsi-lab benchmark-env --config configs/camelyon16.yaml
wsi-lab smoke-test --config configs/camelyon16.yaml
```

The complete idempotent workflow is:

```bash
wsi-lab run --config configs/camelyon16.yaml
```

`train-rl` and `run` perform one guarded official-test reporting operation
after the pipeline completes. This final report is enabled by default and
evaluates both the best-validation checkpoint and the newest compatible regular
checkpoint. Disable it when the official test set must remain unopened, or
while doing exploratory runs:

```bash
wsi-lab train-rl --config configs/camelyon16.yaml --no-final-test-eval
wsi-lab run --config configs/camelyon16.yaml --no-final-test-eval
```

The explicit `--final-test-eval` form is also accepted. A dry run never starts
the final evaluation.

Inspect its stage decisions without writing, preprocessing, or training:

```bash
wsi-lab run --config configs/camelyon16.yaml --dry-run
```

Each stage has a content fingerprint and completion manifest. Compatible complete work is skipped, stale artifacts are rebuilt, and compatible incomplete training checkpoints resume.

RL outputs live under
`workspace/runs/rl/<ppo-fingerprint>/` (or the configured workspace root):

- `tensorboard/` records PPO losses, `training/episode_return`,
  `training/episode_length`, cumulative `training/completed_episodes`, and
  `training/learning_rate`, plus `validation/*`,
  `test/best_validation/*`, and `test/latest/*` metrics;
- `manifest.json` records the final global step, update and completed-episode
  counts, last PPO metrics, dependencies, and best-validation selection;
- `validation_<global-step>.json` contains aggregate and per-slide validation
  results together with the checkpoint-selection decision;
- `checkpoints/best_validation.pt` is the validation winner and
  `checkpoints/best_validation.json` is its selection record;
- `evaluation_ppo_test.json` contains separate `best_validation` and `latest`
  aggregate/per-slide results. Each completed checkpoint keeps PPO diagnosis-head
  and shared trajectory results in `metrics`, and adds
  `ppo_patches_belief_metrics`, which scores the calibrated final belief at a
  `0.5` threshold on the patches selected by PPO. The latter also reports NLL,
  Brier score, ECE, confusion counts, and a counterfactual terminal-diagnosis
  return. Each entry records its resolved checkpoint provenance.

Validation prefers the highest finite AUROC. Accuracy is used only until a
finite AUROC is available, and an exact tie keeps the earlier checkpoint. Test
evaluation always treats that winner and the newest compatible regular
checkpoint as independent roles. If no validation winner exists, its report is
marked unavailable while the latest checkpoint is still evaluated. Test results
never select a checkpoint or feed back into training.

Standalone `eval-rl --split test --allow-test` uses the same two-checkpoint
report and can evaluate checkpoints from an interrupted training experiment;
PPO manifest completion is not required. When `--config` points to a saved
`<run>/config.yaml`, that run is recognized automatically. Use `--run-dir RUN`
to select it explicitly when needed.

View every run while it is training with:

```bash
tensorboard --logdir workspace/runs/rl
```

Use [manual play](docs/MANUAL_PLAY.md) to control the production environment
with the keyboard while viewing the exact actor observations, scalar values,
action mask, and rewards.

## Core invariant

Navigation is bounded by one box enclosing the union of all tissue islands. The
box is represented in level-0 slide coordinates and expanded outward on the
global 0.625× grid, so every finer level retains the original slide origin and
exact quadtree alignment. Cached `tissue_valid` maps define the navigation
graph: pans select the nearest tissue-valid cell in the requested direction,
possibly jumping across background, and zoom-in actions expose only
tissue-valid children.

Embeddings remain sparse attributes on some 2.5×, 5×, 10×, and 20× cells; they
never define the tissue ROI or graph. A tissue-valid cell with no embedding is
still navigable and produces RGB, visits, and normal action cost, but contributes
no diagnostic evidence.

That same snapped box is the spatial extent of every actor-wide observation.
Overview RGB is cropped to it, and the current-location and visit channels are
projected into its coordinate frame before the five-channel image is resized.
The scalar vector is `[magnification, remaining_budget, belief, roi_x, roi_y]`,
where x/y are the current cell center normalized between the nominal snapped
tissue-ROI edges. The current viewport still reads the selected global WSI
cell. Navigation IDs, global patch coordinates, and embedding lookups are never
shifted or reinterpreted by the observation crop.

PPO training can optionally sample one of eight rotated/flipped virtual slide
frames and/or one mild H&E color transform per episode. Spatial channels,
reported coordinates, pan directions, and zoom quadrants remain aligned from
the agent's perspective, while canonical cell IDs and embedding rows remain
unchanged. These runtime transforms reuse the existing preprocessing and raw
RGB caches and are disabled for validation and testing.

The project repository has no license grant. Third-party terms are summarized in [THIRD_PARTY.md](docs/THIRD_PARTY.md).
