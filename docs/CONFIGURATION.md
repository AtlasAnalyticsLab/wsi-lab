# Configuration

Configuration is strict Pydantic v2 YAML. Unknown keys fail before expensive work. `${NAME}` expressions are expanded eagerly; an unset variable is an error. Relative dataset sidecars and embedding roots resolve beneath `dataset.root`; `workspace.root` resolves relative to the YAML file.

`WSILab-v0` fixes the 11 actions, six navigation levels, four belief scales, three actor branches, and magnification one-hot encoding. Episode length, minimum evidence, image sizes, visit saturation, rewards, caches, training hyperparameters, and devices are configurable with paper-protocol defaults.

Tissue sources (`atlaspatch` is the default):

- `atlaspatch`: generate and cache a workspace mask even when a source mask exists;
- `auto`: validate/reuse a compatible configured mask, otherwise run AtlasPatch;
- `existing`: require a compatible configured mask.

Every selected mask is normalized to a cached binary mask for preprocessing.
Tissue detection is independent of tumor masks and annotations; neither may be
used to form the tissue ROI or navigation-validity maps.

AtlasPatch 1.1.0's `detect-tissue` command fixes the segmentation checkpoint to
`AtlasAnalyticsLab/AtlasPatch` / `model.pth` and exposes no model, revision, or
checkpoint option. Accordingly, `model_identifier`, `model_revision`, and
`checkpoint` are intentionally not configuration fields; strict validation
rejects them instead of recording an override that AtlasPatch would ignore.
After a successful call, the adapter resolves the downloaded Hugging Face
snapshot and records its SHA256 together with the pinned SAM2 source commit.

The supplied portable configurations and their dataset-root variables are:

```text
configs/camelyon16.example.yaml    CAMELYON16_ROOT
configs/camelyon17.example.yaml    CAMELYON17_ROOT
configs/camelyon16_17.example.yaml CAMELYON_DATASETS_ROOT
configs/tcga_nsclc.example.yaml    TCGA_NSCLC_ROOT
configs/generic.example.yaml       WSI_DATASET_ROOT
```

For example:

```bash
export CAMELYON16_ROOT=/path/to/CAMELYON16
```

For the combined adapter, point `CAMELYON_DATASETS_ROOT` at the common parent,
not at either individual dataset:

```bash
export CAMELYON_DATASETS_ROOT=/path/to/CAMELYON_DATASETS
# Contains CAMELYON16/ and CAMELYON17/ by default.
```

Set each single-dataset variable to that dataset's top-level native root, never
its `images/`, `masks/`, or embedding subdirectory. Keep the `${...}`
expression in any YAML that will be shared or committed so local filesystem
paths are not published.
TCGA's `embeddings.root` is written relative to each LUAD/LUSC cohort. For
`camelyon16_17`, the relative `embeddings.root` is likewise applied beneath
each native CAMELYON child root; `dataset.labels` selects the CAMELYON16 table
and `dataset.camelyon17_labels` selects the CAMELYON17 table. Generic manifest
image and sidecar paths are root-relative. See
[DATA_FORMAT.md](DATA_FORMAT.md) for the exact adapter contracts.

If a labeled native test cohort exists, it is preserved and
`split.test_fraction` is ignored. Otherwise the package creates a grouped
train/validation/test split using `validation_fraction` and `test_fraction`.
Patient grouping is used when reliable IDs are available for every development
record. Blind `test_unlabeled` records are persisted as unassigned and cannot
be used by training or evaluation.

`camelyon16_17` is the deliberate exception to a single global split. The split
manager constructs the CAMELYON16 and CAMELYON17 splits independently with the
same `split.seed`, `validation_fraction`, `test_fraction`, and
`stratify_by_center` values, then unions partitions with matching names. Thus
CAMELYON16 retains its official labeled test slides while CAMELYON17 retains
patient-grouped generated test slides. With complete native datasets and the
defaults, the result is 516 train, 154 validation, and 229 labeled test slides;
the 500 blind CAMELYON17 official-test slides remain unassigned.

Standalone evaluation defaults to `val`. Direct access to `test` requires both
`--split test` and `--allow-test`. Training and calibration APIs reject test
records unconditionally.

An authorized PPO test evaluation reports two checkpoint roles independently:
`checkpoints/best_validation.pt`, when available, and the newest compatible
regular `checkpoint_*.pt`. The regular checkpoint scan does not require a
complete PPO manifest, so `eval-rl` can report the latest atomic checkpoint
left by interrupted training. A missing or incompatible best checkpoint is
reported as unavailable and is never replaced or mislabeled with the latest
checkpoint. A saved `<run>/config.yaml` selects its parent run automatically;
`eval-rl --run-dir RUN` provides an explicit override.

Navigation baselines have a separate reporting-only command and do not resolve
or load a PPO run:

```bash
wsi-lab eval-baselines \
  --config CONFIG.yaml \
  --split test \
  --allow-test
```

The default sweep compares `random` and `raster` navigators. Confidence
conditions use thresholds `0.5 0.9`, require 32 unique evidence patches, and
allow 500 total actions including the final diagnosis. Fixed conditions stop
after fixed navigation-step counts `1 4 8 16 32 64 128 256 500`, followed by
one additional diagnosis action. A budget counts every navigation action,
including zooms and revisits; unique evidence count remains a separately
reported outcome. Override these values with `--navigators`, `--thresholds`,
`--confidence-min-evidence`, `--confidence-max-total-actions`, and
`--step-budgets`. `--random-seed` defaults to `rl.seed`, and `--output`
selects a non-default report path.

The sweep constructs evaluation environments with diagnosis available from
the start and enough capacity for the largest fixed-step prefix plus its
diagnosis action (501 actions under the defaults). Those are evaluator-local
overrides: the saved configuration, preprocessing and belief fingerprints,
trained PPO environments, and checkpoints are unchanged.
The default atomic report is
`workspace/models/belief/<belief-fingerprint>/evaluation_<split>_navigation_baseline_sweep.json`.

The preprocessing and training pipeline stages materialize only the persisted
train and validation partitions and do not open guarded test HDF5 payloads or
WSI pixels. Planning may stat source inventory paths, sizes, and modification
times to decide whether a full-dataset validation artifact is stale; it does
not read those guarded files. After those stages complete, however, `train-rl`
and `run` default to a separate authorized final test evaluation. Use
`--no-final-test-eval` to suppress that post-pipeline operation and leave test
pixels and embeddings unopened. `--dry-run` also never evaluates test.

Both the default-on final operation and the explicit
`eval-rl --split test --allow-test` command first require complete
train/validation preprocessing. They then add only explicitly authorized test
entries to the sparse deterministic cell-ID table and record the operation in
an immutable manifest beneath
`test_evaluations/<evaluation-fingerprint>/manifest.json`, with
`training_dependency: false`. Neither those artifacts nor any test metric can
affect PPO weights, validation checkpoint selection, stage fingerprints, or
resume compatibility. Each slide receives a fixed non-overlapping int64 block
from its deterministic adapter index, so arbitrary bounded subsets never need
to inspect unrequested embeddings.

Run `wsi-lab prepare-test --config CONFIG --allow-test` once before a cluster
evaluator array. Array jobs should use `--test-artifacts require`, which only
accepts the exact completed manifest and performs no preprocessing writes. The
default `--test-artifacts auto` is intended for individual evaluators: it
reuses a compatible warm cache and otherwise coordinates one builder with a
fingerprint-scoped interprocess lock. Test evaluation uses the separate
immutable slide index as an authority: each slide generation is checked lazily
when evaluation first loads it. A stale or replaced generation therefore fails
closed without adding an up-front 129-slide preprocessing pass. The evaluator
never rewrites the development `slides.json` on a cache hit. Cold preparation
may add deterministic test rows to the shared
`cell_offsets.json` table and create test-only slide directories, but it does
not alter existing development slide artifacts or make them training inputs.
Automatic cross-process preparation assumes the processed workspace provides
POSIX `flock` semantics across cluster nodes; the scheduler dependency plus
read-only `require` mode is the preferred cluster workflow. Do not run a writer
that rebuilds the same fingerprinted slide directory concurrently with the
dependent evaluator array.

`require` still recomputes the current stage fingerprints using read-only file
metadata so source-inventory changes fail closed. It does not preprocess test
slides or open their HDF5/WSI payloads during that check. This is a short
metadata validation rather than the prior per-slide preparation workload.
The extra offset, slide-build, index, and generated-mask coordination is enabled
only for evaluator-driven cold preparation; normal development preprocessing
retains its existing defaults, index publication, fingerprints, and training
dependencies.

The environment normalizes remaining budget by configured `max_episode_steps`;
default 500. Its scalar observation is
`[magnification, remaining_budget, belief, roi_x, roi_y]`; the first three
positions retain their prior meanings and x/y are the current cell center
normalized against the nominal edges of the snapped tissue bounding box. The
final available action is diagnosis-only. Diagnosis actions become available
after `environment.min_diagnostic_evidence` unique valid embeddings; the
default is 32. A masked action supplied by an external caller is a rejected
no-op with `-step_cost` and does not consume budget.

`environment.start_position` controls the initial 0.625x viewport. `center`
(the default) uses the preprocessed tissue-valid center cell. `random` samples
uniformly from all tissue-valid 0.625x cells on the selected slide; the sample
is reproducible through the seed passed to `reset()`.

`environment.geometric_augmentation` and
`environment.photometric_augmentation` independently enable runtime data
augmentation for PPO training. Both default to `false`, are sampled once per
episode from streams derived from the environment reset seed, and are forcibly disabled for
validation, testing, manual play, and other evaluation environments. Geometric
augmentation uniformly samples clockwise 0/90/180/270-degree rotations, with
or without a subsequent horizontal flip, while preserving actions relative to
what the agent observes. Photometric augmentation independently scales the H
and E optical-density concentrations in `[0.9, 1.1]` and applies that one
sampled transform consistently to the episode's overview and viewports.
Dedicated seeded augmentation streams do not consume the RNG used for slide
selection or random starting cells, so enabling augmentation does not change
those sequences for an otherwise identical run.

Augmentation is virtual: it neither creates augmented slides, grids, nor HDF5
files, nor changes preprocessing or belief-model fingerprints. Canonical HDF5
feature vectors remain attached to their logical patches and are reused without
feature extraction. Changing either option creates a distinct PPO fingerprint,
so an augmented PPO run cannot resume from an incompatible unaugmented run. Raw
RGB caching occurs before augmentation, allowing every episode to reuse the
same canonical cached patches.

Preprocessing computes a single level-0 bounding box around the union of all
tissue pixels and expands it outward to the global 0.625× navigation grid. The
box selects an aligned navigation envelope without rebasing or independently
cropping any level; canonical full-slide grid rows and IDs remain unchanged.
Each level caches a `tissue_valid` map whose cells are valid when they have any
binary-mask overlap. Pan actions target the nearest valid cell in the requested
row or column and can cross background gaps; a direction with no valid
destination is masked. Zoom-in quadrants are masked unless the geometrically
aligned child is tissue-valid. Zoom-out retains the aligned parent transition.
Embedding availability does not participate in any of these decisions.

The same `navigation_roi_level0` is also the coordinate extent for all
actor-wide spatial observations. The environment crops overview RGB to it and
projects both location and visits into it before applying `overview_size`.
Changing `overview_size` changes only the common output raster; it does not
change the ROI or navigation geometry. `viewport_size` still controls the
independently read current-cell RGB viewport. ROI projection never rebases the
underlying grids or the level-0 coordinates used to find embedding rows.

`environment.reward_type: default` is unchanged. With the optional `belief`
reward, belief starts at `0.5`; the first unique valid embedding updates ABMIL
but contributes no shaping reward. The existing change-in-true-class-belief
term starts with the second unique valid embedding. Embeddingless moves and
revisits do not advance that evidence count.

## PPO monitoring, validation, and final test

`rl.rollout_steps` is the number of transitions collected by each vector
worker before an update. Therefore one update normally advances the global
counter by `rl.num_envs * rl.rollout_steps`. `rl.validation_interval` and
`rl.checkpoint_interval` are global transition counts; an event runs after the
first completed rollout that crosses its threshold, rather than interrupting a
rollout at the exact number.

TensorBoard is controlled by `logging.tensorboard`. When enabled, the PPO run
directory records:

| Tags | Meaning |
|---|---|
| `ppo/*` | Policy/value/total loss, entropy, KL, clipping, gradient norm, explained variance, and optimizer-work counters |
| `training/episode_return` | Sum of rewards for each completed training episode |
| `training/episode_length` | Actions in each completed training episode |
| `training/completed_episodes` | Cumulative completed training episodes, emitted each rollout |
| `training/learning_rate` | Optimizer learning rate used for the completed PPO update |
| `validation/*` | Aggregate deterministic validation metrics plus `validation/is_best_checkpoint` and `validation/best_selection_value` |
| `test/best_validation/*` | Aggregate final-test metrics for the validation winner |
| `test/latest/*` | Aggregate final-test metrics for the newest compatible regular checkpoint |

Per-worker return and length accumulators continue across rollout boundaries.
Regular checkpoints preserve the cumulative completed-episode count. Because
vector-worker/OpenSlide state is deliberately not serialized, resuming starts
fresh worker episodes and discards only any partial, uncompleted episode totals.
The completed count, last PPO update metrics, dependencies, and current
best-validation record are also included in the PPO stage's `manifest.json`.

Every validation pass evaluates the complete `val` split once with a
deterministic policy. It writes `validation_<global-step>.json` and considers
the current weights for `checkpoints/best_validation.pt`. A finite AUROC is the
primary selection metric; accuracy is the fallback while no finite AUROC is
available. AUROC supersedes an accuracy-selected checkpoint, accuracy cannot
supersede an AUROC-selected checkpoint, and ties retain the earlier result.
The mirrored selection record is `checkpoints/best_validation.json`.

The final official-test evaluation is a command-line lifecycle option, not a
YAML hyperparameter. It is enabled by default for both `train-rl` and `run`:

```bash
wsi-lab train-rl --config CONFIG.yaml --final-test-eval
wsi-lab train-rl --config CONFIG.yaml --no-final-test-eval
```

After successful pipeline completion it evaluates every official test slide
once per available checkpoint role: `best_validation.pt` and the newest
compatible regular checkpoint. Results are written as a two-entry comparison
in `evaluation_ppo_test.json` in the PPO run directory. Numeric aggregates use
the `test/best_validation/*` and `test/latest/*` TensorBoard namespaces for the
PPO diagnosis head. Each checkpoint entry also contains
`ppo_patches_belief_metrics`, computed from its calibrated `final_belief >= 0.5`
on the same PPO-selected trajectory; those numeric values use the nested
`test/<selection>/ppo_patches_belief/*` namespace. Shared trajectory metrics
remain in `metrics`. Each entry records its resolved checkpoint path, global
step, dependency fingerprints, selection role, and embedded best-validation
record. These passes are reporting-only and do not update PPO, select between
the two models, or mark test artifacts as training inputs.
