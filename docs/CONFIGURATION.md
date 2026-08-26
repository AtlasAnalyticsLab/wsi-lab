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
configs/tcga_nsclc.example.yaml    TCGA_NSCLC_ROOT
configs/generic.example.yaml       WSI_DATASET_ROOT
```

For example:

```bash
export CAMELYON16_ROOT=/path/to/CAMELYON16
```

Set each variable to the dataset's top-level native root, never its `images/`,
`masks/`, or embedding subdirectory. Keep the `${...}` expression in any YAML
that will be shared or committed so local filesystem paths are not published.
TCGA's `embeddings.root` is written relative to each LUAD/LUSC cohort; generic
manifest image and sidecar paths are root-relative. See
[DATA_FORMAT.md](DATA_FORMAT.md) for the exact adapter contracts.

If a labeled native test cohort exists, it is preserved and
`split.test_fraction` is ignored. Otherwise the package creates a grouped
train/validation/test split using `validation_fraction` and `test_fraction`.
Patient grouping is used when reliable IDs are available for every development
record. Blind `test_unlabeled` records are persisted as unassigned and cannot
be used by training or evaluation.

Standalone evaluation defaults to `val`. Direct access to `test` requires both
`--split test` and `--allow-test`. Training and calibration APIs reject test
records unconditionally.

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
a separate `test_evaluation_preprocessing.json` manifest with
`training_dependency: false`. Neither those artifacts nor any test metric can
affect PPO weights, validation checkpoint selection, stage fingerprints, or
resume compatibility. Each slide receives a fixed non-overlapping int64 block
from its deterministic adapter index, so arbitrary bounded subsets never need
to inspect unrequested embeddings.

The environment normalizes remaining budget by configured `max_episode_steps`;
default 500. Its scalar observation is
`[magnification, remaining_budget, belief, roi_x, roi_y]`; the first three
positions retain their prior meanings and x/y are the current cell center
normalized against the nominal edges of the snapped tissue bounding box. The
final available action is diagnosis-only. A masked action supplied by an
external caller is a rejected no-op with `-step_cost` and does not consume
budget.

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
| `test/*` | Aggregate metrics from the automatic final test evaluation |

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
once, preferring `best_validation.pt` and falling back to the latest compatible
regular checkpoint if no validation winner exists. Results are written to
`evaluation_ppo_test.json` in the PPO run directory and numeric aggregates are
written as `test/*` TensorBoard tags. The JSON also records the resolved
checkpoint path, global step, dependency fingerprints, selection mode, and
embedded best-validation record. This final pass is reporting-only and does
not update PPO, select a model, or mark test artifacts as training inputs.
