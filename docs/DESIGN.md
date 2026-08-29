# Design

## System boundary

The selected native or manifest dataset adapter owns discovery, labels, patient groups, and source-cohort metadata. The HDF5 provider owns lazy sparse feature access. Preprocessing owns dense WSI geometry, tissue projection, and source-row mappings. The Gym environment consumes processed geometry plus RGB/embedding providers. ABMIL consumes only labels, the persisted split, and embeddings. PPO consumes the processed environment and a frozen calibrated ABMIL accumulator.

This separation is intentional: `train-belief` cannot trigger tissue detection, geometric preprocessing, environment construction, or PPO.

Geometric preprocessing is development-only inside the training pipeline.
Dataset-global cell IDs use a fixed, non-overlapping int64 block per
deterministic adapter slide index. The persisted offset table may therefore be
sparse, and preprocessing an arbitrary bounded development subset never opens
any other HDF5 file. An explicitly authorized test evaluation may add test
entries only after development preprocessing completes; it receives a separate
manifest and cannot become a belief/PPO training dependency. This authorization
comes either from `eval-rl --split test --allow-test` or from the default-on
post-training final evaluation. `--no-final-test-eval` keeps official-test WSI
pixels and HDF5 payloads unopened by `train-rl` and `run`.

## Geometry before evidence

The six levels are fixed at 0.625×, 1.25×, 2.5×, 5×, 10×, and 20×. Their
nominal level-0 cell footprints are 16,384, 8,192, 4,096, 2,048, 1,024, and 512
pixels. Preprocessing computes one box around the union of all tissue-mask
regions, including disconnected islands. It expresses the box in level-0
coordinates and expands it outward on the global 16,384-pixel grid. The box
selects a navigation envelope from the unchanged global hierarchy; full
canonical rows, IDs, and the slide origin remain persisted, and levels are
never independently cropped or rebased. Nominal bounds may overhang a right or
bottom edge, while clipped bounds are used for maps and the RGB reader pads the
rest white.

The binary mask is projected onto every level and persisted as `tissue_valid`;
any mask overlap is sufficient, independently of the legacy thresholded
`tissue_status` field.
Topology is computed from those maps before sparse HDF5 rows are attached.
Parent/child relations remain exact quadtree geometry. Each same-scale pan edge
points to the nearest tissue-valid cell farther in its direction on the same
row or column, allowing a jump across background; it is absent if no such cell
exists. Of the four aligned children, only tissue-valid children can be entered.
Zoom-out keeps the ordinary aligned parent. A missing `embedding_index` never
removes a tissue-valid cell or relationship, and HDF5 rows are mapped by global
slide coordinates rather than ROI-local indices.

## Belief

Gated ABMIL receives `[UNI embedding; four-way magnification one-hot]`. Partial bags are sampled dynamically after the slide split. Magnification composition is chosen before unique rows, so dense scales do not dominate by row count. The trained model is selected on deterministic validation NLL, then scalar-temperature calibrated on validation only.

The frozen accumulator caches each hidden projection and attention logit. Its normalized pooled representation is mathematically identical to a full forward pass over the same evidence set.

The optional belief-shaped reward counts unique valid embedding observations,
not visited cells. The first such observation updates the accumulator from the
initial `0.5` belief but has zero shaping reward; belief deltas are rewarded
starting with the second observation. The default sparse reward is unchanged.

## Environment

The actor's spatial world is the persisted snapped tissue ROI. Its five-channel
global branch is `[ROI overview RGB; current location; scaled visits]`. Overview
RGB is cropped from the WSI overview to that box, while location bounds and
global visit counts are projected through the same global-to-ROI transform;
the aligned result is then resized by the existing observation pipeline. All
three views therefore cover one identical level-0 extent, including one box
around multiple disconnected tissue islands.

The viewport branch is different by design: it remains the actual RGB for the
current globally addressed navigation cell. The five normalized scalars have
the backward-stable order `[magnification, remaining_budget, belief, roi_x,
roi_y]`; x/y are the current cell center normalized against the nominal snapped
ROI edges rather than full-slide dimensions. The inverse ROI mapping returns
original level-0 coordinates. It does not alter cell IDs, grid origins,
parent/child relationships, viewport reads, or embedding lookup coordinates.

Optional training-time augmentation adds a virtual agent-facing frame without
rewriting that canonical geometry. One episode-wide dihedral orientation
rotates/flips the cropped overview, viewport, location, and visit rasters;
agent-facing bounds and normalized x/y follow the same frame. Pan directions
and zoom quadrants are inversely mapped to canonical graph edges, while cell
IDs and HDF5 source rows remain stable. A separately sampled episode-wide H&E
optical-density transform affects RGB only. Validation, test, manual, and
benchmark environments stay canonical.

The historical boolean action mask stays in `info` and rollout storage. It is
generated from the cached tissue-valid navigation graph, so PPO and the manual
controller share exactly the same pan and zoom constraints. Invalid external
actions are no-ops with normal step cost and no budget, visit, or evidence
change.

The final available action is reserved for an explicit diagnosis. Gym termination therefore represents diagnosis; the normal episode path does not time out.

## Recurrent PPO

Separate residual CNNs encode overview and viewport. A scalar MLP joins them before a one-layer GRU and actor/critic heads. Rollouts retain exact masks and recurrent boundary state. PPO minibatches shuffle temporal sequences, never individual timesteps, and reset GRU state at episode starts.

Each vector worker also owns training-only episode return and length
accumulators. They survive rollout boundaries, reset on termination or
truncation, and emit a record only for a completed episode. The trainer logs
those values as `training/episode_return` and `training/episode_length`, logs
the cumulative `training/completed_episodes` count once per rollout, and logs
`training/learning_rate` after each update. Regular checkpoints preserve the
completed count. Worker processes and OpenSlide handles are intentionally not
checkpointed, so resume restores the count but starts clean episodes instead
of attaching an abandoned partial return to a new slide.

Adding ROI x/y changes the scalar encoder input from three values to five. The
PPO stage algorithm fingerprint is therefore versioned forward so legacy
three-scalar policy checkpoints are retrained; belief checkpoints, embeddings,
and processed global coordinates are not reinterpreted.

## Validation selection and final test

At each crossed `rl.validation_interval`, the current policy deterministically
examines every validation slide once. The pass is excluded from the training
global-step counter and gradient updates. Its full aggregate/per-slide result
is stored in `validation_<global-step>.json`, while numeric aggregates use
`validation/*` tags in every enabled scalar-tracking backend.

Validation is also the sole model-selection boundary. A finite AUROC is the
primary objective; accuracy is used only when AUROC is unavailable. The first
valid candidate is retained, AUROC supersedes an accuracy-selected candidate,
accuracy never supersedes an AUROC-selected candidate, and subsequent values
must be strictly greater, so a tie keeps the earlier weights. The winner is
atomically stored as `checkpoints/best_validation.pt`. Its embedded selection
record is mirrored in `checkpoints/best_validation.json`; the checkpoint is
authoritative and can repair a missing or interrupted sidecar. Regular
`checkpoint_<global-step>.pt` files remain the only automatic-resume
candidates, so the best evaluation snapshot cannot accidentally rewind
training. A genuine `RESUME` retains its best-selection state. If the planner
instead starts a fresh `TRAIN` attempt after a crash left only best-selection
files, those files are moved intact beneath `checkpoints/best_validation_archive/`
and the new attempt selects from scratch.

After a successful `train-rl` or `run`, a separate final test pass is enabled
by default. It explicitly authorizes test preprocessing, evaluates all official
test slides exactly once per available checkpoint role, and reports
`best_validation.pt` independently from the newest compatible regular
checkpoint. A missing or invalid best checkpoint is marked unavailable and
does not suppress the latest result. `--no-final-test-eval` disables the pass,
and dry-run mode never performs it. The two-entry result is written to
`evaluation_ppo_test.json`. Existing `metrics` and the
`test/best_validation/*` and `test/latest/*` scalar namespaces continue to
describe the PPO diagnosis head and shared trajectory. Every completed entry
also reports `ppo_patches_belief_metrics`, obtained by thresholding the
calibrated terminal belief on PPO-selected patches at `0.5`; its tracking
metrics are nested under `test/<selection>/ppo_patches_belief/*`. Each entry
identifies the resolved checkpoint, its training step, dependencies, and
embedded validation record.

Standalone `eval-rl --split test --allow-test` uses the same comparison. It
scans atomic regular checkpoints directly rather than requiring a complete PPO
manifest, so the latest compatible checkpoint from interrupted training remains
evaluable. A saved run-local configuration selects its parent run, with
`--run-dir` available as an explicit override.

`eval-baselines` is independent of PPO checkpoint selection. It evaluates the
same frozen calibrated belief model along paired random and serpentine-raster
routes. For each navigator and slide, one diagnosis-free route supplies every
confidence and fixed-step prefix, so changing a stopping condition cannot
change the patches seen by another condition. Random routes use stable
per-slide seeds, making every budget a prefix of the same route.

Confidence endpoints require the configured sweep evidence floor, stop when
the calibrated belief exits `[1-threshold, threshold]`, and otherwise use the
configured total-action cap. Fixed endpoints are indexed by an exact number of
navigation actions. The belief is read after step `K` and one diagnosis action
is added, producing total episode length `K + 1`; zooms and revisits count
toward `K`, while unique evidence remains an independently reported trajectory
outcome. Evaluation-only action-mask and episode-length overrides allow this
shared prefix collection without changing persisted configuration or any
training fingerprint.

This post-training sequence is one-way:

```text
train PPO -> validate/select best + save latest -> freeze both roles -> evaluate test
```

Test labels, rewards, predictions, and metrics cannot update either policy,
select between the two checkpoint roles, change a stage fingerprint, or enter
resume compatibility. Test navigation artifacts are recorded separately under
`test_evaluations/<evaluation-fingerprint>/` with `training_dependency: false`.
The manifest and test index are published atomically after preparation and are
immutable once complete. A fingerprint-scoped process lock and a post-lock
manifest check guarantee one builder for concurrent automatic preparation.
Cluster evaluator arrays can instead use the strictly read-only
`--test-artifacts require` mode after one explicit `prepare-test` job. The
automatic lock relies on POSIX `flock` support from the shared filesystem;
`require` performs no preprocessing writes and therefore avoids that runtime
coordination path entirely. The validated index is passed into the test episode
source, which compares each slide's input fingerprint, deterministic index,
cell offset, cell count, and (for newly created indexes) complete metadata
fingerprint on first load. This pins evaluation to the prepared slide
generation without scanning every slide up front. Concurrency coordination is
an operational option enabled by the evaluator's cold builder; the normal
training preprocessing path leaves that option disabled and retains its prior
artifact semantics.
