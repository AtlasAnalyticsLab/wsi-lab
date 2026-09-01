# Agent-integrated recurrent-PPO efficiency experiment

## Scope

This experiment compares WSI-Lab and HistoGym during native-observation
recurrent-PPO interaction and optimization. It evaluates computational
efficiency only; rewards, diagnostic accuracy, and learning quality are not
compared.

Both backends execute 10,000 balanced episodes over the same 100 CAMELYON16
slides, with 100 externally truncated interactions per episode (1,000,000 PPO
interactions per backend). One invocation runs each backend once in the fixed
order WSI-Lab then HistoGym.

The two agents share the same three-channel visual CNN architecture, GRU,
critic, PPO implementation, hyperparameters, seed, and CUDA device. WSI-Lab
also uses a separate five-channel global-image CNN, a scalar MLP, precomputed
embedding lookup, and a frozen calibrated ABMIL belief model. The belief model
is called but never optimized; UNI-v2 generation is not part of the run.
WSI-Lab diagnosis actions and HistoGym's stay action are masked.

Pinned source revisions:

- WSI-Lab: `0deb7837364add790f9530c41f0ef0b5e616c05c`
- HistoGym: `5b742760321d28a469d2ef9494e886a49d0aa7f7`

The exact Python/PyTorch environment is in `environments/environment.yml`.

## Prerequisites and external inputs

- Linux with Git, Conda, and OpenSlide-compatible system support.
- A CUDA-capable device visible to PyTorch.
- CAMELYON16 and the four UNI-v2 embedding collections in the layout below.
- The frozen calibrated checkpoint identified in `models/README.md`.

The checkpoint is an external experimental input and is not included in this
source-only package. Obtain it from the paper's separately distributed
artifact bundle or from the authors, verify its hash, and store it locally.
The checkpoint cannot be reconstructed from its hash.

Set three shell variables:

```bash
export CAMELYON16_ROOT
export BELIEF_CHECKPOINT
export CUDA_DEVICE
unset CUDA_VISIBLE_DEVICES
```

`CUDA_DEVICE` must be an explicit host-visible PyTorch device such as `cuda:0`.
Leave `CUDA_VISIBLE_DEVICES` unset so PyTorch and resource monitoring use the
same index. The dataset layout is:

```text
CAMELYON16_ROOT/
├── images/tumor_*.tif
├── annotations/tumor_*.xml
├── evaluation/reference.csv
└── Embeddings/
    ├── UNI_v2_2.5x_256/patches/tumor_*.h5
    ├── UNI_v2_5x_256/patches/tumor_*.h5
    ├── UNI_v2_10x_256/patches/tumor_*.h5
    └── UNI_v2_20x_256/patches/tumor_*.h5
```

The configuration step validates the dataset cohort and the checkpoint hash
and size. The embedding collections are required precomputed inputs; their
generation is deliberately outside the timed comparison.

## 1. Verify and create the unified environment

Run from this experiment directory:

```bash
bash scripts/verify_package.sh
bash scripts/setup.sh
```

Setup creates ignored runtime state, checks out the pinned WSI-Lab revision,
clones the pinned HistoGym revision, and creates one shared Python/PyTorch
environment. A narrow in-process Gym-to-Gymnasium alias allows both native
backends to use the identical PPO implementation without editing HistoGym.
The repository's WSI-Lab training and library files are not modified.

## 2. Configure the local inputs

```bash
runtime/unified-env/bin/python scripts/configure.py \
  --dataset-root "${CAMELYON16_ROOT}" \
  --belief-checkpoint "${BELIEF_CHECKPOINT}" \
  --device "${CUDA_DEVICE}"
```

Configuration creates ignored `work/` inputs. It reconstructs and verifies the
seed-1 cohort, creates the balanced 10,000-episode schedule, binds the exact
checkpoint by hash, and records the selected CUDA device.

## 3. Validate before measuring

```bash
bash scripts/validate.sh
```

Validation checks source revisions, workload size, input identities, policy
parameter counts, and bit-identical initialization of the shared viewport CNN,
GRU, and critic. It does not open a WSI or run an episode.

## 4. Run the two measured blocks

```bash
bash scripts/run_measured.sh wsi_lab
bash scripts/run_measured.sh histogym
```

Before each backend, the wrapper performs `sync` and best-effort targeted
`POSIX_FADV_DONTNEED` eviction on the fixed input files and records whether
each eviction succeeded. WSI-Lab performs fresh AtlasPatch/navigation
preprocessing before PPO. HistoGym has no separate preprocessing phase.
Existing output directories are never reused.

## 5. Generate metrics, checks, and figures

```bash
bash scripts/analyze.sh
```

The analysis validates protocol identity and accounting before creating:

```text
analysis_outputs/native_observation_ppo_10k/
├── metrics.csv
├── report.md
├── summary.json
├── cumulative_runtime.png
├── cumulative_runtime.pdf
├── online_components.png
└── online_components.pdf
```

Reported quantities include policy parameter counts; preprocessing, online,
end-to-end, native-step, policy-inference, and PPO-optimization time; latency
and throughput; resource use; and observation payload. No diagnostic metric is
produced.

## Timing definitions

- Native step time wraps the native `env.step()` call. WSI-Lab embedding lookup
  and frozen belief inference are nested subcomponents and are not added twice.
- Online wall time includes native lifecycle and observations, observation
  packing/device transfer, action sampling, rollout bookkeeping, and recurrent
  PPO optimization.
- End-to-end time adds fresh WSI-Lab preprocessing and policy/optimizer setup.
- UNI-v2 generation, result serialization, and final checkpoint writes are
  excluded.
- CUDA is synchronized around timed policy and PPO sections.
- Process-tree resources are sampled every 0.1 seconds; the configured GPU is
  sampled every second.

## Repeating the experiment

One invocation creates one execution per backend. To run another independent
execution, archive `runs/` and `analysis_outputs/` outside the package and
repeat Sections 4 and 5. The deterministic `work/` cohort, schedule, and
protocol can be reused.

## Sharing and privacy

This package contains no precomputed results. Newly generated `runtime/`,
`work/`, and `runs/` files can contain the reproducer's local paths and raw
result metadata and should not be published unreviewed. The analysis summary
stores source hashes rather than source paths, and generated PDFs omit creation
timestamps. Run the package-level privacy check before committing the source
package.
