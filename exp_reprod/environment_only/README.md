# Environment-only efficiency experiment

## Scope

This experiment compares native online execution in WSI-Lab and HistoGym
without a learned policy, visual encoder, belief-model inference, diagnostic
action, rendering, or online embedding reads. It measures computational
efficiency only; it does not evaluate diagnostic accuracy, reward, or sample
efficiency.

The fixed workload contains 100 CAMELYON16 tumor slides, 10,000 balanced
episodes, and 100 accepted navigation actions per episode (1,000,000 accepted
actions per backend). Every episode has 50 pan, 25 zoom-in, and 25 zoom-out
action priorities. The priority sequence is deterministically shuffled using
seed `1 + episode_index`; unavailable classes are postponed. The package runs
each backend once per invocation.

Pinned source revisions:

- WSI-Lab: `0deb7837364add790f9530c41f0ef0b5e616c05c`
- HistoGym: `5b742760321d28a469d2ef9494e886a49d0aa7f7`

The exact Python environments are in `environments/wsi_lab.yml` and
`harness/benchmarks/histogym_comparison/histogym/environment.yml`.

## Prerequisites

- Linux with Git, Conda, and OpenSlide-compatible system support.
- A CUDA-capable device for WSI-Lab preprocessing.
- Sufficient local storage for CAMELYON16, embeddings, and newly generated
  preprocessing artifacts.
- Permission to drop the Linux page cache for the measured cold-cache
  boundaries. The wrapper itself never invokes `sudo`.

Set two shell variables before following the commands below:

```bash
export CAMELYON16_ROOT
export CUDA_DEVICE
unset CUDA_VISIBLE_DEVICES
```

`CAMELYON16_ROOT` must be the local CAMELYON16 directory. `CUDA_DEVICE` must be
an explicit host-visible PyTorch device such as `cuda:0`. Leave
`CUDA_VISIBLE_DEVICES` unset so PyTorch and the GPU resource monitor use the
same index.

The dataset directory must contain:

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

UNI-v2 generation is outside the timed experiment. The HDF5 files are needed
to build fresh WSI-Lab navigation artifacts but are not read during online
interaction.

## 1. Verify and create the environments

Run from this experiment directory:

```bash
bash scripts/verify_package.sh
bash scripts/setup.sh
```

Setup creates ignored state under `runtime/`, checks out WSI-Lab at the pinned
revision, clones HistoGym at its pinned revision, and creates isolated Python
3.11 and Python 3.8 environments. It does not edit the repository's WSI-Lab
training or library files.

## 2. Configure the local inputs

```bash
runtime/wsi-lab-env/bin/python scripts/configure.py \
  --dataset-root "${CAMELYON16_ROOT}" \
  --device "${CUDA_DEVICE}"
```

Configuration creates ignored `work/` inputs. It reconstructs the seed-1
cohort, verifies the authoritative `reference.csv`, checks every required WSI,
XML, and embedding file, and generates the exact 10,000-episode schedule. It
fails if the cohort identity differs.

## 3. Validate before measuring

```bash
bash scripts/validate.sh
```

Validation checks both source revisions, both environments, the 100-slide
cohort, all external inputs, and the 10,000-by-100 action plan without running
an episode.

## 4. Run the two measured blocks

Run WSI-Lab first and HistoGym second. Immediately before each command, drop
the page cache and attest that boundary to the wrapper:

```bash
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
CACHE_DROP_CONFIRMED=1 bash scripts/run_measured.sh wsi_lab

sync
echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
CACHE_DROP_CONFIRMED=1 bash scripts/run_measured.sh histogym
```

WSI-Lab performs fresh AtlasPatch tissue-mask and navigation-artifact
preprocessing before online execution. HistoGym has no separate durable
preprocessing stage. Existing backend output directories are never reused.

## 5. Generate metrics and the cumulative-runtime figure

```bash
bash scripts/analyze.sh
```

The primary publication-ready outputs are:

```text
analysis_outputs/
├── metrics/
│   ├── metrics.csv
│   ├── report.md
│   └── summary.json
└── cumulative_runtime/
    ├── cumulative_runtime.csv
    ├── cumulative_runtime.png
    ├── cumulative_runtime.pdf
    ├── README.md
    └── summary.json
```

The metrics are preprocessing runtime, online runtime, end-to-end runtime,
native-step throughput, and overall-online throughput. The cumulative plot
uses every saved episode boundary and performs no interpolation.

## Timing definitions

- Native step time wraps only the backend's native `env.step()` call.
- Online wall time starts before the first native construction and ends after
  the final close. It includes construction, reset, native observation
  generation, action scheduling, steps, and orchestration.
- End-to-end time adds fresh preprocessing and initial import/setup.
- Cache-drop waiting, result serialization, trace compression, rendering,
  post-hoc analysis, policy/belief inference, and original UNI-v2 generation
  are excluded.
- Process-tree resources are sampled every 0.1 seconds.

## Repeating the experiment

One invocation creates one execution per backend. To make another independent
execution, archive the completed `runs/` and `analysis_outputs/` directories
outside the package, then repeat Sections 4 and 5. Keeping `work/` is safe: it
contains the immutable cohort and episode schedule, while preprocessing lives
inside the newly created WSI-Lab run workspace.

## Sharing and privacy

This package contains no precomputed results. Newly generated `runtime/`,
`work/`, and `runs/` files can contain the reproducer's local paths and raw
result metadata and should not be published unreviewed. The analysis summaries
store source hashes rather than source paths. Run the package-level privacy
check before committing the source package.
