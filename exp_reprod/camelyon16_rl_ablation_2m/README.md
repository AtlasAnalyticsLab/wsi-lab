# Reproducing the six-variant CAMELYON16 RL ablation

This directory is a portable, training-only reproduction package for the
six-variant CAMELYON16 study. It runs from an extracted full-source submission
ZIP on a workstation or server; neither Git history nor access to the WSI-Lab
repository is required. The default launcher executes one process at a time;
optional bounded parallelism is available with `--jobs`.

The committed files contain experiment definitions and implementation code
only. They contain no trained weights, measured metric values, plots, telemetry, run maps,
prior launch records, tracking run IDs, account details, machine names, user names, or
machine-specific paths. Everything produced by these scripts is written under
a user-selected work directory outside this package.

## What this package reproduces

The main matrix contains six variants, two start policies, and three PPO seeds:

```text
6 variants x 2 starts x 3 seeds = 36 runs
starts: center, random
seeds:  1, 2, 3
```

| ID | Policy/environment contract |
|---|---|
| `v1_full` | Global view, viewport, all scalars including belief, tissue-masked navigation, and diagnosis actions |
| `v2_viewport_only` | Viewport is the only policy input; belief still operates internally; tissue masking and diagnosis actions remain |
| `v3_no_tissue_mask` | Full observation, geometric navigation without tissue masking, and RGB-white evidence suppression |
| `v4_no_diagnosis` | Full observation and tissue masking, nine navigation actions, and belief classification at the 500-action horizon |
| `v5_no_belief` | Global view, viewport, non-belief scalars, tissue masking, and diagnosis actions; no belief checkpoint or embedding reads |
| `v6_combined` | Viewport only, geometric navigation, nine navigation actions, no belief, and no horizon classifier |

Each main run requests 2,000,000 environment transitions. PPO collects batches
of `8 x 128 = 1,024` transitions, so a completed main run ends at the first
batch boundary at or above the request. Checkpoints are written every 100,000
requested transitions. The committed `study.json` is the authoritative matrix
and protocol definition; configurations are generated only when the reviewer
runs the launcher.

This protocol intentionally does not perform PPO validation or official-test
evaluation. Training telemetry and deterministic monitoring use the training
split only. The shared calibrated belief prerequisite still uses the normal
development train/validation procedure. Do not use the repository's ordinary
`eval-rl` command for these custom checkpoints: it constructs the standard
policy class rather than the ablation-specific policy classes.

## Data prerequisites

Obtain CAMELYON16 according to its distributor's terms. Keep its native layout
and provide canonical UNI v2 HDF5 embeddings at all four magnifications:

```text
$CAMELYON16_ROOT/
├── images/
├── evaluation/reference.csv
├── Embeddings/
│   ├── UNI_v2_2.5x_256/patches/
│   ├── UNI_v2_5x_256/patches/
│   ├── UNI_v2_10x_256/patches/
│   └── UNI_v2_20x_256/patches/
├── masks/
└── annotations/
```

The HDF5 files must satisfy the schema documented in
[DATA_FORMAT.md](../../docs/DATA_FORMAT.md), including the `coords`,
`passports`, and `features/uni_v2` datasets and their geometry metadata.
WSI-Lab consumes these embeddings but does not generate them. Consequently,
the embeddings are an external data prerequisite rather than an omitted step
in this launcher. Use the same canonical AtlasPatch/UNI v2 HDF5 export as the
recorded study when reproducing numerical values. The launcher validates every
development-slide file and the HDF5 schema/geometry, but this package cannot
establish numerical equivalence for embeddings produced by a different
external extraction pipeline.

The launcher also requires the canonical 399-row reference inventory: 270
development slides and 129 official-test records (the duplicate `test_049` is
absent, and the corrected `tumor_111` replaces `normal_086`). It validates the
development labels and center assignments because they determine the fixed
216/54 development split. Training remains development-only; official-test
WSIs and embeddings are neither required nor opened.

Fresh tissue masks are built with AtlasPatch and stored in the work directory;
a source `background_tissue/` directory is not required. The package downloads
the immutable AtlasPatch snapshot recorded in `study.json` into a cache beneath
`WORK_ROOT`, verifies its size and SHA-256 before use, and also verifies the
pinned SAM2 commit and configuration. It never relies on AtlasPatch's mutable
default model revision. CAMELYON16 tumor-mask and XML annotation sidecars are
reporting-only: they never enter observations,
action masks, rewards, or policy optimization. They are required here to
recreate the deterministic training-monitor cache and its balanced 30-slide
cohort.

## 1. Install the software

From the extracted archive root (the directory containing `pyproject.toml`),
create and activate the reproduction environment:

```bash
conda env create \
  --file exp_reprod/camelyon16_rl_ablation_2m/environment.yml
conda activate wsi-lab-ablation
```

For a CPU-only environment, use `environment.cpu.yml` and activate
`wsi-lab-ablation-cpu` instead. Both recipes target an ordinary Linux x86-64
workstation or server and use packages from the listed public Conda channels
and PyPI. The recorded protocol uses CUDA devices. A CUDA-capable GPU is
strongly recommended for the full matrix. CPU execution is supported for
portability and small checks but preprocessing and 36 full training runs will
be very slow. See [INSTALLATION.md](../../docs/INSTALLATION.md) for OpenSlide,
AtlasPatch, CUDA, and CPU-environment details.

## 2. Select data, work, and source directories

Run the following from the extracted archive root after setting
`CAMELYON16_ROOT`:

```bash
DATA_ROOT="$CAMELYON16_ROOT"
WORK_ROOT="$PWD/workspace/exp_reprod/camelyon16_rl_ablation_2m"
SOURCE_ROOT="$PWD"
```

Install the selected source tree without re-solving the pinned dependencies:

```bash
python -m pip install --no-deps -e "$SOURCE_ROOT"
```

The default `WORK_ROOT` is separated from the publication package and covered
by the archive's generated-workspace ignore rule. Do not select a work
directory inside this reproduction package; the launcher rejects that choice.

The recorded core implementation is identified in `study.json` by a reference
source identifier and a path-independent source-tree checksum. The submitted
ZIP includes those exact `src/wsi_lab` bytes, so `SOURCE_ROOT="$PWD"` is the
archive-compatible setting. `doctor`, `prepare`, and `run` verify the file
contents directly and never invoke Git or contact the WSI-Lab repository. If
that checksum fails, re-extract the unmodified submission ZIP; combining this
folder with a different source archive is intentionally rejected.

The environment contains the Git command only because pip uses it to fetch the
public, commit-pinned SAM2 dependency. It is not used to fetch WSI-Lab.

The environment recipes pin a public, portable reproduction stack, and
`doctor` verifies its computation-relevant Python distribution versions
against `software_reproduction` in `study.json`. The same file separately
retains the result-free software version snapshot of the original run under
`software_reference`. Some original versions were site-built packages without
portable public wheels, so they are provenance rather than installation
instructions. The reproduction stack stays within the source package's
declared dependency ranges and keeps PyTorch, Torchvision, CUDA, AtlasPatch,
and SAM2 at the study versions. Platform libraries, GPU drivers, and hardware
can still differ, so this package targets protocol and statistical
reproduction; it does not promise bitwise-identical floating-point output
across machines.

## 3. Inspect and check the study

Print the complete, deterministic 36-run matrix:

```bash
python exp_reprod/camelyon16_rl_ablation_2m/reproduce.py list
```

Check the source tree, pinned software environment and AtlasPatch wrapper,
native dataset layout, OpenSlide, PyTorch, package imports, canonical reference
inventory, all 270 development
WSIs, all four embeddings per development slide, and the required tumor
sidecars without creating experiment artifacts:

```bash
python exp_reprod/camelyon16_rl_ablation_2m/reproduce.py doctor \
  --source-root "$SOURCE_ROOT" \
  --data-root "$DATA_ROOT" \
  --work-root "$WORK_ROOT"
```

Preview every preparation command without opening data or writing files:

```bash
python exp_reprod/camelyon16_rl_ablation_2m/reproduce.py prepare \
  --source-root "$SOURCE_ROOT" \
  --data-root "$DATA_ROOT" \
  --work-root "$WORK_ROOT" \
  --dry-run
```

The `smoke` and `run` commands also accept `--dry-run`. Dry runs do not require
the dataset to exist and never create the selected work directory.

## 4. Build shared prerequisites once

On a CUDA machine:

```bash
python exp_reprod/camelyon16_rl_ablation_2m/reproduce.py prepare \
  --source-root "$SOURCE_ROOT" \
  --data-root "$DATA_ROOT" \
  --work-root "$WORK_ROOT"
```

For CPU-only execution, make every prerequisite device explicit:

```bash
python exp_reprod/camelyon16_rl_ablation_2m/reproduce.py prepare \
  --source-root "$SOURCE_ROOT" \
  --data-root "$DATA_ROOT" \
  --work-root "$WORK_ROOT" \
  --device cpu \
  --belief-inference-device cpu \
  --tissue-device cpu
```

Preparation is idempotent and resumable. It performs, in order:

1. Download and checksum verification of the pinned AtlasPatch tissue model.
2. Development split creation and embedding validation.
3. Fresh AtlasPatch tissue masking and WSI preprocessing.
4. Belief-model training and calibration.
5. Construction and verification of the train-only analysis/monitor cache.

It deliberately does not call the complete core pipeline because that would
also launch the ordinary PPO configuration and its evaluation workflow.

## 5. Run the six-cell smoke study

The smoke study exercises each variant once for 1,024 transitions while
retaining the real 500-action episode horizon:

```bash
python exp_reprod/camelyon16_rl_ablation_2m/reproduce.py smoke \
  --source-root "$SOURCE_ROOT" \
  --data-root "$DATA_ROOT" \
  --work-root "$WORK_ROOT" \
  --jobs 1
```

If PPO itself must run on CPU, add `--device cpu --belief-inference-device cpu`.
Run configurations automatically inherit the
tissue and belief-training settings saved by `prepare`, preventing accidental
upstream fingerprint changes.

## 6. Run the full 36-cell study

```bash
python exp_reprod/camelyon16_rl_ablation_2m/reproduce.py run \
  --source-root "$SOURCE_ROOT" \
  --data-root "$DATA_ROOT" \
  --work-root "$WORK_ROOT" \
  --jobs 1
```

`--jobs 1` is the safe default because each run already uses eight synchronous
environments and a full PPO model. Increase it only when the server has enough
independent compute and memory. Every run has its own fingerprinted output
directory, so bounded parallel execution is otherwise safe.

Runs can be split across invocations with filters. They may also run on servers
that share the same absolute work directory. Generated manifests intentionally
bind artifact paths, so independently rooted outputs are not relocatable or
mergeable by `summarize.py`. Canonical matrix indices and run IDs do not change:

```bash
python exp_reprod/camelyon16_rl_ablation_2m/reproduce.py run \
  --source-root "$SOURCE_ROOT" \
  --data-root "$DATA_ROOT" \
  --work-root "$WORK_ROOT" \
  --variants v1_full v2_viewport_only \
  --starts center \
  --seeds 1 2 3
```

Re-running the same selection is safe. A completed run becomes a no-op only
after its final checkpoint, telemetry, and monitoring artifacts pass integrity
verification. An interrupted run exits nonzero, remains explicitly marked
`interrupted`, and resumes from its latest compatible checkpoint on the next
invocation.

## 7. Inspect and summarize newly generated outputs

Inspect completion state without modifying it:

```bash
python exp_reprod/camelyon16_rl_ablation_2m/reproduce.py status \
  --work-root "$WORK_ROOT"
```

After all 36 main runs are complete, verify every final checkpoint, telemetry
stream, monitoring milestone, and matrix identity, then aggregate episode
telemetry:

```bash
python exp_reprod/camelyon16_rl_ablation_2m/summarize.py \
  --source-root "$SOURCE_ROOT" \
  --work-root "$WORK_ROOT" \
  --mode main
```

The generated summary is written beneath `$WORK_ROOT/summaries/`. No expected
metric values are embedded in this repository; the summary contains only
measurements produced by the reviewer's run.

The work directory has this shape:

```text
WORK_ROOT/
├── generated_configs/
├── analysis_cache/
├── summaries/
└── shared_workspace/
    ├── splits/
    ├── tissue/
    ├── processed/
    ├── models/belief/
    └── rl_ablation/<run-fingerprint>/
        ├── checkpoints/
        ├── telemetry/
        ├── monitoring/
        ├── tensorboard/
        ├── training_progress.json
        └── run_manifest.json
```

All of these are generated locally and remain outside the committed package.
W&B is disabled in the recorded configuration.

## Publication and implementation checks

The package includes a read-only audit that rejects private machine paths,
credentials, generated artifacts, prior-run metadata, symlinks, and unexpected
binary or large files. When Git metadata is present during authoring it also
rejects ignored publication files; that check is safely skipped in an
extracted source ZIP:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python -B exp_reprod/camelyon16_rl_ablation_2m/audit.py
```

After installing the project dependencies, run the protocol and variant
behavior tests without creating test caches in this package:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python -B -m pytest -q -p no:cacheprovider \
  exp_reprod/camelyon16_rl_ablation_2m/tests
```

These tests cover matrix ordering, dry-run immutability, source provenance,
fixed observation/action contracts, disabled observation branches, belief-free
execution, no-mask geometry, deterministic random starts, and horizon behavior.

## Important interpretation notes

- The ablation runner trains on `train` only. It never requests official-test
  artifacts and never schedules validation or final-test evaluation.
- Tumor masks and annotations are used only to label generated monitoring
  views. The policy cannot observe them.
- `v4_no_diagnosis` uses the calibrated belief at the horizon;
  `v6_combined` has no classifier at the horizon.
- `v5_no_belief` and `v6_combined` neither load the belief checkpoint nor read
  belief embeddings during their episodes.
- Generated configurations preserve the preparation-time tissue and belief
  settings. PPO device selection may change without rebuilding prerequisites.
- This directory is independent of any other experiment package that may later
  appear under `exp_reprod/`; it does not require a shared index or root-level
  helper.
