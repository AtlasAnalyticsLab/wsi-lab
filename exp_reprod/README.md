# WSI-Lab / HistoGym efficiency reproduction package

This directory contains the runnable materials for two computational-efficiency
comparisons. It intentionally contains no measured outputs, result tables,
plots, historical command manifests, author-machine paths, or machine/storage
inventory.

- `environment_only/` measures native environment execution without policy or
  belief-model inference.
- `agent_integrated/` measures native-observation recurrent PPO, including
  policy inference and optimization and WSI-Lab's frozen belief inference.

Both experiments use the same deterministic 100-slide CAMELYON16 cohort,
10,000-episode schedule, and 100 actions/interactions per episode. Each run
script performs one execution per backend in the fixed order WSI-Lab followed
by HistoGym.

## What is and is not included

The package includes the benchmark harness, protocol templates, cohort
identity, pinned source revisions, software environments, setup/configuration
scripts, fail-fast validation, measured-run wrappers, and analysis scripts.
Keep this directory inside a Git clone of WSI-Lab: setup creates a detached
worktree at the pinned source revision without changing the active checkout.

The following licensed or externally managed inputs are not redistributed:

- CAMELYON16 images and XML annotations;
- the four required UNI-v2 HDF5 embedding collections;
- model assets downloaded by AtlasPatch/SAM2;
- the frozen calibrated belief checkpoint used by the agent-integrated run.

The experiment READMEs describe the required layout and integrity checks.

## Privacy and generated state

All runtime state is written beneath `runtime/`, `work/`, `runs/`, or
`analysis_outputs/` inside an experiment. These directories are ignored by
Git. They are created only on the reproducer's machine and may contain that
machine's paths or newly generated results. Do not force-add or redistribute
raw generated directories without reviewing them.

The packaged tree can be checked before publication with:

```bash
python scripts/privacy_check.py
bash verify_all.sh
```

Start with the README in the experiment you want to run. Do not begin the
multi-hour measured blocks until setup, configuration, and validation succeed.
Absolute timings depend on the reproducer's system, so record the system
description separately from this source package when reporting a new run.
