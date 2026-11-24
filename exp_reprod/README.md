# WSI-Lab experiment reproduction package

This directory contains the runnable materials for two computational-efficiency
comparisons and the CAMELYON16 RL ablation study. It contains no measured
outputs, result tables, plots, historical command manifests, author-machine
paths, or machine/storage inventory.

- `environment_only/` measures native environment execution without policy or
  belief-model inference.
- `agent_integrated/` measures native-observation recurrent PPO, including
  policy inference and optimization and WSI-Lab's frozen belief inference.
- `camelyon16_rl_ablation_2m/` reproduces the six-variant training ablation.

Both efficiency experiments use the same deterministic 100-slide CAMELYON16
cohort, 10,000-episode schedule, and 100 actions/interactions per episode. Each run
script performs one execution per backend in the fixed order WSI-Lab followed
by HistoGym.

## What is and is not included

The package includes benchmark harnesses, protocol templates, cohort
identities, software environments, setup/configuration scripts, fail-fast
validation, measured-run wrappers, and analysis scripts. It lives inside the
WSI-Lab repository:

```text
wsi-lab/
  pyproject.toml
  src/
  exp_reprod/
```

Set `WSI_LAB_SOURCE_ROOT` to the repository's absolute path before running the
efficiency workflows; each experiment README shows the commands. Setup
installs that local source tree directly and checks out HistoGym at its pinned
commit. WSI-Lab Git history is not required. The ablation commands also accept
`--source-root` and retain the study-specific source and runtime compatibility
checks described in their README.

The following licensed or externally managed inputs are not redistributed:

- CAMELYON16 images and XML annotations;
- the four required UNI-v2 HDF5 embedding collections;
- model assets downloaded by AtlasPatch/SAM2.

The frozen calibrated belief checkpoint used by the agent-integrated run is
included and bound by its published SHA-256 identity. CAMELYON16 and UNI-v2
embeddings remain external inputs.

The experiment READMEs describe the required layout and integrity checks.

## Privacy and generated state

The efficiency workflows write runtime state beneath `runtime/`, `work/`,
`runs/`, or `analysis_outputs/` inside each experiment. The ablation workflow
uses a separate, user-selected work directory. These generated directories are
not part of the source package. They may contain local paths or newly
generated results; review them before redistribution.

From the repository root, select the source tree and check the package before
publication:

```bash
export WSI_LAB_SOURCE_ROOT="$(pwd)"
python exp_reprod/scripts/privacy_check.py
```

For the ablation study, follow `camelyon16_rl_ablation_2m/README.md` for its
separate audit and protocol tests.

Start with the README in the experiment you want to run. Do not begin the
multi-hour measured blocks until setup, configuration, and validation succeed.
Absolute timings depend on the reproducer's system, so record the system
description separately from this source package when reporting a new run.
