# Configuration

Configuration is strict Pydantic v2 YAML. Unknown keys fail before expensive work. `${NAME}` expressions are expanded eagerly; an unset variable is an error. Relative dataset sidecars and embedding roots resolve beneath `dataset.root`; `workspace.root` resolves relative to the YAML file.

`RL4XSWSI-v0` fixes the 11 actions, six navigation levels, four belief scales, three actor branches, and magnification one-hot encoding. Episode length, minimum evidence, image sizes, visit saturation, rewards, caches, training hyperparameters, and devices are configurable with paper-protocol defaults.

Tissue sources:

- `auto`: validate/reuse a compatible configured mask, otherwise run AtlasPatch;
- `existing`: require a compatible configured mask;
- `atlaspatch`: generate a workspace artifact even when a source mask exists.

AtlasPatch 1.1.0's `detect-tissue` command fixes the segmentation checkpoint to
`AtlasAnalyticsLab/AtlasPatch` / `model.pth` and exposes no model, revision, or
checkpoint option. Accordingly, `model_identifier`, `model_revision`, and
`checkpoint` are intentionally not configuration fields; strict validation
rejects them instead of recording an override that AtlasPatch would ignore.
After a successful call, the adapter resolves the downloaded Hugging Face
snapshot and records its SHA256 together with the pinned SAM2 source commit.

`configs/camelyon16.yaml` expects:

```bash
export CAMELYON16_ROOT=/path/to/CAMELYON16
```

Evaluation defaults to `val`. Accessing `test` requires both `--split test` and `--allow-test`. Training and calibration APIs reject test records unconditionally.

Routine `preprocess`, `run`, and PPO-training preparation materialize only the
development cohort and do not inspect official-test embedding files. The
explicit command `eval-rl --split test --allow-test` first requires that
development preprocessing is complete, then adds the explicitly authorized
official-test entries to the sparse deterministic cell-ID table and records the operation in a separate
`test_evaluation_preprocessing.json` manifest. Those test artifacts are not a
training dependency. Each slide receives a fixed non-overlapping int64 block
from its deterministic adapter index, so arbitrary bounded subsets never need
to inspect unrequested embeddings.

The environment normalizes remaining budget by configured `max_episode_steps`; default 500. The final available action is diagnosis-only. A masked action supplied by an external caller is a rejected no-op with `-step_cost` and does not consume budget.
