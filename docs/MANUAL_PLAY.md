# Manual controller

The manual controller is an interactive debugging dashboard over the production
Gymnasium environment. It loads the same processed slide source and calibrated
ABMIL accumulator as RL evaluation. Keyboard actions are submitted to
`WSIEnv.step()` only when the current `info["action_mask"]` marks them valid;
the utility does not implement navigation, zoom, evidence, reward, budget, or
diagnosis rules itself.

Activate the project environment, configure CAMELYON16, and launch validation
manual play:

```bash
conda activate wsi-lab
export CAMELYON16_ROOT=/path/to/CAMELYON16
wsi-lab preprocess --config configs/camelyon16.yaml
wsi-lab manual-play --config configs/camelyon16.yaml --split val
```

`manual-play` requires completed dense navigation artifacts and deliberately
does not create them. The one-time `preprocess` command above uses the normal
pipeline, including its compatible-mask reuse and AtlasPatch-only tissue
policy. It may be omitted after preprocessing is complete.

Choose an initial slide with `--slide-id` and optionally a reset seed:

```bash
wsi-lab manual-play \
  --config configs/camelyon16.yaml \
  --split val \
  --slide-id tumor_017 \
  --seed 1
```

Official test access remains guarded and uses the same explicit authorized
preprocessing path as RL evaluation:

```bash
wsi-lab manual-play \
  --config configs/camelyon16.yaml \
  --split test \
  --allow-test \
  --slide-id test_001
```

The window shows the current viewport; overview RGB; a composite overview;
the exact location and scaled-visit channels; every actor scalar; the current
environment mask; latest action and reward; cumulative reward; episode step;
and termination state. The ground-truth label is not displayed before a
diagnosis.

## Keyboard controls

| Key | Action |
|---|---|
| Arrow keys | Pan up, down, left, or right |
| `7`, `9`, `1`, `3` | Zoom into top-left, top-right, bottom-left, or bottom-right |
| `-` or Backspace | Zoom out |
| `c` | Diagnose cancerous |
| `b` | Diagnose benign |
| `r` | Reset the current slide |
| `n` or `]` | Select the next slide |
| `[` | Select the previous slide |
| Escape | Close the controller |

Invalid actions are shown but blocked before `step()` is called. Diagnosis is
therefore unavailable until the environment permits it, and the final-budget
diagnosis-only constraint is enforced unchanged.

The controller requires a graphical display and an interactive Matplotlib
backend. On a remote machine, use desktop access or X forwarding rather than a
headless `Agg` backend.
