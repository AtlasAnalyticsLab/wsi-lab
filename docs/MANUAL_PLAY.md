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
pipeline, including its cached AtlasPatch mask and tissue-valid navigation
graph. It may be omitted after preprocessing is complete.

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

The window shows exactly the spatial observations delivered to the actor. The
overview RGB, composite, location channel, and scaled-visit channel all use the
same snapped tissue-ROI extent. The separately labeled current viewport remains
the actual RGB read for the globally addressed navigation cell. The status
panel lists all five actor scalars in order: magnification, remaining budget,
ABMIL belief, tissue-ROI center x, and tissue-ROI center y. It also shows the
current environment mask; latest action and reward; cumulative reward; episode
step; and termination state. The ground-truth label is not displayed before a
diagnosis.

Manual play does not create a separate visualization coordinate system. Its
ROI overview, location, visits, and normalized x/y are the environment's exact
observation arrays and values; no full-WSI fallback is drawn behind them.

## Keyboard controls

| Key | Action |
|---|---|
| Arrow keys | Pan to the nearest tissue-valid cell up, down, left, or right |
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

An arrow-key pan can jump across background between disconnected tissue
regions. It is masked when no tissue-valid cell exists farther in that exact
direction. A zoom-in key is likewise masked when its aligned child quadrant is
entirely non-tissue. Missing embeddings do not mask either action; a
tissue-valid embeddingless cell remains inspectable as raw RGB.

The controller requires a graphical display and an interactive Matplotlib
backend. On a remote machine, use desktop access or X forwarding rather than a
headless `Agg` backend.
