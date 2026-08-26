# RL4XS WSI

`rl4xs-wsi` is a research package for sequential examination of whole-slide images. It builds a dense six-level navigation hierarchy, attaches sparse tissue-filtered UNI v2 evidence, trains a calibrated gated-attention MIL belief model, and supplies a local recurrent-PPO implementation for `RL4XSWSI-v0`.

The package currently supports CAMELYON16 and its existing AtlasPatch/UNI v2 HDF5 embeddings. PEAN is intentionally unsupported.

## Install

Create the dedicated environment; do not reuse another WSI environment:

```bash
git clone <repository-url>
cd wsi-lab
conda env create --file environment.yml
conda activate wsi-lab
```

See [the installation guide](docs/INSTALLATION.md) for verification, CPU setup, AtlasPatch/SAM2 requirements, and troubleshooting.

## Configure

```bash
export CAMELYON16_ROOT=/path/to/CAMELYON16
rl4xs-wsi validate --config configs/camelyon16.yaml
```

The default split preserves every `test_*` slide as official test and creates a deterministic 216/54 train/validation split over development slides (seed 1, stratified by binary label and center). Test access is guarded; evaluation defaults to validation.

## Commands

```bash
rl4xs-wsi validate --config configs/camelyon16.yaml
rl4xs-wsi preprocess --config configs/camelyon16.yaml
rl4xs-wsi train-belief --config configs/camelyon16.yaml
rl4xs-wsi eval-belief --config configs/camelyon16.yaml --split val
rl4xs-wsi train-rl --config configs/camelyon16.yaml
rl4xs-wsi eval-rl --config configs/camelyon16.yaml --split val
rl4xs-wsi render-trajectory --config configs/camelyon16.yaml --trajectory trajectory.json
rl4xs-wsi benchmark-env --config configs/camelyon16.yaml
rl4xs-wsi smoke-test --config configs/camelyon16.yaml
```

The complete idempotent workflow is:

```bash
rl4xs-wsi run --config configs/camelyon16.yaml
```

Inspect its stage decisions without writing, preprocessing, or training:

```bash
rl4xs-wsi run --config configs/camelyon16.yaml --dry-run
```

Each stage has a content fingerprint and completion manifest. Compatible complete work is skipped, stale artifacts are rebuilt, and compatible incomplete training checkpoints resume.

## Core invariant

Navigation geometry covers the entire WSI rectangle, including background. Embeddings are tissue-filtered attributes on some 2.5×, 5×, 10×, and 20× cells. An embeddingless cell remains navigable and produces RGB, visits, and normal action cost; it simply contributes no diagnostic evidence. Embedding presence is never used as a tissue detector or a navigation-validity test.

The project repository has no license grant. Third-party terms are summarized in [THIRD_PARTY.md](docs/THIRD_PARTY.md).
