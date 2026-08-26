# Installation

## GPU environment used by this project

The supported setup is a fresh Conda environment named `wsi-lab`. The pre-existing `wsi-nav-env` is unrelated and must not be activated, modified, or used to validate this package.

```bash
cd wsi-lab
conda env create --file environment.yml
conda activate wsi-lab
```

`environment.yml` installs Python 3.11, native OpenSlide 4.x, PyTorch with CUDA 12.1, torchvision, AtlasPatch 1.1.0, SAM2 at commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`, TIFF support including `imagecodecs` for CAMELYON16's LZW masks, this package in editable mode, development tools, TensorBoard, and optional W&B support.

### Verify environment isolation and installed components

Run these checks only after activating the new environment. The first check fails if Python is outside the expected Conda prefix or if `wsi-nav-env` appears in the environment name, prefix, or executable. `EXPECTED_WSI_LAB_ENV` defaults to `wsi-lab`.

```bash
python - <<'PY'
import os
import sys
from pathlib import Path

expected = os.environ.get("EXPECTED_WSI_LAB_ENV", "wsi-lab")
name = os.environ.get("CONDA_DEFAULT_ENV")
prefix_text = os.environ.get("CONDA_PREFIX")
if not prefix_text:
    raise SystemExit("CONDA_PREFIX is unset; activate the dedicated Conda environment first")
prefix = Path(prefix_text).resolve()
executable = Path(sys.executable).resolve()

print(f"CONDA_DEFAULT_ENV={name}")
print(f"CONDA_PREFIX={prefix}")
print(f"python={executable}")
print(f"python_version={sys.version}")

if name != expected:
    raise SystemExit(f"expected environment {expected!r}, found {name!r}")
if "wsi-nav-env" in " ".join((str(name), str(prefix), str(executable))):
    raise SystemExit("the prohibited wsi-nav-env is active or present in the executable path")
try:
    executable.relative_to(prefix)
except ValueError as exc:
    raise SystemExit(f"Python executable {executable} is outside Conda prefix {prefix}") from exc
PY
```

Verify the numerical stack, CUDA runtime, OpenSlide Python binding/native library, and other core libraries. The GPU environment requires CUDA availability; the CPU variant below sets `EXPECT_CUDA=0` before running this same check.

```bash
python - <<'PY'
import os

import gymnasium
import h5py
import numpy
import openslide
import torch
import torchvision

expect_cuda = os.environ.get("EXPECT_CUDA", "1") != "0"
cuda_available = torch.cuda.is_available()

print(f"torch={torch.__version__}")
print(f"torchvision={torchvision.__version__}")
print(f"torch_cuda_build={torch.version.cuda}")
print(f"cuda_available={cuda_available}")
print(f"cudnn={torch.backends.cudnn.version()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
if cuda_available:
    print(f"cuda_device_0={torch.cuda.get_device_name(0)}")
print(f"gymnasium={gymnasium.__version__}")
print(f"numpy={numpy.__version__}")
print(f"h5py={h5py.__version__}")
print(f"openslide_python={openslide.__version__}")
print(f"openslide_native={getattr(openslide, '__library_version__', 'unavailable')}")
print(f"openslide_module={openslide.__file__}")

if cuda_available != expect_cuda:
    raise SystemExit(
        f"CUDA availability mismatch: expected {expect_cuda}, observed {cuda_available}"
    )
PY
```

Resolve the distributions and import locations. This also verifies that pip recorded the required SAM2 revision or resolved commit.

```bash
python - <<'PY'
import importlib.util
import json
from importlib import metadata

sam2_commit = "2b90b9f5ceec907a1c18123530e92e794ad901a4"

for name in ("atlas-patch", "SAM-2", "wsi-lab"):
    distribution = metadata.distribution(name)
    print(
        f"distribution={distribution.metadata['Name']} "
        f"version={distribution.version} location={distribution.locate_file('')}"
    )

for name in ("atlas_patch", "sam2", "wsi_lab"):
    specification = importlib.util.find_spec(name)
    if specification is None:
        raise SystemExit(f"required import is unavailable: {name}")
    locations = list(specification.submodule_search_locations or ())
    print(f"module={name} origin={specification.origin} locations={locations}")

atlas = metadata.distribution("atlas-patch")
if atlas.version != "1.1.0":
    raise SystemExit(f"expected atlas-patch 1.1.0, found {atlas.version}")

sam2 = metadata.distribution("SAM-2")
direct_url_text = sam2.read_text("direct_url.json")
if not direct_url_text:
    raise SystemExit("SAM2 installation has no direct_url.json commit provenance")
direct_url = json.loads(direct_url_text)
vcs = direct_url.get("vcs_info", {})
requested = vcs.get("requested_revision")
resolved = vcs.get("commit_id")
print(f"sam2_requested_revision={requested}")
print(f"sam2_resolved_commit={resolved}")
if sam2_commit not in {requested, resolved}:
    raise SystemExit(
        f"SAM2 is not pinned to {sam2_commit}: requested={requested!r}, resolved={resolved!r}"
    )
PY
```

Finally, verify the AtlasPatch executable resolved from the active prefix and inspect all required command surfaces:

```bash
ATLASPATCH_EXECUTABLE="$(command -v atlaspatch)"
test -n "${ATLASPATCH_EXECUTABLE}"
case "${ATLASPATCH_EXECUTABLE}" in
  "${CONDA_PREFIX}"/*) ;;
  *) echo "atlaspatch resolved outside ${CONDA_PREFIX}: ${ATLASPATCH_EXECUTABLE}" >&2; exit 1 ;;
esac
echo "atlaspatch=${ATLASPATCH_EXECUTABLE}"
atlaspatch --help
atlaspatch info
atlaspatch detect-tissue --help
```

Keep the complete output with the test report. It provides the environment, CUDA, OpenSlide, AtlasPatch, SAM2, and package provenance required to reproduce preprocessing and training.

## CPU-only development variant

The required project environment remains the GPU-enabled `wsi-lab`. For a separate CPU-only development environment, copy `environment.yml` to `environment.cpu.yml` and make all of these changes:

- change `name` to `wsi-lab-cpu`;
- remove the `nvidia` channel;
- remove `pytorch-cuda=12.1`;
- add `cpuonly` to the Conda dependencies from the `pytorch` channel;
- retain Python, Git, OpenSlide, PyTorch, torchvision, pip, AtlasPatch, pinned SAM2, and the editable project installation.

Then create and verify it independently:

```bash
conda env create --file environment.cpu.yml
conda activate wsi-lab-cpu
export EXPECTED_WSI_LAB_ENV=wsi-lab-cpu
export EXPECT_CUDA=0
```

Run the same isolation, library, provenance, and AtlasPatch CLI checks above. For CPU execution, set `belief.device: cpu`, `rl.device: cpu`, `rl.belief_device: cpu`, and `tissue.atlaspatch.device: cpu` in a copied configuration. Never use the CPU environment as evidence that the required CUDA `wsi-lab` setup works.

AtlasPatch tissue segmentation is substantially slower on CPU. Mocked adapter tests and reuse of validated masks do not run SAM2 inference, so use those paths for routine CPU development. A real CPU AtlasPatch integration run should remain explicitly selected and bounded to one temporary slide artifact.

## AtlasPatch and SAM2

Any tissue detection required by this package is performed through:

```text
atlaspatch detect-tissue <slide> --output <temporary-output> --device cuda:0 --seg-batch-size 1
```

AtlasPatch 1.1.0 downloads its fixed `AtlasAnalyticsLab/AtlasPatch` /
`model.pth` checkpoint from Hugging Face's `main` revision; the command does
not expose a custom checkpoint or revision flag. Ensure that model is already
cached or that the host can reach Hugging Face before requesting a missing
mask. Do not add `checkpoint`, `model_identifier`, or `model_revision` beneath
`tissue.atlaspatch`: strict configuration validation rejects those unsupported
overrides.

After a successful call, the adapter selects AtlasPatch's binary
`*_mask_bw.png` rather than its RGB `*_mask.png` visualization. It then resolves
the exact Hugging Face snapshot used, hashes `model.pth`, verifies the installed
SAM2 VCS commit, and records repository, filename, requested revision, resolved
snapshot, checkpoint size/SHA256, and SAM2 requested/resolved commits in the
tissue manifest. The installed `sam2.1_hiera_t.yaml` filename, size, and
SHA256 are recorded as well. Reuse compares the stored path-independent model
identity to the current local Hugging Face `main` snapshot when that checkpoint
is cached; a changed snapshot rebuilds the mask. When WSI MPP metadata is
unavailable, the adapter supplies a generated `wsi,mpp` CSV based on validated
embedding metadata. Output is
validated and normalized in the workspace; source dataset masks are never
overwritten.

There is deliberately no Otsu, HSV, or other fallback. AtlasPatch failures include captured output and a reproduction command.

## Common issues

- `OpenSlideUnsupportedFormatError`: verify both the Conda `openslide` native library and `openslide-python` binding are installed in `wsi-lab`.
- CUDA/torchvision operator errors: recreate the environment so PyTorch, torchvision, and `pytorch-cuda` are solved together.
- AtlasPatch import/command failure: verify `atlas-patch==1.1.0`, the pinned SAM2 commit, and access to `AtlasAnalyticsLab/AtlasPatch/model.pth` with `python -m pip freeze` and the captured reproduction command.
- Missing dataset-root variable: export the variable named by the selected
  example config (`CAMELYON16_ROOT`, `CAMELYON17_ROOT`, `TCGA_NSCLC_ROOT`, or
  `WSI_DATASET_ROOT`) before loading it.
- AtlasPatch licensing: review [THIRD_PARTY.md](THIRD_PARTY.md) before distribution or commercial use.

## Documentation and configuration in distributions

Source distributions contain `README.md`, `environment.yml`, `configs/*.yaml`, and `docs/*.md`. Wheels install the same user-facing assets under the environment data prefix:

```text
<sys.prefix>/share/wsi-lab/
```

Find that location without assuming a platform-specific prefix:

```bash
python - <<'PY'
import sysconfig
from pathlib import Path

print(Path(sysconfig.get_path("data")) / "share" / "wsi-lab")
PY
```

Creating `wsi-lab` still starts from a source checkout because `environment.yml` installs this project with `-e .`. The distributed copies preserve the exact environment and configuration examples for inspection and archival use.
