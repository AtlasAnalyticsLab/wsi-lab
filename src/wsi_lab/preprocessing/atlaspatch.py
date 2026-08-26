"""Strict subprocess boundary for AtlasPatch tissue detection.

AtlasPatch is deliberately treated as an external tool. Keeping its Torch,
SAM2, and native WSI dependencies out of this process makes preprocessing
workers easier to reproduce and prevents dependency conflicts with training.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import json
import shlex
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from wsi_lab.errors import PreprocessingError
from wsi_lab.utils.atomic import atomic_write_text
from wsi_lab.utils.fingerprints import file_fingerprint

_MASK_SUFFIXES = {".tif", ".tiff", ".png", ".npy"}
SUPPORTED_ATLASPATCH_VERSION = "1.1.0"
PINNED_SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"

# AtlasPatch 1.1.0's ``detect-tissue`` command does not expose model or
# checkpoint flags. Its implementation unconditionally calls
# ``hf_hub_download`` with these coordinates and the default (``main``)
# revision. Keep this contract explicit so generated-mask provenance reflects
# what the subprocess actually loaded rather than user-supplied metadata that
# the CLI would silently ignore.
ATLASPATCH_MODEL_REPOSITORY = "AtlasAnalyticsLab/AtlasPatch"
ATLASPATCH_MODEL_FILENAME = "model.pth"
ATLASPATCH_SAM_CONFIG_FILENAME = "sam2.1_hiera_t.yaml"


def installed_atlaspatch_version() -> str | None:
    try:
        return importlib.metadata.version("atlas-patch")
    except importlib.metadata.PackageNotFoundError:
        return None


@lru_cache(maxsize=1)
def installed_sam2_provenance() -> dict[str, Any]:
    """Return and validate the VCS provenance of the pinned SAM2 install."""

    try:
        distribution = importlib.metadata.distribution("SAM-2")
    except importlib.metadata.PackageNotFoundError as exc:
        raise PreprocessingError(
            "SAM-2 is not installed; AtlasPatch provenance is incomplete"
        ) from exc
    direct_url_text = distribution.read_text("direct_url.json")
    if not direct_url_text:
        raise PreprocessingError("SAM-2 installation has no direct_url.json VCS provenance")
    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError as exc:
        raise PreprocessingError("SAM-2 direct_url.json is invalid") from exc
    vcs = direct_url.get("vcs_info", {})
    requested = vcs.get("requested_revision") if isinstance(vcs, Mapping) else None
    resolved = vcs.get("commit_id") if isinstance(vcs, Mapping) else None
    if PINNED_SAM2_COMMIT not in {requested, resolved}:
        raise PreprocessingError(
            "SAM-2 installation does not match the required pinned commit "
            f"{PINNED_SAM2_COMMIT}: requested={requested!r}, resolved={resolved!r}"
        )
    return {
        "distribution": distribution.metadata["Name"],
        "version": distribution.version,
        "pinned_commit": PINNED_SAM2_COMMIT,
        "requested_revision": requested,
        "resolved_commit": resolved,
    }


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash a file without materializing it in memory."""

    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=32)
def _sha256_unchanged_file(path: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    return sha256_file(path)


def _content_identity(path: Path) -> dict[str, Any]:
    stat_identity = file_fingerprint(path)
    return {
        **stat_identity,
        "sha256": _sha256_unchanged_file(
            str(path.resolve()), int(stat_identity["size"]), int(stat_identity["mtime_ns"])
        ),
    }


def resolve_atlaspatch_checkpoint_identity() -> dict[str, Any]:
    """Resolve and hash the checkpoint used by AtlasPatch 1.1.0.

    This is intentionally called only after a successful ``detect-tissue``
    subprocess. At that point AtlasPatch must have placed its fixed
    Hugging Face model in the local cache. No network lookup or independent
    model download is performed here.
    """

    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError as exc:
        raise PreprocessingError(
            "huggingface_hub is unavailable; cannot record AtlasPatch checkpoint provenance"
        ) from exc
    cached = try_to_load_from_cache(
        repo_id=ATLASPATCH_MODEL_REPOSITORY,
        filename=ATLASPATCH_MODEL_FILENAME,
    )
    if not isinstance(cached, str):
        raise PreprocessingError(
            "AtlasPatch succeeded but its fixed Hugging Face checkpoint could not be resolved "
            f"from the local cache: {ATLASPATCH_MODEL_REPOSITORY}/{ATLASPATCH_MODEL_FILENAME}"
        )
    path = Path(cached).expanduser().absolute()
    if not path.is_file():
        raise PreprocessingError(f"resolved AtlasPatch checkpoint does not exist: {path}")
    parts = path.parts
    try:
        snapshot_index = parts.index("snapshots")
        resolved_snapshot = parts[snapshot_index + 1]
    except (ValueError, IndexError) as exc:
        raise PreprocessingError(
            f"AtlasPatch checkpoint path does not expose a resolved Hugging Face snapshot: {path}"
        ) from exc
    if not resolved_snapshot:
        raise PreprocessingError(f"AtlasPatch checkpoint has an empty resolved snapshot: {path}")
    stat = path.stat()
    return {
        "repository": ATLASPATCH_MODEL_REPOSITORY,
        "filename": ATLASPATCH_MODEL_FILENAME,
        "requested_revision": "main",
        "resolved_snapshot": resolved_snapshot,
        "cache_path": str(path),
        "size": stat.st_size,
        "sha256": _sha256_unchanged_file(str(path), stat.st_size, stat.st_mtime_ns),
    }


@lru_cache(maxsize=1)
def resolve_atlaspatch_sam_config_identity() -> dict[str, Any]:
    """Resolve and hash AtlasPatch 1.1.0's actual default SAM2 config."""

    try:
        from atlas_patch.core.config import default_sam2_config_path
    except ImportError as exc:
        raise PreprocessingError(
            "atlas_patch is unavailable; cannot record its SAM2 config provenance"
        ) from exc
    path = Path(default_sam2_config_path()).expanduser().resolve()
    if not path.is_file():
        raise PreprocessingError(f"AtlasPatch default SAM2 config does not exist: {path}")
    if path.name != ATLASPATCH_SAM_CONFIG_FILENAME:
        raise PreprocessingError(
            "AtlasPatch 1.1.0 resolved an unexpected default SAM2 config: "
            f"expected {ATLASPATCH_SAM_CONFIG_FILENAME}, found {path.name}"
        )
    stat = path.stat()
    return {
        "filename": path.name,
        "installed_path": str(path),
        "size": stat.st_size,
        "sha256": _sha256_unchanged_file(str(path), stat.st_size, stat.st_mtime_ns),
    }


def ensure_mpp_override_csv(
    path: str | Path,
    slide_path: str | Path,
    mpp: float,
) -> Path:
    """Persist AtlasPatch's exact ``wsi,mpp`` override without mtime churn."""

    if not np_is_finite_positive(mpp):
        raise ValueError(f"MPP override must be finite and positive, got {mpp}")
    target = Path(path).expanduser().resolve()
    slide = Path(slide_path).expanduser().resolve()
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("wsi", "mpp"))
    writer.writerow((str(slide), format(float(mpp), ".12g")))
    content = stream.getvalue()
    try:
        if target.read_text(encoding="utf-8") == content:
            return target
    except OSError:
        pass
    atomic_write_text(target, content)
    return target


def np_is_finite_positive(value: float) -> bool:
    # Avoid importing NumPy into the lightweight subprocess adapter.
    return value > 0.0 and value not in (float("inf"), float("-inf")) and value == value


@dataclass(frozen=True, slots=True)
class AtlasPatchOptions:
    executable: str = "atlaspatch"
    device: str = "cuda:0"
    segmentation_batch_size: int = 1
    timeout_seconds: int = 7200
    mpp_csv: Path | None = None

    def __post_init__(self) -> None:
        if not self.executable.strip():
            raise ValueError("AtlasPatch executable cannot be empty")
        if self.segmentation_batch_size < 1:
            raise ValueError("AtlasPatch segmentation_batch_size must be positive")
        if self.timeout_seconds < 1:
            raise ValueError("AtlasPatch timeout_seconds must be positive")

    @classmethod
    def from_config(cls, value: object, *, mpp_csv: str | Path | None = None) -> AtlasPatchOptions:
        """Create options from the shared, duck-typed AtlasPatch config."""

        configured_csv = mpp_csv if mpp_csv is not None else getattr(value, "mpp_csv", None)
        return cls(
            executable=str(getattr(value, "executable", "atlaspatch")),
            device=str(getattr(value, "device", "cuda:0")),
            segmentation_batch_size=int(getattr(value, "segmentation_batch_size", 1)),
            timeout_seconds=int(getattr(value, "timeout_seconds", 7200)),
            mpp_csv=Path(configured_csv).expanduser().resolve() if configured_csv else None,
        )

    def cache_identity(self) -> dict[str, Any]:
        """Configuration that changes generated-mask semantics."""

        return {
            "executable": self.executable,
            "atlaspatch_version": installed_atlaspatch_version(),
            "device": self.device,
            "segmentation_batch_size": self.segmentation_batch_size,
            "mpp_csv": _content_identity(self.mpp_csv)
            if self.mpp_csv and self.mpp_csv.is_file()
            else None,
            "segmentation_model": {
                "repository": ATLASPATCH_MODEL_REPOSITORY,
                "filename": ATLASPATCH_MODEL_FILENAME,
                "requested_revision": "main",
                "sam2_commit": PINNED_SAM2_COMMIT,
                "sam2_config_filename": ATLASPATCH_SAM_CONFIG_FILENAME,
            },
        }


@dataclass(frozen=True, slots=True)
class AtlasPatchResult:
    mask_path: Path
    command: tuple[str, ...]
    reproduction_command: str
    returncode: int
    stdout: str
    stderr: str
    atlaspatch_version: str | None
    provenance: Mapping[str, Any]


class AtlasPatchRunner:
    """Run ``atlaspatch detect-tissue`` using an argv list and no shell."""

    def __init__(self, options: AtlasPatchOptions | None = None) -> None:
        self.options = options or AtlasPatchOptions()

    def command(
        self,
        slide_path: str | Path,
        output_dir: str | Path,
    ) -> tuple[str, ...]:
        slide = Path(slide_path).expanduser().resolve()
        output = Path(output_dir).expanduser().resolve()
        command = [
            self.options.executable,
            "detect-tissue",
            str(slide),
            "--output",
            str(output),
            "--device",
            self.options.device,
            "--seg-batch-size",
            str(self.options.segmentation_batch_size),
        ]
        if self.options.mpp_csv is not None:
            command.extend(("--mpp-csv", str(self.options.mpp_csv)))
        return tuple(command)

    def run(
        self,
        slide_path: str | Path,
        output_dir: str | Path,
        *,
        mpp: tuple[float | None, float | None] | None = None,
    ) -> AtlasPatchResult:
        slide = Path(slide_path).expanduser().resolve()
        output = Path(output_dir).expanduser().resolve()
        if not slide.is_file():
            raise FileNotFoundError(slide)
        if self.options.mpp_csv is not None and not self.options.mpp_csv.is_file():
            raise FileNotFoundError(f"AtlasPatch MPP CSV does not exist: {self.options.mpp_csv}")
        executable = self.options.executable
        if Path(executable).parent != Path("."):
            if not Path(executable).expanduser().is_file():
                raise PreprocessingError(f"AtlasPatch executable does not exist: {executable}")
        elif shutil.which(executable) is None:
            raise PreprocessingError(
                f"AtlasPatch executable {executable!r} is not on PATH; tissue generation cannot continue"
            )
        output.mkdir(parents=True, exist_ok=True)
        command = self.command(slide, output)
        try:
            completed = subprocess.run(
                list(command),
                check=False,
                capture_output=True,
                text=True,
                timeout=self.options.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise PreprocessingError(
                f"AtlasPatch timed out after {self.options.timeout_seconds}s: {shlex.join(command)}"
            ) from exc
        except OSError as exc:
            raise PreprocessingError(
                f"failed to start AtlasPatch: {type(exc).__name__}: {exc}"
            ) from exc
        if completed.returncode != 0:
            raise PreprocessingError(
                "AtlasPatch tissue detection failed "
                f"(exit={completed.returncode})\ncommand: {shlex.join(command)}\n"
                f"stdout:\n{completed.stdout[-65536:]}\nstderr:\n{completed.stderr[-65536:]}"
            )
        mask_path = self._discover_mask(output)
        version = self._version()
        if version != SUPPORTED_ATLASPATCH_VERSION:
            raise PreprocessingError(
                "AtlasPatch checkpoint provenance is defined only for the required "
                f"version {SUPPORTED_ATLASPATCH_VERSION}, found {version!r}"
            )
        checkpoint = resolve_atlaspatch_checkpoint_identity()
        sam2 = installed_sam2_provenance()
        sam2_config = resolve_atlaspatch_sam_config_identity()
        segmentation_model = {
            "implementation": "sam2",
            "repository": ATLASPATCH_MODEL_REPOSITORY,
            "filename": ATLASPATCH_MODEL_FILENAME,
            "requested_revision": "main",
            "resolved_snapshot": checkpoint["resolved_snapshot"],
            "checkpoint": checkpoint,
            "sam2": sam2,
            "sam2_config": sam2_config,
        }
        provenance: dict[str, Any] = {
            "runner": "argv_subprocess",
            "command": list(command),
            "reproduction_command": shlex.join(command),
            "returncode": completed.returncode,
            "stdout": completed.stdout[-65536:],
            "stderr": completed.stderr[-65536:],
            "atlaspatch_version": version,
            "atlaspatch_options": asdict(self.options),
            "segmentation_model": segmentation_model,
            "mpp": list(mpp) if mpp is not None else None,
            "raw_mask": {
                **file_fingerprint(mask_path),
                "sha256": sha256_file(mask_path),
            },
        }
        return AtlasPatchResult(
            mask_path=mask_path,
            command=command,
            reproduction_command=shlex.join(command),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            atlaspatch_version=version,
            provenance=provenance,
        )

    def _version(self) -> str | None:
        installed = installed_atlaspatch_version()
        if installed is not None:
            return installed
        try:
            completed = subprocess.run(
                [self.options.executable, "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        text = (completed.stdout or completed.stderr).strip()
        return text[:1024] or None

    @staticmethod
    def _discover_mask(output_dir: Path) -> Path:
        candidates = sorted(
            (
                path
                for path in output_dir.rglob("*")
                if path.is_file()
                and path.suffix.casefold() in _MASK_SUFFIXES
                and "overlay" not in path.name.casefold()
            ),
            key=lambda path: (
                "tissue" not in path.name.casefold(),
                "mask" not in path.name.casefold(),
                str(path),
            ),
        )
        if not candidates:
            raise PreprocessingError(
                f"AtlasPatch returned success but produced no supported mask under {output_dir}"
            )
        # AtlasPatch 1.1.0 emits both ``*_mask_bw.png`` (the binary tissue
        # mask) and ``*_mask.png`` (an RGB visualization). Prefer the
        # explicitly binary artifact before applying generic name scoring.
        binary_candidates = [
            path
            for path in candidates
            if path.stem.casefold().endswith("_mask_bw")
            or path.stem.casefold().endswith("_tissue_bw")
        ]
        if len(binary_candidates) == 1:
            return binary_candidates[0]
        if len(binary_candidates) > 1:
            raise PreprocessingError(
                "AtlasPatch produced multiple explicit binary tissue masks: "
                + ", ".join(str(path) for path in binary_candidates[:8])
            )
        best_score = (
            "tissue" not in candidates[0].name.casefold(),
            "mask" not in candidates[0].name.casefold(),
        )
        tied = [
            path
            for path in candidates
            if (
                "tissue" not in path.name.casefold(),
                "mask" not in path.name.casefold(),
            )
            == best_score
        ]
        if len(tied) > 1:
            raise PreprocessingError(
                "AtlasPatch produced multiple equally plausible tissue masks: "
                + ", ".join(str(path) for path in tied[:8])
            )
        return candidates[0]


__all__ = [
    "AtlasPatchOptions",
    "AtlasPatchResult",
    "AtlasPatchRunner",
    "ensure_mpp_override_csv",
    "installed_atlaspatch_version",
    "installed_sam2_provenance",
    "resolve_atlaspatch_checkpoint_identity",
    "resolve_atlaspatch_sam_config_identity",
    "sha256_file",
]
