"""Crash-safe artifact writes using same-directory atomic replacement."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any


def _replace_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def atomic_write_text(path: str | Path, text: str) -> None:
    _replace_bytes(Path(path), text.encode("utf-8"))


def atomic_json_dump(path: str | Path, value: Any) -> None:
    atomic_write_text(Path(path), json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def atomic_torch_save(path: str | Path, value: Any) -> None:
    import torch

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise
