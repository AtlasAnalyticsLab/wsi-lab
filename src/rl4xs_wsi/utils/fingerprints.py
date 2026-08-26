"""Stable content and lightweight file fingerprints."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def _canonical(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _canonical(value.model_dump(mode="json"))
    if dataclasses.is_dataclass(value):
        return _canonical(dataclasses.asdict(value))  # type: ignore[arg-type]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda x: str(x[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, set):
        return sorted(_canonical(item) for item in value)
    return value


def fingerprint(value: Any) -> str:
    payload = json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_fingerprint(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    stat = source.stat()
    return {"path": str(source.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
