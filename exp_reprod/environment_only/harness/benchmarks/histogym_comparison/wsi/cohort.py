"""Strict reader for the shared, fixed CAMELYON16 cohort manifest."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

CohortName = Literal["measured", "pilot"]


@dataclass(frozen=True, slots=True)
class SlideSpec:
    ordinal: int
    slide_id: str
    wsi_path: Path
    xml_path: Path
    label: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["wsi_path"] = str(self.wsi_path)
        value["xml_path"] = str(self.xml_path)
        return value


@dataclass(frozen=True, slots=True)
class CohortSelection:
    manifest_path: Path
    manifest_sha256: str
    cohort: CohortName
    slides: tuple[SlideSpec, ...]
    selection_sha256: str
    limited: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "cohort": self.cohort,
            "selection_sha256": self.selection_sha256,
            "limited": self.limited,
            "slide_count": len(self.slides),
            "slide_ids": [slide.slide_id for slide in self.slides],
        }


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _resolve_manifest_path(value: object, manifest_dir: Path, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"manifest field {field!r} must be a non-empty path string")
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else manifest_dir / candidate).resolve()


def load_cohort(
    path: str | Path,
    *,
    cohort: CohortName = "measured",
    limit: int | None = None,
) -> CohortSelection:
    """Load one ordered cohort and fail closed on malformed or drifting rows."""

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid cohort manifest {manifest_path}: {exc}") from exc
    if not isinstance(raw, dict) or int(raw.get("schema_version", -1)) != 1:
        raise ValueError(f"unsupported cohort manifest schema: {manifest_path}")
    key = "slides" if cohort == "measured" else "pilot_slides"
    rows = raw.get(key)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"manifest {key!r} must be a non-empty list")
    if cohort == "measured" and len(rows) != 100:
        raise ValueError(f"measured manifest must contain exactly 100 slides, found {len(rows)}")
    if limit is not None:
        if cohort == "measured":
            raise ValueError("--limit is forbidden for the measured 100-slide cohort")
        if limit < 1 or limit > len(rows):
            raise ValueError(f"limit must be in [1,{len(rows)}], got {limit}")
        rows = rows[:limit]

    slides: list[SlideSpec] = []
    for expected_ordinal, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"manifest row {expected_ordinal} is not an object")
        ordinal = int(row.get("ordinal", -1))
        if ordinal != expected_ordinal:
            raise ValueError(
                f"manifest ordinals must be contiguous in run order; "
                f"row {expected_ordinal} has ordinal {ordinal}"
            )
        slide_id = str(row.get("slide_id", ""))
        if not slide_id or Path(slide_id).name != slide_id:
            raise ValueError(f"unsafe slide_id at ordinal {ordinal}: {slide_id!r}")
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"metadata for {slide_id} must be an object")
        slides.append(
            SlideSpec(
                ordinal=ordinal,
                slide_id=slide_id,
                wsi_path=_resolve_manifest_path(
                    row.get("wsi_path"), manifest_path.parent, f"{slide_id}.wsi_path"
                ),
                xml_path=_resolve_manifest_path(
                    row.get("xml_path"), manifest_path.parent, f"{slide_id}.xml_path"
                ),
                label=int(row.get("label", -1)),
                metadata=dict(metadata),
            )
        )
    ids = [slide.slide_id for slide in slides]
    if len(ids) != len(set(ids)):
        raise ValueError("cohort manifest contains duplicate slide IDs")
    normalized = [slide.to_dict() for slide in slides]
    return CohortSelection(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        cohort=cohort,
        slides=tuple(slides),
        selection_sha256=hashlib.sha256(_canonical_bytes(normalized)).hexdigest(),
        limited=limit is not None,
    )


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def atomic_json_dump(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                value,
                stream,
                default=json_default,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def append_jsonl(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                value,
                default=json_default,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        stream.write("\n")
        stream.flush()


__all__ = [
    "CohortName",
    "CohortSelection",
    "SlideSpec",
    "append_jsonl",
    "atomic_json_dump",
    "load_cohort",
    "sha256_file",
]
