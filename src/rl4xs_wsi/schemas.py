"""Versioned serializable schemas shared by pipeline stages."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


class ManifestStatus(StrEnum):
    INCOMPLETE = "incomplete"
    COMPLETE = "complete"
    FAILED = "failed"


class ArtifactManifest(StrictModel):
    schema_version: int = 1
    artifact_type: str
    fingerprint: str
    status: ManifestStatus
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    package_version: str = "0.1.0"
    git_commit: str | None = None
    dependencies: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ValidationIssue(StrictModel):
    severity: str
    code: str
    message: str
    slide_id: str | None = None
    magnification: float | None = None
    path: str | None = None


class ValidationReport(StrictModel):
    schema_version: int = 1
    fingerprint: str
    complete: bool
    checked_files: int = 0
    feature_dimension: int | None = None
    issues: list[ValidationIssue] = Field(default_factory=list)
    per_file: dict[str, dict[str, Any]] = Field(default_factory=dict)
