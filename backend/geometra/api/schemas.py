"""API schemas using Pydantic models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ConversionDirection(str, Enum):
    TWO_D_TO_THREE_D = "2d_to_3d"
    THREE_D_TO_TWO_D = "3d_to_2d"


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ── Requests ──────────────────────────────────────────────────────────────────


class ConversionRequest(BaseModel):
    """Request to start a new conversion job."""
    direction: ConversionDirection
    file_path: str
    options: dict[str, Any] = Field(default_factory=dict)


# ── Responses ─────────────────────────────────────────────────────────────────


class JobResponse(BaseModel):
    """Response returned when a conversion job is created."""
    job_id: str
    status: JobStatus
    direction: ConversionDirection
    created_at: datetime


class JobStatusResponse(BaseModel):
    """Status of an existing conversion job."""
    job_id: str
    status: JobStatus
    direction: ConversionDirection
    progress: float = 0.0
    message: str = ""
    result_paths: list[str] = Field(default_factory=list)
    validation_report: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime
    error: str | None = None


class HealthResponse(BaseModel):
    """Server health check."""
    status: str = "ok"
    version: str = "0.1.0"


class FileUploadResponse(BaseModel):
    """Response after a file is uploaded."""
    filename: str
    file_path: str
    file_size_bytes: int
    detected_format: str | None = None
    detected_direction: ConversionDirection | None = None
