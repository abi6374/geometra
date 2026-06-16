"""API routes for Geometra."""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from geometra.api.schemas import (
    ConversionDirection,
    ConversionRequest,
    FileUploadResponse,
    HealthResponse,
    JobResponse,
    JobStatus,
    JobStatusResponse,
)
from geometra.config import settings
from geometra.core.orchestrator import ConversionOrchestrator

router = APIRouter()

# Track uploaded files for cleanup and validation
_uploaded_files: dict[str, dict[str, Any]] = {}


def _cleanup_old_uploads(max_age_seconds: int = 3600) -> None:
    """Remove uploaded files older than max_age_seconds."""
    now = time.time()
    stale = []
    for job_id, info in _uploaded_files.items():
        if now - info.get("timestamp", 0) > max_age_seconds:
            path = Path(info["path"])
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass
            stale.append(job_id)
    for job_id in stale:
        _uploaded_files.pop(job_id, None)
    if stale:
        import logging
        logging.getLogger(__name__).debug("Cleaned up %d stale uploaded files", len(stale))


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse()


@router.post("/upload", response_model=FileUploadResponse, tags=["conversion"])
async def upload_file(file: UploadFile = File(...)) -> FileUploadResponse:
    """Upload a drawing or CAD file for processing."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Determine file extension
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    allowed = {*settings.supported_2d_formats, *settings.supported_3d_formats}
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Supported: {', '.join(sorted(allowed))}",
        )

    # Check file size
    contents = await file.read()
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds {settings.max_file_size_mb} MB limit",
        )

    # Save file
    job_id = str(uuid.uuid4())
    safe_name = f"{job_id}_{file.filename}"
    dest = settings.upload_dir / safe_name
    dest.write_bytes(contents)

    # Detect direction
    direction = (
        ConversionDirection.THREE_D_TO_TWO_D
        if ext in settings.supported_3d_formats
        else ConversionDirection.TWO_D_TO_THREE_D
    )

    # Track for validation and cleanup
    _uploaded_files[job_id] = {
        "path": str(dest),
        "timestamp": time.time(),
        "filename": file.filename,
    }

    return FileUploadResponse(
        filename=file.filename,
        file_path=str(dest),
        file_size_bytes=len(contents),
        detected_format=ext,
        detected_direction=direction,
    )


@router.post("/convert", response_model=JobResponse, tags=["conversion"])
async def start_conversion(req: ConversionRequest) -> JobResponse:
    """Start a conversion job."""
    # Validate file path: must exist and belong to a previous upload
    file_path = Path(req.file_path)
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"File not found: {req.file_path}. Upload the file first via /upload.",
        )
    # Optional: verify the file was uploaded through our system
    is_known_upload = any(
        info["path"] == req.file_path
        for info in _uploaded_files.values()
    )
    if not is_known_upload and req.direction != ConversionDirection.TWO_D_TO_THREE_D:
        # Allow arbitrary file paths only for 2D→3D (user's own files)
        # For 3D→2D, require upload through our system
        import logging
        logging.getLogger(__name__).warning(
            "Conversion requested for unknown file: %s (3D→2D pipeline)",
            req.file_path,
        )

    job_id = str(uuid.uuid4())
    orchestrator = ConversionOrchestrator()
    orchestrator.start_conversion(
        job_id=job_id,
        direction=req.direction,
        file_path=req.file_path,
        options=req.options,
    )
    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        direction=req.direction,
        created_at=datetime.now(timezone.utc),
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["conversion"])
async def get_job_status(job_id: str) -> JobStatusResponse:
    """Get the status of a conversion job."""
    orchestrator = ConversionOrchestrator()
    status = orchestrator.get_job_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


@router.get("/jobs", response_model=list[JobStatusResponse], tags=["conversion"])
async def list_jobs() -> list[JobStatusResponse]:
    """List all conversion jobs."""
    orchestrator = ConversionOrchestrator()
    return orchestrator.list_jobs()
