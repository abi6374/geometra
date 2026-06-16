"""Geometra FastAPI application."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from geometra.api.routes import router
from geometra.config import settings

app = FastAPI(
    title="Geometra API",
    description="AI-Powered Bidirectional Engineering Drawing and CAD Reconstruction Platform",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
async def startup() -> None:
    """Initialize application on startup."""
    settings.ensure_dirs()

    # Start periodic cleanup of uploaded files (runs every hour)
    # Using a simple in-memory tracker - calls to _cleanup_old_uploads
    # happen on every upload. For persistent cleanup, a background task
    # would be needed.
    import asyncio

    async def periodic_cleanup() -> None:
        while True:
            await asyncio.sleep(3600)
            from geometra.api.routes import _cleanup_old_uploads
            _cleanup_old_uploads()

    asyncio.create_task(periodic_cleanup())


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Geometra API - AI-Powered Engineering Drawing Platform"}
