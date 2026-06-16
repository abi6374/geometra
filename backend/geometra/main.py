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


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Geometra API - AI-Powered Engineering Drawing Platform"}
