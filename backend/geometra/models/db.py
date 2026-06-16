"""MongoDB database models and connection management.

Uses Motor for async MongoDB access.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import IndexModel

from geometra.config import settings

client: AsyncIOMotorClient | None = None
db: AsyncIOMotorDatabase | None = None


async def connect_db() -> AsyncIOMotorDatabase:
    """Connect to MongoDB and return the database instance."""
    global client, db
    client = AsyncIOMotorClient(settings.mongodb_url)
    db = client[settings.mongodb_database]
    await _ensure_indexes(db)
    return db


async def close_db() -> None:
    """Close the MongoDB connection."""
    global client
    if client:
        client.close()
        client = None


async def get_db() -> AsyncIOMotorDatabase:
    """Get the current database instance, connecting if necessary."""
    global db
    if db is None:
        db = await connect_db()
    return db


async def _ensure_indexes(database: AsyncIOMotorDatabase) -> None:
    """Ensure indexes exist on collections."""
    jobs = database["jobs"]
    await jobs.create_indexes(
        [
            IndexModel("job_id", unique=True),
            IndexModel("status"),
            IndexModel("created_at"),
        ]
    )


# ── Job Document Helpers ──────────────────────────────────────────────────────


async def create_job_document(job_id: str, direction: str, file_path: str) -> str:
    """Create a job document in MongoDB."""
    db = await get_db()
    doc = {
        "job_id": job_id,
        "status": "pending",
        "direction": direction,
        "file_path": file_path,
        "progress": 0.0,
        "message": "Job created",
        "result_paths": [],
        "validation_report": None,
        "error": None,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    await db["jobs"].insert_one(doc)
    return job_id


async def update_job_document(job_id: str, update: dict[str, Any]) -> None:
    """Update a job document."""
    db = await get_db()
    update["updated_at"] = datetime.now(timezone.utc)
    await db["jobs"].update_one({"job_id": job_id}, {"$set": update})


async def get_job_document(job_id: str) -> dict[str, Any] | None:
    """Get a job document by ID."""
    db = await get_db()
    return await db["jobs"].find_one({"job_id": job_id}, {"_id": 0})
