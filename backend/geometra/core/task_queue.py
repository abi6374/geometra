"""Agent pipeline orchestration and Celery task queue."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from celery import Celery

from geometra.api.schemas import ConversionDirection, JobStatus, JobStatusResponse
from geometra.config import settings

logger = logging.getLogger(__name__)

celery_app = Celery(
    "geometra",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)


# ── Hybrid Job Store (MongoDB + in-memory fallback) ──────────────────────────

# In-memory fallback job store (used when MongoDB is not available)
_jobs: dict[str, dict[str, Any]] = {}

# Lazy MongoDB client for sync access (Celery tasks are sync)
_mongo_client = None


def _get_job_collection() -> Any | None:
    """Get the MongoDB jobs collection for sync access.

    Uses pymongo (sync) for Celery task access, as Motor (async)
    cannot be used from sync tasks.
    """
    global _mongo_client
    if _mongo_client is None:
        try:
            from pymongo import MongoClient
            _mongo_client = MongoClient(settings.mongodb_url, serverSelectionTimeoutMS=2000)
            # Verify connection
            _mongo_client.admin.command("ping")
            db = _mongo_client[settings.mongodb_database]
            # Ensure index
            db["jobs"].create_index("job_id", unique=True)
            db["jobs"].create_index("status")
            logger.info("Connected to MongoDB for sync job store")
            return db["jobs"]
        except ImportError:
            logger.warning("pymongo not installed; using in-memory job store")
            _mongo_client = None
        except Exception as exc:
            logger.warning("MongoDB not available; using in-memory job store: %s", exc)
            _mongo_client = None

    if _mongo_client is not None:
        try:
            return _mongo_client[settings.mongodb_database]["jobs"]
        except Exception:
            pass
    return None


def _build_job_doc(
    job_id: str,
    direction: ConversionDirection,
    status: JobStatus = JobStatus.PENDING,
    progress: float = 0.0,
    message: str = "",
    result_paths: list[str] | None = None,
    validation_report: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "job_id": job_id,
        "status": status,
        "direction": direction,
        "progress": progress,
        "message": message,
        "result_paths": result_paths or [],
        "validation_report": validation_report,
        "created_at": now,
        "updated_at": now,
        "error": error,
    }


class ConversionOrchestrator:
    """Orchestrates the execution of the multi-agent conversion pipeline.

    Uses MongoDB when available (via pymongo), falling back to an
    in-memory store when MongoDB is not connected. This ensures the
    system works in development without external dependencies while
    supporting production deployments with MongoDB.
    """

    def start_conversion(
        self,
        job_id: str,
        direction: ConversionDirection,
        file_path: str,
        options: dict[str, Any] | None = None,
    ) -> str:
        doc = _build_job_doc(job_id, direction, status=JobStatus.PENDING, message="Job created")

        # Try MongoDB first
        collection = _get_job_collection()
        if collection is not None:
            try:
                collection.insert_one(doc.copy())
            except Exception as exc:
                logger.warning("Failed to write job to MongoDB: %s; falling back to in-memory", exc)
                _jobs[job_id] = doc
        else:
            _jobs[job_id] = doc

        # Dispatch to Celery
        pipeline_job.delay(job_id=job_id, direction=direction.value, file_path=file_path, options=options or {})
        return job_id

    def get_job_status(self, job_id: str) -> JobStatusResponse | None:
        # Try MongoDB first
        collection = _get_job_collection()
        if collection is not None:
            try:
                job = collection.find_one({"job_id": job_id}, {"_id": 0})
                if job is not None:
                    return JobStatusResponse(**job)
            except Exception:
                pass

        # Fallback to in-memory
        job = _jobs.get(job_id)
        if job is None:
            return None
        return JobStatusResponse(**job)

    def list_jobs(self) -> list[JobStatusResponse]:
        # Try MongoDB first
        results: list[JobStatusResponse] = []
        collection = _get_job_collection()
        if collection is not None:
            try:
                for job in collection.find({}, {"_id": 0}).sort("created_at", -1).limit(100):
                    results.append(JobStatusResponse(**job))
                return results
            except Exception:
                pass

        # Fallback to in-memory
        return [JobStatusResponse(**j) for j in _jobs.values()]

    def update_job(
        self,
        job_id: str,
        status: JobStatus | None = None,
        progress: float | None = None,
        message: str | None = None,
        result_paths: list[str] | None = None,
        validation_report: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        update: dict[str, Any] = {"updated_at": datetime.now(timezone.utc)}
        if status is not None:
            update["status"] = status
        if progress is not None:
            update["progress"] = progress
        if message is not None:
            update["message"] = message
        if result_paths is not None:
            update["result_paths"] = result_paths
        if validation_report is not None:
            update["validation_report"] = validation_report
        if error is not None:
            update["error"] = error

        # Try MongoDB first
        collection = _get_job_collection()
        if collection is not None:
            try:
                collection.update_one({"job_id": job_id}, {"$set": update})
                return
            except Exception:
                pass

        # Fallback to in-memory
        job = _jobs.get(job_id)
        if job is not None:
            job.update(update)


# ── Helper: validate agent input/output ──────────────────────────────────────


def _safe_validate(agent: Any, input_data: Any, stage: str, job_id: str, orchestrator: ConversionOrchestrator) -> None:
    """Validate agent input and raise a clear error if invalid."""
    try:
        if hasattr(agent, "validate_input") and callable(agent.validate_input):
            if not agent.validate_input(input_data):
                raise ValueError(
                    f"Agent {agent.name} rejected input at stage '{stage}': "
                    f"input validation failed (missing required keys or invalid format)"
                )
    except Exception as exc:
        orchestrator.update_job(
            job_id,
            status=JobStatus.FAILED,
            error=f"Input validation failed at {stage}: {exc}",
            message=f"Validation error at {stage}",
        )
        raise


# ── Celery Task ───────────────────────────────────────────────────────────────


@celery_app.task(bind=True, max_retries=2, default_retry_delay=10)
def pipeline_job(
    self,
    job_id: str,
    direction: str,
    file_path: str,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute the full agent pipeline for a conversion job.

    Retries with exponential backoff: 10s, 30s, 90s (max 300s).
    """
    from geometra.agents.agent_01_input_processing import InputProcessingAgent
    from geometra.agents.agent_02_drawing_understanding import DrawingUnderstandingAgent
    from geometra.agents.agent_03_ocr_annotation import OCRAnnotationAgent
    from geometra.agents.agent_04_feature_recognition import FeatureRecognitionAgent
    from geometra.agents.agent_05_engineering_reasoning import EngineeringReasoningAgent
    from geometra.agents.agent_06 import CADGenerationAgent
    from geometra.agents.agent_07_validation import ValidationAgent
    from geometra.agents.agent_08_3d_to_2d import ThreeDToTwoDConversionAgent

    orchestrator = ConversionOrchestrator()
    options = options or {}
    direction_enum = ConversionDirection(direction)

    # Calculate retry countdown (exponential backoff: 10s, 60s, 300s)
    retry_count = self.request.retries if hasattr(self, "request") else 0
    countdown = min(10 * (3 ** retry_count), 300)

    try:
        # ── Step 1: Input Processing ──
        orchestrator.update_job(job_id, status=JobStatus.PROCESSING, progress=0.05, message="Processing input file")
        agent_1 = InputProcessingAgent()
        _safe_validate(agent_1, file_path, "agent_1_input", job_id, orchestrator)
        doc = agent_1.process(file_path)

        if direction_enum == ConversionDirection.TWO_D_TO_THREE_D:
            # ── Step 2: Drawing Understanding ──
            orchestrator.update_job(job_id, progress=0.15, message="Understanding drawing geometry")
            agent_2 = DrawingUnderstandingAgent()
            _safe_validate(agent_2, doc, "agent_2_input", job_id, orchestrator)
            primitives = agent_2.process(doc)

            # ── Step 3: OCR & Annotations ──
            orchestrator.update_job(job_id, progress=0.30, message="Extracting annotations and dimensions")
            agent_3 = OCRAnnotationAgent()
            _safe_validate(agent_3, doc, "agent_3_input", job_id, orchestrator)
            annotations = agent_3.process(doc)

            # ── Step 4: Feature Recognition ──
            orchestrator.update_job(job_id, progress=0.45, message="Recognizing manufacturing features")
            agent_4 = FeatureRecognitionAgent()
            _safe_validate(agent_4, primitives, "agent_4_input", job_id, orchestrator)
            features = agent_4.process(primitives)

            # ── Step 5: Engineering Reasoning ──
            orchestrator.update_job(job_id, progress=0.60, message="Applying engineering reasoning")
            agent_5 = EngineeringReasoningAgent()
            reason_input = {
                "primitives": primitives,
                "annotations": annotations,
                "features": features,
            }
            _safe_validate(agent_5, reason_input, "agent_5_input", job_id, orchestrator)
            param_tree = agent_5.process(reason_input)

            # ── Step 6: CAD Generation ──
            orchestrator.update_job(job_id, progress=0.75, message="Generating CAD model")
            agent_6 = CADGenerationAgent()
            _safe_validate(agent_6, param_tree, "agent_6_input", job_id, orchestrator)
            cad_result = agent_6.generate(param_tree)

            # ── Step 7: Validation ──
            orchestrator.update_job(job_id, progress=0.90, message="Validating generated CAD")
            agent_7 = ValidationAgent()
            validation = agent_7.validate(doc, cad_result)

        else:
            # 3D → 2D pipeline is simpler
            # ── Step 2: Input Processing (3D) ──
            orchestrator.update_job(job_id, progress=0.30, message="Processing 3D model")

            # ── Step 8: 3D-to-2D Conversion ──
            orchestrator.update_job(job_id, progress=0.60, message="Generating 2D projections")
            agent_8 = ThreeDToTwoDConversionAgent()
            cad_result = agent_8.convert(doc)

            # ── Step 7: Validation ──
            orchestrator.update_job(job_id, progress=0.90, message="Validating projections")
            agent_7 = ValidationAgent()
            validation = agent_7.validate(doc, cad_result)

        orchestrator.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            progress=1.0,
            message="Conversion completed successfully",
            result_paths=cad_result.get("output_paths", []),
            validation_report=validation,
        )
        return {"status": "completed", "job_id": job_id}

    except Exception as exc:
        orchestrator.update_job(
            job_id,
            status=JobStatus.FAILED,
            error=str(exc),
            message=f"Conversion failed: {exc}",
        )
        # Exponential backoff: retry with increasing delays
        # countdown = min(10 * (3 ** retries), 300) → 10s, 30s, 90s
        raise self.retry(exc=exc, countdown=countdown) from exc
