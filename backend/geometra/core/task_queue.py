"""Agent pipeline orchestration and Celery task queue."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from celery import Celery

from geometra.api.schemas import ConversionDirection, JobStatus, JobStatusResponse
from geometra.config import settings

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


# In-memory job store (to be replaced with MongoDB)
_jobs: dict[str, dict[str, Any]] = {}


class ConversionOrchestrator:
    """Orchestrates the execution of the multi-agent conversion pipeline."""

    def start_conversion(
        self,
        job_id: str,
        direction: ConversionDirection,
        file_path: str,
        options: dict[str, Any] | None = None,
    ) -> str:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "direction": direction,
            "progress": 0.0,
            "message": "Job created",
            "result_paths": [],
            "validation_report": None,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "error": None,
        }
        # Dispatch to Celery
        pipeline_job.delay(job_id=job_id, direction=direction.value, file_path=file_path, options=options or {})
        return job_id

    def get_job_status(self, job_id: str) -> JobStatusResponse | None:
        job = _jobs.get(job_id)
        if job is None:
            return None
        return JobStatusResponse(**job)

    def list_jobs(self) -> list[JobStatusResponse]:
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
        job = _jobs.get(job_id)
        if job is None:
            return
        if status is not None:
            job["status"] = status
        if progress is not None:
            job["progress"] = progress
        if message is not None:
            job["message"] = message
        if result_paths is not None:
            job["result_paths"] = result_paths
        if validation_report is not None:
            job["validation_report"] = validation_report
        if error is not None:
            job["error"] = error
        job["updated_at"] = datetime.now(timezone.utc)


# ── Celery Task ───────────────────────────────────────────────────────────────


@celery_app.task(bind=True, max_retries=1)
def pipeline_job(
    self,
    job_id: str,
    direction: str,
    file_path: str,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute the full agent pipeline for a conversion job."""
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

    try:
        # ── Step 1: Input Processing ──
        orchestrator.update_job(job_id, status=JobStatus.PROCESSING, progress=0.05, message="Processing input file")
        agent_1 = InputProcessingAgent()
        doc = agent_1.process(file_path)

        if direction_enum == ConversionDirection.TWO_D_TO_THREE_D:
            # ── Step 2: Drawing Understanding ──
            orchestrator.update_job(job_id, progress=0.15, message="Understanding drawing geometry")
            agent_2 = DrawingUnderstandingAgent()
            primitives = agent_2.process(doc)

            # ── Step 3: OCR & Annotations ──
            orchestrator.update_job(job_id, progress=0.30, message="Extracting annotations and dimensions")
            agent_3 = OCRAnnotationAgent()
            annotations = agent_3.process(doc)

            # ── Step 4: Feature Recognition ──
            orchestrator.update_job(job_id, progress=0.45, message="Recognizing manufacturing features")
            agent_4 = FeatureRecognitionAgent()
            features = agent_4.process(primitives)

            # ── Step 5: Engineering Reasoning ──
            orchestrator.update_job(job_id, progress=0.60, message="Applying engineering reasoning")
            agent_5 = EngineeringReasoningAgent()
            param_tree = agent_5.process({
                "primitives": primitives,
                "annotations": annotations,
                "features": features,
            })

            # ── Step 6: CAD Generation ──
            orchestrator.update_job(job_id, progress=0.75, message="Generating CAD model")
            agent_6 = CADGenerationAgent()
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
        raise self.retry(exc=exc) from exc
