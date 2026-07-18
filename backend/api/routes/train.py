"""BridgeGuardian AI — /train endpoint"""
from __future__ import annotations
import asyncio
import logging
import uuid
from datetime import datetime
from fastapi import APIRouter, BackgroundTasks, HTTPException
from backend.schemas.request import TrainRequest
from backend.schemas.response import TrainingStatusResponse

logger = logging.getLogger("bridgeguardian.api.train")
router = APIRouter()

_training_jobs: dict = {}


def get_pipeline():
    from backend.main import inference_pipeline
    return inference_pipeline


async def _run_training(job_id: str, config_path: str):
    """Background training task."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from ml_pipeline.train import train_pipeline

    _training_jobs[job_id]["status"] = "running"
    _training_jobs[job_id]["started_at"] = datetime.utcnow().isoformat()

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, train_pipeline, config_path)
        _training_jobs[job_id]["status"] = "completed"
        _training_jobs[job_id]["result"] = result
        _training_jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()

        # Reload inference pipeline with new models
        from backend.main import inference_pipeline
        inference_pipeline.load()
        logger.info(f"Training job {job_id} completed. Pipeline reloaded.")

    except Exception as e:
        logger.error(f"Training job {job_id} failed: {e}", exc_info=True)
        _training_jobs[job_id]["status"] = "failed"
        _training_jobs[job_id]["error"] = str(e)


@router.post(
    "/train",
    response_model=TrainingStatusResponse,
    summary="Trigger model training",
    tags=["Training"],
)
async def train(
    request: TrainRequest,
    background_tasks: BackgroundTasks,
) -> TrainingStatusResponse:
    """
    Triggers the full ML training pipeline in the background.
    Returns a job ID to poll for status.
    """
    # Check if training is already running
    running = [j for j in _training_jobs.values() if j.get("status") == "running"]
    if running and not request.force_retrain:
        raise HTTPException(status_code=409, detail="Training already in progress.")

    job_id = str(uuid.uuid4())[:8]
    _training_jobs[job_id] = {"status": "queued", "job_id": job_id}
    background_tasks.add_task(_run_training, job_id, request.config_path)

    return TrainingStatusResponse(
        status="queued",
        message=f"Training job {job_id} queued. This may take several minutes.",
        job_id=job_id,
    )


@router.get(
    "/train/status/{job_id}",
    summary="Check training job status",
    tags=["Training"],
)
async def training_status(job_id: str):
    """Returns the status of a training job."""
    if job_id not in _training_jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return _training_jobs[job_id]


@router.get(
    "/train/status",
    summary="List all training jobs",
    tags=["Training"],
)
async def all_training_status():
    return {"jobs": _training_jobs}
