"""BridgeGuardian AI — /health, /model-info, /evaluate, /history endpoints"""
from __future__ import annotations
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import desc
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.core.models import PredictionRecord, InspectionRecord
from backend.schemas.response import (
    EvaluationResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionHistoryItem,
    PredictionHistoryResponse,
)

logger = logging.getLogger("bridgeguardian.api.misc")
router = APIRouter()


def get_pipeline():
    from backend.main import inference_pipeline
    return inference_pipeline


# ─────────────────────────── /health ────────────────────────────────────── #

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="API health check",
    tags=["System"],
)
async def health_check(
    db: Session = Depends(get_db),
    pipeline=Depends(get_pipeline),
) -> HealthResponse:
    """Returns system health: API status, model readiness, and database connectivity."""
    db_ok = True
    try:
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
    except Exception:
        db_ok = False

    return HealthResponse(
        status="healthy" if (pipeline.is_ready and db_ok) else "degraded",
        version="1.0.0",
        model_ready=pipeline.is_ready,
        database_ok=db_ok,
    )


@router.get(
    "/health/liveness",
    summary="Kubernetes / Cloud Liveness Probe",
    tags=["System"],
)
async def liveness_probe():
    """Returns HTTP 200 OK if server process is running."""
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}


@router.get(
    "/health/readiness",
    summary="Kubernetes / Cloud Readiness Probe",
    tags=["System"],
)
async def readiness_probe(
    db: Session = Depends(get_db),
    pipeline=Depends(get_pipeline),
):
    """Returns HTTP 200 OK if database is connected and models are warmed in RAM."""
    db_ok = True
    try:
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
    except Exception:
        db_ok = False

    if not db_ok or not pipeline.is_ready:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "database_ok": db_ok,
                "model_ready": pipeline.is_ready,
            },
        )

    return {
        "status": "ready",
        "database_ok": True,
        "model_ready": True,
    }



# ─────────────────────────── /model-info ────────────────────────────────── #

@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Get trained model metadata",
    tags=["System"],
)
async def model_info(pipeline=Depends(get_pipeline)) -> ModelInfoResponse:
    """Returns metadata about the currently loaded ML models."""
    training_results = None
    results_path = Path(pipeline.models_dir) / "training_results.json"
    if results_path.exists():
        with open(results_path) as f:
            training_results = json.load(f)

    return ModelInfoResponse(
        is_ready=pipeline.is_ready,
        model_version=pipeline._model_version,
        models_available=list(pipeline._models.keys()),
        feature_count=len(pipeline._feature_columns),
        training_results=training_results,
    )


# ─────────────────────────── /evaluate ──────────────────────────────────── #

@router.get(
    "/evaluate",
    summary="Get model evaluation metrics",
    tags=["System"],
)
async def evaluate(pipeline=Depends(get_pipeline)):
    """Returns training evaluation metrics for all trained models."""
    results_path = Path(pipeline.models_dir) / "training_results.json"
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="No training results found. Train first.")
    with open(results_path) as f:
        return json.load(f)


# ─────────────────────────── /history ───────────────────────────────────── #

@router.get(
    "/history",
    response_model=PredictionHistoryResponse,
    summary="Get prediction history",
    tags=["Prediction"],
)
async def prediction_history(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
) -> PredictionHistoryResponse:
    """Returns paginated prediction history from the database, including drone campaigns."""
    pred_records = db.query(PredictionRecord).all()
    inspection_records = db.query(InspectionRecord).filter(InspectionRecord.status == "completed").all()

    combined_items = []

    for r in pred_records:
        h_score = r.health_score * 100 if (r.health_score is not None and r.health_score <= 1.0) else r.health_score
        f_prob = r.failure_probability * 100 if (r.failure_probability is not None and r.failure_probability <= 1.0) else r.failure_probability
        combined_items.append(
            PredictionHistoryItem(
                id=r.id,
                created_at=r.created_at,
                health_score=h_score,
                failure_probability=f_prob,
                rul_days=r.rul_days,
                risk_category=r.risk_category,
                maintenance_priority=r.maintenance_priority,
                model_version=r.model_version or "Vibration & Telemetry ML",
            )
        )

    for i in inspection_records:
        h_score = i.health_score
        f_prob = i.failure_probability
        combined_items.append(
            PredictionHistoryItem(
                id=10000 + i.id,
                created_at=i.created_at,
                health_score=h_score,
                failure_probability=f_prob,
                rul_days=i.rul_days,
                risk_category=i.risk_category,
                maintenance_priority=i.maintenance_priority,
                model_version="Drone Campaign YOLOv11/SAM2",
            )
        )

    def get_sort_key(item: PredictionHistoryItem) -> datetime:
        if not item.created_at:
            return datetime.min
        if item.created_at.tzinfo is not None:
            return item.created_at.replace(tzinfo=None)
        return item.created_at

    # Sort descending by creation timestamp
    combined_items.sort(key=get_sort_key, reverse=True)
    total = len(combined_items)
    paginated_items = combined_items[offset : offset + limit]

    return PredictionHistoryResponse(items=paginated_items, total=total)
