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
from backend.core.models import PredictionRecord
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
    """Returns paginated prediction history from the database."""
    total = db.query(PredictionRecord).count()
    records = (
        db.query(PredictionRecord)
        .order_by(desc(PredictionRecord.created_at))
        .offset(offset)
        .limit(limit)
        .all()
    )

    items = [
        PredictionHistoryItem(
            id=r.id,
            created_at=r.created_at,
            health_score=r.health_score * 100 if r.health_score else None,
            failure_probability=r.failure_probability * 100 if r.failure_probability else None,
            rul_days=r.rul_days,
            risk_category=r.risk_category,
            maintenance_priority=r.maintenance_priority,
            model_version=r.model_version,
        )
        for r in records
    ]
    return PredictionHistoryResponse(items=items, total=total)
