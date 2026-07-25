"""
BridgeGuardian AI — ML Governance & Drift Endpoints (/ml/drift)
Exposes statistical data drift detection and model performance monitoring endpoints.
"""
from __future__ import annotations

from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.ml.drift_detector import DataDriftDetector
from backend.core.database import get_db
from backend.core.models import PredictionRecord

router = APIRouter()


class DriftStatusResponse(BaseModel):
    dataset_drift_detected: bool
    drift_share: float
    drifted_features_count: int
    total_features_evaluated: int
    feature_metrics: Dict[str, Any]


@router.get(
    "/ml/drift-status",
    response_model=DriftStatusResponse,
    summary="Get telemetry feature drift status",
    tags=["ML Governance"],
)
async def get_drift_status(db: Session = Depends(get_db)) -> Any:
    """
    Computes statistical Kolmogorov-Smirnov feature drift by comparing recent telemetry records
    against baseline distributions.
    """
    records = (
        db.query(PredictionRecord)
        .order_by(PredictionRecord.created_at.desc())
        .limit(100)
        .all()
    )

    if not records:
        # Return clean baseline report if no records are logged yet
        return DriftStatusResponse(
            dataset_drift_detected=False,
            drift_share=0.0,
            drifted_features_count=0,
            total_features_evaluated=0,
            feature_metrics={},
        )

    # Extract numerical fields from historical records
    import json
    import pandas as pd

    data_rows = []
    for rec in records:
        try:
            row = json.loads(rec.input_data)
            data_rows.append(row)
        except Exception:
            continue

    if not data_rows:
        return DriftStatusResponse(
            dataset_drift_detected=False,
            drift_share=0.0,
            drifted_features_count=0,
            total_features_evaluated=0,
            feature_metrics={},
        )

    df_prod = pd.DataFrame(data_rows)
    detector = DataDriftDetector()
    results = detector.compute_feature_drift(df_prod)

    return DriftStatusResponse(**results)
