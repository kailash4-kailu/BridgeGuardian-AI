"""
BridgeGuardian AI — /predict endpoint
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.core.models import PredictionRecord
from backend.schemas.request import BridgeSensorInput
from backend.schemas.response import PredictionResponse

logger = logging.getLogger("bridgeguardian.api.predict")
router = APIRouter()


def get_pipeline():
    """Dependency that returns the global inference pipeline."""
    from backend.main import inference_pipeline
    return inference_pipeline


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict bridge health",
    description="Run the full prediction pipeline on a bridge sensor reading.",
    tags=["Prediction"],
)
async def predict(
    sensor_input: BridgeSensorInput,
    db: Session = Depends(get_db),
    pipeline=Depends(get_pipeline),
) -> PredictionResponse:
    """
    Accepts bridge sensor readings and returns:
    - Health Score (0–100)
    - Failure Probability (%)
    - Remaining Useful Life (days)
    - Risk Category
    - Maintenance Priority & Recommendation
    - Prediction Confidence
    """
    if not pipeline.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. POST to /train to train the model first.",
        )

    input_dict = sensor_input.model_dump(exclude_none=False)
    # Fill None values with dataset medians / defaults for inference
    input_dict = {k: (v if v is not None else _get_default(k)) for k, v in input_dict.items()}

    try:
        result = pipeline.predict(input_dict)
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    # Persist to database
    record = PredictionRecord(
        input_data=json.dumps({k: v for k, v in input_dict.items() if not isinstance(v, float) or v == v}),
        health_score=result["health_score_raw"],
        failure_probability=result["failure_probability_raw"],
        rul_days=result["rul_days"],
        risk_category=result["risk_category"],
        maintenance_priority=result["maintenance_priority"],
        maintenance_recommendation=result["maintenance_recommendation"],
        prediction_confidence=result["prediction_confidence"],
        model_version=result["model_version"],
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return PredictionResponse(
        prediction_id=record.id,
        **result,
    )


def _get_default(feature: str) -> float:
    """Return sensible default value for missing features at inference time."""
    DEFAULTS = {
        "Strain_microstrain": 734.5,
        "Deflection_mm": 14.99,
        "Vibration_ms2": 1.20,
        "Tilt_deg": 0.72,
        "Displacement_mm": 22.36,
        "Crack_Propagation_mm": 0.015,
        "Corrosion_Level_percent": 0.15,
        "Cable_Member_Tension_kN": 447.9,
        "Bearing_Joint_Forces_kN": 260.1,
        "Fatigue_Accumulation_au": 0.30,
        "Modal_Frequency_Hz": 1.90,
        "Temperature_C": 15.0,
        "Humidity_percent": 60.3,
        "Wind_Speed_ms": 6.5,
        "Wind_Direction_deg": 180.0,
        "Precipitation_mmh": 0.0,
        "Water_Level_m": 2.0,
        "Seismic_Activity_ms2": 0.0,
        "Solar_Radiation_Wm2": 446.5,
        "Air_Quality_Index_AQI": 55.0,
        "Soil_Settlement_mm": 0.30,
        "Vehicle_Load_tons": 16.4,
        "Traffic_Volume_vph": 853.2,
        "Pedestrian_Load_pph": 96.3,
        "Impact_Events_g": 0.0,
        "Dynamic_Load_Distribution_percent": 90.1,
        "Axle_Counts_pmin": 43.4,
        "Anomaly_Detection_Score": 0.0,
        "Energy_Dissipation_au": 0.156,
        "Acoustic_Emissions_levels": 10.45,
        "Visual_Analysis_Defect_Score": 0.004,
        "Electrical_Resistance_ohms": 0.282,
        "Localized_Strain_Hotspot": 0.0,
        "Bridge_Mood_Meter": "Healthy",
        "Vibration_Anomaly_Location": "Unknown",
        "Flood_Event_Flag": 0.0,
        "High_Winds_Storms": 0.0,
        "Landslide_Ground_Movement": 0.0,
        "Abnormal_Traffic_Load_Surges": 0.0,
        "SHI_Predicted_7d_Ahead": None,
        "SHI_Predicted_30d_Ahead": None,
    }
    return DEFAULTS.get(feature, 0.0)
