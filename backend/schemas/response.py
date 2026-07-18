"""
BridgeGuardian AI — Pydantic Response Schemas
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class FeatureImportanceItem(BaseModel):
    feature: str
    shap_value: float
    direction: str


class ShapExplanation(BaseModel):
    base_value: float
    shap_values: List[float]
    feature_names: List[str]
    feature_importances: List[FeatureImportanceItem]
    top_positive_features: List[FeatureImportanceItem]
    top_negative_features: List[FeatureImportanceItem]
    prediction_contribution: float
    note: Optional[str] = None


class PredictionResponse(BaseModel):
    """Full prediction response."""
    prediction_id: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Core predictions
    health_score: float = Field(description="Bridge health score 0–100")
    health_score_raw: float = Field(description="Raw SHI value 0–1")
    failure_probability: float = Field(description="Failure probability %")
    failure_probability_raw: float

    # RUL
    rul_days: float = Field(description="Estimated remaining useful life in days")
    rul_degradation_rate: float
    rul_confidence: str
    rul_message: str

    # Risk & maintenance
    risk_category: str = Field(description="Excellent / Good / Fair / Poor / Critical")
    maintenance_priority: str = Field(description="Routine / Low / Medium / High / Emergency")
    maintenance_recommendation: str
    maintenance_alert: bool

    # Confidence
    prediction_confidence: float
    model_version: str

    class Config:
        json_schema_extra = {
            "example": {
                "health_score": 82.16,
                "failure_probability": 2.7,
                "rul_days": 423.5,
                "risk_category": "Good",
                "maintenance_priority": "Low",
            }
        }


class ExplainResponse(BaseModel):
    """SHAP explanation response."""
    target: str
    explanation: ShapExplanation
    prediction_value: Optional[float] = None


class GlobalImportanceItem(BaseModel):
    feature: str
    importance: float


class ModelInfoResponse(BaseModel):
    """Model metadata response."""
    is_ready: bool
    model_version: str
    models_available: List[str]
    feature_count: int
    training_results: Optional[Dict[str, Any]] = None


class TrainingStatusResponse(BaseModel):
    """Training job status response."""
    status: str
    message: str
    job_id: Optional[str] = None


class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    version: str
    model_ready: bool
    database_ok: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PredictionHistoryItem(BaseModel):
    id: int
    created_at: datetime
    health_score: Optional[float]
    failure_probability: Optional[float]
    rul_days: Optional[float]
    risk_category: Optional[str]
    maintenance_priority: Optional[str]
    model_version: Optional[str]


class PredictionHistoryResponse(BaseModel):
    items: List[PredictionHistoryItem]
    total: int


class EvaluationResponse(BaseModel):
    """Model evaluation results response."""
    models: Dict[str, Dict[str, Any]]
    best_model: str
    evaluation_timestamp: datetime = Field(default_factory=datetime.utcnow)
