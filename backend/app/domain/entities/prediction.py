"""
BridgeGuardian AI — Domain Entity: PredictionEntity
Pure Python domain representation of a structural health prediction request and result.
Free from framework (FastAPI, SQLAlchemy) dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class PredictionEntity:
    """Domain model representing a prediction record."""
    input_data: Dict[str, Any]
    health_score: Optional[float] = None
    failure_probability: Optional[float] = None
    rul_days: Optional[float] = None
    risk_category: Optional[str] = None
    maintenance_priority: Optional[str] = None
    maintenance_recommendation: Optional[str] = None
    prediction_confidence: Optional[float] = None
    repair_cost_estimate: Optional[float] = None
    model_version: Optional[str] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entity attributes to standard dictionary format."""
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "input_data": self.input_data,
            "health_score": self.health_score,
            "failure_probability": self.failure_probability,
            "rul_days": self.rul_days,
            "risk_category": self.risk_category,
            "maintenance_priority": self.maintenance_priority,
            "maintenance_recommendation": self.maintenance_recommendation,
            "prediction_confidence": self.prediction_confidence,
            "repair_cost_estimate": self.repair_cost_estimate,
            "model_version": self.model_version,
        }
