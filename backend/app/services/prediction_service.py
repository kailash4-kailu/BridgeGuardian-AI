"""
BridgeGuardian AI — Service Layer: PredictionService
Application use case orchestrator that decouples FastAPI controllers from ML inference and persistence logic.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.app.domain.entities.prediction import PredictionEntity
from backend.app.domain.interfaces.iprediction_repository import IPredictionRepository
from backend.ml.inference import InferencePipeline


class PredictionService:
    """Orchestrates prediction execution, business rule validation, and persistence."""

    def __init__(
        self,
        repository: IPredictionRepository,
        pipeline: Optional[InferencePipeline] = None,
    ) -> None:
        self.repository = repository
        self.pipeline = pipeline

    async def execute_prediction(
        self, input_data: Dict[str, Any], pipeline_instance: Optional[InferencePipeline] = None
    ) -> Dict[str, Any]:
        """
        Executes prediction use case:
        1. Calls ML inference pipeline.
        2. Constructs PredictionEntity domain model.
        3. Persists record via repository port.
        4. Returns prediction dictionary matching API contracts.
        """
        active_pipeline = pipeline_instance or self.pipeline
        if active_pipeline is None or not active_pipeline.is_ready:
            raise RuntimeError("Inference pipeline is not ready or loaded.")

        # 1. Run ML inference
        pred_result = active_pipeline.predict(input_data)

        # 2. Build domain entity
        entity = PredictionEntity(
            input_data=input_data,
            health_score=pred_result.get("health_score"),
            failure_probability=pred_result.get("failure_probability"),
            rul_days=pred_result.get("rul_days"),
            risk_category=pred_result.get("risk_category"),
            maintenance_priority=pred_result.get("maintenance_priority"),
            maintenance_recommendation=pred_result.get("maintenance_recommendation"),
            prediction_confidence=pred_result.get("prediction_confidence"),
            model_version=pred_result.get("model_version"),
        )

        # 3. Persist via repository
        saved_entity = await self.repository.save(entity)

        # 4. Attach generated database ID
        pred_result["id"] = saved_entity.id
        pred_result["created_at"] = (
            saved_entity.created_at.isoformat() if saved_entity.created_at else None
        )
        return pred_result

    async def get_prediction_history(
        self, limit: int = 50, offset: int = 0
    ) -> Dict[str, Any]:
        """Fetch paginated historical predictions via repository."""
        items = await self.repository.get_history(limit=limit, offset=offset)
        total = await self.repository.count_total()
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "items": [item.to_dict() for item in items],
        }

    async def get_prediction_by_id(self, prediction_id: int) -> Optional[PredictionEntity]:
        """Fetch single prediction entity by ID."""
        return await self.repository.get_by_id(prediction_id)
