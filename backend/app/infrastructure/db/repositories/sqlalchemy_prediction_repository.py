"""
BridgeGuardian AI — Infrastructure Adapter: SQLAlchemyPredictionRepository
Concrete implementation of IPredictionRepository using SQLAlchemy ORM.
"""
from __future__ import annotations

import json
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from backend.app.domain.entities.prediction import PredictionEntity
from backend.app.domain.interfaces.iprediction_repository import IPredictionRepository
from backend.core.models import PredictionRecord


class SQLAlchemyPredictionRepository(IPredictionRepository):
    """SQLAlchemy implementation of the IPredictionRepository port."""

    def __init__(self, db: Session) -> None:
        self.db = db

    async def save(self, entity: PredictionEntity) -> PredictionEntity:
        """Persist a PredictionEntity to the database via SQLAlchemy."""
        input_data_str = (
            json.dumps(entity.input_data)
            if isinstance(entity.input_data, dict)
            else str(entity.input_data)
        )

        record = PredictionRecord(
            input_data=input_data_str,
            health_score=entity.health_score,
            failure_probability=entity.failure_probability,
            rul_days=entity.rul_days,
            risk_category=entity.risk_category,
            maintenance_priority=entity.maintenance_priority,
            maintenance_recommendation=entity.maintenance_recommendation,
            prediction_confidence=entity.prediction_confidence,
            repair_cost_estimate=entity.repair_cost_estimate,
            model_version=entity.model_version,
        )

        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)

        entity.id = record.id
        entity.created_at = record.created_at
        return entity

    async def get_by_id(self, prediction_id: int) -> Optional[PredictionEntity]:
        """Fetch prediction entity by primary key."""
        stmt = select(PredictionRecord).where(PredictionRecord.id == prediction_id)
        record = self.db.execute(stmt).scalar_one_or_none()
        if not record:
            return None
        return self._to_entity(record)

    async def get_history(self, limit: int = 50, offset: int = 0) -> List[PredictionEntity]:
        """Fetch paginated historical records."""
        stmt = (
            select(PredictionRecord)
            .order_by(PredictionRecord.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        records = self.db.execute(stmt).scalars().all()
        return [self._to_entity(rec) for rec in records]

    async def count_total(self) -> int:
        """Return total count of prediction records."""
        stmt = select(func.count(PredictionRecord.id))
        return self.db.execute(stmt).scalar() or 0

    @staticmethod
    def _to_entity(record: PredictionRecord) -> PredictionEntity:
        """Helper to convert ORM model to domain entity."""
        try:
            input_dict = json.loads(record.input_data) if record.input_data else {}
        except Exception:
            input_dict = {"raw": record.input_data}

        return PredictionEntity(
            id=record.id,
            created_at=record.created_at,
            input_data=input_dict,
            health_score=record.health_score,
            failure_probability=record.failure_probability,
            rul_days=record.rul_days,
            risk_category=record.risk_category,
            maintenance_priority=record.maintenance_priority,
            maintenance_recommendation=record.maintenance_recommendation,
            prediction_confidence=record.prediction_confidence,
            repair_cost_estimate=record.repair_cost_estimate,
            model_version=record.model_version,
        )
