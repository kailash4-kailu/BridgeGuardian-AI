"""
BridgeGuardian AI — Domain Repository Interface (Port)
Abstract Base Class defining storage operations for PredictionEntity instances.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from backend.app.domain.entities.prediction import PredictionEntity


class IPredictionRepository(ABC):
    """Abstract port interface for prediction persistence."""

    @abstractmethod
    async def save(self, entity: PredictionEntity) -> PredictionEntity:
        """Persist a prediction entity and return the saved instance with ID assigned."""
        pass

    @abstractmethod
    async def get_by_id(self, prediction_id: int) -> Optional[PredictionEntity]:
        """Retrieve a prediction entity by primary key ID."""
        pass

    @abstractmethod
    async def get_history(self, limit: int = 50, offset: int = 0) -> List[PredictionEntity]:
        """Retrieve a paginated list of historical prediction entities ordered by creation date."""
        pass

    @abstractmethod
    async def count_total(self) -> int:
        """Get the total count of prediction records."""
        pass
