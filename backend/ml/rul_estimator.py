"""
BridgeGuardian AI — RUL (Remaining Useful Life) Estimator

Since the dataset has no explicit RUL ground truth, RUL is derived from
the Structural Health Index (SHI) degradation trajectory.

Method:
  - SHI_current is the current health score
  - SHI_critical = 0.40 (industry standard threshold for bridge intervention)
  - Daily degradation rate = estimated from SHI_7d_ahead vs current SHI
  - RUL_days = (SHI_current - SHI_critical) / daily_degradation_rate

This is mathematically sound and used in real bridge monitoring systems
(analogous to remaining capacity estimation in battery health monitoring).
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("bridgeguardian.rul_estimator")

SHI_CRITICAL_THRESHOLD = 0.40  # Bridge requires intervention below this
DEFAULT_DEGRADATION_RATE = 0.001  # per minute (conservative estimate)
MINUTES_PER_DAY = 1440


class RULEstimator:
    """
    Estimates Remaining Useful Life (RUL) of a bridge structure
    based on the SHI degradation trajectory.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        thresholds = config.get("thresholds", {})
        self.shi_critical = thresholds.get("shi_critical", SHI_CRITICAL_THRESHOLD)

    def estimate(
        self,
        shi_current: float,
        shi_7d_ahead: Optional[float] = None,
        shi_30d_ahead: Optional[float] = None,
    ) -> dict:
        """
        Estimate RUL in days and generate a risk assessment.

        Args:
            shi_current: Current SHI value (0–1).
            shi_7d_ahead: Predicted SHI 7 days from now.
            shi_30d_ahead: Predicted SHI 30 days from now.

        Returns:
            Dict with rul_days, degradation_rate_per_day, confidence, method.
        """
        shi_current = max(0.0, min(1.0, shi_current))

        if shi_current <= self.shi_critical:
            return {
                "rul_days": 0.0,
                "degradation_rate_per_day": 0.0,
                "confidence": "high",
                "method": "threshold_breach",
                "message": "Structure has reached critical threshold. Immediate intervention required.",
            }

        # Priority: use 7-day prediction (most reliable forward estimate)
        if shi_7d_ahead is not None:
            shi_7d = max(0.0, min(1.0, shi_7d_ahead))
            daily_rate = max((shi_current - shi_7d) / 7.0, 1e-6)
            method = "7day_forecast"
            confidence = "high"
        elif shi_30d_ahead is not None:
            shi_30d = max(0.0, min(1.0, shi_30d_ahead))
            daily_rate = max((shi_current - shi_30d) / 30.0, 1e-6)
            method = "30day_forecast"
            confidence = "medium"
        else:
            # Fallback to conservative default rate
            daily_rate = DEFAULT_DEGRADATION_RATE * MINUTES_PER_DAY
            method = "default_rate"
            confidence = "low"

        headroom = shi_current - self.shi_critical
        rul_days = headroom / daily_rate if daily_rate > 0 else 9999.0
        rul_days = round(min(rul_days, 3650.0), 1)  # Cap at 10 years

        return {
            "rul_days": rul_days,
            "degradation_rate_per_day": round(daily_rate, 6),
            "confidence": confidence,
            "method": method,
            "message": self._rul_message(rul_days),
        }

    @staticmethod
    def _rul_message(rul_days: float) -> str:
        if rul_days <= 0:
            return "CRITICAL: Immediate intervention required"
        elif rul_days <= 7:
            return f"URGENT: ~{int(rul_days)} days to critical threshold. Schedule emergency inspection."
        elif rul_days <= 30:
            return f"HIGH RISK: ~{int(rul_days)} days remaining. Plan maintenance within 2 weeks."
        elif rul_days <= 90:
            return f"MODERATE RISK: ~{int(rul_days)} days remaining. Schedule maintenance within 1 month."
        elif rul_days <= 365:
            return f"LOW RISK: ~{int(rul_days)} days remaining. Routine inspection recommended."
        else:
            return f"HEALTHY: ~{int(rul_days)} days remaining. Continue regular monitoring."
