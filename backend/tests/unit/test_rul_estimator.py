"""
BridgeGuardian AI — Unit Tests: RULEstimator
Tests Remaining Useful Life (RUL) estimation mathematical calculations and edge cases.
"""
from __future__ import annotations

import pytest
from backend.ml.rul_estimator import RULEstimator


@pytest.fixture
def rul_estimator():
    config = {"thresholds": {"shi_critical": 0.40}}
    return RULEstimator(config)


def test_rul_threshold_breach(rul_estimator):
    """If SHI is at or below critical threshold (0.40), RUL must be 0.0 with high confidence."""
    result = rul_estimator.estimate(shi_current=0.35)
    assert result["rul_days"] == 0.0
    assert result["confidence"] == "high"
    assert result["method"] == "threshold_breach"
    assert "CRITICAL" in result["message"] or "critical" in result["message"].lower()


def test_rul_7day_forecast_trajectory(rul_estimator):
    """RUL using 7-day forward forecast trajectory."""
    # Current SHI = 0.80, 7d ahead SHI = 0.73 -> Daily rate = (0.80 - 0.73) / 7 = 0.01/day
    # Headroom = 0.80 - 0.40 = 0.40 -> RUL days = 0.40 / 0.01 = 40.0 days
    result = rul_estimator.estimate(shi_current=0.80, shi_7d_ahead=0.73)
    assert result["rul_days"] == 40.0
    assert result["degradation_rate_per_day"] == 0.01
    assert result["confidence"] == "high"
    assert result["method"] == "7day_forecast"


def test_rul_30day_forecast_fallback(rul_estimator):
    """RUL using 30-day forward forecast fallback when 7-day is absent."""
    # Current SHI = 0.90, 30d ahead SHI = 0.75 -> Daily rate = (0.90 - 0.75) / 30 = 0.005/day
    # Headroom = 0.90 - 0.40 = 0.50 -> RUL days = 0.50 / 0.005 = 100.0 days
    result = rul_estimator.estimate(shi_current=0.90, shi_7d_ahead=None, shi_30d_ahead=0.75)
    assert result["rul_days"] == 100.0
    assert result["confidence"] == "medium"
    assert result["method"] == "30day_forecast"


def test_rul_default_rate_fallback(rul_estimator):
    """RUL default degradation rate fallback when forward forecasts are absent."""
    result = rul_estimator.estimate(shi_current=0.85, shi_7d_ahead=None, shi_30d_ahead=None)
    assert result["rul_days"] > 0.0
    assert result["confidence"] == "low"
    assert result["method"] == "default_rate"


def test_rul_max_safety_cap(rul_estimator):
    """RUL should be capped at 3650 days (10 years maximum)."""
    # Extremely slow degradation rate
    result = rul_estimator.estimate(shi_current=0.99, shi_7d_ahead=0.9899999)
    assert result["rul_days"] <= 3650.0
