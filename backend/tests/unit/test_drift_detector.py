"""
BridgeGuardian AI — Unit Tests: DataDriftDetector
Tests Kolmogorov-Smirnov statistical feature drift calculations and threshold detection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.app.ml.drift_detector import DataDriftDetector


def test_drift_detector_no_drift():
    """Verify that identical distributions yield no drift detection."""
    np.random.seed(42)
    baseline_df = pd.DataFrame({
        "Strain_microstrain": np.random.normal(700, 50, 100),
        "Vibration_ms2": np.random.normal(1.2, 0.1, 100),
    })

    prod_df = pd.DataFrame({
        "Strain_microstrain": np.random.normal(700, 50, 100),
        "Vibration_ms2": np.random.normal(1.2, 0.1, 100),
    })

    detector = DataDriftDetector(baseline_data=baseline_df)
    results = detector.compute_feature_drift(prod_df)

    assert results["dataset_drift_detected"] is False
    assert results["drifted_features_count"] == 0
    assert results["total_features_evaluated"] == 2


def test_drift_detector_significant_drift():
    """Verify that shifted distributions trigger drift detection."""
    np.random.seed(42)
    baseline_df = pd.DataFrame({
        "Strain_microstrain": np.random.normal(700, 50, 100),
        "Vibration_ms2": np.random.normal(1.2, 0.1, 100),
    })

    # Shifted production data
    prod_df = pd.DataFrame({
        "Strain_microstrain": np.random.normal(950, 50, 100),
        "Vibration_ms2": np.random.normal(3.5, 0.1, 100),
    })

    detector = DataDriftDetector(baseline_data=baseline_df)
    results = detector.compute_feature_drift(prod_df)

    assert results["dataset_drift_detected"] is True
    assert results["drifted_features_count"] == 2
    assert results["feature_metrics"]["Strain_microstrain"]["drift_detected"] is True
