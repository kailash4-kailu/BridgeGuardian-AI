"""
BridgeGuardian AI — Unit Tests: FeatureEngineer
Tests domain structural feature calculations and transformations.
"""
from __future__ import annotations

import pandas as pd
import pytest
from backend.ml.feature_engineer import FeatureEngineer


@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {
            "Strain_microstrain": 500.0,
            "Vibration_ms2": 2.0,
            "Temperature_C": 25.0,
            "Traffic_Volume_vph": 1000.0,
            "Tilt_deg": 0.5,
            "Deflection_mm": 10.0,
        }
    ])


def test_feature_engineer_transform(sample_df):
    """Verify that domain interaction features are correctly appended to the DataFrame."""
    config = {"training": {"rolling_windows": [5], "lag_features": [1]}}
    fe = FeatureEngineer(config=config)
    df_transformed = fe.transform(sample_df)

    assert len(df_transformed) == len(sample_df)
    assert "Strain_microstrain" in df_transformed.columns
    assert "Vibration_ms2" in df_transformed.columns
