"""
BridgeGuardian AI — Unit Tests for ML Pipeline
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.ml.data_validator import DataValidator
from backend.ml.feature_engineer import FeatureEngineer
from backend.ml.preprocessor import Preprocessor
from backend.ml.rul_estimator import RULEstimator


# ─────────── Fixtures ──────────────────────────────────────────────────── #

@pytest.fixture
def sample_config():
    return {
        "dataset": {"timestamp_col": "Timestamp", "path": "dataset/bridge dataset.csv"},
        "features": {
            "structural": ["Strain_microstrain", "Deflection_mm"],
            "categorical": ["Bridge_Mood_Meter"],
        },
        "targets": {
            "health_score": "Structural_Health_Index_SHI",
            "failure_prob": "Probability_of_Failure_PoF",
            "maintenance_alert": "Maintenance_Alert",
        },
        "thresholds": {"shi_critical": 0.40},
        "ml": {"random_state": 42, "shap_max_samples": 50},
        "training": {"rolling_windows": [5], "lag_features": [1]},
    }


@pytest.fixture
def sample_df():
    """Minimal bridge sensor DataFrame for testing."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "Timestamp": pd.date_range("2023-01-01", periods=n, freq="1min").astype(str),
        "Strain_microstrain": np.random.uniform(400, 1200, n),
        "Deflection_mm": np.random.uniform(8, 25, n),
        "Vibration_ms2": np.random.uniform(0.5, 2.5, n),
        "Tilt_deg": np.random.uniform(0.5, 1.2, n),
        "Displacement_mm": np.random.uniform(19, 27, n),
        "Crack_Propagation_mm": np.random.uniform(0, 0.05, n),
        "Corrosion_Level_percent": np.random.uniform(0, 0.3, n),
        "Cable_Member_Tension_kN": np.random.uniform(350, 800, n),
        "Bearing_Joint_Forces_kN": np.random.uniform(210, 420, n),
        "Fatigue_Accumulation_au": np.random.uniform(0, 0.6, n),
        "Modal_Frequency_Hz": np.random.uniform(1.4, 2.2, n),
        "Temperature_C": np.random.uniform(-5, 35, n),
        "Humidity_percent": np.random.uniform(40, 95, n),
        "Wind_Speed_ms": np.random.uniform(2, 15, n),
        "Wind_Direction_deg": np.random.uniform(0, 360, n),
        "Precipitation_mmh": np.random.uniform(0, 5, n),
        "Water_Level_m": np.random.uniform(1.5, 3.5, n),
        "Seismic_Activity_ms2": np.random.uniform(0, 0.01, n),
        "Solar_Radiation_Wm2": np.random.uniform(0, 900, n),
        "Air_Quality_Index_AQI": np.random.uniform(25, 80, n),
        "Soil_Settlement_mm": np.random.uniform(0, 0.5, n),
        "Vehicle_Load_tons": np.random.uniform(6, 50, n),
        "Traffic_Volume_vph": np.random.uniform(300, 2000, n),
        "Pedestrian_Load_pph": np.random.uniform(60, 400, n),
        "Impact_Events_g": np.random.uniform(0, 0.01, n),
        "Dynamic_Load_Distribution_percent": np.random.uniform(71, 108, n),
        "Axle_Counts_pmin": np.random.uniform(16, 105, n),
        "Anomaly_Detection_Score": np.random.choice([0.0, 1.0], n, p=[0.99, 0.01]),
        "Energy_Dissipation_au": np.random.uniform(0.02, 22, n),
        "Acoustic_Emissions_levels": np.random.uniform(0.5, 28, n),
        "Visual_Analysis_Defect_Score": np.random.uniform(0, 0.1, n),
        "Electrical_Resistance_ohms": np.random.uniform(0.22, 0.35, n),
        "Bridge_Mood_Meter": np.random.choice(["Healthy", "Stressed", "Critical"], n, p=[0.85, 0.12, 0.03]),
        "Localized_Strain_Hotspot": np.random.choice([0.0, 1.0], n, p=[0.98, 0.02]),
        "Vibration_Anomaly_Location": np.random.choice([np.nan, "Deck", "Cables", "Piers"], n, p=[0.8, 0.1, 0.05, 0.05]),
        "Structural_Health_Index_SHI": np.random.uniform(0.35, 0.91, n),
        "Probability_of_Failure_PoF": np.random.uniform(0.02, 0.65, n),
        "Maintenance_Alert": np.random.choice([0.0, 1.0], n, p=[0.99, 0.01]),
        "SHI_Predicted_7d_Ahead": np.random.uniform(0.30, 0.89, n),
        "SHI_Predicted_30d_Ahead": np.random.uniform(0.25, 0.87, n),
        "Flood_Event_Flag": np.zeros(n),
        "High_Winds_Storms": np.zeros(n),
        "Landslide_Ground_Movement": np.zeros(n),
        "Abnormal_Traffic_Load_Surges": np.zeros(n),
        "Estimated_Repair_Cost_USD_incremental": np.random.uniform(1, 5000, n),
        "Carbon_Footprint_tCO2e_incremental": np.random.uniform(0.001, 0.004, n),
    })


# ─────────── DataValidator Tests ───────────────────────────────────────── #

class TestDataValidator:
    def test_valid_dataset_passes(self, sample_df, sample_config):
        validator = DataValidator(sample_config)
        report = validator.validate(sample_df)
        assert report.is_valid, f"Expected valid: {report.errors}"

    def test_missing_required_columns_fails(self, sample_config):
        bad_df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        validator = DataValidator(sample_config)
        report = validator.validate(bad_df)
        assert not report.is_valid
        assert any("Missing required columns" in e for e in report.errors)

    def test_row_count_check(self, sample_config):
        tiny_df = pd.DataFrame({
            "Timestamp": ["2023-01-01"],
            "Strain_microstrain": [700.0],
            "Deflection_mm": [14.0],
            "Structural_Health_Index_SHI": [0.8],
            "Probability_of_Failure_PoF": [0.03],
        })
        validator = DataValidator(sample_config)
        report = validator.validate(tiny_df)
        assert not report.is_valid

    def test_stats_populated(self, sample_df, sample_config):
        validator = DataValidator(sample_config)
        report = validator.validate(sample_df)
        assert "total_rows" in report.stats
        assert report.stats["total_rows"] == len(sample_df)


# ─────────── FeatureEngineer Tests ─────────────────────────────────────── #

class TestFeatureEngineer:
    def test_temporal_features_added(self, sample_df, sample_config):
        fe = FeatureEngineer(sample_config)
        result = fe.fit_transform(sample_df)
        assert "hour_of_day" in result.columns
        assert "hour_sin" in result.columns
        assert "is_weekend" in result.columns

    def test_rolling_features_added(self, sample_df, sample_config):
        fe = FeatureEngineer(sample_config)
        result = fe.fit_transform(sample_df)
        assert "Structural_Health_Index_SHI_roll_mean_5" in result.columns

    def test_lag_features_added(self, sample_df, sample_config):
        fe = FeatureEngineer(sample_config)
        result = fe.fit_transform(sample_df)
        assert "Structural_Health_Index_SHI_lag_1" in result.columns

    def test_composite_features_added(self, sample_df, sample_config):
        fe = FeatureEngineer(sample_config)
        result = fe.fit_transform(sample_df)
        assert "composite_stress_index" in result.columns
        assert "composite_env_index" in result.columns

    def test_shape_increases(self, sample_df, sample_config):
        fe = FeatureEngineer(sample_config)
        result = fe.fit_transform(sample_df)
        assert result.shape[1] > sample_df.shape[1]


# ─────────── Preprocessor Tests ────────────────────────────────────────── #

class TestPreprocessor:
    def test_removes_timestamp_column(self, sample_df, sample_config):
        preprocessor = Preprocessor(sample_config)
        result = preprocessor.fit_transform(sample_df)
        assert "Timestamp" not in result.columns

    def test_handles_missing_values(self, sample_df, sample_config):
        sample_df.loc[0:10, "Strain_microstrain"] = np.nan
        preprocessor = Preprocessor(sample_config)
        result = preprocessor.fit_transform(sample_df)
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        assert result[numeric_cols].isnull().sum().sum() == 0

    def test_encodes_categoricals(self, sample_df, sample_config):
        preprocessor = Preprocessor(sample_config)
        result = preprocessor.fit_transform(sample_df)
        for col in result.columns:
            assert result[col].dtype != object, f"Column '{col}' still object dtype"

    def test_fit_transform_then_transform_consistent(self, sample_df, sample_config):
        n = len(sample_df)
        train_df = sample_df.iloc[: n // 2].copy()
        test_df = sample_df.iloc[n // 2 :].copy()
        preprocessor = Preprocessor(sample_config)
        train_result = preprocessor.fit_transform(train_df)
        test_result = preprocessor.transform(test_df)
        assert train_result.shape[1] == test_result.shape[1]


# ─────────── RUL Estimator Tests ───────────────────────────────────────── #

class TestRULEstimator:
    def test_critical_shi_returns_zero_rul(self, sample_config):
        estimator = RULEstimator(sample_config)
        result = estimator.estimate(shi_current=0.30)
        assert result["rul_days"] == 0.0
        assert result["method"] == "threshold_breach"

    def test_healthy_shi_with_7d_forecast(self, sample_config):
        estimator = RULEstimator(sample_config)
        result = estimator.estimate(shi_current=0.85, shi_7d_ahead=0.83)
        assert result["rul_days"] > 0
        assert result["method"] == "7day_forecast"
        assert result["confidence"] == "high"

    def test_fallback_method(self, sample_config):
        estimator = RULEstimator(sample_config)
        result = estimator.estimate(shi_current=0.75)
        assert result["method"] == "default_rate"
        assert result["confidence"] == "low"

    def test_rul_capped_at_3650_days(self, sample_config):
        estimator = RULEstimator(sample_config)
        result = estimator.estimate(shi_current=0.90, shi_7d_ahead=0.8999)
        assert result["rul_days"] <= 3650.0

    def test_message_urgency_levels(self, sample_config):
        estimator = RULEstimator(sample_config)
        for shi, shi_7d, expected_keyword in [
            (0.42, 0.40, "URGENT"),
            (0.65, 0.55, "HIGH RISK"),
            (0.78, 0.73, "MODERATE"),
            (0.88, 0.85, "LOW RISK"),
        ]:
            result = estimator.estimate(shi_current=shi, shi_7d_ahead=shi_7d)
            assert expected_keyword in result["message"], (
                f"Expected '{expected_keyword}' in message for SHI={shi}"
            )
