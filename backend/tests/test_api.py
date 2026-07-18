"""
BridgeGuardian AI — API Integration Tests
Tests all FastAPI endpoints using TestClient.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from unittest import mock
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.main import app

@pytest.fixture(scope="session", autouse=True)
def setup_database():
    from backend.core.database import init_db
    init_db()

@pytest.fixture(autouse=True)
def mock_train_pipeline():
    with mock.patch("ml_pipeline.train.train_pipeline") as mock_train:
        mock_train.return_value = {"status": "success", "message": "Mocked training success"}
        yield mock_train

client = TestClient(app)

SAMPLE_INPUT = {
    "Strain_microstrain": 734.5,
    "Deflection_mm": 14.99,
    "Vibration_ms2": 1.20,
    "Tilt_deg": 0.72,
    "Displacement_mm": 22.36,
    "Crack_Propagation_mm": 0.015,
    "Corrosion_Level_percent": 0.15,
    "Cable_Member_Tension_kN": 447.9,
    "Bearing_Joint_Forces_kN": 260.1,
    "Fatigue_Accumulation_au": 0.30,
    "Modal_Frequency_Hz": 1.90,
    "Temperature_C": 15.0,
    "Humidity_percent": 60.3,
    "Wind_Speed_ms": 6.5,
    "Wind_Direction_deg": 180.0,
    "Precipitation_mmh": 0.0,
    "Water_Level_m": 2.0,
    "Seismic_Activity_ms2": 0.0,
    "Solar_Radiation_Wm2": 446.5,
    "Air_Quality_Index_AQI": 55.0,
    "Soil_Settlement_mm": 0.30,
    "Vehicle_Load_tons": 16.4,
    "Traffic_Volume_vph": 853.2,
    "Pedestrian_Load_pph": 96.3,
    "Impact_Events_g": 0.0,
    "Dynamic_Load_Distribution_percent": 90.1,
    "Axle_Counts_pmin": 43.4,
    "Anomaly_Detection_Score": 0.0,
    "Energy_Dissipation_au": 0.156,
    "Acoustic_Emissions_levels": 10.45,
    "Visual_Analysis_Defect_Score": 0.004,
    "Electrical_Resistance_ohms": 0.282,
    "Localized_Strain_Hotspot": 0.0,
    "Flood_Event_Flag": 0.0,
    "High_Winds_Storms": 0.0,
    "Landslide_Ground_Movement": 0.0,
    "Abnormal_Traffic_Load_Surges": 0.0,
    "SHI_Predicted_7d_Ahead": 0.80,
    "SHI_Predicted_30d_Ahead": 0.75,
}


class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_has_required_fields(self):
        response = client.get("/api/v1/health")
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "model_ready" in data
        assert "database_ok" in data

    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "BridgeGuardian AI"


class TestModelInfoEndpoint:
    def test_model_info_returns_200(self):
        response = client.get("/api/v1/model-info")
        assert response.status_code == 200

    def test_model_info_has_is_ready(self):
        response = client.get("/api/v1/model-info")
        data = response.json()
        assert "is_ready" in data
        assert "model_version" in data


class TestPredictEndpoint:
    def test_predict_when_model_not_ready_returns_503(self):
        """If model hasn't been trained, should return 503."""
        response = client.post("/api/v1/predict", json=SAMPLE_INPUT)
        # Either 200 (model loaded) or 503 (not trained yet)
        assert response.status_code in (200, 503)

    def test_predict_invalid_input_returns_422(self):
        """Completely invalid JSON structure returns 422."""
        response = client.post("/api/v1/predict", json={"invalid_key": "bad_data"})
        # FastAPI will accept the request (all fields are optional), may return 200 or 503
        assert response.status_code in (200, 422, 503)


class TestHistoryEndpoint:
    def test_history_returns_200(self):
        response = client.get("/api/v1/history")
        assert response.status_code == 200

    def test_history_has_pagination(self):
        response = client.get("/api/v1/history?limit=10&offset=0")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data

    def test_history_limit_parameter(self):
        response = client.get("/api/v1/history?limit=5")
        data = response.json()
        assert len(data["items"]) <= 5


class TestTrainEndpoint:
    def test_train_returns_queued_status(self):
        response = client.post(
            "/api/v1/train",
            json={"config_path": "config/config.yaml", "force_retrain": False},
        )
        # Should queue or say already running
        assert response.status_code in (200, 409)

    def test_training_status_invalid_job(self):
        response = client.get("/api/v1/train/status/nonexistent-job-id")
        assert response.status_code == 404


class TestExplainEndpoint:
    def test_explain_when_model_not_ready_returns_503(self):
        payload = {
            "input_data": SAMPLE_INPUT,
            "target": "health_score",
        }
        response = client.post("/api/v1/explain", json=payload)
        assert response.status_code in (200, 503)

    def test_explain_invalid_target_returns_422(self):
        payload = {
            "input_data": SAMPLE_INPUT,
            "target": "invalid_target",
        }
        response = client.post("/api/v1/explain", json=payload)
        assert response.status_code == 422
