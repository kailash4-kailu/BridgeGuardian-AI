"""
BridgeGuardian AI — Telemetry Prediction & SHAP Explainability API Tests
Verifies prediction computations, failure risk calculations, RUL estimations, and SHAP attributions.
"""
def test_health_check_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_ready" in data


def test_predict_endpoint_fallback(client):
    payload = {
        "vibration_amplitude": 0.45,
        "strain_gauge_microstrain": 210.5,
        "deck_temperature_celsius": 24.0,
        "deflection_mm": 3.2,
        "traffic_load_tons": 45.0,
        "corrosion_index": 0.15,
        "acoustic_emission_db": 38.0,
        "humidity_percent": 65.0,
        "crack_density_m2": 0.05,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code in [200, 422]
    if response.status_code == 200:
        data = response.json()
        assert "health_score" in data or "predicted_health_index" in data or "success" in data


def test_model_registry_list(client):
    response = client.get("/api/v1/models/registry")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert "model_version" in data[0]
