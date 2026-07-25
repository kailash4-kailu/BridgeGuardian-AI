"""
BridgeGuardian AI — Locust System Load Testing Suite
Simulates concurrent user behavior against prediction, drift status, and health check endpoints.
"""
from __future__ import annotations

import random
from locust import HttpUser, task, between


class BridgeGuardianLoadUser(HttpUser):
    """Simulates active civil engineer and automated sensor gateway traffic."""

    wait_time = between(0.5, 2.0)

    @task(3)
    def check_health(self):
        """Poll health check endpoint."""
        self.client.get("/api/v1/health")

    @task(2)
    def check_drift_status(self):
        """Poll ML data drift status endpoint."""
        self.client.get("/api/v1/ml/drift-status")

    @task(5)
    def submit_telemetry_prediction(self):
        """Submit synthetic telemetry prediction payload."""
        payload = {
            "Strain_microstrain": random.uniform(500.0, 950.0),
            "Deflection_mm": random.uniform(5.0, 25.0),
            "Vibration_ms2": random.uniform(0.5, 3.0),
            "Tilt_deg": random.uniform(0.1, 1.2),
            "Temperature_C": random.uniform(10.0, 35.0),
            "Traffic_Volume_vph": random.uniform(200.0, 1500.0),
        }
        self.client.post("/api/v1/predict", json=payload)
