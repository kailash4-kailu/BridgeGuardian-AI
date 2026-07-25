"""
BridgeGuardian AI — Integration Tests: Health Probes & Security Headers
Tests /health/liveness, /health/readiness endpoints and HTTP security header injection.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.main import app

client = TestClient(app)


def test_liveness_probe_returns_200():
    """Verify that /health/liveness returns HTTP 200 alive."""
    response = client.get("/api/v1/health/liveness")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "alive"
    assert "timestamp" in data


def test_readiness_probe_returns_200_or_503():
    """Verify that /health/readiness returns 200 or 503 depending on model readiness."""
    response = client.get("/api/v1/health/readiness")
    assert response.status_code in (200, 503)


def test_security_headers_injection():
    """Verify that HTTP security headers are injected into response headers."""
    response = client.get("/")
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert "Strict-Transport-Security" in response.headers
