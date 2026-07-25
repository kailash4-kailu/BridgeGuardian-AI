"""
BridgeGuardian AI — Integration Tests: Drone Campaigns API & GIS Heatmap
Tests campaign creation, heatmap generation, and component health timeline endpoints.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.main import app

client = TestClient(app)


def test_create_campaign_endpoint():
    """Verify POST /api/v1/campaigns/upload creates new campaign."""
    payload = {
        "name": "Golden Gate Bridge Inspection Block A",
        "bridge_id": "BRIDGE_SF_001",
        "total_images": 25,
    }
    response = client.post("/api/v1/campaigns/upload", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert "cmp_" in data["campaign_id"]
    assert data["name"] == payload["name"]
    assert data["status"] == "IN_PROGRESS"


def test_campaign_heatmap_endpoint():
    """Verify GET /api/v1/campaigns/{id}/heatmap returns spatial density grid."""
    # First create campaign
    res = client.post("/api/v1/campaigns/upload", json={"name": "Test Heatmap", "bridge_id": "B1"})
    cid = res.json()["campaign_id"]

    response = client.get(f"/api/v1/campaigns/{cid}/heatmap")
    assert response.status_code == 200
    data = response.json()
    assert data["campaign_id"] == cid
    assert "heatmap" in data
    assert data["heatmap"]["grid_size"] == 50


def test_campaign_timeline_endpoint():
    """Verify GET /api/v1/campaigns/{id}/timeline returns health records."""
    res = client.post("/api/v1/campaigns/upload", json={"name": "Test Timeline", "bridge_id": "B2"})
    cid = res.json()["campaign_id"]

    response = client.get(f"/api/v1/campaigns/{cid}/timeline")
    assert response.status_code == 200
    data = response.json()
    assert data["campaign_id"] == cid
    assert "component_health_timeline" in data
