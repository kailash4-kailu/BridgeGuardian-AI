"""
BridgeGuardian AI — Integration Tests: Authentication API
Tests /auth/register, /auth/login, and /auth/me endpoints using TestClient.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.main import app

client = TestClient(app)


def test_auth_registration_and_login_flow():
    """Test full user lifecycle: Registration -> Login -> Access Profile."""
    unique_id = uuid.uuid4().hex[:8]
    email = f"engineer_{unique_id}@bridgeguardian.ai"
    password = "SecurePassword123!"

    # 1. Registration
    reg_response = client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": password,
            "full_name": "Test Structural Engineer",
            "role": "structural_engineer",
        },
    )
    assert reg_response.status_code == 201
    reg_data = reg_response.json()
    assert reg_data["email"] == email
    assert reg_data["role"] == "structural_engineer"

    # 2. Login
    login_response = client.post(
        "/api/v1/auth/login",
        data={"username": email, "password": password},
    )
    assert login_response.status_code == 200
    token_data = login_response.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "bearer"
    token = token_data["access_token"]

    # 3. Access Protected /auth/me
    me_response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert me_response.status_code == 200
    me_data = me_response.json()
    assert me_data["email"] == email
    assert me_data["role"] == "structural_engineer"


def test_auth_login_invalid_credentials():
    """Verify login failure on bad password."""
    response = client.post(
        "/api/v1/auth/login",
        data={"username": "nonexistent@bridgeguardian.ai", "password": "WrongPassword!"},
    )
    assert response.status_code == 401
