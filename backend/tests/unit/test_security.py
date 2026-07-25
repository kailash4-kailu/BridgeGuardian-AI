"""
BridgeGuardian AI — Unit Tests: Security Module
Tests password hashing, Verification, and JWT access token encoding/decoding.
"""
from __future__ import annotations

import time
from datetime import timedelta
import pytest

from backend.app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_password_hashing_and_verification():
    """Verify that password hashing generates unique salts and verifies correctly."""
    password = "SuperSecretPassword123!"
    hashed = hash_password(password)

    assert hashed != password
    assert verify_password(password, hashed) is True
    assert verify_password("WrongPassword!", hashed) is False


def test_jwt_access_token_creation_and_decoding():
    """Verify that JWT tokens encode subject & role claims correctly."""
    user_id = 42
    role = "structural_engineer"
    token = create_access_token(subject=user_id, role=role)

    assert isinstance(token, str)
    payload = decode_access_token(token)

    assert payload is not None
    assert payload.get("sub") == "42"
    assert payload.get("role") == "structural_engineer"
    assert "exp" in payload


def test_jwt_access_token_expiration():
    """Verify that expired tokens fail decoding."""
    token = create_access_token(subject=1, role="field_inspector", expires_delta=timedelta(seconds=-10))
    payload = decode_access_token(token)
    assert payload is None
