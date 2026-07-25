"""
BridgeGuardian AI — Unit Tests: RFC 7807 Exception Framework
Tests problem details payload generation and status code assignments.
"""
from __future__ import annotations

import pytest
from backend.app.core.exceptions import (
    BridgeGuardianException,
    ResourceNotFoundException,
    UnauthorizedException,
    ValidationException,
)


def test_base_exception_to_problem_details():
    """Verify base exception produces valid RFC 7807 JSON structure."""
    exc = BridgeGuardianException(detail="Database query timeout", status_code=504, title="Gateway Timeout")
    problem = exc.to_problem_details(request_path="/api/v1/predict")

    assert problem["status"] == 504
    assert problem["title"] == "Gateway Timeout"
    assert problem["detail"] == "Database query timeout"
    assert problem["instance"] == "/api/v1/predict"
    assert "timestamp" in problem


def test_resource_not_found_exception():
    """Verify ResourceNotFoundException produces 404 status code."""
    exc = ResourceNotFoundException(detail="Bridge inspection #999 not found")
    problem = exc.to_problem_details()

    assert problem["status"] == 404
    assert problem["title"] == "Resource Not Found"
    assert "not found" in problem["detail"]


def test_unauthorized_exception():
    """Verify UnauthorizedException produces 401 status code."""
    exc = UnauthorizedException()
    problem = exc.to_problem_details()

    assert problem["status"] == 401
    assert problem["title"] == "Unauthorized"


def test_validation_exception():
    """Verify ValidationException produces 422 status code."""
    exc = ValidationException(detail="Strain microstrain cannot be negative")
    problem = exc.to_problem_details()

    assert problem["status"] == 422
    assert problem["title"] == "Unprocessable Entity"
