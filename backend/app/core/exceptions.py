"""
BridgeGuardian AI — RFC 7807 Problem Details Exception Framework
Implements standardized IETF RFC 7807 error payloads across all API exception types.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional
from fastapi import Request
from fastapi.responses import JSONResponse


class BridgeGuardianException(Exception):
    """Base class for all domain-specific application exceptions."""

    def __init__(
        self,
        detail: str,
        status_code: int = 500,
        title: str = "Internal Server Error",
        type_url: str = "about:blank",
        instance: Optional[str] = None,
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code
        self.title = title
        self.type_url = type_url
        self.instance = instance

    def to_problem_details(self, request_path: Optional[str] = None) -> Dict[str, Any]:
        """Format exception as an RFC 7807 Problem Details JSON object."""
        return {
            "type": self.type_url,
            "title": self.title,
            "status": self.status_code,
            "detail": self.detail,
            "instance": request_path or self.instance or "about:blank",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }


class ResourceNotFoundException(BridgeGuardianException):
    def __init__(self, detail: str = "Requested resource was not found") -> None:
        super().__init__(
            detail=detail,
            status_code=404,
            title="Resource Not Found",
            type_url="https://bridgeguardian.ai/errors/not-found",
        )


class UnauthorizedException(BridgeGuardianException):
    def __init__(self, detail: str = "Authentication credentials missing or invalid") -> None:
        super().__init__(
            detail=detail,
            status_code=401,
            title="Unauthorized",
            type_url="https://bridgeguardian.ai/errors/unauthorized",
        )


class ValidationException(BridgeGuardianException):
    def __init__(self, detail: str = "Invalid input request payload") -> None:
        super().__init__(
            detail=detail,
            status_code=422,
            title="Unprocessable Entity",
            type_url="https://bridgeguardian.ai/errors/validation-error",
        )


async def rfc7807_exception_handler(request: Request, exc: BridgeGuardianException) -> JSONResponse:
    """FastAPI global exception handler mapping BridgeGuardianException to RFC 7807 JSONResponse."""
    payload = exc.to_problem_details(request_path=request.url.path)
    return JSONResponse(
        status_code=exc.status_code,
        content=payload,
        headers={"Content-Type": "application/problem+json"},
    )
