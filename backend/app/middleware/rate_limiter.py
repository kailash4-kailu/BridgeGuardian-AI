"""
BridgeGuardian AI — Rate Limiting Middleware
Enforces sliding window rate limits per client IP address.
"""
from __future__ import annotations

import time
from typing import Dict, List
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from backend.core.config import get_settings

settings = get_settings()

_IP_REQUEST_LOGS: Dict[str, List[float]] = {}


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """Middleware enforcing sliding window rate limit per client IP address."""

    def __init__(self, app, max_requests: int = 120, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    async def dispatch(self, request: Request, call_next):
        # Exclude metrics, static files, and OPTIONS preflights from rate limiting
        if request.method == "OPTIONS" or request.url.path in ["/metrics", "/health", "/docs", "/openapi.json"] or request.url.path.startswith("/static/"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "127.0.0.1"
        now = time.time()

        if client_ip not in _IP_REQUEST_LOGS:
            _IP_REQUEST_LOGS[client_ip] = []

        # Remove expired timestamps outside window
        window_start = now - self.window_seconds
        _IP_REQUEST_LOGS[client_ip] = [t for t in _IP_REQUEST_LOGS[client_ip] if t > window_start]

        if len(_IP_REQUEST_LOGS[client_ip]) >= self.max_requests:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded. Please wait before sending more requests.",
                    "error": "Too Many Requests",
                    "max_requests": self.max_requests,
                    "window_seconds": self.window_seconds,
                },
                headers={"Retry-After": str(self.window_seconds)},
            )

        _IP_REQUEST_LOGS[client_ip].append(now)
        return await call_next(request)
