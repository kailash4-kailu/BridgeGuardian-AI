"""
BridgeGuardian AI — Rate Limiting Middleware
Enforces sliding window rate limits per client IP address using pure ASGI middleware to prevent stream deadlocks.
"""
from __future__ import annotations

import time
from typing import Dict, List
from fastapi import status
from fastapi.responses import JSONResponse

from backend.core.config import get_settings

settings = get_settings()

_IP_REQUEST_LOGS: Dict[str, List[float]] = {}


class RateLimiterMiddleware:
    """Pure ASGI middleware enforcing sliding window rate limit per client IP address."""

    def __init__(self, app, max_requests: int = 120, window_seconds: int = 60):
        self.app = app
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope.get("path", "")
            method = scope.get("method", "")
            if method == "OPTIONS" or path in ["/metrics", "/health", "/docs", "/openapi.json"] or path.startswith("/static/"):
                await self.app(scope, receive, send)
                return

            client = scope.get("client")
            client_ip = client[0] if client else "127.0.0.1"
            now = time.time()

            if client_ip not in _IP_REQUEST_LOGS:
                _IP_REQUEST_LOGS[client_ip] = []

            window_start = now - self.window_seconds
            _IP_REQUEST_LOGS[client_ip] = [t for t in _IP_REQUEST_LOGS[client_ip] if t > window_start]

            if len(_IP_REQUEST_LOGS[client_ip]) >= self.max_requests:
                response = JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "detail": "Rate limit exceeded. Please wait before sending more requests.",
                        "error": "Too Many Requests",
                        "max_requests": self.max_requests,
                        "window_seconds": self.window_seconds,
                    },
                    headers={"Retry-After": str(self.window_seconds)},
                )
                await response(scope, receive, send)
                return

            _IP_REQUEST_LOGS[client_ip].append(now)

        await self.app(scope, receive, send)
