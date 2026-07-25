"""
BridgeGuardian AI — Idempotency Middleware
Validates Idempotency-Key headers on state-modifying POST requests to prevent duplicate execution.
"""
from __future__ import annotations

import json
from typing import Callable, Dict, Any
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# In-memory idempotency cache fallback (if Redis is not active)
_IDEMPOTENCY_CACHE: Dict[str, Dict[str, Any]] = {}


class IdempotencyMiddleware(BaseHTTPMiddleware):
    """
    Middleware that checks for 'Idempotency-Key' in HTTP request headers.
    If the key has been processed within the expiration window, returns the cached response.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.method not in ("POST", "PUT", "PATCH"):
            return await call_next(request)

        idempotency_key = request.headers.get("Idempotency-Key")
        if not idempotency_key:
            return await call_next(request)

        cache_key = f"idempotency:{idempotency_key}"

        # Check cache
        if cache_key in _IDEMPOTENCY_CACHE:
            cached_data = _IDEMPOTENCY_CACHE[cache_key]
            return Response(
                content=cached_data["content"],
                status_code=cached_data["status_code"],
                headers={"X-Cache-Hit": "true", "Content-Type": "application/json"},
            )

        # Execute request
        response = await call_next(request)

        # Cache successful response body
        if 200 <= response.status_code < 300:
            response_body = [section async for section in response.body_iterator]
            response.body_iterator = iterate_in_threadpool(response_body)
            body_bytes = b"".join(response_body)

            _IDEMPOTENCY_CACHE[cache_key] = {
                "content": body_bytes.decode("utf-8"),
                "status_code": response.status_code,
            }

        return response


def iterate_in_threadpool(bytes_list):
    async def iterator():
        for b in bytes_list:
            yield b
    return iterator()
