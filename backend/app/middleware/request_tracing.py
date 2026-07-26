"""
BridgeGuardian AI — Request Tracing & Structured Logging Middleware
Injects unique X-Request-ID headers, tracks execution duration, and logs HTTP activity.
"""
from __future__ import annotations

import time
import uuid
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request

logger = logging.getLogger("bridgeguardian.request")


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Middleware attaching request_id, tracking response time, and emitting structured logs."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        start_time = time.perf_counter()

        response = await call_next(request)

        process_time_ms = (time.perf_counter() - start_time) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-MS"] = f"{process_time_ms:.2f}"

        logger.info(
            f"[{request_id[:8]}] {request.method} {request.url.path} "
            f"-> Status: {response.status_code} ({process_time_ms:.2f}ms)"
        )
        return response
