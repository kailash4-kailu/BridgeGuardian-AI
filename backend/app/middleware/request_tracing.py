"""
BridgeGuardian AI — Request Tracing & Structured Logging Middleware
Injects unique X-Request-ID headers, tracks execution duration, and logs HTTP activity using pure ASGI.
"""
from __future__ import annotations

import time
import uuid
import logging

logger = logging.getLogger("bridgeguardian.request")


class RequestTracingMiddleware:
    """Pure ASGI Middleware attaching request_id, tracking response time, and emitting structured logs."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers_dict = dict(scope.get("headers", []))
        req_id_bytes = headers_dict.get(b"x-request-id", str(uuid.uuid4()).encode("utf-8"))
        req_id_str = req_id_bytes.decode("utf-8")
        start_time = time.perf_counter()

        async def send_with_tracing(message):
            if message["type"] == "http.response.start":
                process_time_ms = (time.perf_counter() - start_time) * 1000
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", req_id_str.encode("utf-8")))
                headers.append((b"x-response-time-ms", f"{process_time_ms:.2f}".encode("utf-8")))
                message["headers"] = headers
                status_code = message.get("status", 200)
                logger.info(
                    f"[{req_id_str[:8]}] {scope.get('method')} {scope.get('path')} "
                    f"-> Status: {status_code} ({process_time_ms:.2f}ms)"
                )
            await send(message)

        await self.app(scope, receive, send_with_tracing)
