"""
BridgeGuardian AI — Prometheus & Telemetry Metrics Module
Tracks HTTP request counters, latency distributions, and ML inference timing metrics.
"""
from __future__ import annotations

import time
from typing import Callable, Dict
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# In-memory Prometheus metric counter fallbacks
_METRICS_STORE: Dict[str, Any] = {
    "http_requests_total": 0,
    "http_errors_total": 0,
    "ml_predictions_total": 0,
    "latency_samples": [],
}


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware that measures HTTP request latency and increments request metrics."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        _METRICS_STORE["http_requests_total"] += 1

        try:
            response = await call_next(request)
            duration = time.time() - start_time
            _METRICS_STORE["latency_samples"].append(round(duration, 4))
            
            # Keep sample size bounded
            if len(_METRICS_STORE["latency_samples"]) > 1000:
                _METRICS_STORE["latency_samples"] = _METRICS_STORE["latency_samples"][-1000:]

            if response.status_code >= 400:
                _METRICS_STORE["http_errors_total"] += 1

            response.headers["X-Response-Time"] = f"{duration * 1000:.2f}ms"
            return response
        except Exception as exc:
            _METRICS_STORE["http_errors_total"] += 1
            raise exc


def get_prometheus_metrics() -> Dict[str, Any]:
    """Return aggregated metric snapshot."""
    samples = _METRICS_STORE["latency_samples"]
    avg_latency = round(sum(samples) / max(len(samples), 1), 4) if samples else 0.0
    return {
        "http_requests_total": _METRICS_STORE["http_requests_total"],
        "http_errors_total": _METRICS_STORE["http_errors_total"],
        "ml_predictions_total": _METRICS_STORE["ml_predictions_total"],
        "avg_latency_seconds": avg_latency,
    }
