"""
BridgeGuardian AI — Prometheus Metrics & Monitoring Engine
Exposes enterprise Prometheus metrics for HTTP requests, ML inference latency, prediction counts, and errors using pure ASGI middleware.
"""
from __future__ import annotations

import time
import logging
from typing import Dict, Any
from fastapi import Response

logger = logging.getLogger("bridgeguardian.telemetry")

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    HAS_PROMETHEUS = True

    HTTP_REQUESTS_TOTAL = Counter(
        "bridgeguardian_http_requests_total",
        "Total HTTP requests handled by BridgeGuardian API",
        ["method", "endpoint", "status"]
    )
    HTTP_ERRORS_TOTAL = Counter(
        "bridgeguardian_http_errors_total",
        "Total HTTP error responses returned",
        ["method", "endpoint", "status"]
    )
    ML_PREDICTIONS_TOTAL = Counter(
        "bridgeguardian_ml_predictions_total",
        "Total machine learning predictions executed",
        ["model_type"]
    )
    MODEL_LATENCY_SECONDS = Histogram(
        "bridgeguardian_model_latency_seconds",
        "Machine learning model inference latency in seconds",
        ["model_type"]
    )
    ACTIVE_WEBSOCKET_CONNECTIONS = Gauge(
        "bridgeguardian_active_websockets",
        "Number of active WebSocket real-time monitor connections"
    )

except Exception as e:
    HAS_PROMETHEUS = False
    logger.warning(f"Prometheus client setup warning: {e}")

_METRICS_STORE: Dict[str, Any] = {
    "http_requests_total": 0,
    "http_errors_total": 0,
    "ml_predictions_total": 0,
    "latency_samples": [],
}


class MetricsMiddleware:
    """Pure ASGI Middleware tracking HTTP request counts, status codes, and latencies."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.time()
        _METRICS_STORE["http_requests_total"] += 1
        method = scope.get("method", "GET")
        endpoint = scope.get("path", "/")

        async def send_with_metrics(message):
            if message["type"] == "http.response.start":
                duration = time.time() - start_time
                _METRICS_STORE["latency_samples"].append(round(duration, 4))
                if len(_METRICS_STORE["latency_samples"]) > 1000:
                    _METRICS_STORE["latency_samples"] = _METRICS_STORE["latency_samples"][-1000:]

                status_code = message.get("status", 200)
                status_str = str(status_code)

                if status_code >= 400:
                    _METRICS_STORE["http_errors_total"] += 1
                    if HAS_PROMETHEUS:
                        HTTP_ERRORS_TOTAL.labels(method=method, endpoint=endpoint, status=status_str).inc()

                if HAS_PROMETHEUS:
                    HTTP_REQUESTS_TOTAL.labels(method=method, endpoint=endpoint, status=status_str).inc()

                headers = list(message.get("headers", []))
                headers.append((b"x-response-time", f"{duration * 1000:.2f}ms".encode("utf-8")))
                message["headers"] = headers

            await send(message)

        try:
            await self.app(scope, receive, send_with_metrics)
        except Exception as exc:
            _METRICS_STORE["http_errors_total"] += 1
            if HAS_PROMETHEUS:
                HTTP_ERRORS_TOTAL.labels(method=method, endpoint=endpoint, status="500").inc()
            raise exc


def record_prediction(model_type: str, duration_seconds: float):
    """Record an ML prediction execution event."""
    _METRICS_STORE["ml_predictions_total"] += 1
    if HAS_PROMETHEUS:
        ML_PREDICTIONS_TOTAL.labels(model_type=model_type).inc()
        MODEL_LATENCY_SECONDS.labels(model_type=model_type).observe(duration_seconds)


def get_prometheus_metrics_raw() -> Response:
    """Return raw Prometheus format metrics payload."""
    if HAS_PROMETHEUS:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    samples = _METRICS_STORE["latency_samples"]
    avg_lat = round(sum(samples) / max(len(samples), 1), 4) if samples else 0.0
    text = (
        f"# HELP bridgeguardian_http_requests_total Total HTTP requests\n"
        f"bridgeguardian_http_requests_total {_METRICS_STORE['http_requests_total']}\n"
        f"# HELP bridgeguardian_http_errors_total Total HTTP errors\n"
        f"bridgeguardian_http_errors_total {_METRICS_STORE['http_errors_total']}\n"
        f"# HELP bridgeguardian_ml_predictions_total Total predictions\n"
        f"bridgeguardian_ml_predictions_total {_METRICS_STORE['ml_predictions_total']}\n"
        f"bridgeguardian_avg_latency_seconds {avg_lat}\n"
    )
    return Response(content=text, media_type="text/plain")
