"""
BridgeGuardian AI — FastAPI Application Entry Point
Production-grade enterprise API with CORS, lifespan management, Prometheus metrics, and structured logging.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.router import api_router
from backend.core.config import get_config, get_settings
from backend.core.database import init_db
from backend.core.logging_config import setup_logging
from backend.ml.inference import InferencePipeline

# ── Logging ───────────────────────────────────────────────────────────── #
settings = get_settings()
logger = setup_logging(level=settings.log_level, name="bridgeguardian")

# ── Global inference pipeline (singleton) ─────────────────────────────── #
inference_pipeline = InferencePipeline(models_dir=settings.models_dir)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    logger.info("=" * 60)
    logger.info("  BridgeGuardian AI — Enterprise Platform Startup")
    logger.info("=" * 60)

    for directory in [
        settings.upload_dir,
        settings.processed_dir,
        settings.reports_dir,
        settings.logs_dir,
        settings.models_dir,
    ]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    logger.info("Configured storage directories verified.")

    init_db()

    try:
        inference_pipeline.load()
        logger.info(f"Inference pipeline loaded OK (version: {inference_pipeline._model_version})")
    except FileNotFoundError:
        logger.warning("No trained models found. Trigger POST /api/v1/train to initialize pipeline.")
    except Exception as e:
        logger.error(f"Failed to load inference pipeline: {e}")

    yield

    logger.info("BridgeGuardian AI — Shutting down gracefully")


# ── Upload Limit Middleware ─────────────────────────────────────────── #
from starlette.middleware.base import BaseHTTPMiddleware

class LimitUploadSizeMiddleware:
    def __init__(self, app, max_upload_size: int):
        self.app = app
        self.max_upload_size = max_upload_size

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("method") == "POST":
            headers = dict(scope.get("headers", []))
            content_length = headers.get(b"content-length")
            if content_length:
                try:
                    length = int(content_length.decode("latin1"))
                    if length > self.max_upload_size:
                        response = JSONResponse(
                            status_code=413,
                            content={"detail": f"Request entity too large. Max allowed is {self.max_upload_size} bytes."}
                        )
                        await response(scope, receive, send)
                        return
                except ValueError:
                    pass
        await self.app(scope, receive, send)


# ── FastAPI Application Builder ───────────────────────────────────────── #
def create_app() -> FastAPI:
    config = get_config()
    app_cfg = config.get("app", {})

    tags_metadata = [
        {"name": "System", "description": "Operational health check, readiness, liveness probes, and Prometheus metrics."},
        {"name": "Authentication", "description": "User registration, OAuth2 JWT token login, refresh, logout, and profile extraction."},
        {"name": "Model Registry", "description": "Production model tracking, accuracy metrics, and version rollbacks."},
        {"name": "Prediction", "description": "Tabular sensor health predictions, failure probabilities, and RUL estimations."},
        {"name": "Explainability", "description": "SHAP feature contribution attributions."},
        {"name": "Computer Vision", "description": "Drone imagery defect segmentation and morphological defect measurement."},
        {"name": "Inspection Campaigns", "description": "Batch drone campaign processing and ReportLab PDF compilation."},
        {"name": "ML Governance", "description": "Kolmogorov-Smirnov statistical data drift monitoring."},
        {"name": "Real-Time Streaming", "description": "WebSocket live multi-stage inspection progress streams."},
    ]

    app = FastAPI(
        title=app_cfg.get("name", "BridgeGuardian AI — Structural Health Platform"),
        description="Enterprise Predictive Maintenance & Explainable Structural Health Monitoring Platform API",
        version=app_cfg.get("version", "1.0.0"),
        terms_of_service="https://bridge-guardian-ai.vercel.app/terms",
        contact={
            "name": "BridgeGuardian Infrastructure Engineering",
            "url": "https://bridge-guardian-ai.vercel.app",
            "email": "support@bridgeguardian.ai",
        },
        license_info={"name": "MIT License"},
        openapi_tags=tags_metadata,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Exceptions & Middlewares
    from backend.app.middleware.security_headers import SecurityHeadersMiddleware
    from backend.app.middleware.request_tracing import RequestTracingMiddleware
    from backend.app.middleware.rate_limiter import RateLimiterMiddleware
    from backend.app.core.telemetry import MetricsMiddleware, get_prometheus_metrics_raw

    app.add_middleware(LimitUploadSizeMiddleware, max_upload_size=settings.max_upload_size)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestTracingMiddleware)
    app.add_middleware(RateLimiterMiddleware, max_requests=settings.rate_limit_per_minute)
    app.add_middleware(MetricsMiddleware)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Router
    app.include_router(api_router, prefix="/api/v1")

    # Static Assets
    from fastapi.staticfiles import StaticFiles
    static_dir = Path("backend/static")
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory="backend/static"), name="static")

    # Prometheus Metrics & System Endpoints
    @app.get("/metrics", tags=["System"])
    async def metrics():
        """Expose raw Prometheus telemetry metrics."""
        return get_prometheus_metrics_raw()

    @app.get("/", tags=["System"])
    async def root():
        return {
            "name": "BridgeGuardian AI",
            "version": "1.0.0",
            "status": "running",
            "docs": "/docs",
            "health": "/api/v1/health",
            "metrics": "/metrics",
        }

    # Global Exception Handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        err_msg = str(exc) if settings.is_development else "Internal server error"
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error": err_msg},
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.is_development,
        log_level=settings.log_level.lower(),
    )
