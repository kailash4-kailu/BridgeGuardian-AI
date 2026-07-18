"""
BridgeGuardian AI — FastAPI Application Entry Point
Production-grade API with CORS, lifespan management, and structured logging.
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
    logger.info("  BridgeGuardian AI - Starting up")
    logger.info("=" * 60)

    # Ensure all configurable directories exist
    for directory in [
        settings.upload_dir,
        settings.processed_dir,
        settings.reports_dir,
        settings.logs_dir,
        settings.models_dir,
    ]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    logger.info("Configured directories ensured on startup.")

    # Initialise database tables
    init_db()
    logger.info("Database initialised OK")

    # Try to load pre-trained models (may not exist on first run)
    try:
        inference_pipeline.load()
        logger.info(f"Inference pipeline loaded OK (version: {inference_pipeline._model_version})")
    except FileNotFoundError:
        logger.warning(
            "No trained models found. POST /train to train models before predicting."
        )
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")

    yield

    logger.info("BridgeGuardian AI - Shutting down gracefully")


# ── Limit Upload Size Middleware ────────────────────────────────────────── #
from starlette.middleware.base import BaseHTTPMiddleware

class LimitUploadSizeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_upload_size: int):
        super().__init__(app)
        self.max_upload_size = max_upload_size

    async def dispatch(self, request, call_next):
        if request.method == "POST":
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > self.max_upload_size:
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"Request entity too large. Max size is {self.max_upload_size} bytes."}
                )
        return await call_next(request)


# ── FastAPI Application ────────────────────────────────────────────────── #
def create_app() -> FastAPI:
    config = get_config()
    app_cfg = config.get("app", {})

    app = FastAPI(
        title=app_cfg.get("name", "BridgeGuardian AI"),
        description=app_cfg.get("description", "Explainable Predictive Maintenance for Bridge SHM"),
        version=app_cfg.get("version", "1.0.0"),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── Upload Limit Middleware ─────────────────────────────────────────── #
    app.add_middleware(LimitUploadSizeMiddleware, max_upload_size=settings.max_upload_size)

    # ── CORS ────────────────────────────────────────────────────────────── #
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ──────────────────────────────────────────────────────────── #
    app.include_router(api_router, prefix="/api/v1")

    # ── Static Files ────────────────────────────────────────────────────── #
    from fastapi.staticfiles import StaticFiles
    # Map the general static folder (or settings.upload_dir/etc.)
    # We mount "backend/static" so frontend can query images and pdfs using standard paths
    static_dir = Path("backend/static")
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory="backend/static"), name="static")

    # ── Root ────────────────────────────────────────────────────────────── #
    @app.get("/", tags=["System"])
    async def root():
        return {
            "name": "BridgeGuardian AI",
            "version": "1.0.0",
            "status": "running",
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    # ── Global exception handler ────────────────────────────────────────── #
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
