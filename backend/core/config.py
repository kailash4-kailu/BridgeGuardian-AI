"""
BridgeGuardian AI — Application Configuration
Loads settings from environment variables and config YAML files.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Enterprise application settings with PostgreSQL, Redis, Celery, and Security defaults."""

    # Core Application Settings
    app_env: str = "development"
    secret_key: str = "bridgeguardian-super-secret-key-change-in-production"
    log_level: str = "INFO"
    models_dir: str = "models"
    config_path: str = "config"

    # Database Configuration (PostgreSQL with SQLite dev fallback)
    database_url: str = "sqlite:///./bridgeguardian.db"
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_recycle: int = 1800

    # Redis & Cache Settings
    redis_url: str = "redis://localhost:6379/0"
    redis_enabled: bool = True
    cache_ttl_seconds: int = 300

    # Celery Background Task Queue
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # JWT Authentication & Security
    jwt_secret_key: str = "bridgeguardian-jwt-secret-key-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 7
    rate_limit_per_minute: int = 120

    # Cloud Image Storage (Cloudinary / AWS S3 / Local fallback)
    storage_provider: str = "local"  # "local", "cloudinary", "s3"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-east-1"
    aws_s3_bucket: Optional[str] = None
    cloudinary_url: Optional[str] = None

    # Static Directories
    upload_dir: str = "backend/static/uploads"
    processed_dir: str = "backend/static/processed"
    reports_dir: str = "backend/static/reports"
    logs_dir: str = "logs"

    # CORS & Performance
    cors_origins: str = "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173,http://localhost:8000,http://127.0.0.1:8000,https://bridge-guardian-ai.vercel.app,*"
    max_file_size: int = 20971520   # 20 MB per single image
    max_upload_size: int = 524288000  # 500 MB total batch upload payload
    host: str = "0.0.0.0"
    port: int = 8000
    demo_mode: bool = True
    api_url: str = "http://localhost:8000"
    frontend_url: str = "http://localhost:3000"
    gunicorn_workers: int = 4

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_development(self) -> bool:
        return self.app_env.lower() == "development"


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()


@lru_cache()
def get_config() -> dict:
    """Load and cache YAML configuration files, merging them into a single dictionary."""
    settings = get_settings()
    config_dir = Path(settings.config_path)

    env_file = config_dir / f"{settings.app_env.lower()}.yaml"
    if not env_file.exists():
        env_file = config_dir / "development.yaml"
        if not env_file.exists():
            env_file = config_dir / "config.yaml"

    config_dict = {}
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if loaded:
                config_dict.update(loaded)

    for config_name in ["prediction", "vision", "report"]:
        cfg_file = config_dir / f"{config_name}.yaml"
        if cfg_file.exists():
            with open(cfg_file, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    config_dict.update(loaded)

    return config_dict
