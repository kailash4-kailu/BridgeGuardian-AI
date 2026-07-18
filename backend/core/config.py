"""
BridgeGuardian AI — Application Configuration
Loads settings from config.yaml and environment variables.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List

import yaml
from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables and config.yaml."""

    # Core
    app_env: str = "development"
    secret_key: str = "change-me-in-production"
    database_url: str = "sqlite:///./bridgeguardian.db"
    cors_origins: str = "http://localhost:3000,http://localhost:5173"
    log_level: str = "INFO"
    models_dir: str = "models"
    config_path: str = "config"
    
    # Static Directories
    upload_dir: str = "backend/static/uploads"
    processed_dir: str = "backend/static/processed"
    reports_dir: str = "backend/static/reports"
    logs_dir: str = "logs"
    
    # Performance & API limits
    max_upload_size: int = 10485760  # 10 MB in bytes
    host: str = "0.0.0.0"
    port: int = 8000
    demo_mode: bool = True
    api_url: str = "http://localhost:8000"
    frontend_url: str = "http://localhost:3000"
    gunicorn_workers: int = 4

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

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
    
    # Base configuration depending on environment (development.yaml or production.yaml)
    env_file = config_dir / f"{settings.app_env.lower()}.yaml"
    if not env_file.exists():
        # Fallback to development.yaml if env specific file is missing
        env_file = config_dir / "development.yaml"
        if not env_file.exists():
            env_file = config_dir / "config.yaml"  # fallback to original monolith if exists
            
    config_dict = {}
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if loaded:
                config_dict.update(loaded)
            
    # Load and merge sub-configurations
    for config_name in ["prediction", "vision", "report"]:
        cfg_file = config_dir / f"{config_name}.yaml"
        if cfg_file.exists():
            with open(cfg_file, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    config_dict.update(loaded)
                    
    return config_dict

