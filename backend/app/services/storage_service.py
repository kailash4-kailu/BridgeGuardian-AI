"""
BridgeGuardian AI — Cloud & Local Image Storage Service
Abstracts image storage across Local Static Filesystem, Cloudinary, and AWS S3.
Stores public access URLs and metadata in PostgreSQL database records.
"""
from __future__ import annotations

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

from backend.core.config import get_settings

logger = logging.getLogger("bridgeguardian.storage")
settings = get_settings()


class StorageService:
    """Enterprise Storage Service handling Cloudinary, AWS S3, and Local fallback uploads."""

    @staticmethod
    def upload_image(file_bytes: bytes, filename: str, folder: str = "drone_inspections") -> Tuple[str, str]:
        """
        Uploads image file bytes to configured storage provider.
        Returns tuple of (storage_key_or_path, public_url).
        """
        provider = settings.storage_provider.lower()

        # Cloudinary Storage Provider
        if provider == "cloudinary" and settings.cloudinary_url:
            try:
                import cloudinary
                import cloudinary.uploader
                cloudinary.config(cloudinary_url=settings.cloudinary_url)
                res = cloudinary.uploader.upload(file_bytes, folder=folder, public_id=Path(filename).stem)
                return res.get("public_id"), res.get("secure_url")
            except Exception as e:
                logger.warning(f"Cloudinary upload failed ({e}). Falling back to local storage.")

        # AWS S3 Storage Provider
        elif provider == "s3" and settings.aws_s3_bucket:
            try:
                import boto3
                s3_client = boto3.client(
                    "s3",
                    aws_access_key_id=settings.aws_access_key_id,
                    aws_secret_access_key=settings.aws_secret_access_key,
                    region_name=settings.aws_region,
                )
                s3_key = f"{folder}/{filename}"
                s3_client.put_object(
                    Bucket=settings.aws_s3_bucket,
                    Key=s3_key,
                    Body=file_bytes,
                    ContentType="image/jpeg",
                )
                url = f"https://{settings.aws_s3_bucket}.s3.{settings.aws_region}.amazonaws.com/{s3_key}"
                return s3_key, url
            except Exception as e:
                logger.warning(f"AWS S3 upload failed ({e}). Falling back to local storage.")

        # Local Static Storage Provider (Default Fallback)
        local_dir = Path(settings.upload_dir) / folder
        local_dir.mkdir(parents=True, exist_ok=True)
        file_path = local_dir / filename
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        rel_url = f"/static/uploads/{folder}/{filename}"
        return str(file_path), rel_url

    @staticmethod
    def get_public_url(storage_key_or_path: str) -> str:
        """Resolves public URL for stored image asset."""
        if storage_key_or_path.startswith("http://") or storage_key_or_path.startswith("https://"):
            return storage_key_or_path
        if storage_key_or_path.startswith("/static/"):
            return storage_key_or_path
        return f"/static/{storage_key_or_path.lstrip('/')}"
