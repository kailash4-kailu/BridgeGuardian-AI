"""
BridgeGuardian AI — Celery Application Configuration
Configures Celery distributed background worker queues with Redis broker and result backend.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from backend.core.config import get_settings

logger = logging.getLogger("bridgeguardian.celery")
settings = get_settings()

try:
    from celery import Celery
    HAS_CELERY = True
    celery_app = Celery(
        "bridgeguardian",
        broker=settings.celery_broker_url,
        backend=settings.celery_result_backend,
        include=["backend.app.tasks.celery_tasks"],
    )
    celery_app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_time_limit=600,       # 10 minute hard execution limit
        task_soft_time_limit=500,  # 8.3 minute soft limit
    )
except Exception as exc:
    HAS_CELERY = False
    celery_app = None
    logger.warning(f"Celery setup notice: {exc}")


def dispatch_async_task(task_func: Any, *args, **kwargs) -> Dict[str, Any]:
    """
    Dispatches task asynchronously via Celery worker if available.
    Falls back gracefully to synchronous execution if Celery or Redis is unreachable.
    """
    if HAS_CELERY and celery_app is not None:
        try:
            async_result = task_func.delay(*args, **kwargs)
            return {
                "task_id": async_result.id,
                "status": "QUEUED",
                "mode": "celery",
            }
        except Exception as e:
            logger.warning(f"Celery dispatch failed ({e}). Falling back to synchronous execution.")

    # Synchronous execution fallback
    try:
        if hasattr(task_func, "run"):
            res = task_func.run(*args, **kwargs)
        else:
            res = task_func(*args, **kwargs)
        return {
            "task_id": "sync-fallback",
            "status": "COMPLETED",
            "result": res,
            "mode": "sync",
        }
    except Exception as err:
        logger.error(f"Task execution failed: {err}")
        return {
            "task_id": "sync-fallback",
            "status": "FAILED",
            "error": str(err),
            "mode": "sync",
        }
