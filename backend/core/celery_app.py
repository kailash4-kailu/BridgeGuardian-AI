"""
BridgeGuardian AI — Celery Application Configuration
Configures Celery distributed worker queues with Redis message broker and fallback sync modes.
"""
from __future__ import annotations

import os
from typing import Any, Dict

# Redis connection URL from environment variable or default localhost fallback
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

try:
    from celery import Celery
    HAS_CELERY = True
    celery_app = Celery(
        "bridgeguardian",
        broker=REDIS_URL,
        backend=REDIS_URL,
        include=[
            "backend.app.tasks.vision_tasks",
            "backend.app.tasks.report_tasks",
        ],
    )
    celery_app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_time_limit=300,        # 5 minutes hard limit
        task_soft_time_limit=240,   # 4 minutes soft limit
    )
except ImportError:
    HAS_CELERY = False
    celery_app = None  # Fallback for environments without Celery installed


def dispatch_async_task(task_func: Any, *args, **kwargs) -> Dict[str, Any]:
    """
    Helper function to dispatch Celery tasks if available,
    falling back to synchronous execution if Celery or Redis is unavailable.
    """
    if HAS_CELERY and celery_app is not None:
        try:
            async_result = task_func.delay(*args, **kwargs)
            return {
                "task_id": async_result.id,
                "status": "QUEUED",
                "mode": "async",
            }
        except Exception:
            pass  # Fallback to sync execution on connection error

    # Synchronous execution fallback
    result = task_func(*args, **kwargs)
    return {
        "task_id": "sync-execution",
        "status": "COMPLETED",
        "result": result,
        "mode": "sync",
    }
