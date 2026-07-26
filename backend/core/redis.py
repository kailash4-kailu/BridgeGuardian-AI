"""
BridgeGuardian AI — Redis Client Integration
Async Redis connection manager with health check and memory cache fallback.
"""
from __future__ import annotations

import logging
from typing import Optional, Any
import json

from backend.core.config import get_settings

logger = logging.getLogger("bridgeguardian.redis")
settings = get_settings()

_redis_client = None
_local_cache = {}


async def get_redis_client():
    """
    Get or initialize async Redis client connection.
    Gracefully falls back if Redis server is unavailable.
    """
    global _redis_client
    if not settings.redis_enabled:
        return None

    if _redis_client is None:
        try:
            import redis.asyncio as aioredis
            client = aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=3.0,
            )
            await client.ping()
            _redis_client = client
            logger.info("Connected to Redis server successfully.")
        except Exception as e:
            logger.warning(f"Redis connection unavailable ({e}). Using in-memory fallback cache.")
            _redis_client = False  # Mark as unavailable

    return _redis_client if _redis_client is not False else None


class CacheService:
    """Enterprise caching service with Redis or memory fallback."""

    @staticmethod
    async def get(key: str) -> Optional[Any]:
        """Fetch cached item by key."""
        client = await get_redis_client()
        if client:
            try:
                val = await client.get(key)
                return json.loads(val) if val else None
            except Exception as e:
                logger.warning(f"Redis GET failed for key {key}: {e}")

        # Memory fallback
        return _local_cache.get(key)

    @staticmethod
    async def set(key: str, value: Any, ttl: int = 300) -> bool:
        """Store item in cache with TTL."""
        client = await get_redis_client()
        val_str = json.dumps(value)
        if client:
            try:
                await client.setex(key, ttl, val_str)
                return True
            except Exception as e:
                logger.warning(f"Redis SET failed for key {key}: {e}")

        # Memory fallback
        _local_cache[key] = value
        return True

    @staticmethod
    async def delete(key: str) -> bool:
        """Delete key from cache."""
        client = await get_redis_client()
        if client:
            try:
                await client.delete(key)
            except Exception as e:
                logger.warning(f"Redis DELETE failed for key {key}: {e}")

        _local_cache.pop(key, None)
        return True
