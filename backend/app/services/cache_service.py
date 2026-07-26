"""
BridgeGuardian AI — Cache Service Helpers
Provides caching abstractions for API routes, session management, and dashboard data.
"""
from __future__ import annotations

import functools
from typing import Any, Callable
from backend.core.redis import CacheService

cache_service = CacheService()


def cache_response(ttl: int = 300, key_prefix: str = "api_cache"):
    """
    Decorator for caching API route response payloads in Redis.
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate deterministic cache key from parameters
            args_str = ":".join(str(a) for a in args)
            kwargs_str = ":".join(f"{k}={v}" for k, v in sorted(kwargs.items()) if k not in ["db", "current_user"])
            cache_key = f"{key_prefix}:{func.__name__}:{args_str}:{kwargs_str}"

            cached_val = await CacheService.get(cache_key)
            if cached_val is not None:
                return cached_val

            result = await func(*args, **kwargs)
            if result is not None:
                await CacheService.set(cache_key, result, ttl=ttl)
            return result

        return wrapper

    return decorator
