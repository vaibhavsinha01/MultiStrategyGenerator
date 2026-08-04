"""Best-effort Redis cache used by the web application.

The cache is deliberately optional: a Redis outage must never make a strategy
or download unavailable.
"""
from __future__ import annotations

import json
import os
from typing import Any

try:
    import redis
except ImportError:  # pragma: no cover - dependency is optional at import time
    redis = None


class RedisCache:
    def __init__(self) -> None:
        self._client = None
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        if redis is None or not url:
            return
        try:
            client = redis.Redis.from_url(url, socket_connect_timeout=0.25, socket_timeout=0.5)
            client.ping()
            self._client = client
        except Exception:
            self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def get_json(self, key: str) -> dict[str, Any] | None:
        try:
            raw = self._client.get(key) if self._client else None
            return json.loads(raw) if raw else None
        except Exception:
            return None

    def set_json(self, key: str, value: dict[str, Any], ttl: int) -> None:
        try:
            if self._client:
                self._client.setex(key, ttl, json.dumps(value, default=str))
        except Exception:
            pass

    def get_bytes(self, key: str) -> bytes | None:
        try:
            return self._client.get(key) if self._client else None
        except Exception:
            return None

    def set_bytes(self, key: str, value: bytes, ttl: int) -> None:
        try:
            if self._client:
                self._client.setex(key, ttl, value)
        except Exception:
            pass

    def delete_prefix(self, prefix: str) -> None:
        try:
            if self._client:
                keys = list(self._client.scan_iter(f"{prefix}*"))
                if keys:
                    self._client.delete(*keys)
        except Exception:
            pass


cache = RedisCache()
