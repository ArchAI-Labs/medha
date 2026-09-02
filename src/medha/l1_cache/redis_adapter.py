"""Redis-backed L1 cache for distributed deployments.

Requires: ``pip install medha[redis]``
"""

from __future__ import annotations

import logging

from medha.interfaces.l1_cache import L1CacheBackend
from medha.types import CacheHit

logger = logging.getLogger(__name__)


def _serialise(hit: CacheHit) -> str:
    """Serialise a ``CacheHit`` to its JSON form.

    Derived from the model rather than a hand-written field list, so every
    field ``CacheHit`` gains travels through the shared L1 automatically.
    The whitelist this replaced silently dropped ``expires_at``, which meant
    the per-entry expiry check in ``Medha._check_l1_cache`` never fired on
    this backend: an expired entry was served until the Redis-level ``ttl``
    (a single global value, when set at all) removed the key.
    """
    return hit.model_dump_json()


def _deserialise(data: str) -> CacheHit:
    """Rebuild a ``CacheHit`` from its JSON form.

    Payloads written by an earlier version simply lack the newer keys and
    fall back to the model defaults, so a populated cache keeps working
    across the upgrade.
    """
    return CacheHit.model_validate_json(data)


class RedisL1Cache(L1CacheBackend):
    """Redis-backed L1 cache.

    Enables sharing the L1 cache across multiple service instances
    (horizontal scaling).  Each entry is stored as a JSON-serialised
    ``CacheHit`` with an optional TTL.  LRU eviction is delegated to Redis
    — configure ``maxmemory-policy allkeys-lru`` on the Redis server for
    automatic eviction when memory is full.

    Args:
        url:      Redis connection URL (e.g. ``"redis://localhost:6379/0"``).
        prefix:   Key namespace prefix (default: ``"medha:l1"``).
        ttl:      Optional entry TTL in seconds.  ``None`` = no expiry.
        max_size: Soft local size hint used for statistics.  Eviction is
                  handled by Redis, not by this adapter.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        prefix: str = "medha:l1",
        ttl: int | None = None,
        max_size: int = 1000,
    ) -> None:
        try:
            import redis.asyncio as aioredis
        except ImportError as exc:
            raise ImportError(
                "RedisL1Cache requires the 'redis' package. "
                "Install it with: pip install 'medha[redis]'"
            ) from exc

        self._prefix = prefix
        self._ttl = ttl
        self._max_size = max_size
        self._size_hint = 0  # Approximate; not decremented on Redis-side eviction
        self._client = aioredis.from_url(url, decode_responses=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _key(self, key: str) -> str:
        return f"{self._prefix}:{key}"

    def _serialise(self, hit: CacheHit) -> str:
        return _serialise(hit)

    def _deserialise(self, data: str) -> CacheHit:
        return _deserialise(data)

    # ------------------------------------------------------------------
    # L1CacheBackend interface
    # ------------------------------------------------------------------

    async def get(self, key: str) -> CacheHit | None:
        try:
            data = await self._client.get(self._key(key))
            if data is None:
                return None
            return self._deserialise(data)
        except Exception as exc:
            logger.warning("RedisL1Cache.get failed (key=%s…): %s", key[:8], exc)
            return None

    async def set(self, key: str, value: CacheHit) -> None:
        try:
            payload = self._serialise(value)
            rkey = self._key(key)
            if self._ttl:
                await self._client.setex(rkey, self._ttl, payload)
            else:
                await self._client.set(rkey, payload)
            self._size_hint += 1
        except Exception as exc:
            logger.warning("RedisL1Cache.set failed (key=%s…): %s", key[:8], exc)

    async def clear(self) -> None:
        try:
            keys = await self._client.keys(f"{self._prefix}:*")
            if keys:
                await self._client.delete(*keys)
            self._size_hint = 0
        except Exception as exc:
            logger.warning("RedisL1Cache.clear failed: %s", exc)

    async def invalidate(self, key: str) -> None:
        try:
            await self._client.delete(self._key(key))
        except Exception as exc:
            logger.warning("RedisL1Cache.invalidate failed (key=%s…): %s", key[:8], exc)

    async def invalidate_prefix(self, prefix: str) -> None:
        """Delete the keys under *prefix* with a cursor scan, not a flush.

        ``scan_iter`` is used rather than ``keys``: it does not block the
        server on a large keyspace, and the base-class default this replaces
        would have dropped every other instance's entries too.

        Keys are collected and deleted in batches so a scan that matches
        thousands of entries does not issue thousands of round-trips.
        """
        batch: list[str] = []
        try:
            async for rkey in self._client.scan_iter(match=f"{self._key(prefix)}*", count=500):
                batch.append(rkey)
                if len(batch) >= 500:
                    await self._client.delete(*batch)
                    batch.clear()
            if batch:
                await self._client.delete(*batch)
        except Exception as exc:
            logger.warning(
                "RedisL1Cache.invalidate_prefix failed (prefix=%s…): %s", prefix[:8], exc
            )

    @property
    def size(self) -> int:
        """Local size hint — not decremented when Redis evicts entries."""
        return self._size_hint

    async def close(self) -> None:
        """Close the underlying Redis connection."""
        await self._client.aclose()
