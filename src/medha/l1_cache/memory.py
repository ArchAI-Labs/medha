"""In-process LRU L1 cache backend (default)."""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict

from medha.interfaces.l1_cache import L1CacheBackend
from medha.types import CacheHit

logger = logging.getLogger(__name__)


class InMemoryL1Cache(L1CacheBackend):
    """Thread-safe, in-process LRU cache.

    Args:
        max_size: Maximum number of entries.  0 disables the cache.
    """

    def __init__(self, max_size: int = 1000) -> None:
        self._max_size = max_size
        self._cache: OrderedDict[str, CacheHit] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> CacheHit | None:
        async with self._lock:
            if key in self._cache:
                hit = self._cache.pop(key)
                self._cache[key] = hit  # LRU bump
                return hit
        return None

    async def set(self, key: str, value: CacheHit) -> None:
        if self._max_size <= 0:
            return
        async with self._lock:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)  # Evict oldest
                logger.debug("L1 cache evicted oldest entry (max_size=%d)", self._max_size)
            self._cache[key] = value

    async def clear(self) -> None:
        async with self._lock:
            self._cache.clear()

    async def invalidate(self, key: str) -> None:
        async with self._lock:
            self._cache.pop(key, None)

    async def invalidate_prefix(self, prefix: str) -> None:
        """Drop every key starting with *prefix* instead of flushing the cache.

        A linear scan: the cache is an ``OrderedDict`` bounded by ``max_size``,
        so there is nothing cheaper to scan and nothing to index.
        """
        async with self._lock:
            for key in [k for k in self._cache if k.startswith(prefix)]:
                del self._cache[key]

    @property
    def size(self) -> int:
        return len(self._cache)
