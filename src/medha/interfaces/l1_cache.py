"""Abstract base class for L1 cache backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from medha.types import CacheHit


class L1CacheBackend(ABC):
    """Interface for L1 (fast-lookup) cache backends.

    L1 cache sits in front of the vector backend and provides sub-millisecond
    responses for recently seen questions.  The default implementation is
    in-memory (``InMemoryL1Cache``); a Redis-backed implementation
    (``RedisL1Cache``) enables sharing the cache across multiple service
    instances in a horizontally-scaled deployment.
    """

    @abstractmethod
    async def get(self, key: str) -> CacheHit | None:
        """Return the cached hit for *key*, or ``None`` on a miss."""
        ...

    @abstractmethod
    async def set(self, key: str, value: CacheHit) -> None:
        """Store *value* under *key*.  Implementations handle eviction internally."""
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Remove all entries from the cache."""
        ...

    @abstractmethod
    async def invalidate(self, key: str) -> None:
        """Remove a single entry by *key*. No-op if key is absent."""
        ...

    async def invalidate_prefix(self, prefix: str) -> None:
        """Remove every entry whose key starts with *prefix*.

        A search that declares metadata filters is cached under a key derived
        from the question *and* those filters, so one question can own several
        L1 keys sharing the question's hash as their prefix. Invalidating that
        question has to reach all of them, or a filtered lookup keeps being
        served an entry the caller has just invalidated.

        The default clears the whole cache. It is blunt — correctness before
        efficiency, since leaving stale entries behind is the one outcome
        invalidation must not have — and both shipped backends override it
        with a real prefix scan. A third-party backend keeps working unchanged,
        at the cost of a full flush on ``Medha.invalidate()``.
        """
        await self.clear()

    async def invalidate_all(self) -> None:
        """Remove all entries. Delegates to :meth:`clear` by default."""
        await self.clear()

    @property
    @abstractmethod
    def size(self) -> int:
        """Current number of entries.  May be approximate for distributed backends."""
        ...

    async def close(self) -> None:
        """Release any resources held by this backend. No-op by default."""
        return
