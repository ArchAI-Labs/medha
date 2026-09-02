# Interfaces (ABCs)

Medha is built around three abstract base classes. Implement these to add new backends, embedders, or L1 cache implementations.

---

## BaseEmbedder

`BaseEmbedder` defines the contract for any embedding provider. Implement `embed()` at minimum; override `embed_batch()` for efficiency.

```python
from medha.interfaces.embedder import BaseEmbedder

class MyEmbedder(BaseEmbedder):
    async def embed(self, text: str) -> list[float]:
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]
```

::: medha.interfaces.embedder.BaseEmbedder

---

## VectorStorageBackend

`VectorStorageBackend` defines the contract for any vector store. Implement all abstract methods, then register your class in `medha/backends/__init__.py`.

```python
from medha.interfaces.storage import VectorStorageBackend

class MyBackend(VectorStorageBackend):
    async def initialize(self, collection: str, dimension: int) -> None: ...
    async def upsert(self, entries: list[CacheEntry]) -> None: ...
    async def query(self, vector: list[float], top_k: int) -> list[tuple[CacheEntry, float]]: ...
    async def delete(self, entry_ids: list[str]) -> None: ...
    async def count(self) -> int: ...
    async def close(self) -> None: ...
```

Not every member is abstract. `load_stats()` / `save_stats()`, `search_filtered()` and the `supports_metadata` flag ship with defaults, so a backend written against an older release keeps working — it simply opts out of stats persistence and of [metadata filters](../user_guide/metadata_filters.md), the latter by refusing them rather than mishandling them.

::: medha.interfaces.storage.VectorStorageBackend

---

## L1CacheBackend

`L1CacheBackend` defines the contract for the fast in-process cache layer (Tier 0 of the waterfall).

```python
from medha.interfaces.l1_cache import L1CacheBackend

class MyL1Cache(L1CacheBackend):
    async def get(self, key: str) -> CacheEntry | None: ...
    async def set(self, key: str, entry: CacheEntry, ttl: int | None = None) -> None: ...
    async def delete(self, key: str) -> None: ...
    async def clear(self) -> None: ...
```

!!! warning "Override `invalidate_prefix()`"
    `Medha.invalidate()` calls it to reach every key a question owns — one per
    set of [metadata filters](../user_guide/metadata_filters.md) it has been
    searched under. The default implementation clears the whole cache, which is
    correct but blunt; a prefix scan restores per-key invalidation. Both
    shipped backends override it.

::: medha.interfaces.l1_cache.L1CacheBackend
