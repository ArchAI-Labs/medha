"""VectorStorageBackend abstract class defining the storage interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from medha.types import CacheEntry, CacheResult, MetadataDict, PersistedStats
from medha.utils.metadata import filter_fetch_size, verify_filters


class VectorStorageBackend(ABC):
    """Abstract base class for vector storage backends."""

    #: Whether this backend stores and returns ``CacheEntry.metadata``.
    #: ``Medha`` refuses a filtered search against a backend that does not,
    #: because every entry would come back with empty metadata and every
    #: filtered search would silently return NO_MATCH. Backends set this to
    #: True once their upsert/read path round-trips metadata.
    supports_metadata: bool = False

    @abstractmethod
    async def initialize(self, collection_name: str, dimension: int, **kwargs: Any) -> None:
        """Set up the storage backend (create collection, indexes, quantization).

        This method is idempotent: calling it twice with the same arguments
        must not raise or duplicate data.

        Args:
            collection_name: Name of the vector collection.
            dimension: Vector dimensionality (must match the embedder).
            **kwargs: Backend-specific configuration (quantization, HNSW params, etc.).

        Raises:
            StorageInitializationError: If setup fails.
        """
        ...

    @abstractmethod
    async def search(
        self,
        collection_name: str,
        vector: list[float],
        limit: int = 5,
        score_threshold: float = 0.0,
    ) -> list[CacheResult]:
        """Search for similar vectors.

        Args:
            collection_name: Collection to search.
            vector: Query vector.
            limit: Max number of results.
            score_threshold: Minimum similarity score (0.0 - 1.0).

        Returns:
            List of CacheResult, sorted by descending score.

        Raises:
            StorageError: If the search fails.
        """
        ...

    @abstractmethod
    async def upsert(self, collection_name: str, entries: list[CacheEntry]) -> None:
        """Insert or update cache entries.

        Args:
            collection_name: Target collection.
            entries: List of CacheEntry objects to upsert.

        Raises:
            StorageError: If the upsert fails.
        """
        ...

    @abstractmethod
    async def scroll(
        self,
        collection_name: str,
        limit: int = 100,
        offset: str | None = None,
        with_vectors: bool = False,
    ) -> tuple[list[CacheResult], str | None]:
        """Iterate over all points in a collection.

        Used by fuzzy search (Tier 4) and admin operations.

        Args:
            collection_name: Collection to scroll.
            limit: Batch size per scroll.
            offset: Pagination token from a previous scroll.
            with_vectors: Whether to include vectors in results.

        Returns:
            Tuple of (results, next_offset). next_offset is None when done.

        Raises:
            StorageError: If the scroll fails.
        """
        ...

    @abstractmethod
    async def count(self, collection_name: str) -> int:
        """Return the number of points in a collection.

        Raises:
            StorageError: If the count fails.
        """
        ...

    @abstractmethod
    async def delete(self, collection_name: str, ids: list[str]) -> None:
        """Delete points by ID.

        Args:
            collection_name: Target collection.
            ids: List of point IDs to delete.

        Raises:
            StorageError: If the delete fails.
        """
        ...

    @abstractmethod
    async def find_expired(self, collection_name: str) -> list[str]:
        """Return IDs of entries with expires_at < now(UTC).

        Raises:
            StorageError: If the query fails.
        """
        ...

    @abstractmethod
    async def search_by_normalized_question(
        self, collection_name: str, normalized_question: str
    ) -> CacheResult | None:
        """Find one entry by exact normalized_question match.

        Nothing enforces uniqueness on ``normalized_question``, and
        ``Medha.store()`` mints a new id on every call, so several entries can
        share one normalized question. **Which** of them is returned is
        backend-dependent and may differ between calls.

        Two consequences for callers: one that must reach every match cannot
        rely on a single call (``Medha.invalidate`` loops until the question is
        gone), and one that must reach a *specific* entry cannot use this
        method at all — it has no way to say which.

        Returns:
            One matching CacheResult, or None if nothing matches.
        """
        ...

    @abstractmethod
    async def find_by_query_hash(
        self, collection_name: str, query_hash: str
    ) -> list[str]:
        """Return all point IDs whose payload.query_hash matches *query_hash*.

        Returns:
            List of string IDs (may be empty).
        """
        ...

    @abstractmethod
    async def find_by_template_id(
        self, collection_name: str, template_id: str
    ) -> list[str]:
        """Return all point IDs whose payload.template_id matches *template_id*.

        Returns:
            List of string IDs (may be empty).
        """
        ...

    @abstractmethod
    async def drop_collection(self, collection_name: str) -> None:
        """Permanently delete the entire collection and all its data.

        Raises:
            StorageError: If the drop fails.
        """
        ...

    @abstractmethod
    async def update_feedback(
        self,
        collection_name: str,
        point_id: str,
        correct: bool,
    ) -> int:
        """Increment feedback_correct or feedback_incorrect for a stored entry.

        Args:
            collection_name: Target collection.
            point_id:        ID of the entry to update.
            correct:         True → increment feedback_correct;
                             False → increment feedback_incorrect.

        Returns:
            The new value of the incremented counter after the update.
            Returns 0 if the entry is not found (no exception raised).

        Raises:
            StorageError: If the update fails.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release resources (close connections, etc.)."""
        ...

    # --- Optional overrides (non-abstract: subclasses keep working unchanged) ---

    async def search_filtered(
        self,
        collection_name: str,
        vector: list[float],
        limit: int = 5,
        score_threshold: float = 0.0,
        filters: MetadataDict | None = None,
        overfetch: int = 10,
    ) -> list[CacheResult]:
        """Search for similar vectors, keeping only entries matching *filters*.

        This is the method ``Medha`` calls; :meth:`search` stays the primitive
        every backend must implement. The default implementation over-fetches
        and filters in Python, so a backend gains metadata filtering as soon as
        it round-trips ``CacheEntry.metadata`` — no change to its ``search()``
        signature, and none at all for a third-party backend.

        Backends whose query language can express the filter override this and
        push it down, which is both faster and exact. The default is neither:
        it can only filter what the unfiltered top-N happened to contain, so a
        match ranked below ``limit * overfetch`` is missed. Recall loss, never
        a wrong answer — a returned entry always satisfies the filter.

        Args:
            collection_name: Collection to search.
            vector: Query vector.
            limit: Max number of results to return *after* filtering.
            score_threshold: Minimum similarity score (0.0 - 1.0).
            filters: Exact-match metadata constraints, ANDed. None or empty
                means no filtering, in which case this is exactly ``search()``.
            overfetch: How many candidates to retrieve per requested result.
                Ignored by backends that filter natively.

        Returns:
            List of CacheResult, sorted by descending score, each matching
            every filter.

        Raises:
            StorageError: If the search fails.
        """
        if not filters:
            return await self.search(collection_name, vector, limit, score_threshold)

        fetch = filter_fetch_size(limit, filters, overfetch)
        candidates = await self.search(collection_name, vector, fetch, score_threshold)
        return verify_filters(candidates, filters, limit)

    async def load_stats(self, collection_name: str) -> PersistedStats | None:
        """Load persisted statistics for the collection, or None if not yet saved.

        Default implementation returns None (stats persistence not supported).
        Backends that support it store the stats as a JSON blob under the key
        f"_medha_stats_{collection_name}" (in a dedicated metadata location
        appropriate for the backend — not in the main vector index).

        Returns:
            PersistedStats if previously saved, None otherwise.

        Raises:
            StorageError: If the load fails (not if simply absent).
        """
        return None

    async def save_stats(
        self,
        collection_name: str,
        stats: PersistedStats,
    ) -> None:
        """Persist statistics for the collection.

        Default implementation is a no-op (stats persistence not supported).

        Args:
            collection_name: Target collection.
            stats:           Snapshot to persist.

        Raises:
            StorageError: If the save fails.
        """
        return

    async def connect(self) -> None:
        """Establish connection. No-op for backends that don't require it."""
        return

    # --- Context manager support ---

    async def __aenter__(self) -> VectorStorageBackend:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> bool:
        await self.close()
        return False
