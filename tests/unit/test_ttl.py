"""Unit tests for TTL (Time-To-Live) cache lifecycle support."""

import hashlib
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from medha.backends.memory import InMemoryBackend
from medha.config import Settings
from medha.types import CacheEntry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(
    question: str = "test question",
    query: str = "SELECT 1",
    expires_at: datetime | None = None,
    dim: int = 8,
) -> CacheEntry:
    vec = [0.1] * dim
    # Make vectors unique per question so cosine similarity differs
    vec[0] = abs(hash(question) % 100) / 100.0 + 0.01
    magnitude = sum(v**2 for v in vec) ** 0.5
    vec = [v / magnitude for v in vec]
    return CacheEntry(
        id=str(uuid.uuid4()),
        vector=vec,
        original_question=question,
        normalized_question=question.lower(),
        generated_query=query,
        query_hash=hashlib.md5(query.encode()).hexdigest(),
        expires_at=expires_at,
    )


def _past(seconds: int = 10) -> datetime:
    return datetime.now(timezone.utc) - timedelta(seconds=seconds)


def _future(seconds: int = 3600) -> datetime:
    return datetime.now(timezone.utc) + timedelta(seconds=seconds)


COLL = "ttl_test"


# ---------------------------------------------------------------------------
# InMemoryBackend TTL tests
# ---------------------------------------------------------------------------

class TestInMemoryTTL:
    async def test_expired_entry_excluded_from_search(self):
        b = InMemoryBackend()
        await b.initialize(COLL, 8)

        expired = _make_entry("expired question", "SELECT expired", expires_at=_past())
        await b.upsert(COLL, [expired])

        results = await b.search(COLL, expired.vector, limit=10, score_threshold=0.0)
        assert all(r.id != expired.id for r in results), "Expired entry must not appear in search"

    async def test_non_expired_entry_appears_in_search(self):
        b = InMemoryBackend()
        await b.initialize(COLL, 8)

        valid = _make_entry("valid question", "SELECT valid", expires_at=_future())
        await b.upsert(COLL, [valid])

        results = await b.search(COLL, valid.vector, limit=10, score_threshold=0.0)
        ids = [r.id for r in results]
        assert valid.id in ids, "Non-expired entry must appear in search"

    async def test_no_ttl_entry_never_expires(self):
        b = InMemoryBackend()
        await b.initialize(COLL, 8)

        immortal = _make_entry("immortal question", "SELECT immortal", expires_at=None)
        await b.upsert(COLL, [immortal])

        results = await b.search(COLL, immortal.vector, limit=10, score_threshold=0.0)
        ids = [r.id for r in results]
        assert immortal.id in ids, "Entry without TTL must never be excluded"

    async def test_find_expired_returns_expired_ids(self):
        b = InMemoryBackend()
        await b.initialize(COLL, 8)

        expired = _make_entry("will expire", "SELECT exp", expires_at=_past())
        valid = _make_entry("will not expire", "SELECT ok", expires_at=_future())
        await b.upsert(COLL, [expired, valid])

        expired_ids = await b.find_expired(COLL)
        assert expired.id in expired_ids
        assert valid.id not in expired_ids

    async def test_find_expired_ignores_no_ttl_entries(self):
        b = InMemoryBackend()
        await b.initialize(COLL, 8)

        immortal = _make_entry("immortal", "SELECT 1", expires_at=None)
        await b.upsert(COLL, [immortal])

        expired_ids = await b.find_expired(COLL)
        assert immortal.id not in expired_ids

    async def test_find_expired_empty_collection(self):
        b = InMemoryBackend()
        await b.initialize(COLL, 8)

        expired_ids = await b.find_expired(COLL)
        assert expired_ids == []

    async def test_find_expired_missing_collection_returns_empty(self):
        b = InMemoryBackend()
        result = await b.find_expired("nonexistent_collection")
        assert result == []


# ---------------------------------------------------------------------------
# Core Medha TTL tests (using InMemoryBackend + MockEmbedder)
# ---------------------------------------------------------------------------

class TestMedhaTTL:
    @pytest.fixture
    async def medha(self, mock_embedder):
        from medha.backends.memory import InMemoryBackend
        from medha.core import Medha

        settings = Settings(
            backend_type="memory",
            score_threshold_exact=0.99,
            score_threshold_semantic=0.50,
            l1_cache_max_size=0,  # disable L1 so we test backend filtering
        )
        backend = InMemoryBackend()
        m = Medha(
            collection_name="ttl_core_test",
            embedder=mock_embedder,
            backend=backend,
            settings=settings,
        )
        await m.start()
        yield m
        await m.close()

    async def test_store_with_ttl_expired_not_found(self, medha):
        """Entry stored with already-expired TTL must not appear in search."""
        await medha.store(
            "how many users", "SELECT COUNT(*) FROM users", ttl=-1
        )
        hit = await medha.search("how many users")
        from medha.types import SearchStrategy
        assert hit.strategy == SearchStrategy.NO_MATCH

    async def test_store_without_ttl_immortal(self, medha):
        """Entry stored without TTL must appear in search indefinitely."""
        await medha.store("list all products", "SELECT * FROM products")
        hit = await medha.search("list all products")
        from medha.types import SearchStrategy
        assert hit.strategy != SearchStrategy.NO_MATCH
        assert hit.generated_query == "SELECT * FROM products"

    async def test_store_ttl_none_explicit_overrides_default(self, mock_embedder):
        """store(ttl=None) must produce an immortal entry even if default_ttl_seconds is set."""
        from medha.backends.memory import InMemoryBackend
        from medha.core import Medha

        settings = Settings(
            backend_type="memory",
            default_ttl_seconds=1,
            score_threshold_exact=0.99,
            score_threshold_semantic=0.50,
            l1_cache_max_size=0,
        )
        backend = InMemoryBackend()
        m = Medha(
            collection_name="ttl_none_test",
            embedder=mock_embedder,
            backend=backend,
            settings=settings,
        )
        await m.start()
        try:
            await m.store("count orders", "SELECT COUNT(*) FROM orders", ttl=None)
            # The entry should be immortal (expires_at=None), so it appears in search
            hit = await m.search("count orders")
            from medha.types import SearchStrategy
            assert hit.strategy != SearchStrategy.NO_MATCH
        finally:
            await m.close()

    async def test_store_ttl_param_overrides_default(self, mock_embedder):
        """store(ttl=3600) must override default_ttl_seconds=1."""
        from medha.backends.memory import InMemoryBackend
        from medha.core import Medha

        settings = Settings(
            backend_type="memory",
            default_ttl_seconds=1,  # very short default
            score_threshold_exact=0.99,
            score_threshold_semantic=0.50,
            l1_cache_max_size=0,
        )
        backend = InMemoryBackend()
        m = Medha(
            collection_name="ttl_override_test",
            embedder=mock_embedder,
            backend=backend,
            settings=settings,
        )
        await m.start()
        try:
            await m.store("list employees", "SELECT * FROM employees", ttl=3600)
            hit = await m.search("list employees")
            from medha.types import SearchStrategy
            assert hit.strategy != SearchStrategy.NO_MATCH
        finally:
            await m.close()

    async def test_expire_removes_expired_entries(self, mock_embedder):
        """expire() must delete expired entries and return count."""
        from medha.backends.memory import InMemoryBackend
        from medha.core import Medha

        settings = Settings(
            backend_type="memory",
            score_threshold_exact=0.99,
            score_threshold_semantic=0.50,
            l1_cache_max_size=0,
        )
        backend = InMemoryBackend()
        m = Medha(
            collection_name="expire_test",
            embedder=mock_embedder,
            backend=backend,
            settings=settings,
        )
        await m.start()
        try:
            # Store one entry that will be already expired (ttl=-1 → expires in past)
            await m.store("expired query", "SELECT expired", ttl=-1)
            # Store one immortal entry
            await m.store("valid query", "SELECT valid")

            deleted = await m.expire()
            assert deleted == 1

            # Immortal entry still accessible via backend count
            count = await backend.count("expire_test")
            assert count == 1
        finally:
            await m.close()

    async def test_expire_does_not_touch_valid_entries(self, mock_embedder):
        """expire() must not delete entries that have not yet expired."""
        from medha.backends.memory import InMemoryBackend
        from medha.core import Medha

        settings = Settings(
            backend_type="memory",
            score_threshold_exact=0.99,
            score_threshold_semantic=0.50,
            l1_cache_max_size=0,
        )
        backend = InMemoryBackend()
        m = Medha(
            collection_name="no_expire_test",
            embedder=mock_embedder,
            backend=backend,
            settings=settings,
        )
        await m.start()
        try:
            await m.store("future ttl query", "SELECT future", ttl=3600)
            deleted = await m.expire()
            assert deleted == 0
            assert await backend.count("no_expire_test") == 1
        finally:
            await m.close()

    async def test_expire_collection_name_param(self, mock_embedder):
        """expire(collection_name=...) must operate on that specific collection."""
        from medha.backends.memory import InMemoryBackend
        from medha.core import Medha

        settings = Settings(
            backend_type="memory",
            score_threshold_exact=0.99,
            score_threshold_semantic=0.50,
            l1_cache_max_size=0,
        )
        backend = InMemoryBackend()
        m = Medha(
            collection_name="expire_coll_test",
            embedder=mock_embedder,
            backend=backend,
            settings=settings,
        )
        await m.start()
        try:
            await m.store("expired item", "SELECT exp", ttl=-1)
            deleted = await m.expire(collection_name="expire_coll_test")
            assert deleted == 1
        finally:
            await m.close()
