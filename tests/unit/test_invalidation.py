"""Unit tests for the selective invalidation API."""

import asyncio
import hashlib
import uuid

import pytest

from medha.backends.memory import InMemoryBackend
from medha.config import Settings
from medha.types import CacheEntry, SearchStrategy

COLL = "inv_test"
DIM = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vec(seed: int, dim: int = DIM) -> list[float]:
    vec = [(((seed + i) % 7) + 1) / 7.0 for i in range(dim)]
    mag = sum(v * v for v in vec) ** 0.5
    return [v / mag for v in vec]


def _make_entry(
    question: str = "test question",
    query: str = "SELECT 1",
    template_id: str | None = None,
    dim: int = DIM,
) -> CacheEntry:
    vec = _unit_vec(abs(hash(question)) % 100, dim)
    return CacheEntry(
        id=str(uuid.uuid4()),
        vector=vec,
        original_question=question,
        normalized_question=question.lower().strip(),
        generated_query=query,
        query_hash=hashlib.md5(query.encode()).hexdigest(),
        template_id=template_id,
    )


# ---------------------------------------------------------------------------
# InMemoryBackend — search_by_normalized_question
# ---------------------------------------------------------------------------

class TestSearchByNormalizedQuestion:
    async def test_finds_existing_entry(self):
        b = InMemoryBackend()
        await b.initialize(COLL, DIM)
        entry = _make_entry("How many users?", "SELECT COUNT(*) FROM users")
        await b.upsert(COLL, [entry])

        result = await b.search_by_normalized_question(COLL, entry.normalized_question)
        assert result is not None
        assert result.id == entry.id

    async def test_returns_none_for_missing(self):
        b = InMemoryBackend()
        await b.initialize(COLL, DIM)

        result = await b.search_by_normalized_question(COLL, "no such question")
        assert result is None

    async def test_raises_on_missing_collection(self):
        b = InMemoryBackend()
        from medha.exceptions import StorageError
        with pytest.raises(StorageError):
            await b.search_by_normalized_question("nonexistent", "q")


# ---------------------------------------------------------------------------
# InMemoryBackend — find_by_query_hash
# ---------------------------------------------------------------------------

class TestFindByQueryHash:
    async def test_returns_matching_ids(self):
        b = InMemoryBackend()
        await b.initialize(COLL, DIM)
        qhash = hashlib.md5(b"SELECT COUNT(*) FROM users").hexdigest()
        e1 = _make_entry("how many users", "SELECT COUNT(*) FROM users")
        e2 = _make_entry("how many users v2", "SELECT COUNT(*) FROM users")
        e3 = _make_entry("list products", "SELECT * FROM products")
        await b.upsert(COLL, [e1, e2, e3])

        ids = await b.find_by_query_hash(COLL, qhash)
        assert set(ids) == {e1.id, e2.id}

    async def test_returns_empty_for_unknown_hash(self):
        b = InMemoryBackend()
        await b.initialize(COLL, DIM)
        await b.upsert(COLL, [_make_entry()])

        ids = await b.find_by_query_hash(COLL, "deadbeef" * 4)
        assert ids == []


# ---------------------------------------------------------------------------
# InMemoryBackend — find_by_template_id
# ---------------------------------------------------------------------------

class TestFindByTemplateId:
    async def test_returns_matching_ids(self):
        b = InMemoryBackend()
        await b.initialize(COLL, DIM)
        e1 = _make_entry("how many users", "SELECT COUNT(*) FROM users", template_id="count_users")
        e2 = _make_entry("how many orders", "SELECT COUNT(*) FROM orders", template_id="count_users")
        e3 = _make_entry("list products", "SELECT * FROM products", template_id="list_products")
        await b.upsert(COLL, [e1, e2, e3])

        ids = await b.find_by_template_id(COLL, "count_users")
        assert set(ids) == {e1.id, e2.id}

    async def test_returns_empty_for_unknown_template(self):
        b = InMemoryBackend()
        await b.initialize(COLL, DIM)
        await b.upsert(COLL, [_make_entry(template_id="other")])

        ids = await b.find_by_template_id(COLL, "nonexistent_template")
        assert ids == []


# ---------------------------------------------------------------------------
# InMemoryBackend — drop_collection
# ---------------------------------------------------------------------------

class TestDropCollection:
    async def test_drop_removes_collection(self):
        b = InMemoryBackend()
        await b.initialize(COLL, DIM)
        await b.upsert(COLL, [_make_entry()])

        await b.drop_collection(COLL)
        # Collection should no longer exist
        from medha.exceptions import StorageError
        with pytest.raises(StorageError):
            await b.search(COLL, _unit_vec(1))

    async def test_drop_missing_collection_is_noop(self):
        b = InMemoryBackend()
        # Should not raise
        await b.drop_collection("does_not_exist")


# ---------------------------------------------------------------------------
# Medha.invalidate
# ---------------------------------------------------------------------------

class TestMedhaInvalidate:
    @pytest.fixture
    async def medha(self, mock_embedder):
        from medha.backends.memory import InMemoryBackend
        from medha.core import Medha

        settings = Settings(
            backend_type="memory",
            score_threshold_exact=0.99,
            score_threshold_semantic=0.50,
            l1_cache_max_size=100,
        )
        backend = InMemoryBackend()
        m = Medha(
            collection_name="inv_core",
            embedder=mock_embedder,
            backend=backend,
            settings=settings,
        )
        await m.start()
        yield m
        await m.close()

    async def test_invalidate_removes_entry_and_l1(self, medha):
        await medha.store("how many users", "SELECT COUNT(*) FROM users")

        # Confirm it's cached in L1
        hit_before = await medha.search("how many users")
        assert hit_before.strategy != SearchStrategy.NO_MATCH

        result = await medha.invalidate("how many users")
        assert result is True

        # L1 should be cleared; backend entry gone → NO_MATCH
        hit_after = await medha.search("how many users")
        assert hit_after.strategy == SearchStrategy.NO_MATCH

    async def test_invalidate_returns_false_when_not_found(self, medha):
        result = await medha.invalidate("question never stored")
        assert result is False

    async def test_invalidate_removes_every_duplicate(self, medha):
        """One question can map to several entries, and all must go.

        ``store()`` mints a new id per call, so storing the same question
        twice leaves two entries. Deleting only the one
        ``search_by_normalized_question()`` happens to return left the other
        answering a question the caller had just invalidated — it came back as
        a confident EXACT_MATCH.
        """
        question = "how many active users"
        await medha.store(question, "SELECT count(*) FROM users WHERE active")
        await medha.store(question, "SELECT count(*) FROM users WHERE active = true")
        assert await medha._backend.count("inv_core") == 2

        assert await medha.invalidate(question) is True

        assert await medha._backend.count("inv_core") == 0
        hit = await medha.search(question)
        assert hit.strategy == SearchStrategy.NO_MATCH, (
            f"invalidated question still answered by {hit.strategy.value}: "
            f"{hit.generated_query!r}"
        )

    async def test_invalidate_terminates_on_stale_reads(self, medha):
        """A backend that keeps serving a deleted entry must not loop forever.

        Elasticsearch and Azure refresh asynchronously, so a lookup right
        after a delete can still return the document just removed. The loop
        stops when it sees an id it has already deleted rather than waiting
        for the refresh.
        """
        await medha.store("how many users", "SELECT COUNT(*) FROM users")
        backend = medha._backend
        stale = await backend.search_by_normalized_question("inv_core", "how many users")
        assert stale is not None

        # Delete is a no-op and the lookup always returns the same entry:
        # the worst case an eventually-consistent backend can present.
        async def _never_deletes(collection_name, ids):
            return None

        async def _always_finds(collection_name, normalized_question):
            return stale

        backend.delete = _never_deletes
        backend.search_by_normalized_question = _always_finds

        assert await asyncio.wait_for(medha.invalidate("how many users"), timeout=5) is True

    async def test_invalidate_by_query_hash_removes_all_matching(self, medha):
        query = "SELECT COUNT(*) FROM users"
        qhash = hashlib.md5(query.encode()).hexdigest()
        await medha.store("how many users", query)
        await medha.store("user count please", query)
        await medha.store("list products", "SELECT * FROM products")

        deleted = await medha.invalidate_by_query_hash(qhash)
        assert deleted == 2

        # Remaining entry untouched
        from medha.backends.memory import InMemoryBackend
        backend = medha._backend
        assert isinstance(backend, InMemoryBackend)
        count = await backend.count("inv_core")
        assert count == 1

    async def test_invalidate_by_template_removes_only_template_entries(self, medha):
        await medha.store("how many users", "SELECT COUNT(*) FROM users", template_id="count_tmpl")
        await medha.store("how many orders", "SELECT COUNT(*) FROM orders", template_id="count_tmpl")
        await medha.store("list products", "SELECT * FROM products", template_id="list_tmpl")

        deleted = await medha.invalidate_by_template("count_tmpl")
        assert deleted == 2

        backend = medha._backend
        assert isinstance(backend, InMemoryBackend)
        count = await backend.count("inv_core")
        assert count == 1

    async def test_invalidate_collection_drops_and_reinitializes(self, medha):
        await medha.store("question one", "SELECT 1")
        await medha.store("question two", "SELECT 2")

        backend = medha._backend
        assert isinstance(backend, InMemoryBackend)
        count_before = await backend.count("inv_core")
        assert count_before == 2

        dropped = await medha.invalidate_collection()
        assert dropped == 2

        # Collection re-initialized and usable
        count_after = await backend.count("inv_core")
        assert count_after == 0

        # Can store new entries after invalidation
        ok = await medha.store("fresh question", "SELECT 3")
        assert ok is True
