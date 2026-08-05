"""End-to-end integration tests: MockEmbedder + ChromaBackend (ephemeral) + Medha pipeline.

Chromadb in ephemeral mode requires no external service — runs fully in-process.
"""

import pytest

chromadb = pytest.importorskip("chromadb")

from medha.backends.chroma import ChromaBackend
from medha.config import Settings
from medha.core import Medha
from medha.types import SearchStrategy

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def chroma_backend():
    settings = Settings(chroma_mode="ephemeral")
    b = ChromaBackend(settings)
    await b.connect()
    yield b
    await b.close()


@pytest.fixture
async def medha_chroma(mock_embedder):
    settings = Settings(
        backend_type="chroma",
        chroma_mode="ephemeral",
        score_threshold_exact=0.99,
        score_threshold_semantic=0.85,
        l1_cache_max_size=100,
    )
    b = ChromaBackend(settings)
    await b.connect()
    m = Medha(
        collection_name="chroma_e2e",
        embedder=mock_embedder,
        backend=b,
        settings=settings,
    )
    await m.start()
    yield m
    await m.close()


# ---------------------------------------------------------------------------
# Backend-level CRUD
# ---------------------------------------------------------------------------


class TestBackendCRUD:
    async def test_initialize_and_count(self, chroma_backend):
        await chroma_backend.initialize("test_col", 8)
        assert await chroma_backend.count("test_col") == 0

    async def test_upsert_and_count(self, chroma_backend):
        from tests.conftest import make_entry

        await chroma_backend.initialize("test_col", 8)
        entries = [make_entry() for _ in range(5)]
        await chroma_backend.upsert("test_col", entries)
        assert await chroma_backend.count("test_col") == 5

    async def test_upsert_same_id_overwrites(self, chroma_backend):
        import uuid

        from tests.conftest import make_entry

        await chroma_backend.initialize("test_col", 8)
        eid = str(uuid.uuid4())
        await chroma_backend.upsert("test_col", [make_entry(id=eid, query="SELECT 1")])
        await chroma_backend.upsert("test_col", [make_entry(id=eid, query="SELECT 2")])
        assert await chroma_backend.count("test_col") == 1

        results, _ = await chroma_backend.scroll("test_col")
        assert results[0].generated_query == "SELECT 2"

    async def test_search_returns_sorted_results(self, chroma_backend):
        from tests.conftest import make_entry

        await chroma_backend.initialize("test_col", 4)
        await chroma_backend.upsert("test_col", [
            make_entry(id="a", vector=[1.0, 0.0, 0.0, 0.0], query="SELECT A"),
            make_entry(id="b", vector=[0.0, 0.0, 0.0, 1.0], query="SELECT B"),
        ])

        results = await chroma_backend.search("test_col", [1.0, 0.0, 0.0, 0.0], limit=5, score_threshold=0.0)

        assert len(results) == 2
        assert results[0].score >= results[1].score
        assert results[0].generated_query == "SELECT A"

    async def test_scroll_pagination(self, chroma_backend):
        from tests.conftest import make_entry

        await chroma_backend.initialize("test_col", 8)
        entries = [make_entry() for _ in range(7)]
        await chroma_backend.upsert("test_col", entries)

        page1, next_offset = await chroma_backend.scroll("test_col", limit=4)
        assert len(page1) == 4
        assert next_offset == "4"

        page2, next_offset2 = await chroma_backend.scroll("test_col", limit=4, offset=next_offset)
        assert len(page2) == 3
        assert next_offset2 is None

    async def test_delete(self, chroma_backend):
        from tests.conftest import make_entry

        await chroma_backend.initialize("test_col", 8)
        e1, e2 = make_entry(), make_entry()
        await chroma_backend.upsert("test_col", [e1, e2])
        await chroma_backend.delete("test_col", [e1.id])
        assert await chroma_backend.count("test_col") == 1

    async def test_search_by_query_hash(self, chroma_backend):
        from tests.conftest import make_entry

        await chroma_backend.initialize("test_col", 8)
        entry = make_entry(query="SELECT 42")
        await chroma_backend.upsert("test_col", [entry])

        result = await chroma_backend.search_by_query_hash("test_col", entry.query_hash)

        assert result is not None
        assert result.generated_query == "SELECT 42"

    async def test_update_usage_count(self, chroma_backend):
        from tests.conftest import make_entry

        await chroma_backend.initialize("test_col", 8)
        entry = make_entry()
        await chroma_backend.upsert("test_col", [entry])

        await chroma_backend.update_usage_count("test_col", entry.id)

        results, _ = await chroma_backend.scroll("test_col")
        assert results[0].usage_count == 2


# ---------------------------------------------------------------------------
# Medha pipeline tests
# ---------------------------------------------------------------------------


class TestStoreAndSearch:
    async def test_store_and_exact_search(self, medha_chroma):
        await medha_chroma.store("How many users are there", "SELECT COUNT(*) FROM users")
        await medha_chroma.clear_caches()

        hit = await medha_chroma.search("How many users are there")
        assert hit.strategy == SearchStrategy.EXACT_MATCH
        assert hit.generated_query == "SELECT COUNT(*) FROM users"
        assert hit.confidence >= 0.99

    async def test_l1_cache_hit(self, medha_chroma):
        await medha_chroma.store("Get user count", "SELECT COUNT(*) FROM users")
        await medha_chroma.clear_caches()

        hit1 = await medha_chroma.search("Get user count")
        assert hit1.strategy != SearchStrategy.NO_MATCH

        hit2 = await medha_chroma.search("Get user count")
        assert hit2.strategy == SearchStrategy.L1_CACHE

    async def test_no_match_empty(self, medha_chroma):
        hit = await medha_chroma.search("What is the meaning of life")
        assert hit.strategy == SearchStrategy.NO_MATCH
        assert not hit.generated_query

    async def test_store_batch(self, medha_chroma):
        entries = [
            {"question": f"Question number {i}", "generated_query": f"SELECT {i}"}
            for i in range(5)
        ]
        ok = await medha_chroma.store_batch(entries)
        assert ok is True

        results, _ = await medha_chroma._backend.scroll("chroma_e2e", limit=10)
        stored_queries = {r.generated_query for r in results}
        for i in range(5):
            assert f"SELECT {i}" in stored_queries


class TestBackendTypeViaSettings:
    async def test_backend_type_chroma_via_settings(self, mock_embedder):
        settings = Settings(
            backend_type="chroma",
            chroma_mode="ephemeral",
            score_threshold_exact=0.99,
            score_threshold_semantic=0.85,
        )
        m = Medha(
            collection_name="auto_backend_test",
            embedder=mock_embedder,
            settings=settings,
        )
        assert isinstance(m._backend, ChromaBackend)
        await m.start()
        await m.close()


class TestContextManager:
    async def test_context_manager(self, mock_embedder):
        settings = Settings(backend_type="chroma", chroma_mode="ephemeral")
        async with Medha(
            collection_name="ctx_chroma_test",
            embedder=mock_embedder,
            settings=settings,
        ) as m:
            await m.store("test question", "SELECT 1")
            hit = await m.search("test question")
            assert hit.strategy != SearchStrategy.NO_MATCH


class TestTTL:
    async def test_expired_entry_excluded_from_search(self, mock_embedder):
        from datetime import datetime, timedelta, timezone

        from tests.conftest import make_entry

        settings = Settings(chroma_mode="ephemeral")
        b = ChromaBackend(settings)
        await b.connect()
        await b.initialize("ttl_test", 4)

        try:
            expired_vec = [1.0, 0.0, 0.0, 0.0]
            expired = make_entry(
                vector=expired_vec, question="expired q", dim=4,
            )
            # Manually set expires_at to past via model reconstruction
            from medha.types import CacheEntry
            expired = CacheEntry(
                id=expired.id, vector=expired_vec,
                original_question="expired q",
                normalized_question="expired q",
                generated_query="SELECT expired",
                query_hash=expired.query_hash,
                expires_at=datetime.now(timezone.utc) - timedelta(seconds=10),
            )
            await b.upsert("ttl_test", [expired])

            results = await b.search("ttl_test", expired_vec, limit=5, score_threshold=0.0)
            assert all(r.id != expired.id for r in results), "Expired entry must not appear in search"
        finally:
            await b.close()

    async def test_find_expired_returns_expired_ids(self, mock_embedder):
        from datetime import datetime, timedelta, timezone

        from medha.types import CacheEntry
        from tests.conftest import make_entry

        settings = Settings(chroma_mode="ephemeral")
        b = ChromaBackend(settings)
        await b.connect()
        await b.initialize("ttl_find_test", 8)

        try:
            vec = [0.1] * 8
            expired = make_entry(vector=vec, question="will expire", dim=8)
            expired = CacheEntry(
                id=expired.id, vector=vec,
                original_question="will expire",
                normalized_question="will expire",
                generated_query="SELECT expired",
                query_hash=expired.query_hash,
                expires_at=datetime.now(timezone.utc) - timedelta(seconds=10),
            )
            valid = make_entry(vector=[0.2] * 8, question="will stay", dim=8)
            valid = CacheEntry(
                id=valid.id, vector=[0.2] * 8,
                original_question="will stay",
                normalized_question="will stay",
                generated_query="SELECT valid",
                query_hash=valid.query_hash,
                expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            )
            await b.upsert("ttl_find_test", [expired, valid])

            expired_ids = await b.find_expired("ttl_find_test")
            assert expired.id in expired_ids
            assert valid.id not in expired_ids
        finally:
            await b.close()


class TestInvalidation:
    async def test_find_by_query_hash(self, chroma_backend):
        from tests.conftest import make_entry

        await chroma_backend.initialize("inv_test", 8)
        query = "SELECT COUNT(*) FROM users"
        import hashlib
        qhash = hashlib.md5(query.encode()).hexdigest()
        e1 = make_entry(question="how many users", query=query)
        e2 = make_entry(question="user count", query=query)
        e3 = make_entry(question="list products", query="SELECT * FROM products")
        await chroma_backend.upsert("inv_test", [e1, e2, e3])

        ids = await chroma_backend.find_by_query_hash("inv_test", qhash)
        assert set(ids) == {e1.id, e2.id}

    async def test_find_by_template_id(self, chroma_backend):
        from medha.types import CacheEntry
        from tests.conftest import make_entry

        await chroma_backend.initialize("inv_test2", 8)
        e1 = make_entry(question="q1", query="SELECT 1")
        e1 = CacheEntry(
            id=e1.id, vector=e1.vector, original_question="q1",
            normalized_question="q1", generated_query="SELECT 1",
            query_hash=e1.query_hash, template_id="tmpl_a",
        )
        e2 = make_entry(question="q2", query="SELECT 2")
        e2 = CacheEntry(
            id=e2.id, vector=e2.vector, original_question="q2",
            normalized_question="q2", generated_query="SELECT 2",
            query_hash=e2.query_hash, template_id="tmpl_a",
        )
        e3 = make_entry(question="q3", query="SELECT 3")
        e3 = CacheEntry(
            id=e3.id, vector=e3.vector, original_question="q3",
            normalized_question="q3", generated_query="SELECT 3",
            query_hash=e3.query_hash, template_id="tmpl_b",
        )
        await chroma_backend.upsert("inv_test2", [e1, e2, e3])

        ids = await chroma_backend.find_by_template_id("inv_test2", "tmpl_a")
        assert set(ids) == {e1.id, e2.id}

    async def test_drop_collection(self, chroma_backend):
        from tests.conftest import make_entry

        await chroma_backend.initialize("drop_test", 8)
        await chroma_backend.upsert("drop_test", [make_entry()])

        await chroma_backend.drop_collection("drop_test")

        await chroma_backend.initialize("drop_test", 8)
        assert await chroma_backend.count("drop_test") == 0


class TestCollectionIsolation:
    async def test_multiple_collections_isolated(self, mock_embedder):
        settings = Settings(backend_type="chroma", chroma_mode="ephemeral")

        m1 = Medha(collection_name="col_chroma_alpha", embedder=mock_embedder, settings=settings)
        m2 = Medha(collection_name="col_chroma_beta", embedder=mock_embedder, settings=settings)

        await m1.start()
        await m2.start()

        try:
            await m1.store("alpha question", "SELECT 'alpha'")
            await m2.store("beta question", "SELECT 'beta'")

            await m1.clear_caches()
            await m2.clear_caches()

            hit_alpha = await m1.search("alpha question")
            hit_beta = await m2.search("beta question")

            assert hit_alpha.generated_query == "SELECT 'alpha'"
            assert hit_beta.generated_query == "SELECT 'beta'"

            miss = await m1.search("beta question")
            assert miss.strategy == SearchStrategy.NO_MATCH
        finally:
            await m1.close()
            await m2.close()
