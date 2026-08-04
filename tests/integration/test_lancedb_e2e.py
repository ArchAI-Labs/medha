"""End-to-end integration tests: MockEmbedder + LanceDBBackend (local) + Medha pipeline.

LanceDB in local mode requires no external service — runs fully on local disk
using a temporary directory per test.
"""

import hashlib
import uuid

import pytest

lancedb = pytest.importorskip("lancedb")

from medha.backends.lancedb import LanceDBBackend
from medha.config import Settings
from medha.core import Medha
from medha.types import SearchStrategy

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def lancedb_backend(tmp_path):
    settings = Settings(
        backend_type="lancedb",
        lancedb_uri=str(tmp_path / "lancedb"),
    )
    b = LanceDBBackend(settings)
    await b.connect()
    yield b
    await b.close()


@pytest.fixture
async def medha_lancedb(mock_embedder, tmp_path):
    settings = Settings(
        backend_type="lancedb",
        lancedb_uri=str(tmp_path / "lancedb_medha"),
        score_threshold_exact=0.99,
        score_threshold_semantic=0.85,
        l1_cache_max_size=100,
    )
    b = LanceDBBackend(settings)
    await b.connect()
    m = Medha(
        collection_name="lancedb_e2e",
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
    async def test_initialize_and_count(self, lancedb_backend):
        await lancedb_backend.initialize("test_col", 8)
        assert await lancedb_backend.count("test_col") == 0

    async def test_upsert_and_count(self, lancedb_backend):
        from tests.conftest import make_entry

        await lancedb_backend.initialize("test_col", 8)
        entries = [make_entry() for _ in range(5)]
        await lancedb_backend.upsert("test_col", entries)
        assert await lancedb_backend.count("test_col") == 5

    async def test_upsert_same_id_overwrites(self, lancedb_backend):
        from tests.conftest import make_entry

        await lancedb_backend.initialize("test_col", 8)
        eid = str(uuid.uuid4())
        await lancedb_backend.upsert("test_col", [make_entry(id=eid, query="SELECT 1")])
        await lancedb_backend.upsert("test_col", [make_entry(id=eid, query="SELECT 2")])
        assert await lancedb_backend.count("test_col") == 1

        results, _ = await lancedb_backend.scroll("test_col")
        assert results[0].generated_query == "SELECT 2"

    async def test_search_returns_sorted_results(self, lancedb_backend):
        from tests.conftest import make_entry

        await lancedb_backend.initialize("test_col", 4)
        await lancedb_backend.upsert("test_col", [
            make_entry(id="a", vector=[1.0, 0.0, 0.0, 0.0], query="SELECT A", dim=4),
            make_entry(id="b", vector=[0.0, 0.0, 0.0, 1.0], query="SELECT B", dim=4),
        ])

        results = await lancedb_backend.search(
            "test_col", [1.0, 0.0, 0.0, 0.0], limit=5, score_threshold=0.0
        )

        assert len(results) == 2
        assert results[0].score >= results[1].score
        assert results[0].generated_query == "SELECT A"

    async def test_scroll_pagination(self, lancedb_backend):
        from tests.conftest import make_entry

        await lancedb_backend.initialize("test_col", 8)
        entries = [make_entry() for _ in range(7)]
        await lancedb_backend.upsert("test_col", entries)

        page1, next_offset = await lancedb_backend.scroll("test_col", limit=4)
        assert len(page1) == 4
        assert next_offset == "4"

        page2, next_offset2 = await lancedb_backend.scroll(
            "test_col", limit=4, offset=next_offset
        )
        assert len(page2) == 3
        assert next_offset2 is None

    async def test_delete(self, lancedb_backend):
        from tests.conftest import make_entry

        await lancedb_backend.initialize("test_col", 8)
        e1, e2 = make_entry(), make_entry()
        await lancedb_backend.upsert("test_col", [e1, e2])
        await lancedb_backend.delete("test_col", [e1.id])
        assert await lancedb_backend.count("test_col") == 1

    async def test_search_by_normalized_question(self, lancedb_backend):
        from tests.conftest import make_entry

        await lancedb_backend.initialize("test_col", 8)
        entry = make_entry(question="how many users")
        await lancedb_backend.upsert("test_col", [entry])

        result = await lancedb_backend.search_by_normalized_question(
            "test_col", entry.normalized_question
        )

        assert result is not None
        assert result.id == entry.id

    async def test_search_by_normalized_question_not_found(self, lancedb_backend):
        await lancedb_backend.initialize("test_col", 8)
        result = await lancedb_backend.search_by_normalized_question("test_col", "xyz")
        assert result is None


# ---------------------------------------------------------------------------
# Medha pipeline tests
# ---------------------------------------------------------------------------


class TestStatsPersistence:
    """Real round-trips against an on-disk LanceDB meta table."""

    async def test_load_stats_returns_none_when_never_saved(self, lancedb_backend):
        await lancedb_backend.initialize("test_col", 8)

        assert await lancedb_backend.load_stats("test_col") is None

    async def test_save_then_load_round_trip(self, lancedb_backend):
        from medha.types import PersistedStats

        await lancedb_backend.initialize("test_col", 8)
        stats = PersistedStats(
            total_requests=20,
            total_hits=15,
            total_misses=4,
            total_errors=1,
            hits_by_strategy={"exact": 9, "semantic": 6},
        )

        await lancedb_backend.save_stats("test_col", stats)
        loaded = await lancedb_backend.load_stats("test_col")

        assert loaded is not None
        assert loaded.total_requests == 20
        assert loaded.total_hits == 15
        assert loaded.total_errors == 1
        assert loaded.hits_by_strategy == {"exact": 9, "semantic": 6}

    async def test_save_stats_overwrites_previous_snapshot(self, lancedb_backend):
        from medha.types import PersistedStats

        await lancedb_backend.initialize("test_col", 8)
        await lancedb_backend.save_stats("test_col", PersistedStats(total_requests=1))
        await lancedb_backend.save_stats("test_col", PersistedStats(total_requests=42))

        loaded = await lancedb_backend.load_stats("test_col")

        assert loaded is not None
        assert loaded.total_requests == 42

    async def test_stats_are_isolated_per_collection(self, lancedb_backend):
        from medha.types import PersistedStats

        await lancedb_backend.initialize("col_a", 8)
        await lancedb_backend.initialize("col_b", 8)

        await lancedb_backend.save_stats("col_a", PersistedStats(total_requests=3))
        await lancedb_backend.save_stats("col_b", PersistedStats(total_requests=7))

        loaded_a = await lancedb_backend.load_stats("col_a")
        loaded_b = await lancedb_backend.load_stats("col_b")
        assert loaded_a is not None and loaded_a.total_requests == 3
        assert loaded_b is not None and loaded_b.total_requests == 7

    async def test_stats_do_not_pollute_the_data_table(self, lancedb_backend):
        from medha.types import PersistedStats
        from tests.conftest import make_entry

        await lancedb_backend.initialize("test_col", 8)
        await lancedb_backend.upsert("test_col", [make_entry()])

        await lancedb_backend.save_stats("test_col", PersistedStats(total_requests=5))

        assert await lancedb_backend.count("test_col") == 1

    async def test_stats_survive_a_reconnect(self, tmp_path):
        from medha.backends.lancedb import LanceDBBackend
        from medha.types import PersistedStats

        settings = Settings(
            backend_type="lancedb",
            lancedb_uri=str(tmp_path / "lancedb_stats"),
        )

        b = LanceDBBackend(settings)
        await b.connect()
        await b.initialize("test_col", 8)
        await b.save_stats("test_col", PersistedStats(total_requests=17))
        await b.close()

        b2 = LanceDBBackend(settings)
        await b2.connect()
        try:
            loaded = await b2.load_stats("test_col")
            assert loaded is not None
            assert loaded.total_requests == 17
        finally:
            await b2.close()

    async def test_quotes_in_collection_name_are_escaped(self, lancedb_backend):
        from medha.types import PersistedStats

        weird = "bad' OR '1'='1"
        await lancedb_backend.save_stats(weird, PersistedStats(total_requests=2))

        loaded = await lancedb_backend.load_stats(weird)
        assert loaded is not None
        assert loaded.total_requests == 2
        # The injected predicate did not match a different row
        assert await lancedb_backend.load_stats("something_else") is None


class TestStoreAndSearch:
    async def test_store_and_exact_search(self, medha_lancedb):
        await medha_lancedb.store("How many users are there", "SELECT COUNT(*) FROM users")
        await medha_lancedb.clear_caches()

        hit = await medha_lancedb.search("How many users are there")
        assert hit.strategy == SearchStrategy.EXACT_MATCH
        assert hit.generated_query == "SELECT COUNT(*) FROM users"
        assert hit.confidence >= 0.99

    async def test_l1_cache_hit(self, medha_lancedb):
        await medha_lancedb.store("Get user count", "SELECT COUNT(*) FROM users")
        await medha_lancedb.clear_caches()

        hit1 = await medha_lancedb.search("Get user count")
        assert hit1.strategy != SearchStrategy.NO_MATCH

        hit2 = await medha_lancedb.search("Get user count")
        assert hit2.strategy == SearchStrategy.L1_CACHE

    async def test_no_match_empty(self, medha_lancedb):
        hit = await medha_lancedb.search("What is the meaning of life")
        assert hit.strategy == SearchStrategy.NO_MATCH
        assert not hit.generated_query

    async def test_store_batch(self, medha_lancedb):
        entries = [
            {"question": f"Question number {i}", "generated_query": f"SELECT {i}"}
            for i in range(5)
        ]
        ok = await medha_lancedb.store_batch(entries)
        assert ok is True

        results, _ = await medha_lancedb._backend.scroll("lancedb_e2e", limit=10)
        stored_queries = {r.generated_query for r in results}
        for i in range(5):
            assert f"SELECT {i}" in stored_queries


class TestBackendTypeViaSettings:
    async def test_backend_type_lancedb_via_settings(self, mock_embedder, tmp_path):
        settings = Settings(
            backend_type="lancedb",
            lancedb_uri=str(tmp_path / "lancedb_auto"),
            score_threshold_exact=0.99,
            score_threshold_semantic=0.85,
        )
        m = Medha(
            collection_name="auto_backend_test",
            embedder=mock_embedder,
            settings=settings,
        )
        assert isinstance(m._backend, LanceDBBackend)
        await m.start()
        await m.close()


class TestContextManager:
    async def test_context_manager(self, mock_embedder, tmp_path):
        settings = Settings(
            backend_type="lancedb",
            lancedb_uri=str(tmp_path / "lancedb_ctx"),
        )
        async with Medha(
            collection_name="ctx_lancedb_test",
            embedder=mock_embedder,
            settings=settings,
        ) as m:
            await m.store("test question", "SELECT 1")
            hit = await m.search("test question")
            assert hit.strategy != SearchStrategy.NO_MATCH


class TestTTL:
    async def test_expired_entry_excluded_from_search(self, mock_embedder, tmp_path):
        from datetime import datetime, timedelta, timezone

        from medha.types import CacheEntry
        from tests.conftest import make_entry

        settings = Settings(lancedb_uri=str(tmp_path / "lancedb_ttl"))
        b = LanceDBBackend(settings)
        await b.connect()
        await b.initialize("ttl_test", 4)

        try:
            expired_vec = [1.0, 0.0, 0.0, 0.0]
            base = make_entry(vector=expired_vec, question="expired q", dim=4)
            expired = CacheEntry(
                id=base.id,
                vector=expired_vec,
                original_question="expired q",
                normalized_question="expired q",
                generated_query="SELECT expired",
                query_hash=base.query_hash,
                expires_at=datetime.now(timezone.utc) - timedelta(seconds=10),
            )
            await b.upsert("ttl_test", [expired])

            results = await b.search("ttl_test", expired_vec, limit=5, score_threshold=0.0)
            assert all(r.id != expired.id for r in results)
        finally:
            await b.close()

    async def test_find_expired_returns_expired_ids(self, mock_embedder, tmp_path):
        from datetime import datetime, timedelta, timezone

        from medha.types import CacheEntry
        from tests.conftest import make_entry

        settings = Settings(lancedb_uri=str(tmp_path / "lancedb_ttl_find"))
        b = LanceDBBackend(settings)
        await b.connect()
        await b.initialize("ttl_find_test", 8)

        try:
            vec = [0.1] * 8
            base_expired = make_entry(vector=vec, question="will expire", dim=8)
            expired = CacheEntry(
                id=base_expired.id,
                vector=vec,
                original_question="will expire",
                normalized_question="will expire",
                generated_query="SELECT expired",
                query_hash=base_expired.query_hash,
                expires_at=datetime.now(timezone.utc) - timedelta(seconds=10),
            )
            base_valid = make_entry(vector=[0.2] * 8, question="will stay", dim=8)
            valid = CacheEntry(
                id=base_valid.id,
                vector=[0.2] * 8,
                original_question="will stay",
                normalized_question="will stay",
                generated_query="SELECT valid",
                query_hash=base_valid.query_hash,
                expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            )
            await b.upsert("ttl_find_test", [expired, valid])

            expired_ids = await b.find_expired("ttl_find_test")
            assert expired.id in expired_ids
            assert valid.id not in expired_ids
        finally:
            await b.close()


class TestInvalidation:
    async def test_find_by_query_hash(self, lancedb_backend):
        from tests.conftest import make_entry

        await lancedb_backend.initialize("inv_test", 8)
        query = "SELECT COUNT(*) FROM users"
        qhash = hashlib.md5(query.encode()).hexdigest()
        e1 = make_entry(question="how many users", query=query)
        e2 = make_entry(question="user count", query=query)
        e3 = make_entry(question="list products", query="SELECT * FROM products")
        await lancedb_backend.upsert("inv_test", [e1, e2, e3])

        ids = await lancedb_backend.find_by_query_hash("inv_test", qhash)
        assert set(ids) == {e1.id, e2.id}

    async def test_find_by_template_id(self, lancedb_backend):
        from medha.types import CacheEntry
        from tests.conftest import make_entry

        await lancedb_backend.initialize("inv_test2", 8)
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
        await lancedb_backend.upsert("inv_test2", [e1, e2, e3])

        ids = await lancedb_backend.find_by_template_id("inv_test2", "tmpl_a")
        assert set(ids) == {e1.id, e2.id}

    async def test_drop_collection(self, lancedb_backend):
        from tests.conftest import make_entry

        await lancedb_backend.initialize("drop_test", 8)
        await lancedb_backend.upsert("drop_test", [make_entry()])

        await lancedb_backend.drop_collection("drop_test")

        await lancedb_backend.initialize("drop_test", 8)
        assert await lancedb_backend.count("drop_test") == 0


class TestCollectionIsolation:
    async def test_multiple_collections_isolated(self, mock_embedder, tmp_path):
        settings = Settings(
            backend_type="lancedb",
            lancedb_uri=str(tmp_path / "lancedb_isolation"),
        )

        m1 = Medha(collection_name="col_lance_alpha", embedder=mock_embedder, settings=settings)
        m2 = Medha(collection_name="col_lance_beta", embedder=mock_embedder, settings=settings)

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
