"""Integration tests for QdrantBackend with in-memory Qdrant."""

import uuid

import pytest

from medha.backends.qdrant import QdrantBackend
from medha.config import Settings
from medha.types import CacheEntry
from medha.utils.normalization import query_hash
from tests.conftest import MockEmbedder

COLLECTION = "test_integration"
DIMENSION = 384


@pytest.fixture
async def backend():
    settings = Settings(qdrant_mode="memory")
    b = QdrantBackend(settings)
    await b.connect()
    yield b
    await b.close()


@pytest.fixture
def embedder():
    return MockEmbedder(dimension=DIMENSION)


async def _make_entry(embedder: MockEmbedder, question: str, query: str) -> CacheEntry:
    vec = await embedder.aembed(question.lower())
    return CacheEntry(
        id=str(uuid.uuid4()),
        vector=vec,
        original_question=question,
        normalized_question=question.lower(),
        generated_query=query,
        query_hash=query_hash(query),
    )


class TestInitialize:
    async def test_initialize_creates_collection(self, backend):
        await backend.initialize(COLLECTION, DIMENSION)
        collections = await backend.client.get_collections()
        names = {c.name for c in collections.collections}
        assert COLLECTION in names

    async def test_initialize_idempotent(self, backend):
        await backend.initialize(COLLECTION, DIMENSION)
        await backend.initialize(COLLECTION, DIMENSION)
        count = await backend.count(COLLECTION)
        assert count == 0


class TestUpsertAndSearch:
    async def test_upsert_and_search(self, backend, embedder):
        await backend.initialize(COLLECTION, DIMENSION)

        entry = await _make_entry(embedder, "How many users?", "SELECT COUNT(*) FROM users")
        await backend.upsert(COLLECTION, [entry])

        vec = await embedder.aembed("how many users?")
        results = await backend.search(COLLECTION, vec, limit=5, score_threshold=0.0)
        assert len(results) >= 1
        assert results[0].generated_query == "SELECT COUNT(*) FROM users"
        assert results[0].score > 0.9

    async def test_search_threshold(self, backend, embedder):
        await backend.initialize(COLLECTION, DIMENSION)

        entry = await _make_entry(embedder, "How many users?", "SELECT COUNT(*) FROM users")
        await backend.upsert(COLLECTION, [entry])

        # Search with an unrelated vector — should not match at high threshold
        unrelated_vec = await embedder.aembed("completely unrelated query xyz")
        results = await backend.search(COLLECTION, unrelated_vec, limit=5, score_threshold=0.99)
        assert len(results) == 0


class TestScrollPagination:
    async def test_scroll_pagination(self, backend, embedder):
        await backend.initialize(COLLECTION, DIMENSION)

        # Insert 5 entries
        entries = []
        for i in range(5):
            e = await _make_entry(embedder, f"question {i}", f"SELECT {i}")
            entries.append(e)
        await backend.upsert(COLLECTION, entries)

        # Scroll with limit=2 to test pagination
        all_results = []
        offset = None
        while True:
            batch, offset = await backend.scroll(COLLECTION, limit=2, offset=offset)
            all_results.extend(batch)
            if offset is None:
                break

        assert len(all_results) == 5


class TestSearchByQueryHash:
    async def test_search_by_query_hash(self, backend, embedder):
        await backend.initialize(COLLECTION, DIMENSION)

        query = "SELECT COUNT(*) FROM users"
        entry = await _make_entry(embedder, "How many users?", query)
        await backend.upsert(COLLECTION, [entry])

        result = await backend.search_by_query_hash(COLLECTION, query_hash(query))
        assert result is not None
        assert result.generated_query == query

    async def test_search_by_query_hash_not_found(self, backend, embedder):
        await backend.initialize(COLLECTION, DIMENSION)

        result = await backend.search_by_query_hash(COLLECTION, "nonexistent_hash")
        assert result is None


class TestDelete:
    async def test_delete(self, backend, embedder):
        await backend.initialize(COLLECTION, DIMENSION)

        entry = await _make_entry(embedder, "to delete", "SELECT 1")
        await backend.upsert(COLLECTION, [entry])
        assert await backend.count(COLLECTION) == 1

        await backend.delete(COLLECTION, [entry.id])
        assert await backend.count(COLLECTION) == 0


class TestCount:
    async def test_count(self, backend, embedder):
        await backend.initialize(COLLECTION, DIMENSION)
        assert await backend.count(COLLECTION) == 0

        entries = []
        for i in range(3):
            e = await _make_entry(embedder, f"q{i}", f"SELECT {i}")
            entries.append(e)
        await backend.upsert(COLLECTION, entries)
        assert await backend.count(COLLECTION) == 3


class TestUpdateUsageCount:
    async def test_update_usage_count(self, backend, embedder):
        await backend.initialize(COLLECTION, DIMENSION)

        entry = await _make_entry(embedder, "usage test", "SELECT 1")
        await backend.upsert(COLLECTION, [entry])

        vec = await embedder.aembed("usage test")
        results = await backend.search(COLLECTION, vec, limit=1)
        initial_count = results[0].usage_count

        # Increment usage count
        await backend.update_usage_count(COLLECTION, entry.id)

        results = await backend.search(COLLECTION, vec, limit=1)
        assert results[0].usage_count == initial_count + 1

    async def test_update_usage_count_nonexistent(self, backend):
        await backend.initialize(COLLECTION, DIMENSION)
        # Should not raise — just logs a warning
        await backend.update_usage_count(COLLECTION, "nonexistent-id")


class TestStats:
    async def test_load_stats_returns_none_when_never_saved(self, backend):
        await backend.initialize(COLLECTION, DIMENSION)

        assert await backend.load_stats(COLLECTION) is None

    async def test_save_then_load_round_trip(self, backend):
        from medha.types import PersistedStats

        await backend.initialize(COLLECTION, DIMENSION)
        stats = PersistedStats(
            total_requests=12,
            total_hits=9,
            total_misses=3,
            hits_by_strategy={"exact": 5, "semantic": 4},
        )

        await backend.save_stats(COLLECTION, stats)
        loaded = await backend.load_stats(COLLECTION)

        assert loaded is not None
        assert loaded.total_requests == 12
        assert loaded.total_hits == 9
        assert loaded.hits_by_strategy == {"exact": 5, "semantic": 4}

    async def test_save_stats_overwrites_previous_snapshot(self, backend):
        from medha.types import PersistedStats

        await backend.initialize(COLLECTION, DIMENSION)
        await backend.save_stats(COLLECTION, PersistedStats(total_requests=1))
        await backend.save_stats(COLLECTION, PersistedStats(total_requests=99))

        loaded = await backend.load_stats(COLLECTION)

        assert loaded is not None
        assert loaded.total_requests == 99

    async def test_stats_do_not_pollute_the_data_collection(self, backend, embedder):
        from medha.types import PersistedStats

        await backend.initialize(COLLECTION, DIMENSION)
        await backend.upsert(
            COLLECTION, [await _make_entry(embedder, "q", "SELECT 1")]
        )

        await backend.save_stats(COLLECTION, PersistedStats(total_requests=4))

        assert await backend.count(COLLECTION) == 1

    async def test_stats_are_isolated_per_collection(self, backend):
        from medha.types import PersistedStats

        await backend.initialize(COLLECTION, DIMENSION)
        await backend.initialize("other_collection", DIMENSION)

        await backend.save_stats(COLLECTION, PersistedStats(total_requests=7))

        assert await backend.load_stats("other_collection") is None
        loaded = await backend.load_stats(COLLECTION)
        assert loaded is not None
        assert loaded.total_requests == 7


class TestCloseAndReopen:
    async def test_close_and_reopen(self):
        settings = Settings(qdrant_mode="memory")
        b = QdrantBackend(settings)
        await b.connect()
        await b.initialize(COLLECTION, DIMENSION)
        await b.close()

        # Reopen — new in-memory instance, so collection is gone
        await b.connect()
        # Re-initializing should work without error
        await b.initialize(COLLECTION, DIMENSION)
        count = await b.count(COLLECTION)
        assert count == 0
        await b.close()
