"""Unit tests for the PersistedStats model and stats persistence (0.5.0)."""

import asyncio

import pytest

from medha.config import Settings
from medha.core import Medha
from medha.types import PersistedStats, SearchStrategy


class TestPersistedStatsModel:
    def test_default_hit_rate_is_zero(self):
        s = PersistedStats()
        assert s.hit_rate == 0.0

    def test_default_miss_rate_is_zero(self):
        s = PersistedStats()
        assert s.miss_rate == 0.0

    def test_hit_rate_calculation(self):
        s = PersistedStats(total_requests=10, total_hits=7)
        assert s.hit_rate == pytest.approx(0.7)

    def test_miss_rate_sum_not_required_to_be_1(self):
        """Errors are neither hits nor misses, so the two rates need not add to 1."""
        s = PersistedStats(total_requests=10, total_hits=7, total_misses=2, total_errors=1)
        assert s.miss_rate == pytest.approx(0.2)
        assert s.hit_rate + s.miss_rate == pytest.approx(0.9)

    def test_defaults_are_timezone_aware(self):
        s = PersistedStats()
        assert s.last_reset_at.tzinfo is not None
        assert s.updated_at.tzinfo is not None

    def test_json_roundtrip_preserves_counts(self):
        s = PersistedStats(
            total_requests=5,
            total_hits=3,
            hits_by_strategy={"semantic_match": 2, "l1_cache": 1},
        )
        restored = PersistedStats.model_validate_json(s.model_dump_json())
        assert restored.total_requests == 5
        assert restored.hits_by_strategy == {"semantic_match": 2, "l1_cache": 1}


class TestInMemoryBackendStats:
    async def test_load_returns_none_before_save(self, inmemory_backend):
        assert await inmemory_backend.load_stats("col") is None

    async def test_save_and_load_roundtrip(self, inmemory_backend):
        s = PersistedStats(total_requests=5, total_hits=3, hits_by_strategy={"semantic_match": 3})
        await inmemory_backend.save_stats("col", s)

        loaded = await inmemory_backend.load_stats("col")
        assert loaded is not None
        assert loaded.total_requests == 5
        assert loaded.hits_by_strategy["semantic_match"] == 3

    async def test_save_overwrites_previous(self, inmemory_backend):
        await inmemory_backend.save_stats("col", PersistedStats(total_requests=1))
        await inmemory_backend.save_stats("col", PersistedStats(total_requests=2))

        loaded = await inmemory_backend.load_stats("col")
        assert loaded is not None
        assert loaded.total_requests == 2

    async def test_collections_are_isolated(self, inmemory_backend):
        await inmemory_backend.save_stats("a", PersistedStats(total_requests=4))

        assert await inmemory_backend.load_stats("b") is None
        loaded = await inmemory_backend.load_stats("a")
        assert loaded is not None and loaded.total_requests == 4


class TestStatsCollectorSnapshot:
    """_StatsCollector must round-trip its accumulators through PersistedStats."""

    async def test_build_persisted_reflects_recorded_hits(self, medha_memory):
        await medha_memory._stats.record(SearchStrategy.SEMANTIC_MATCH, 1.0)
        await medha_memory._stats.record(SearchStrategy.L1_CACHE, 1.0)
        await medha_memory._stats.record(SearchStrategy.NO_MATCH, 1.0)
        await medha_memory._stats.record(SearchStrategy.ERROR, 1.0)

        snapshot = medha_memory._build_persisted_stats()

        assert snapshot.total_requests == 4
        assert snapshot.total_hits == 2
        assert snapshot.total_misses == 1
        assert snapshot.total_errors == 1
        assert snapshot.hits_by_strategy["semantic_match"] == 1
        assert snapshot.hits_by_strategy["l1_cache"] == 1

    async def test_build_persisted_excludes_misses_and_errors_from_strategies(self, medha_memory):
        await medha_memory._stats.record(SearchStrategy.NO_MATCH, 1.0)
        await medha_memory._stats.record(SearchStrategy.ERROR, 1.0)

        snapshot = medha_memory._build_persisted_stats()

        assert "no_match" not in snapshot.hits_by_strategy
        assert "error" not in snapshot.hits_by_strategy

    async def test_load_persisted_seeds_accumulators(self, medha_memory):
        medha_memory._load_persisted_stats(
            PersistedStats(
                total_requests=10,
                total_hits=6,
                total_misses=3,
                total_errors=1,
                hits_by_strategy={"semantic_match": 6},
            )
        )

        snapshot = medha_memory._build_persisted_stats()
        assert snapshot.total_requests == 10
        assert snapshot.total_hits == 6
        assert snapshot.total_misses == 3
        assert snapshot.total_errors == 1
        assert snapshot.hits_by_strategy["semantic_match"] == 6

    async def test_load_persisted_then_record_accumulates(self, medha_memory):
        medha_memory._load_persisted_stats(PersistedStats(total_requests=10, total_hits=6))
        await medha_memory._stats.record(SearchStrategy.SEMANTIC_MATCH, 1.0)

        snapshot = medha_memory._build_persisted_stats()
        assert snapshot.total_requests == 11
        assert snapshot.total_hits == 7


class TestStatsPersistenceLifecycle:
    """start() restores a snapshot; search() flushes one every N requests."""

    async def test_start_loads_persisted_stats(self, mock_embedder):
        from medha.backends.memory import InMemoryBackend

        backend = InMemoryBackend()
        await backend.connect()
        await backend.save_stats(
            "lifecycle", PersistedStats(total_requests=42, total_hits=30)
        )

        m = Medha(
            collection_name="lifecycle",
            embedder=mock_embedder,
            backend=backend,
            settings=Settings(backend_type="memory"),
        )
        await m.start()
        try:
            snapshot = m._build_persisted_stats()
            assert snapshot.total_requests == 42
            assert snapshot.total_hits == 30
        finally:
            await m.close()

    async def test_start_without_snapshot_starts_cold(self, mock_embedder):
        from medha.backends.memory import InMemoryBackend

        backend = InMemoryBackend()
        await backend.connect()

        m = Medha(
            collection_name="cold",
            embedder=mock_embedder,
            backend=backend,
            settings=Settings(backend_type="memory"),
        )
        await m.start()
        try:
            assert m._build_persisted_stats().total_requests == 0
        finally:
            await m.close()

    async def test_search_persists_on_interval(self, mock_embedder):
        from medha.backends.memory import InMemoryBackend

        backend = InMemoryBackend()
        await backend.connect()

        m = Medha(
            collection_name="interval",
            embedder=mock_embedder,
            backend=backend,
            settings=Settings(backend_type="memory", stats_persist_interval=2),
        )
        await m.start()
        try:
            await m.search("first question")
            # Interval is 2: nothing flushed yet.
            assert await backend.load_stats("interval") is None

            await m.search("second question")
            await asyncio.gather(*m._stats_persist_tasks)  # fire-and-forget task

            persisted = await backend.load_stats("interval")
            assert persisted is not None
            assert persisted.total_requests == 2
        finally:
            await m.close()

    async def test_persist_failure_does_not_break_search(self, mock_embedder):
        """Stats persistence is best-effort: a failing backend must not surface."""
        from medha.backends.memory import InMemoryBackend

        backend = InMemoryBackend()
        await backend.connect()

        async def _boom(collection_name, stats):
            raise RuntimeError("stats backend down")

        backend.save_stats = _boom

        m = Medha(
            collection_name="failing",
            embedder=mock_embedder,
            backend=backend,
            settings=Settings(backend_type="memory", stats_persist_interval=1),
        )
        await m.start()
        try:
            result = await m.search("a question")
            await asyncio.gather(*m._stats_persist_tasks)
            assert result.strategy == SearchStrategy.NO_MATCH  # search still worked
        finally:
            await m.close()

    async def test_stats_not_persisted_when_collection_disabled(self, mock_embedder):
        from medha.backends.memory import InMemoryBackend

        backend = InMemoryBackend()
        await backend.connect()

        m = Medha(
            collection_name="nostats",
            embedder=mock_embedder,
            backend=backend,
            settings=Settings(
                backend_type="memory", collect_stats=False, stats_persist_interval=1
            ),
        )
        await m.start()
        try:
            await m.search("a question")
            assert not m._stats_persist_tasks
            assert await backend.load_stats("nostats") is None
        finally:
            await m.close()
