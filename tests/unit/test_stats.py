"""Unit tests for the Medha observability API (stats / reset_stats)."""

import pytest

from medha.config import Settings
from medha.core import Medha, _StatsCollector
from medha.types import CacheEntry, CacheResult, CacheStats, SearchStrategy, StrategyStats
from medha.interfaces.storage import VectorStorageBackend
from tests.conftest import MockEmbedder


# ---------------------------------------------------------------------------
# Minimal mock backend
# ---------------------------------------------------------------------------

class _SimpleBackend(VectorStorageBackend):
    def __init__(self):
        self._collections: dict[str, list[CacheEntry]] = {}

    async def initialize(self, collection_name: str, dimension: int, **kwargs) -> None:
        self._collections.setdefault(collection_name, [])

    async def search(self, collection_name, vector, limit=5, score_threshold=0.0):
        return []

    async def upsert(self, collection_name, entries):
        self._collections.setdefault(collection_name, []).extend(entries)

    async def scroll(self, collection_name, limit=100, offset=None, with_vectors=False):
        return [], None

    async def count(self, collection_name: str) -> int:
        return len(self._collections.get(collection_name, []))

    async def delete(self, collection_name, ids):
        entries = self._collections.get(collection_name, [])
        id_set = set(ids)
        self._collections[collection_name] = [e for e in entries if e.id not in id_set]

    async def find_expired(self, collection_name):
        return []

    async def search_by_normalized_question(self, collection_name, normalized_question):
        return None

    async def find_by_query_hash(self, collection_name, query_hash):
        return []

    async def find_by_template_id(self, collection_name, template_id):
        return []

    async def drop_collection(self, collection_name):
        self._collections.pop(collection_name, None)

    async def update_feedback(self, collection_name: str, point_id: str, correct: bool) -> int:
        return 0

    async def close(self):
        pass


def _make_medha(collect_stats: bool = True, max_samples: int = 10_000) -> Medha:
    settings = Settings(
        backend_type="memory",
        score_threshold_exact=0.99,
        score_threshold_semantic=0.85,
        collect_stats=collect_stats,
        stats_max_latency_samples=max_samples,
    )
    return Medha(
        collection_name="test_stats",
        embedder=MockEmbedder(dimension=8),
        backend=_SimpleBackend(),
        settings=settings,
    )


# ---------------------------------------------------------------------------
# _StatsCollector unit tests
# ---------------------------------------------------------------------------

class TestStatsCollector:
    @pytest.mark.asyncio
    async def test_initial_snapshot_is_zero(self):
        collector = _StatsCollector()
        snap = await collector.snapshot(backend_count=0)
        assert snap.total_requests == 0
        assert snap.total_hits == 0
        assert snap.total_misses == 0
        assert snap.total_errors == 0
        assert snap.hit_rate == 0.0
        assert snap.miss_rate == 0.0
        assert snap.avg_latency_ms == 0.0
        assert snap.p50_latency_ms == 0.0
        assert snap.p95_latency_ms == 0.0
        assert snap.p99_latency_ms == 0.0
        assert snap.backend_count == 0

    @pytest.mark.asyncio
    async def test_record_hit_increments_hits(self):
        collector = _StatsCollector()
        await collector.record(SearchStrategy.EXACT_MATCH, 10.0)
        snap = await collector.snapshot(backend_count=0)
        assert snap.total_requests == 1
        assert snap.total_hits == 1
        assert snap.total_misses == 0
        assert snap.total_errors == 0

    @pytest.mark.asyncio
    async def test_record_miss_increments_misses(self):
        collector = _StatsCollector()
        await collector.record(SearchStrategy.NO_MATCH, 5.0)
        snap = await collector.snapshot(backend_count=0)
        assert snap.total_misses == 1
        assert snap.total_hits == 0

    @pytest.mark.asyncio
    async def test_record_error_increments_errors(self):
        collector = _StatsCollector()
        await collector.record(SearchStrategy.ERROR, 1.0)
        snap = await collector.snapshot(backend_count=0)
        assert snap.total_errors == 1
        assert snap.total_hits == 0

    @pytest.mark.asyncio
    async def test_all_hit_strategies_are_counted_as_hits(self):
        collector = _StatsCollector()
        for strategy in (
            SearchStrategy.L1_CACHE,
            SearchStrategy.TEMPLATE_MATCH,
            SearchStrategy.EXACT_MATCH,
            SearchStrategy.SEMANTIC_MATCH,
            SearchStrategy.FUZZY_MATCH,
        ):
            await collector.record(strategy, 1.0)
        snap = await collector.snapshot(backend_count=0)
        assert snap.total_hits == 5
        assert snap.total_misses == 0
        assert snap.total_errors == 0

    @pytest.mark.asyncio
    async def test_by_strategy_populated(self):
        collector = _StatsCollector()
        await collector.record(SearchStrategy.EXACT_MATCH, 10.0)
        await collector.record(SearchStrategy.NO_MATCH, 5.0)
        snap = await collector.snapshot(backend_count=0)
        assert "exact_match" in snap.by_strategy
        assert "no_match" in snap.by_strategy
        assert snap.by_strategy["exact_match"].count == 1
        assert snap.by_strategy["no_match"].count == 1

    @pytest.mark.asyncio
    async def test_percentile_calculation(self):
        collector = _StatsCollector()
        # Record 100 latencies: 1ms, 2ms, ..., 100ms
        for i in range(1, 101):
            await collector.record(SearchStrategy.EXACT_MATCH, float(i))
        snap = await collector.snapshot(backend_count=0)
        # p50 should be around 50ms, p95 around 95ms, p99 around 99ms
        assert 45.0 <= snap.p50_latency_ms <= 55.0
        assert 90.0 <= snap.p95_latency_ms <= 100.0
        assert 95.0 <= snap.p99_latency_ms <= 100.0

    @pytest.mark.asyncio
    async def test_reset_clears_all_state(self):
        collector = _StatsCollector()
        await collector.record(SearchStrategy.EXACT_MATCH, 10.0)
        await collector.record(SearchStrategy.NO_MATCH, 5.0)
        await collector.reset()
        snap = await collector.snapshot(backend_count=7)
        assert snap.total_requests == 0
        assert snap.total_hits == 0
        assert snap.total_misses == 0
        assert snap.total_errors == 0
        assert snap.total_latency_ms == 0.0
        assert snap.by_strategy == {}
        assert snap.backend_count == 7

    @pytest.mark.asyncio
    async def test_disabled_collector_is_noop(self):
        collector = _StatsCollector(enabled=False)
        await collector.record(SearchStrategy.EXACT_MATCH, 100.0)
        snap = await collector.snapshot(backend_count=0)
        assert snap.total_requests == 0

    @pytest.mark.asyncio
    async def test_latency_sample_cap(self):
        collector = _StatsCollector(max_latency_samples=10)
        for i in range(50):
            await collector.record(SearchStrategy.EXACT_MATCH, float(i))
        snap = await collector.snapshot(backend_count=0)
        # total_requests counts all 50, but percentiles are based on last 10 samples
        assert snap.total_requests == 50
        # last 10 samples are 40..49, so p50 should be in that range
        assert snap.p50_latency_ms >= 40.0

    @pytest.mark.asyncio
    async def test_avg_latency_ms_property(self):
        collector = _StatsCollector()
        await collector.record(SearchStrategy.EXACT_MATCH, 20.0)
        await collector.record(SearchStrategy.NO_MATCH, 40.0)
        snap = await collector.snapshot(backend_count=0)
        assert snap.avg_latency_ms == pytest.approx(30.0)

    @pytest.mark.asyncio
    async def test_strategy_stats_avg_latency(self):
        collector = _StatsCollector()
        await collector.record(SearchStrategy.EXACT_MATCH, 10.0)
        await collector.record(SearchStrategy.EXACT_MATCH, 30.0)
        snap = await collector.snapshot(backend_count=0)
        s = snap.by_strategy["exact_match"]
        assert s.count == 2
        assert s.total_latency_ms == pytest.approx(40.0)
        assert s.avg_latency_ms == pytest.approx(20.0)

    @pytest.mark.asyncio
    async def test_snapshot_backend_count_reflected(self):
        collector = _StatsCollector()
        snap = await collector.snapshot(backend_count=42)
        assert snap.backend_count == 42

    @pytest.mark.asyncio
    async def test_hit_rate_miss_rate_sum_to_one_no_errors(self):
        collector = _StatsCollector()
        await collector.record(SearchStrategy.EXACT_MATCH, 1.0)
        await collector.record(SearchStrategy.NO_MATCH, 1.0)
        snap = await collector.snapshot(backend_count=0)
        assert snap.hit_rate + snap.miss_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Medha integration-level stats tests
# ---------------------------------------------------------------------------

class TestMedhaStats:
    @pytest.mark.asyncio
    async def test_stats_hit_rate_zero_before_search(self):
        m = _make_medha()
        await m.start()
        s = await m.stats()
        assert s.hit_rate == 0.0
        assert s.total_requests == 0
        await m.close()

    @pytest.mark.asyncio
    async def test_hit_rate_correct_after_searches(self):
        m = _make_medha()
        await m.start()
        # store() puts the entry in L1 cache; subsequent same-question search hits L1
        await m.store("How many users?", "SELECT COUNT(*) FROM users")
        await m.search("How many users?")   # L1 hit (stored → L1)
        await m.search("xyz no match abc")  # miss
        await m.search("xyz no match def")  # miss

        s = await m.stats()
        assert s.total_requests == 3
        assert s.total_hits >= 1
        assert s.hit_rate > 0.0
        await m.close()

    @pytest.mark.asyncio
    async def test_reset_stats_zeroes_counters(self):
        m = _make_medha()
        await m.start()
        await m.search("question one")
        await m.search("question two")

        await m.reset_stats()
        s = await m.stats()
        assert s.total_requests == 0
        assert s.total_hits == 0
        assert s.total_misses == 0
        assert s.total_errors == 0
        assert s.by_strategy == {}
        await m.close()

    @pytest.mark.asyncio
    async def test_collect_stats_false_is_noop(self):
        m = _make_medha(collect_stats=False)
        await m.start()
        await m.search("test question")
        s = await m.stats()
        assert s.total_requests == 0
        await m.close()

    @pytest.mark.asyncio
    async def test_backend_count_reflects_stored_entries(self):
        m = _make_medha()
        await m.start()
        await m.store("q1", "SELECT 1")
        await m.store("q2", "SELECT 2")
        s = await m.stats()
        assert s.backend_count == 2
        await m.close()

    @pytest.mark.asyncio
    async def test_by_strategy_populated_per_strategy(self):
        m = _make_medha()
        await m.start()
        await m.search("no match a")
        await m.search("no match b")
        s = await m.stats()
        assert "no_match" in s.by_strategy
        assert s.by_strategy["no_match"].count == 2
        await m.close()

    @pytest.mark.asyncio
    async def test_stats_since_before_until(self):
        m = _make_medha()
        await m.start()
        await m.search("test")
        s = await m.stats()
        assert s.since <= s.until
        await m.close()

    @pytest.mark.asyncio
    async def test_cache_stats_str_human_readable(self):
        m = _make_medha()
        await m.start()
        await m.search("q")
        s = await m.stats()
        text = str(s)
        assert "requests=" in text
        assert "hit_rate=" in text
        assert "backend_count=" in text
        await m.close()

    @pytest.mark.asyncio
    async def test_stats_frozen_model(self):
        s = CacheStats()
        with pytest.raises(Exception):
            s.total_requests = 99  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_strategy_stats_frozen_model(self):
        s = StrategyStats()
        with pytest.raises(Exception):
            s.count = 99  # type: ignore[misc]
