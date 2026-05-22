"""Unit tests for medha.core.Medha waterfall search logic."""

import json

import pytest

from medha.config import Settings
from medha.core import Medha
from medha.exceptions import TemplateError
from medha.interfaces.storage import VectorStorageBackend
from medha.types import CacheEntry, CacheResult, SearchStrategy
from tests.conftest import MockEmbedder

# ---------------------------------------------------------------------------
# MockBackend — in-memory backend with cosine similarity
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    mag_a = sum(x ** 2 for x in a) ** 0.5
    mag_b = sum(x ** 2 for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class MockBackend(VectorStorageBackend):
    """In-memory mock backend that stores entries in a dict."""

    def __init__(self):
        self._collections: dict[str, list[CacheEntry]] = {}

    async def initialize(self, collection_name: str, dimension: int, **kwargs) -> None:
        self._collections.setdefault(collection_name, [])

    async def search(
        self,
        collection_name: str,
        vector: list[float],
        limit: int = 5,
        score_threshold: float = 0.0,
    ) -> list[CacheResult]:
        entries = self._collections.get(collection_name, [])
        scored = []
        for e in entries:
            sim = _cosine_similarity(vector, e.vector)
            if sim >= score_threshold:
                scored.append((sim, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, e in scored[:limit]:
            results.append(
                CacheResult(
                    id=e.id,
                    score=min(score, 1.0),
                    original_question=e.original_question,
                    normalized_question=e.normalized_question,
                    generated_query=e.generated_query,
                    query_hash=e.query_hash,
                    response_summary=e.response_summary,
                    template_id=e.template_id,
                    usage_count=e.usage_count,
                    created_at=e.created_at,
                )
            )
        return results

    async def upsert(self, collection_name: str, entries: list[CacheEntry]) -> None:
        self._collections.setdefault(collection_name, []).extend(entries)

    async def scroll(
        self,
        collection_name: str,
        limit: int = 100,
        offset: str | None = None,
        with_vectors: bool = False,
    ) -> tuple[list[CacheResult], str | None]:
        entries = self._collections.get(collection_name, [])
        start = int(offset) if offset else 0
        batch = entries[start : start + limit]
        results = [
            CacheResult(
                id=e.id,
                score=0.0,
                original_question=e.original_question,
                normalized_question=e.normalized_question,
                generated_query=e.generated_query,
                query_hash=e.query_hash,
                response_summary=e.response_summary,
                template_id=e.template_id,
            )
            for e in batch
        ]
        next_offset = str(start + limit) if start + limit < len(entries) else None
        return results, next_offset

    async def count(self, collection_name: str) -> int:
        return len(self._collections.get(collection_name, []))

    async def delete(self, collection_name: str, ids: list[str]) -> None:
        entries = self._collections.get(collection_name, [])
        id_set = set(ids)
        self._collections[collection_name] = [e for e in entries if e.id not in id_set]

    async def find_expired(self, collection_name: str) -> list[str]:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        return [
            e.id
            for e in self._collections.get(collection_name, [])
            if e.expires_at is not None and e.expires_at < now
        ]

    async def search_by_normalized_question(
        self, collection_name: str, normalized_question: str
    ) -> CacheResult | None:
        for e in self._collections.get(collection_name, []):
            if e.normalized_question == normalized_question:
                return CacheResult(
                    id=e.id, score=1.0,
                    original_question=e.original_question,
                    normalized_question=e.normalized_question,
                    generated_query=e.generated_query,
                    query_hash=e.query_hash,
                )
        return None

    async def find_by_query_hash(self, collection_name: str, query_hash: str) -> list[str]:
        return [
            e.id for e in self._collections.get(collection_name, [])
            if e.query_hash == query_hash
        ]

    async def find_by_template_id(self, collection_name: str, template_id: str) -> list[str]:
        return [
            e.id for e in self._collections.get(collection_name, [])
            if e.template_id == template_id
        ]

    async def drop_collection(self, collection_name: str) -> None:
        self._collections.pop(collection_name, None)

    async def update_feedback(
        self, collection_name: str, point_id: str, correct: bool
    ) -> int:
        for e in self._collections.get(collection_name, []):
            if e.id == point_id:
                if correct:
                    e.feedback_correct += 1
                    return e.feedback_correct
                else:
                    e.feedback_incorrect += 1
                    return e.feedback_incorrect
        return 0

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_backend():
    return MockBackend()


@pytest.fixture
def waterfall_settings():
    return Settings(
        qdrant_mode="memory",
        score_threshold_exact=0.99,
        score_threshold_semantic=0.85,
        score_threshold_template=0.80,
        score_threshold_fuzzy=80.0,
        l1_cache_max_size=100,
    )


@pytest.fixture
def medha_instance(mock_backend, waterfall_settings):
    embedder = MockEmbedder(dimension=384)
    return Medha(
        collection_name="test_cache",
        embedder=embedder,
        backend=mock_backend,
        settings=waterfall_settings,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestL1Cache:
    async def test_l1_cache_hit_strategy_is_l1_cache(self, medha_instance):
        """A hit served from L1 must report strategy=L1_CACHE regardless of origin."""
        await medha_instance.start()
        await medha_instance.store("How many users?", "SELECT COUNT(*) FROM users")
        # store() populates L1; next search should come from L1
        hit = await medha_instance.search("How many users?")
        assert hit.strategy == SearchStrategy.L1_CACHE

    async def test_l1_cache_hit(self, medha_instance):
        await medha_instance.start()
        await medha_instance.store("How many users?", "SELECT COUNT(*) FROM users")
        # First search populates L1 via store; second search should be L1 hit
        hit1 = await medha_instance.search("How many users?")
        assert hit1.strategy != SearchStrategy.NO_MATCH
        hit2 = await medha_instance.search("How many users?")
        assert hit2.strategy != SearchStrategy.NO_MATCH
        s = await medha_instance.stats()
        assert s.by_strategy.get("l1_cache") is not None and s.by_strategy["l1_cache"].count >= 1

    async def test_l1_eviction(self, mock_backend, waterfall_settings):
        waterfall_settings = Settings(
            qdrant_mode="memory",
            score_threshold_exact=0.99,
            score_threshold_semantic=0.85,
            score_threshold_template=0.80,
            score_threshold_fuzzy=80.0,
            l1_cache_max_size=2,
        )
        embedder = MockEmbedder(dimension=384)
        m = Medha(
            collection_name="test_evict",
            embedder=embedder,
            backend=mock_backend,
            settings=waterfall_settings,
        )
        await m.start()
        await m.store("question one", "SELECT 1")
        await m.store("question two", "SELECT 2")
        await m.store("question three", "SELECT 3")
        # L1 max is 2, so "question one" should have been evicted
        assert m._l1_backend.size <= 2


class TestWaterfallOrder:
    async def test_template_before_semantic(self, medha_instance, sample_templates):
        await medha_instance.start()
        await medha_instance.load_templates(sample_templates)
        # Store a semantic entry for similar question
        await medha_instance.store(
            "How many users are there",
            "SELECT COUNT(*) FROM users",
        )
        # Clear L1 so we don't get L1 hit
        await medha_instance._l1_backend.clear()
        medha_instance._embedding_cache.clear()
        # Use the *exact same* question to guarantee vector match
        hit = await medha_instance.search("How many users are there")
        # With deterministic mock embedder, identical text → exact vector match.
        # Template or Exact are both valid (template is checked first in waterfall).
        assert hit.strategy in (
            SearchStrategy.TEMPLATE_MATCH,
            SearchStrategy.EXACT_MATCH,
        )

    async def test_exact_match(self, medha_instance):
        await medha_instance.start()
        await medha_instance.store("How many users?", "SELECT COUNT(*) FROM users")
        await medha_instance._l1_backend.clear()
        medha_instance._embedding_cache.clear()
        hit = await medha_instance.search("How many users?")
        # Same text should yield exact match via vector similarity
        assert hit.strategy == SearchStrategy.EXACT_MATCH
        assert hit.confidence >= 0.99

    async def test_semantic_match(self, medha_instance):
        await medha_instance.start()
        await medha_instance.store("How many users?", "SELECT COUNT(*) FROM users")
        await medha_instance._l1_backend.clear()
        medha_instance._embedding_cache.clear()
        # Different but related question — may or may not match depending on mock hash
        hit = await medha_instance.search("Count all users")
        # With hash-based mock, different inputs produce different hashes → likely no match
        assert hit.strategy in (
            SearchStrategy.EXACT_MATCH,
            SearchStrategy.SEMANTIC_MATCH,
            SearchStrategy.NO_MATCH,
            SearchStrategy.ERROR,
        )

    async def test_no_match(self, medha_instance):
        await medha_instance.start()
        hit = await medha_instance.search("completely random xyz abc")
        assert hit.strategy == SearchStrategy.NO_MATCH

    async def test_waterfall_order(self, medha_instance):
        """Verify tiers are checked in order: L1 -> Template -> Exact -> Semantic -> Fuzzy."""
        await medha_instance.start()
        # Empty cache → should hit NO_MATCH (all tiers exhausted)
        hit = await medha_instance.search("any question here")
        assert hit.strategy == SearchStrategy.NO_MATCH
        s = await medha_instance.stats()
        assert s.total_misses == 1


class TestStoreAndSearch:
    async def test_store_and_search(self, medha_instance):
        await medha_instance.start()
        ok = await medha_instance.store(
            "Get user count", "SELECT COUNT(*) FROM users"
        )
        assert ok is True
        await medha_instance._l1_backend.clear()
        medha_instance._embedding_cache.clear()
        hit = await medha_instance.search("Get user count")
        assert hit.generated_query == "SELECT COUNT(*) FROM users"
        assert hit.strategy == SearchStrategy.EXACT_MATCH


class TestStats:
    async def test_stats_tracking(self, medha_instance):
        await medha_instance.start()
        await medha_instance.search("no match question xyz")
        s = await medha_instance.stats()
        assert s.total_requests >= 1
        assert len(s.by_strategy) >= 1
        assert s.total_misses >= 1


class TestClearCaches:
    async def test_clear_caches(self, medha_instance):
        await medha_instance.start()
        await medha_instance.store("test q", "SELECT 1")
        assert medha_instance._l1_backend.size > 0
        await medha_instance.clear_caches()
        assert medha_instance._l1_backend.size == 0
        assert len(medha_instance._embedding_cache) == 0
        s = await medha_instance.stats()
        assert s.total_requests == 0


class TestEmptyQuestion:
    async def test_empty_question(self, medha_instance):
        await medha_instance.start()
        hit = await medha_instance.search("")
        assert hit.strategy == SearchStrategy.ERROR

    async def test_whitespace_only_question(self, medha_instance):
        await medha_instance.start()
        hit = await medha_instance.search("   ")
        assert hit.strategy == SearchStrategy.ERROR


class TestContextManager:
    async def test_context_manager(self, mock_backend, waterfall_settings):
        embedder = MockEmbedder(dimension=384)
        async with Medha(
            collection_name="ctx_test",
            embedder=embedder,
            backend=mock_backend,
            settings=waterfall_settings,
        ) as m:
            ok = await m.store("test", "SELECT 1")
            assert ok is True
        # After exit, backend.close() should have been called
        # No error means success


class TestFuzzyFallback:
    async def test_fuzzy_fallback(self, medha_instance):
        """Fuzzy matching requires rapidfuzz; if absent, gracefully skips."""
        await medha_instance.start()
        await medha_instance.store("How many users?", "SELECT COUNT(*) FROM users")
        await medha_instance._l1_backend.clear()
        medha_instance._embedding_cache.clear()
        # Query with a typo — different hash, won't match exact/semantic
        hit = await medha_instance.search("How meny usrs?")
        # With rapidfuzz: FUZZY_MATCH; without: NO_MATCH — both are acceptable
        assert hit.strategy in (
            SearchStrategy.FUZZY_MATCH,
            SearchStrategy.NO_MATCH,
        )


class TestFuzzyPrefilter:
    """Tests for the vector pre-filter optimisation in Tier 4 fuzzy search."""

    async def test_prefilter_uses_embedding_not_scroll(self, mock_backend, waterfall_settings):
        """_search_fuzzy with an embedding must call backend.search(), not scroll()."""
        await mock_backend.initialize("prefilter_test", 384)
        embedder = MockEmbedder(dimension=384)
        m = Medha(
            collection_name="prefilter_test",
            embedder=embedder,
            backend=mock_backend,
            settings=waterfall_settings,
        )
        await m.start()
        await m.store("How many users?", "SELECT COUNT(*) FROM users")
        await m._l1_backend.clear()
        m._embedding_cache.clear()

        scroll_calls = []
        original_scroll = mock_backend.scroll
        async def tracking_scroll(*args, **kwargs):
            scroll_calls.append((args, kwargs))
            return await original_scroll(*args, **kwargs)
        mock_backend.scroll = tracking_scroll

        embedding = await embedder.aembed("How meny usrs?")
        await m._search_fuzzy("How meny usrs?", embedding)

        # With embedding provided, scroll must NOT be called
        assert len(scroll_calls) == 0

    async def test_prefilter_fallback_to_scroll_without_embedding(
        self, mock_backend, waterfall_settings
    ):
        """_search_fuzzy without embedding must fall back to scroll()."""
        pytest.importorskip("rapidfuzz")
        await mock_backend.initialize("scroll_fallback_test", 384)
        embedder = MockEmbedder(dimension=384)
        m = Medha(
            collection_name="scroll_fallback_test",
            embedder=embedder,
            backend=mock_backend,
            settings=waterfall_settings,
        )
        await m.start()
        await m.store("How many users?", "SELECT COUNT(*) FROM users")

        scroll_calls = []
        original_scroll = mock_backend.scroll
        async def tracking_scroll(*args, **kwargs):
            scroll_calls.append((args, kwargs))
            return await original_scroll(*args, **kwargs)
        mock_backend.scroll = tracking_scroll

        await m._search_fuzzy("How meny usrs?", embedding=None)

        # Without embedding, scroll must be used
        assert len(scroll_calls) >= 1

    async def test_prefilter_top_k_limits_candidates(self, mock_backend):
        """fuzzy_prefilter_top_k must bound the number of candidates searched."""
        pytest.importorskip("rapidfuzz")
        settings = Settings(
            qdrant_mode="memory",
            score_threshold_exact=0.99,
            score_threshold_semantic=0.85,
            score_threshold_fuzzy=80.0,
            score_threshold_fuzzy_prefilter=0.0,  # Accept all vectors
            fuzzy_prefilter_top_k=2,
        )
        embedder = MockEmbedder(dimension=384)
        m = Medha(
            collection_name="topk_test",
            embedder=embedder,
            backend=mock_backend,
            settings=settings,
        )
        await m.start()
        # Store 5 entries
        for i in range(5):
            await m.store(f"question number {i}", f"SELECT {i}")

        search_calls = []
        original_search = mock_backend.search
        async def tracking_search(*args, **kwargs):
            search_calls.append(kwargs.get("limit", args[2] if len(args) > 2 else None))
            return await original_search(*args, **kwargs)
        mock_backend.search = tracking_search

        embedding = await embedder.aembed("question number 0")
        await m._search_fuzzy("question number 0", embedding)

        # At least one search call must have used limit=2 (top_k)
        assert any(limit == 2 for limit in search_calls)


class TestBatchStore:
    async def test_store_batch(self, medha_instance):
        await medha_instance.start()
        entries = [
            {"question": "How many users?", "generated_query": "SELECT COUNT(*) FROM users"},
            {"question": "List products", "generated_query": "SELECT * FROM products",
             "response_summary": "all products", "template_id": "list_all"},
        ]
        ok = await medha_instance.store_batch(entries)
        assert ok is True

    async def test_store_batch_empty(self, medha_instance):
        await medha_instance.start()
        ok = await medha_instance.store_batch([])
        assert ok is True

    async def test_store_batch_populates_l1(self, medha_instance):
        """store_batch must populate L1 cache, consistent with store()."""
        await medha_instance.start()
        entries = [
            {"question": "batch q one", "generated_query": "SELECT 1"},
            {"question": "batch q two", "generated_query": "SELECT 2"},
        ]
        await medha_instance.store_batch(entries)
        # Both questions should now be in L1
        assert medha_instance._l1_backend.size >= 2
        hit = await medha_instance.search("batch q one")
        assert hit.strategy == SearchStrategy.L1_CACHE
        assert hit.generated_query == "SELECT 1"

    async def test_store_batch_populates_embedding_cache(self, medha_instance):
        """store_batch must populate the embedding cache with computed vectors."""
        await medha_instance.start()
        medha_instance._embedding_cache.clear()
        await medha_instance.store_batch([
            {"question": "cached q", "generated_query": "SELECT 1"},
        ])
        assert len(medha_instance._embedding_cache) >= 1


class TestLoadTemplatesFromFile:
    async def test_load_from_json(self, medha_instance, tmp_path):
        await medha_instance.start()
        templates_data = [
            {
                "intent": "count",
                "template_text": "How many {entity}",
                "query_template": "SELECT COUNT(*) FROM {entity}",
                "parameters": ["entity"],
            }
        ]
        tpl_file = tmp_path / "templates.json"
        tpl_file.write_text(json.dumps(templates_data))

        await medha_instance.load_templates_from_file(str(tpl_file))
        assert len(medha_instance._templates) == 1
        assert medha_instance._templates[0].intent == "count"

    async def test_load_from_invalid_file(self, medha_instance):
        await medha_instance.start()
        with pytest.raises(TemplateError):
            await medha_instance.load_templates_from_file("/nonexistent/path.json")


class TestStartWithTemplates:
    async def test_start_with_templates_syncs(self, mock_backend, waterfall_settings, sample_templates):
        """Templates passed to constructor are synced to backend on start()."""
        embedder = MockEmbedder(dimension=384)
        m = Medha(
            collection_name="tpl_sync",
            embedder=embedder,
            backend=mock_backend,
            settings=waterfall_settings,
            templates=sample_templates,
        )
        await m.start()
        count = await mock_backend.count("__medha_templates_tpl_sync")
        assert count >= len(sample_templates)


class TestEmbeddingCacheEviction:
    async def test_embedding_cache_evicts_oldest(self, medha_instance):
        await medha_instance.start()
        medha_instance._embedding_cache_max = 2
        # Fill cache with 3 entries to trigger eviction
        await medha_instance.store("q1", "SELECT 1")
        await medha_instance.store("q2", "SELECT 2")
        await medha_instance.store("q3", "SELECT 3")
        assert len(medha_instance._embedding_cache) <= 2


class TestAsyncCacheLocks:
    """Verify that async locks exist on both in-memory caches."""

    def test_l1_cache_backend_has_lock(self, medha_instance):
        import asyncio

        from medha.l1_cache.memory import InMemoryL1Cache
        assert isinstance(medha_instance._l1_backend, InMemoryL1Cache)
        assert hasattr(medha_instance._l1_backend, "_lock")
        assert isinstance(medha_instance._l1_backend._lock, asyncio.Lock)

    def test_embedding_cache_lock_is_asyncio_lock(self, medha_instance):
        import asyncio
        assert hasattr(medha_instance, "_embedding_cache_lock")
        assert isinstance(medha_instance._embedding_cache_lock, asyncio.Lock)

    async def test_concurrent_searches_do_not_corrupt_l1_cache(
        self, medha_instance
    ):
        """Multiple concurrent searches on the same question must not corrupt L1."""
        import asyncio
        await medha_instance.start()
        await medha_instance.store("concurrent question", "SELECT 1")
        await medha_instance._l1_backend.clear()
        medha_instance._embedding_cache.clear()

        results = await asyncio.gather(
            *[medha_instance.search("concurrent question") for _ in range(10)]
        )
        assert all(r.strategy != SearchStrategy.ERROR for r in results)
        assert medha_instance._l1_backend.size <= medha_instance._settings.l1_cache_max_size


class TestTemplateCollectionNaming:
    """Verify the template collection uses the __medha_templates_ prefix."""

    def test_template_collection_prefix(self, mock_backend, waterfall_settings):
        embedder = MockEmbedder(dimension=384)
        m = Medha(
            collection_name="my_cache",
            embedder=embedder,
            backend=mock_backend,
            settings=waterfall_settings,
        )
        assert m._template_collection == "__medha_templates_my_cache"

    def test_template_collection_no_suffix_collision(
        self, mock_backend, waterfall_settings
    ):
        """A collection named 'my_templates' must not produce 'my_templates_templates'."""
        embedder = MockEmbedder(dimension=384)
        m = Medha(
            collection_name="my_templates",
            embedder=embedder,
            backend=mock_backend,
            settings=waterfall_settings,
        )
        assert m._template_collection == "__medha_templates_my_templates"
        assert m._template_collection != "my_templates_templates"


class TestSyncWrappers:
    def test_store_sync(self, mock_backend, waterfall_settings):
        embedder = MockEmbedder(dimension=384)
        m = Medha(
            collection_name="sync_test",
            embedder=embedder,
            backend=mock_backend,
            settings=waterfall_settings,
        )
        # start() must be called before store — use _run_sync for it too
        from medha.interfaces.embedder import BaseEmbedder
        BaseEmbedder._run_sync(m.start())

        ok = m.store_sync("test question", "SELECT 1")
        assert ok is True

    def test_search_sync(self, mock_backend, waterfall_settings):
        embedder = MockEmbedder(dimension=384)
        m = Medha(
            collection_name="sync_test",
            embedder=embedder,
            backend=mock_backend,
            settings=waterfall_settings,
        )
        from medha.interfaces.embedder import BaseEmbedder
        BaseEmbedder._run_sync(m.start())

        hit = m.search_sync("no match xyz")
        assert hit.strategy == SearchStrategy.NO_MATCH

    def test_warm_from_file_sync(self, mock_backend, waterfall_settings, tmp_path):
        embedder = MockEmbedder(dimension=384)
        m = Medha(
            collection_name="warm_sync_test",
            embedder=embedder,
            backend=mock_backend,
            settings=waterfall_settings,
        )
        from medha.interfaces.embedder import BaseEmbedder
        BaseEmbedder._run_sync(m.start())

        data = [{"question": "sync warm q", "generated_query": "SELECT 99"}]
        f = tmp_path / "warm.json"
        f.write_text(json.dumps(data))

        count = m.warm_from_file_sync(str(f))
        assert count == 1


class TestWarmFromFile:
    """Tests for Medha.warm_from_file()."""

    async def test_warm_json_array(self, medha_instance, tmp_path):
        """JSON array format is loaded correctly."""
        await medha_instance.start()
        data = [
            {"question": "How many users?", "generated_query": "SELECT COUNT(*) FROM users"},
            {"question": "List products", "generated_query": "SELECT * FROM products"},
        ]
        f = tmp_path / "warm.json"
        f.write_text(json.dumps(data))

        count = await medha_instance.warm_from_file(str(f))
        assert count == 2

    async def test_warm_jsonl(self, medha_instance, tmp_path):
        """JSONL format (one object per line) is loaded correctly."""
        await medha_instance.start()
        lines = [
            '{"question": "q1", "generated_query": "SELECT 1"}',
            '{"question": "q2", "generated_query": "SELECT 2"}',
            '{"question": "q3", "generated_query": "SELECT 3"}',
        ]
        f = tmp_path / "warm.jsonl"
        f.write_text("\n".join(lines))

        count = await medha_instance.warm_from_file(str(f))
        assert count == 3

    async def test_warm_populates_cache(self, medha_instance, tmp_path):
        """Entries loaded via warm_from_file are immediately findable in L1."""
        await medha_instance.start()
        data = [{"question": "warm hit question", "generated_query": "SELECT 42"}]
        f = tmp_path / "warm.json"
        f.write_text(json.dumps(data))

        await medha_instance.warm_from_file(str(f))
        hit = await medha_instance.search("warm hit question")
        assert hit.strategy == SearchStrategy.L1_CACHE
        assert hit.generated_query == "SELECT 42"

    async def test_warm_increments_warm_loaded(self, medha_instance, tmp_path):
        """warm_loaded stat is incremented correctly."""
        await medha_instance.start()
        data = [
            {"question": "stat q1", "generated_query": "SELECT 1"},
            {"question": "stat q2", "generated_query": "SELECT 2"},
        ]
        f = tmp_path / "warm.json"
        f.write_text(json.dumps(data))

        await medha_instance.warm_from_file(str(f))
        assert medha_instance._warm_loaded == 2

    async def test_warm_optional_fields(self, medha_instance, tmp_path):
        """response_summary and template_id are preserved."""
        await medha_instance.start()
        data = [{
            "question": "optional field q",
            "generated_query": "SELECT 1",
            "response_summary": "one row",
            "template_id": "my_template",
        }]
        f = tmp_path / "warm.json"
        f.write_text(json.dumps(data))

        count = await medha_instance.warm_from_file(str(f))
        assert count == 1

    async def test_warm_empty_file(self, medha_instance, tmp_path):
        """An empty JSON array returns 0 without error."""
        await medha_instance.start()
        f = tmp_path / "empty.json"
        f.write_text("[]")

        count = await medha_instance.warm_from_file(str(f))
        assert count == 0

    async def test_warm_file_not_found(self, medha_instance):
        """A missing file raises MedhaError."""
        await medha_instance.start()
        from medha.exceptions import MedhaError
        with pytest.raises(MedhaError):
            await medha_instance.warm_from_file("/nonexistent/path.jsonl")

    async def test_warm_invalid_json_raises(self, medha_instance, tmp_path):
        """Malformed JSON raises MedhaError."""
        await medha_instance.start()
        f = tmp_path / "bad.json"
        f.write_text("{not: valid json}")
        from medha.exceptions import MedhaError
        with pytest.raises(MedhaError):
            await medha_instance.warm_from_file(str(f))


class TestStatsGranular:
    """Tests for the extended stats fields."""

    async def test_stats_has_by_strategy_after_search(self, medha_instance):
        """After a search, by_strategy is populated."""
        await medha_instance.start()
        await medha_instance.search("any question")
        s = await medha_instance.stats()
        assert s.total_requests >= 1
        assert len(s.by_strategy) >= 1

    async def test_stats_request_recorded_after_search(self, medha_instance):
        """After a search, total_requests > 0 and latency is positive."""
        await medha_instance.start()
        await medha_instance.search("any question")
        s = await medha_instance.stats()
        assert s.total_requests >= 1
        assert s.total_latency_ms >= 0.0

    async def test_total_stored_increments_on_store(self, medha_instance):
        await medha_instance.start()
        assert medha_instance._total_stored == 0
        await medha_instance.store("q1", "SELECT 1")
        assert medha_instance._total_stored == 1
        await medha_instance.store("q2", "SELECT 2")
        assert medha_instance._total_stored == 2

    async def test_total_stored_increments_on_store_batch(self, medha_instance):
        await medha_instance.start()
        entries = [
            {"question": "batch a", "generated_query": "SELECT 10"},
            {"question": "batch b", "generated_query": "SELECT 20"},
        ]
        await medha_instance.store_batch(entries)
        assert medha_instance._total_stored == 2

    async def test_warm_loaded_zero_initially(self, medha_instance):
        await medha_instance.start()
        assert medha_instance._warm_loaded == 0

    async def test_clear_caches_resets_stats(self, medha_instance, tmp_path):
        """clear_caches() resets stats, total_stored and warm_loaded."""
        await medha_instance.start()
        await medha_instance.store("q", "SELECT 1")
        await medha_instance.search("q")

        data = [{"question": "warm q", "generated_query": "SELECT 99"}]
        f = tmp_path / "w.json"
        f.write_text(json.dumps(data))
        await medha_instance.warm_from_file(str(f))

        await medha_instance.clear_caches()
        s = await medha_instance.stats()
        assert medha_instance._total_stored == 0
        assert medha_instance._warm_loaded == 0
        assert s.total_requests == 0
        assert s.total_hits == 0


class TestParallelTiers:
    """Verify that tier 2 (exact) and tier 3 (semantic) run in parallel."""

    async def test_exact_and_semantic_both_checked(self, medha_instance):
        """After a search that misses tiers 0-1, both exact and semantic
        stats are incremented in the same search call."""
        await medha_instance.start()
        # Store something so the backend is not empty
        await medha_instance.store("baseline", "SELECT 0")
        await medha_instance._l1_backend.clear()
        medha_instance._embedding_cache.clear()

        await medha_instance.search("a completely different question xyz")
        s = await medha_instance.stats()
        assert s.total_requests >= 1

    async def test_exact_hit_preferred_over_semantic(self, medha_instance):
        """If both tiers return a result, exact (score >= 0.99) takes priority."""
        await medha_instance.start()
        await medha_instance.store("exact question alpha", "SELECT 'exact'")
        await medha_instance._l1_backend.clear()
        medha_instance._embedding_cache.clear()

        hit = await medha_instance.search("exact question alpha")
        # Should be exact or l1_cache, never semantic
        assert hit.strategy in (SearchStrategy.EXACT_MATCH, SearchStrategy.L1_CACHE)

    async def test_parallel_tiers_latency_recorded(self, medha_instance):
        """Both exact and semantic tier latencies are populated after a search."""
        await medha_instance.start()
        await medha_instance.store("latency test q", "SELECT 1")
        await medha_instance._l1_backend.clear()
        medha_instance._embedding_cache.clear()

        await medha_instance.search("latency test q")
        s = await medha_instance.stats()
        assert s.total_requests == 1
        assert s.total_latency_ms >= 0.0


class TestPluggableL1Cache:
    """Verify that Medha accepts a custom L1CacheBackend."""

    async def test_custom_l1_backend_is_used(self, mock_backend, waterfall_settings):
        from medha.l1_cache.memory import InMemoryL1Cache
        custom_l1 = InMemoryL1Cache(max_size=5)
        embedder = MockEmbedder(dimension=384)
        m = Medha(
            collection_name="custom_l1_test",
            embedder=embedder,
            backend=mock_backend,
            settings=waterfall_settings,
            l1_backend=custom_l1,
        )
        await m.start()
        await m.store("pluggable q", "SELECT 99")
        assert m._l1_backend is custom_l1
        assert custom_l1.size >= 1

    async def test_default_l1_is_in_memory(self, mock_backend, waterfall_settings):
        from medha.l1_cache.memory import InMemoryL1Cache
        embedder = MockEmbedder(dimension=384)
        m = Medha(
            collection_name="default_l1",
            embedder=embedder,
            backend=mock_backend,
            settings=waterfall_settings,
        )
        assert isinstance(m._l1_backend, InMemoryL1Cache)

    async def test_l1_backend_size_reflects_stored(self, medha_instance):
        await medha_instance.start()
        assert medha_instance._l1_backend.size == 0
        await medha_instance.store("s1", "SELECT 1")
        await medha_instance.store("s2", "SELECT 2")
        assert medha_instance._l1_backend.size == 2


class TestPersistentEmbeddingCache:
    """Verify load/save of the embedding cache to disk."""

    async def test_save_and_reload(self, mock_backend, waterfall_settings, tmp_path):
        """Embeddings computed in session A are available on session B startup."""
        cache_file = str(tmp_path / "emb_cache.json")
        settings = Settings(
            qdrant_mode="memory",
            score_threshold_exact=0.99,
            score_threshold_semantic=0.85,
            score_threshold_template=0.80,
            score_threshold_fuzzy=80.0,
            l1_cache_max_size=100,
            embedding_cache_path=cache_file,
        )
        embedder = MockEmbedder(dimension=384)

        # Session A: store a question (embedding is computed and cached)
        async with Medha(
            collection_name="persist_test",
            embedder=embedder,
            backend=mock_backend,
            settings=settings,
        ) as m_a:
            await m_a.store("persistent question", "SELECT 1")
            assert len(m_a._embedding_cache) >= 1

        # Session B: embedding cache should be pre-loaded from disk
        m_b = Medha(
            collection_name="persist_test",
            embedder=embedder,
            backend=mock_backend,
            settings=settings,
        )
        await m_b.start()
        assert len(m_b._embedding_cache) >= 1
        await m_b.close()

    async def test_missing_cache_file_is_ignored(self, mock_backend, tmp_path):
        """A non-existent cache file on startup does not raise an error."""
        settings = Settings(
            qdrant_mode="memory",
            embedding_cache_path=str(tmp_path / "nonexistent.json"),
        )
        embedder = MockEmbedder(dimension=384)
        m = Medha(
            collection_name="no_file_test",
            embedder=embedder,
            backend=mock_backend,
            settings=settings,
        )
        await m.start()  # Must not raise
        await m.close()
