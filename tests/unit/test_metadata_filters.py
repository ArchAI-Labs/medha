"""Unit tests for metadata filtering across the waterfall.

The scenario throughout is the one that motivated the feature: two questions
that are structurally identical and differ only by the period they ask about.
Their embeddings sit on top of each other, so the semantic tier alone cannot
tell them apart — only the metadata can.
"""

import pytest

from medha.backends.memory import InMemoryBackend
from medha.config import Settings
from medha.core import Medha
from medha.exceptions import ConfigurationError
from medha.interfaces.storage import VectorStorageBackend
from medha.types import CacheResult, SearchStrategy
from medha.utils.normalization import question_hash

QUESTION = "total revenue for the period"
QUERY_12 = "SELECT SUM(amount) FROM sales WHERE day = '2026-08-12'"
QUERY_13 = "SELECT SUM(amount) FROM sales WHERE day = '2026-08-13'"


def _settings(**overrides):
    base = dict(
        backend_type="memory",
        score_threshold_exact=0.99,
        score_threshold_semantic=0.50,
        l1_cache_max_size=100,
    )
    base.update(overrides)
    return Settings(**base)


async def _medha(mock_embedder, settings=None, backend=None, collection="meta_filters"):
    m = Medha(
        collection_name=collection,
        embedder=mock_embedder,
        backend=backend or InMemoryBackend(),
        settings=settings or _settings(),
    )
    await m.start()
    return m


class _NoMetadataBackend(InMemoryBackend):
    """A backend that has not adopted metadata yet."""

    supports_metadata = False


@pytest.fixture
async def medha(mock_embedder):
    m = await _medha(mock_embedder)
    yield m
    await m.close()


# ---------------------------------------------------------------------------
# The guardrail itself
# ---------------------------------------------------------------------------

class TestFilteredSearch:
    async def test_entries_differing_only_in_metadata_are_never_confused(self, medha):
        """The headline case: same question text, two days, two queries."""
        await medha.store(QUESTION, QUERY_12, metadata={"resolved_date": "2026-08-12"})
        await medha.store(QUESTION, QUERY_13, metadata={"resolved_date": "2026-08-13"})

        hit_12 = await medha.search(QUESTION, filters={"resolved_date": "2026-08-12"})
        hit_13 = await medha.search(QUESTION, filters={"resolved_date": "2026-08-13"})

        assert hit_12.generated_query == QUERY_12
        assert hit_13.generated_query == QUERY_13

    async def test_unmatched_filter_returns_no_match_not_a_near_neighbour(self, medha):
        await medha.store(QUESTION, QUERY_12, metadata={"resolved_date": "2026-08-12"})

        hit = await medha.search(QUESTION, filters={"resolved_date": "2026-08-14"})

        assert hit.strategy == SearchStrategy.NO_MATCH
        assert hit.generated_query == ""

    async def test_entry_without_metadata_never_answers_a_filtered_search(self, medha):
        """Entries written before metadata existed carry none, and stay out."""
        await medha.store(QUESTION, QUERY_12)

        hit = await medha.search(QUESTION, filters={"resolved_date": "2026-08-12"})

        assert hit.strategy == SearchStrategy.NO_MATCH

    async def test_filter_on_a_key_the_entry_lacks_is_a_mismatch(self, medha):
        await medha.store(QUESTION, QUERY_12, metadata={"resolved_date": "2026-08-12"})

        hit = await medha.search(QUESTION, filters={"tenant": "acme"})

        assert hit.strategy == SearchStrategy.NO_MATCH

    async def test_every_filter_key_must_hold(self, medha):
        await medha.store(
            QUESTION, QUERY_12,
            metadata={"resolved_date": "2026-08-12", "tenant": "acme"},
        )

        both = await medha.search(
            QUESTION, filters={"resolved_date": "2026-08-12", "tenant": "acme"}
        )
        one_wrong = await medha.search(
            QUESTION, filters={"resolved_date": "2026-08-12", "tenant": "other"}
        )

        assert both.generated_query == QUERY_12
        assert one_wrong.strategy == SearchStrategy.NO_MATCH

    async def test_hit_reports_the_scope_it_was_served_from(self, medha):
        await medha.store(QUESTION, QUERY_12, metadata={"resolved_date": "2026-08-12"})

        hit = await medha.search(QUESTION, filters={"resolved_date": "2026-08-12"})

        assert hit.metadata == {"resolved_date": "2026-08-12"}


# ---------------------------------------------------------------------------
# Nothing changes for callers that do not use the feature
# ---------------------------------------------------------------------------

class TestUnfilteredIsUnchanged:
    async def test_unfiltered_search_still_matches_an_entry_with_metadata(self, medha):
        await medha.store(QUESTION, QUERY_12, metadata={"resolved_date": "2026-08-12"})

        hit = await medha.search(QUESTION)

        assert hit.generated_query == QUERY_12

    async def test_unfiltered_search_does_not_call_search_filtered(self, medha, monkeypatch):
        """The common path stays on the method every backend implements."""
        calls = []
        original = medha._backend.search_filtered

        async def tracking(*args, **kwargs):
            calls.append(kwargs)
            return await original(*args, **kwargs)

        monkeypatch.setattr(medha._backend, "search_filtered", tracking)
        await medha.store(QUESTION, QUERY_12)
        await medha._l1_backend.clear()
        await medha.search(QUESTION)

        assert calls == []

    async def test_store_without_metadata_works_on_a_backend_without_support(
        self, mock_embedder
    ):
        m = await _medha(mock_embedder, backend=_NoMetadataBackend(), collection="no_meta")
        try:
            assert await m.store(QUESTION, QUERY_12) is True
            assert (await m.search(QUESTION)).generated_query == QUERY_12
        finally:
            await m.close()


# ---------------------------------------------------------------------------
# L1 (Tier 0)
# ---------------------------------------------------------------------------

class TestL1RespectsFilters:
    async def test_filtered_search_is_not_served_an_unfiltered_l1_entry(self, medha):
        """The regression this feature would be worthless without.

        L1 is keyed by question alone, and it runs before every tier that
        knows about metadata. Without namespacing the key, the unfiltered
        answer cached by the first search would be handed straight back to a
        search asking for a different day.
        """
        await medha.store(QUESTION, QUERY_12, metadata={"resolved_date": "2026-08-12"})
        unfiltered = await medha.search(QUESTION)
        assert unfiltered.generated_query == QUERY_12

        hit = await medha.search(QUESTION, filters={"resolved_date": "2026-08-13"})

        assert hit.strategy == SearchStrategy.NO_MATCH

    async def test_filtered_searches_do_not_leak_into_each_other(self, medha):
        await medha.store(QUESTION, QUERY_12, metadata={"resolved_date": "2026-08-12"})
        await medha.store(QUESTION, QUERY_13, metadata={"resolved_date": "2026-08-13"})

        first = await medha.search(QUESTION, filters={"resolved_date": "2026-08-12"})
        second = await medha.search(QUESTION, filters={"resolved_date": "2026-08-13"})
        # Both are L1 hits by now; they must still be the right ones.
        again_first = await medha.search(QUESTION, filters={"resolved_date": "2026-08-12"})
        again_second = await medha.search(QUESTION, filters={"resolved_date": "2026-08-13"})

        assert again_first.strategy == SearchStrategy.L1_CACHE
        assert again_second.strategy == SearchStrategy.L1_CACHE
        assert again_first.generated_query == first.generated_query == QUERY_12
        assert again_second.generated_query == second.generated_query == QUERY_13

    async def test_unfiltered_key_is_unchanged(self, medha):
        await medha.store(QUESTION, QUERY_12)

        assert medha._l1_key(QUESTION) == question_hash(QUESTION)
        assert await medha._l1_backend.get(question_hash(QUESTION)) is not None

    async def test_filtered_key_is_prefixed_by_the_question_hash(self, medha):
        key = medha._l1_key(QUESTION, {"resolved_date": "2026-08-12"})

        assert key.startswith(question_hash(QUESTION))
        assert key != question_hash(QUESTION)

    async def test_filter_key_order_does_not_matter(self, medha):
        a = medha._l1_key(QUESTION, {"tenant": "acme", "resolved_date": "2026-08-12"})
        b = medha._l1_key(QUESTION, {"resolved_date": "2026-08-12", "tenant": "acme"})

        assert a == b

    async def test_invalidate_clears_the_filtered_variants_too(self, medha):
        await medha.store(QUESTION, QUERY_12, metadata={"resolved_date": "2026-08-12"})
        await medha.search(QUESTION, filters={"resolved_date": "2026-08-12"})
        assert medha._l1_backend.size >= 2  # unfiltered (from store) + filtered

        await medha.invalidate(QUESTION)

        hit = await medha.search(QUESTION, filters={"resolved_date": "2026-08-12"})
        assert hit.strategy == SearchStrategy.NO_MATCH


class TestL1InvalidatePrefix:
    async def test_in_memory_drops_only_the_matching_keys(self):
        from medha.l1_cache.memory import InMemoryL1Cache
        from medha.types import CacheHit

        cache = InMemoryL1Cache(max_size=10)
        await cache.set("aaaa", CacheHit(generated_query="a"))
        await cache.set("aaaa:1234", CacheHit(generated_query="a-filtered"))
        await cache.set("bbbb", CacheHit(generated_query="b"))

        await cache.invalidate_prefix("aaaa")

        assert await cache.get("aaaa") is None
        assert await cache.get("aaaa:1234") is None
        assert await cache.get("bbbb") is not None

    async def test_default_implementation_clears_everything(self):
        """A third-party L1 keeps working, at the cost of a full flush."""
        from medha.interfaces.l1_cache import L1CacheBackend
        from medha.types import CacheHit

        class MinimalL1(L1CacheBackend):
            def __init__(self):
                self.data = {}

            async def get(self, key):
                return self.data.get(key)

            async def set(self, key, value):
                self.data[key] = value

            async def clear(self):
                self.data.clear()

            async def invalidate(self, key):
                self.data.pop(key, None)

            @property
            def size(self):
                return len(self.data)

        cache = MinimalL1()
        await cache.set("aaaa", CacheHit(generated_query="a"))
        await cache.set("bbbb", CacheHit(generated_query="b"))

        await cache.invalidate_prefix("aaaa")

        assert cache.size == 0


# ---------------------------------------------------------------------------
# The default post-filter on the base class
# ---------------------------------------------------------------------------

class _RecordingBackend(InMemoryBackend):
    """Records the limit each search() was asked for."""

    def __init__(self):
        super().__init__()
        self.limits: list[int] = []

    async def search(self, collection_name, vector, limit=5, score_threshold=0.0):
        self.limits.append(limit)
        return await super().search(collection_name, vector, limit, score_threshold)


class TestDefaultPostFilter:
    async def test_over_fetches_then_trims_to_limit(self, mock_embedder):
        backend = _RecordingBackend()
        await backend.initialize("post_filter", 384)
        m = await _medha(mock_embedder, backend=backend, collection="post_filter")
        try:
            for i in range(5):
                await m.store(f"question number {i}", f"SELECT {i}", metadata={"tenant": "acme"})
            backend.limits.clear()

            results = await backend.search_filtered(
                collection_name="post_filter",
                vector=await mock_embedder.aembed("question number 1"),
                limit=2,
                filters={"tenant": "acme"},
                overfetch=10,
            )

            assert backend.limits == [20]
            assert len(results) == 2
            assert all(r.metadata == {"tenant": "acme"} for r in results)
        finally:
            await m.close()

    async def test_no_filters_is_a_plain_search(self, mock_embedder):
        backend = _RecordingBackend()
        await backend.initialize("plain", 384)

        await backend.search_filtered(
            collection_name="plain",
            vector=await mock_embedder.aembed("anything"),
            limit=3,
        )

        assert backend.limits == [3]

    async def test_returned_rows_always_satisfy_the_filter(self, mock_embedder):
        backend = InMemoryBackend()
        await backend.initialize("mixed", 384)
        m = await _medha(mock_embedder, backend=backend, collection="mixed")
        try:
            await m.store("q one", "SELECT 1", metadata={"tenant": "acme"})
            await m.store("q two", "SELECT 2", metadata={"tenant": "globex"})
            await m.store("q three", "SELECT 3")

            results = await backend.search_filtered(
                collection_name="mixed",
                vector=await mock_embedder.aembed("q one"),
                limit=10,
                filters={"tenant": "globex"},
            )

            assert [r.generated_query for r in results] == ["SELECT 2"]
        finally:
            await m.close()


# ---------------------------------------------------------------------------
# Backends that do not support metadata
# ---------------------------------------------------------------------------

class TestUnsupportedBackend:
    async def test_search_with_filters_raises(self, mock_embedder):
        m = await _medha(mock_embedder, backend=_NoMetadataBackend(), collection="unsup1")
        try:
            with pytest.raises(ConfigurationError, match="does not store entry metadata"):
                await m.search(QUESTION, filters={"resolved_date": "2026-08-12"})
        finally:
            await m.close()

    async def test_store_with_metadata_raises(self, mock_embedder):
        m = await _medha(mock_embedder, backend=_NoMetadataBackend(), collection="unsup2")
        try:
            with pytest.raises(ConfigurationError, match="does not store entry metadata"):
                await m.store(QUESTION, QUERY_12, metadata={"resolved_date": "2026-08-12"})
        finally:
            await m.close()

    async def test_error_is_raised_before_any_work_is_done(self, mock_embedder):
        """A filtered search must fail loudly, not return NO_MATCH forever."""
        m = await _medha(mock_embedder, backend=_NoMetadataBackend(), collection="unsup3")
        try:
            with pytest.raises(ConfigurationError):
                await m.search(QUESTION, filters={"x": "y"})
            # The failure is not swallowed into the ERROR strategy either.
            assert m._request_count == 0
        finally:
            await m.close()


# ---------------------------------------------------------------------------
# Validation on the public API
# ---------------------------------------------------------------------------

class TestValidationAtTheBoundary:
    async def test_search_rejects_an_unsafe_filter_key(self, medha):
        with pytest.raises(ValueError, match="filters key"):
            await medha.search(QUESTION, filters={"drop table": "x"})

    async def test_store_rejects_nested_metadata(self, medha):
        with pytest.raises(ValueError, match="must be str, int, float or bool"):
            await medha.store(QUESTION, QUERY_12, metadata={"window": {"from": "10:00"}})

    async def test_store_many_names_the_offending_entry(self, medha):
        with pytest.raises(ValueError, match="entry 1 has invalid metadata"):
            await medha.store_many([
                {"question": "q a", "generated_query": "SELECT 1"},
                {"question": "q b", "generated_query": "SELECT 2", "metadata": {"bad key": 1}},
            ])


# ---------------------------------------------------------------------------
# Batch paths
# ---------------------------------------------------------------------------

class TestBatchPaths:
    async def test_store_batch_carries_metadata(self, medha):
        await medha.store_batch([
            {"question": QUESTION, "generated_query": QUERY_12,
             "metadata": {"resolved_date": "2026-08-12"}},
            {"question": QUESTION, "generated_query": QUERY_13,
             "metadata": {"resolved_date": "2026-08-13"}},
        ])

        hit = await medha.search(QUESTION, filters={"resolved_date": "2026-08-13"})

        assert hit.generated_query == QUERY_13

    async def test_store_many_carries_metadata(self, medha):
        await medha.store_many([
            {"question": QUESTION, "generated_query": QUERY_12,
             "metadata": {"resolved_date": "2026-08-12"}},
        ])

        hit = await medha.search(QUESTION, filters={"resolved_date": "2026-08-12"})

        assert hit.generated_query == QUERY_12

    async def test_search_batch_broadcasts_one_filter(self, medha):
        await medha.store("q a", "SELECT 1", metadata={"tenant": "acme"})
        await medha.store("q b", "SELECT 2", metadata={"tenant": "globex"})
        await medha._l1_backend.clear()

        hits = await medha.search_batch(["q a", "q b"], filters={"tenant": "acme"})

        assert hits[0].generated_query == "SELECT 1"
        assert hits[1].strategy == SearchStrategy.NO_MATCH

    async def test_search_batch_takes_one_filter_per_question(self, medha):
        """The scoped-batch case: each question resolves to its own day."""
        await medha.store(QUESTION, QUERY_12, metadata={"resolved_date": "2026-08-12"})
        await medha.store(QUESTION, QUERY_13, metadata={"resolved_date": "2026-08-13"})
        await medha._l1_backend.clear()

        hits = await medha.search_batch(
            [QUESTION, QUESTION],
            filters=[{"resolved_date": "2026-08-12"}, {"resolved_date": "2026-08-13"}],
        )

        assert hits[0].generated_query == QUERY_12
        assert hits[1].generated_query == QUERY_13

    async def test_search_batch_accepts_none_in_a_slot(self, medha):
        await medha.store(QUESTION, QUERY_12, metadata={"resolved_date": "2026-08-12"})
        await medha._l1_backend.clear()

        hits = await medha.search_batch(
            [QUESTION, QUESTION],
            filters=[None, {"resolved_date": "2026-08-99"}],
        )

        assert hits[0].generated_query == QUERY_12
        assert hits[1].strategy == SearchStrategy.NO_MATCH

    async def test_search_batch_rejects_a_length_mismatch(self, medha):
        with pytest.raises(ValueError, match="2 questions"):
            await medha.search_batch(["a", "b"], filters=[{"x": "y"}])


# ---------------------------------------------------------------------------
# strict vs soft
# ---------------------------------------------------------------------------

class TestFilterModes:
    async def test_strict_is_the_default(self):
        assert Settings().metadata_filter_mode == "strict"

    async def test_soft_keeps_a_mismatch_that_still_clears_the_threshold(
        self, mock_embedder
    ):
        settings = _settings(
            metadata_filter_mode="soft",
            metadata_filter_soft_penalty=0.9,
            score_threshold_semantic=0.50,
        )
        m = await _medha(mock_embedder, settings=settings, collection="soft_mode")
        try:
            await m.store(QUESTION, QUERY_12, metadata={"resolved_date": "2026-08-12"})
            await m._l1_backend.clear()

            hit = await m.search(QUESTION, filters={"resolved_date": "2026-08-13"})

            assert hit.generated_query == QUERY_12
            assert hit.strategy == SearchStrategy.SEMANTIC_MATCH
        finally:
            await m.close()

    async def test_soft_drops_a_mismatch_the_penalty_pushes_under_the_threshold(
        self, mock_embedder
    ):
        settings = _settings(
            metadata_filter_mode="soft",
            metadata_filter_soft_penalty=0.1,
            score_threshold_semantic=0.50,
        )
        m = await _medha(mock_embedder, settings=settings, collection="soft_drop")
        try:
            await m.store(QUESTION, QUERY_12, metadata={"resolved_date": "2026-08-12"})
            await m._l1_backend.clear()

            hit = await m.search(QUESTION, filters={"resolved_date": "2026-08-13"})

            assert hit.strategy == SearchStrategy.NO_MATCH
        finally:
            await m.close()

    async def test_soft_prefers_the_matching_entry_over_a_closer_mismatch(
        self, mock_embedder
    ):
        settings = _settings(
            metadata_filter_mode="soft",
            metadata_filter_soft_penalty=0.5,
            score_threshold_semantic=0.30,
        )
        m = await _medha(mock_embedder, settings=settings, collection="soft_rank")
        try:
            await m.store(QUESTION, QUERY_12, metadata={"resolved_date": "2026-08-12"})
            await m.store(QUESTION, QUERY_13, metadata={"resolved_date": "2026-08-13"})
            await m._l1_backend.clear()

            hit = await m.search(QUESTION, filters={"resolved_date": "2026-08-13"})

            assert hit.generated_query == QUERY_13
        finally:
            await m.close()


# ---------------------------------------------------------------------------
# Deduplication identity
# ---------------------------------------------------------------------------

class TestDedupIdentity:
    async def test_same_query_under_two_scopes_is_two_entries(self, medha):
        """The same SQL for two tenants is not a duplicate."""
        same_sql = "SELECT SUM(amount) FROM sales"
        await medha.store("q acme", same_sql, metadata={"tenant": "acme"})
        await medha.store("q globex", same_sql, metadata={"tenant": "globex"})

        deleted = await medha.dedup_collection()

        assert deleted == 0
        assert await medha._backend.count(medha._collection_name) == 2

    async def test_same_query_and_scope_still_deduplicates(self, medha):
        same_sql = "SELECT SUM(amount) FROM sales"
        await medha.store("q one", same_sql, metadata={"tenant": "acme"})
        await medha.store("q two", same_sql, metadata={"tenant": "acme"})

        deleted = await medha.dedup_collection()

        assert deleted == 1

    async def test_entries_without_metadata_deduplicate_as_before(self, medha):
        same_sql = "SELECT SUM(amount) FROM sales"
        await medha.store("q one", same_sql)
        await medha.store("q two", same_sql)

        assert await medha.dedup_collection() == 1


# ---------------------------------------------------------------------------
# Round-trip through the storage layer
# ---------------------------------------------------------------------------

class TestStorageRoundTrip:
    async def test_scroll_returns_metadata(self, medha):
        await medha.store(QUESTION, QUERY_12, metadata={"resolved_date": "2026-08-12"})

        results, _ = await medha._backend.scroll(medha._collection_name, limit=10)

        assert results[0].metadata == {"resolved_date": "2026-08-12"}

    async def test_every_scalar_type_survives(self, medha):
        meta = {"date": "2026-08-12", "hour": 10, "ratio": 0.25, "draft": True}
        await medha.store(QUESTION, QUERY_12, metadata=meta)

        hit = await medha.search(QUESTION, filters=meta)

        assert hit.metadata == meta

    async def test_stored_metadata_is_not_aliased_to_the_caller_dict(self, medha):
        meta = {"tenant": "acme"}
        await medha.store(QUESTION, QUERY_12, metadata=meta)
        meta["tenant"] = "globex"

        hit = await medha.search(QUESTION, filters={"tenant": "acme"})

        assert hit.generated_query == QUERY_12

    async def test_default_is_an_empty_map_not_none(self):
        result = CacheResult(
            id="x", score=1.0, original_question="q", normalized_question="q",
            generated_query="SELECT 1", query_hash="h",
        )
        assert result.metadata == {}

    async def test_interface_declares_no_metadata_support_by_default(self):
        assert VectorStorageBackend.supports_metadata is False
        assert InMemoryBackend.supports_metadata is True
