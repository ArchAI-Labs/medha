"""Unit tests for the feedback loop (Spec 11 — v0.4.0)."""

import asyncio
import hashlib
import uuid

import pytest
from pydantic import ValidationError

from medha.config import Settings
from medha.types import CacheEntry, CacheResult, SearchStrategy
from medha.utils.normalization import normalize_question, query_hash


class TestFeedbackTypes:
    def test_cache_entry_feedback_defaults_zero(self):
        entry = CacheEntry(
            id=str(uuid.uuid4()),
            vector=[0.1] * 8,
            original_question="q",
            normalized_question="q",
            generated_query="SELECT 1",
            query_hash=hashlib.md5(b"SELECT 1").hexdigest(),
        )
        assert entry.feedback_correct == 0
        assert entry.feedback_incorrect == 0

    def test_cache_result_feedback_defaults_zero(self):
        result = CacheResult(
            id=str(uuid.uuid4()),
            score=0.9,
            original_question="q",
            normalized_question="q",
            generated_query="SELECT 1",
            query_hash=hashlib.md5(b"SELECT 1").hexdigest(),
        )
        assert result.feedback_correct == 0
        assert result.feedback_incorrect == 0

    def test_cache_entry_backward_compat_no_feedback_fields(self):
        data = {
            "id": str(uuid.uuid4()),
            "vector": [0.1] * 8,
            "original_question": "old entry",
            "normalized_question": "old entry",
            "generated_query": "SELECT old",
            "query_hash": hashlib.md5(b"SELECT old").hexdigest(),
        }
        entry = CacheEntry(**data)
        assert entry.feedback_correct == 0
        assert entry.feedback_incorrect == 0


class TestInMemoryBackendUpdateFeedback:
    async def test_update_feedback_correct_returns_new_count(self, inmemory_backend):
        from tests.conftest import make_entry
        entry = make_entry(question="feedback q correct", query="SELECT fb_c")
        await inmemory_backend.initialize("fb_test", 8)
        await inmemory_backend.upsert("fb_test", [entry])

        result = await inmemory_backend.update_feedback("fb_test", entry.id, correct=True)

        assert result == 1

    async def test_update_feedback_incorrect_returns_new_count(self, inmemory_backend):
        from tests.conftest import make_entry
        entry = make_entry(question="feedback q incorrect", query="SELECT fb_i")
        await inmemory_backend.initialize("fb_test", 8)
        await inmemory_backend.upsert("fb_test", [entry])

        result = await inmemory_backend.update_feedback("fb_test", entry.id, correct=False)

        assert result == 1

    async def test_update_feedback_accumulates_and_returns_correct_count(self, inmemory_backend):
        from tests.conftest import make_entry
        entry = make_entry(question="accum feedback q", query="SELECT accum")
        await inmemory_backend.initialize("fb_test", 8)
        await inmemory_backend.upsert("fb_test", [entry])

        r1 = await inmemory_backend.update_feedback("fb_test", entry.id, correct=True)
        r2 = await inmemory_backend.update_feedback("fb_test", entry.id, correct=True)
        r3 = await inmemory_backend.update_feedback("fb_test", entry.id, correct=False)

        assert r1 == 1
        assert r2 == 2
        assert r3 == 1  # incorrect counter starts from 0

        results, _ = await inmemory_backend.scroll("fb_test")
        matching = [r for r in results if r.id == entry.id]
        assert matching[0].feedback_correct == 2
        assert matching[0].feedback_incorrect == 1

    async def test_update_feedback_missing_id_returns_zero_no_exception(self, inmemory_backend):
        await inmemory_backend.initialize("fb_test", 8)

        result = await inmemory_backend.update_feedback(
            "fb_test", "nonexistent-id-000", correct=True
        )

        assert result == 0


class TestSeveralEntriesPerQuestion:
    """One question can map to several entries, and today they are indistinguishable.

    ``store()`` takes the query as a separate argument, so nothing stops the
    same question being stored with two different queries. Both entries then
    carry the same normalized question *and* the same vector — the embedding
    is computed from the question alone — so they tie at score 1.0 and which
    one answers is decided by whatever order the backend returns ties in.

    These tests pin the contract as it actually stands: *one of* the entries,
    never *which one*. Issue #36 gives entries distinguishable metadata, which
    is what makes a deterministic answer possible; when it lands, these
    assertions get tightened deliberately rather than breaking by surprise.
    """

    QUESTION = "how many active users"
    QUERIES = (
        "SELECT count(*) FROM users WHERE active",
        "SELECT count(*) FROM users WHERE active = true",
        "SELECT count(*) FROM u WHERE flag",
    )

    async def _store_variants(self, medha) -> None:
        for query in self.QUERIES:
            await medha.store(self.QUESTION, query)

    async def test_variants_are_kept_as_separate_entries(self, medha_memory):
        await self._store_variants(medha_memory)
        count = await medha_memory._backend.count(medha_memory._collection_name)
        assert count == len(self.QUERIES)

    async def test_search_answers_with_one_of_the_stored_queries(self, medha_memory):
        await self._store_variants(medha_memory)
        # Drop L1, which would otherwise short-circuit to the last write and
        # hide the fact that the backend is the one resolving the tie.
        await medha_memory.clear_caches()

        hit = await medha_memory.search(self.QUESTION)

        assert hit.strategy is not SearchStrategy.NO_MATCH
        assert hit.generated_query in self.QUERIES, (
            f"answered with a query that was never stored: {hit.generated_query!r}"
        )

    async def test_feedback_marks_exactly_one_entry(self, medha_memory):
        """Not which one — that is undefined — but that it is not all of them."""
        await self._store_variants(medha_memory)

        assert await medha_memory.feedback(self.QUESTION, correct=False) is True

        results, _ = await medha_memory._backend.scroll(
            medha_memory._collection_name, limit=100
        )
        marked = [r for r in results if r.feedback_incorrect]
        assert len(marked) == 1
        assert marked[0].feedback_incorrect == 1

    async def test_invalidate_still_removes_all_of_them(self, medha_memory):
        """The one place ambiguity is already resolved: invalidate is exhaustive."""
        await self._store_variants(medha_memory)

        assert await medha_memory.invalidate(self.QUESTION) is True

        assert await medha_memory._backend.count(medha_memory._collection_name) == 0


class TestMedhaFeedback:
    async def test_feedback_correct_returns_true(self, medha_memory):
        await medha_memory.store("How many users are registered?", "SELECT COUNT(*) FROM users")
        result = await medha_memory.feedback("How many users are registered?", correct=True)
        assert result is True

    async def test_feedback_incorrect_returns_true(self, medha_memory):
        await medha_memory.store("How many orders exist?", "SELECT COUNT(*) FROM orders")
        result = await medha_memory.feedback("How many orders exist?", correct=False)
        assert result is True

    async def test_feedback_returns_false_when_not_found(self, medha_memory):
        result = await medha_memory.feedback("This question was never stored", correct=True)
        assert result is False

    async def test_feedback_after_invalidate_returns_false(self, medha_memory):
        question = "What is the total revenue?"
        await medha_memory.store(question, "SELECT SUM(amount) FROM sales")
        await medha_memory.invalidate(question)
        result = await medha_memory.feedback(question, correct=True)
        assert result is False

    async def test_feedback_counters_visible_in_cache_result(self, medha_memory):
        from medha.utils.normalization import normalize_question
        question = "How many products are in the catalog?"
        await medha_memory.store(question, "SELECT COUNT(*) FROM products")
        await medha_memory.feedback(question, correct=True)
        await medha_memory.feedback(question, correct=False)

        normalized = normalize_question(question)
        backend_result = await medha_memory._backend.search_by_normalized_question(
            medha_memory._collection_name, normalized
        )
        assert backend_result is not None
        assert backend_result.feedback_correct == 1
        assert backend_result.feedback_incorrect == 1

    async def test_feedback_on_l1_hit_updates_backend(self, medha_memory):
        from medha.utils.normalization import normalize_question
        question = "How many employees are active?"
        await medha_memory.store(question, "SELECT COUNT(*) FROM employees WHERE active = 1")
        # Populate L1 cache
        await medha_memory.search(question)
        # Feedback must still reach the backend even though L1 is warm
        result = await medha_memory.feedback(question, correct=True)
        assert result is True

        normalized = normalize_question(question)
        backend_result = await medha_memory._backend.search_by_normalized_question(
            medha_memory._collection_name, normalized
        )
        assert backend_result is not None
        assert backend_result.feedback_correct == 1


class TestMedhaFeedbackAutoInvalidation:
    @pytest.fixture
    async def medha_threshold(self, mock_embedder):
        from medha.backends.memory import InMemoryBackend
        from medha.core import Medha
        settings = Settings(
            backend_type="memory",
            score_threshold_exact=0.99,
            score_threshold_semantic=0.85,
            feedback_incorrect_threshold=3,
        )
        m = Medha("fb_threshold", mock_embedder, InMemoryBackend(), settings)
        await m.start()
        yield m
        await m.close()

    async def test_no_auto_invalidation_when_threshold_is_none(self, medha_memory):
        from medha.utils.normalization import normalize_question
        question = "How many tables exist in the schema?"
        await medha_memory.store(question, "SELECT COUNT(*) FROM information_schema.tables")
        for _ in range(5):
            await medha_memory.feedback(question, correct=False)

        normalized = normalize_question(question)
        result = await medha_memory._backend.search_by_normalized_question(
            medha_memory._collection_name, normalized
        )
        assert result is not None

    async def test_auto_invalidation_fires_at_threshold(self, medha_threshold):
        from medha.utils.normalization import normalize_question
        question = "List all active sessions in the database"
        await medha_threshold.store(question, "SELECT * FROM sessions WHERE active = 1")
        await medha_threshold.feedback(question, correct=False)
        await medha_threshold.feedback(question, correct=False)
        await medha_threshold.feedback(question, correct=False)  # hits threshold=3

        normalized = normalize_question(question)
        result = await medha_threshold._backend.search_by_normalized_question(
            medha_threshold._collection_name, normalized
        )
        assert result is None

    async def test_auto_invalidation_does_not_fire_below_threshold(self, medha_threshold):
        from medha.utils.normalization import normalize_question
        question = "Show all pending tasks in the queue"
        await medha_threshold.store(question, "SELECT * FROM tasks WHERE status = 'pending'")
        await medha_threshold.feedback(question, correct=False)
        await medha_threshold.feedback(question, correct=False)  # 2 < threshold=3

        normalized = normalize_question(question)
        result = await medha_threshold._backend.search_by_normalized_question(
            medha_threshold._collection_name, normalized
        )
        assert result is not None

    async def test_auto_invalidation_clears_l1(self, medha_threshold):
        from medha.types import SearchStrategy
        question = "Count all invoices created this month"
        await medha_threshold.store(question, "SELECT COUNT(*) FROM invoices")
        # Populate L1 cache via a search
        await medha_threshold.search(question)
        # Trigger auto-invalidation
        for _ in range(3):
            await medha_threshold.feedback(question, correct=False)
        # L1 should be cleared; next search must return NO_MATCH
        hit = await medha_threshold.search(question)
        assert hit.strategy == SearchStrategy.NO_MATCH

    async def test_auto_invalidation_is_idempotent(self, medha_threshold):
        question = "Show all active user accounts"
        await medha_threshold.store(question, "SELECT * FROM users WHERE active = 1")
        for _ in range(3):
            await medha_threshold.feedback(question, correct=False)
        # Entry is gone; a further call must return False and not raise
        result = await medha_threshold.feedback(question, correct=False)
        assert result is False

    async def test_correct_feedback_never_triggers_invalidation(self, medha_threshold):
        from medha.utils.normalization import normalize_question
        question = "Count all available products in stock"
        await medha_threshold.store(question, "SELECT COUNT(*) FROM products WHERE in_stock = 1")
        for _ in range(10):
            await medha_threshold.feedback(question, correct=True)

        normalized = normalize_question(question)
        result = await medha_threshold._backend.search_by_normalized_question(
            medha_threshold._collection_name, normalized
        )
        assert result is not None


class TestFeedbackByEntryId:
    """Id-addressed feedback (issue #43): the plumbing that lets a caller mark
    the entry that actually answered, not the question that was asked."""

    async def test_entry_id_feedback_increments_that_entry(self, medha_memory):
        from medha.utils.normalization import normalize_question
        question = "How many active subscriptions exist?"
        await medha_memory.store(question, "SELECT COUNT(*) FROM subscriptions")
        normalized = normalize_question(question)
        stored = await medha_memory._backend.search_by_normalized_question(
            medha_memory._collection_name, normalized
        )

        result = await medha_memory.feedback(
            question, correct=False, entry_id=stored.id
        )

        assert result is True
        updated = await medha_memory._backend.search_by_normalized_question(
            medha_memory._collection_name, normalized
        )
        assert updated.feedback_incorrect == 1

    async def test_entry_id_feedback_works_for_a_misdirected_hit(self, medha_memory):
        """The whole point of entry_id: the asked question was never stored
        under its own normalized form, so the question-based lookup that
        feedback() falls back to would find nothing — but the id it was
        actually served under still resolves."""
        stored_question = "How many active subscriptions exist?"
        asked_question = "something else entirely, never stored"
        await medha_memory.store(stored_question, "SELECT COUNT(*) FROM subscriptions")
        from medha.utils.normalization import normalize_question
        stored = await medha_memory._backend.search_by_normalized_question(
            medha_memory._collection_name, normalize_question(stored_question)
        )

        # The plain question-based path finds nothing for the asked question.
        assert await medha_memory.feedback(asked_question, correct=False) is False

        # entry_id addresses the entry that actually answered directly.
        result = await medha_memory.feedback(
            asked_question, correct=False, entry_id=stored.id
        )
        assert result is True

        updated = await medha_memory._backend.search_by_normalized_question(
            medha_memory._collection_name, normalize_question(stored_question)
        )
        assert updated.feedback_incorrect == 1

    async def test_entry_id_not_found_returns_false(self, medha_memory):
        result = await medha_memory.feedback(
            "irrelevant", correct=True, entry_id="no-such-entry-id"
        )
        assert result is False

    async def test_question_based_feedback_unchanged(self, medha_memory):
        """feedback(question, correct) without entry_id behaves exactly as before."""
        question = "How many active carts are there?"
        await medha_memory.store(question, "SELECT COUNT(*) FROM carts")

        assert await medha_memory.feedback(question, correct=True) is True
        assert await medha_memory.feedback("never stored", correct=True) is False


class TestFeedbackByEntryIdAutoInvalidation:
    @pytest.fixture
    async def medha_threshold(self, mock_embedder):
        from medha.backends.memory import InMemoryBackend
        from medha.core import Medha
        settings = Settings(
            backend_type="memory",
            score_threshold_exact=0.99,
            score_threshold_semantic=0.85,
            feedback_incorrect_threshold=2,
        )
        m = Medha("fb_id_threshold", mock_embedder, InMemoryBackend(), settings)
        await m.start()
        yield m
        await m.close()

    async def test_auto_invalidation_removes_only_the_addressed_entry(self, medha_threshold):
        """Two entries share a question; feedback addressed to one id must not
        take the other down with it."""
        question = "how many active orders"
        await medha_threshold.store(question, "SELECT count(*) FROM orders WHERE active")
        await medha_threshold.store(question, "SELECT count(*) FROM orders WHERE flag = 1")

        results, _ = await medha_threshold._backend.scroll(
            medha_threshold._collection_name, limit=100
        )
        assert len(results) == 2
        target_id = results[0].id

        await medha_threshold.feedback(question, correct=False, entry_id=target_id)
        await medha_threshold.feedback(question, correct=False, entry_id=target_id)  # hits threshold=2

        remaining, _ = await medha_threshold._backend.scroll(
            medha_threshold._collection_name, limit=100
        )
        remaining_ids = {r.id for r in remaining}
        assert target_id not in remaining_ids
        assert remaining_ids == {results[1].id}

    async def test_auto_invalidation_clears_l1_for_the_asked_question(self, medha_threshold):
        from medha.types import SearchStrategy
        question = "count pending refunds"
        await medha_threshold.store(question, "SELECT COUNT(*) FROM refunds")
        hit = await medha_threshold.search(question)  # populates L1
        assert hit.entry_id is not None

        await medha_threshold.feedback(question, correct=False, entry_id=hit.entry_id)
        await medha_threshold.feedback(question, correct=False, entry_id=hit.entry_id)  # threshold=2

        after = await medha_threshold.search(question)
        assert after.strategy == SearchStrategy.NO_MATCH


class TestFeedbackByEntryIdRaceCondition:
    """A dedicated concurrency test for the entry_id auto-invalidation path.

    Plain asyncio.gather() of two feedback() calls turns out *not* to
    interleave here: InMemoryBackend.update_feedback()/.delete() take the
    backend's asyncio.Lock, and acquiring an uncontended asyncio.Lock does not
    yield control back to the event loop — so the first call runs start to
    finish (increment, threshold check, delete) before the second one gets
    scheduled at all. Confirmed empirically: 50/50 runs landed on the same
    (True, False) ordering, never (True, True). Asserting against whichever
    outcome that produces would test nothing about concurrency.

    To exercise the actual race — both calls crossing the threshold and both
    reaching the delete step for the same id — a real suspension point is
    injected at backend.delete(), the moment that matters. That reliably
    produces genuine interleaving: both calls observe a threshold-crossing
    count and both attempt to delete the same entry, which is exactly the
    scenario this test guards. What must hold regardless of backend timing:
    neither call raises, and the entry ends up deleted exactly once — the
    second physical delete on an id the first already removed is a no-op
    (see InMemoryBackend.delete(), a plain dict.pop(id_, None))."""

    async def test_concurrent_entry_id_feedback_does_not_raise(self, mock_embedder):
        from medha.backends.memory import InMemoryBackend
        from medha.core import Medha

        settings = Settings(
            backend_type="memory",
            score_threshold_exact=0.99,
            score_threshold_semantic=0.85,
            feedback_incorrect_threshold=1,
        )
        m = Medha("fb_race", mock_embedder, InMemoryBackend(), settings)
        await m.start()
        question = "how many concurrent races"
        await m.store(question, "SELECT COUNT(*) FROM races")
        stored = await m._backend.search_by_normalized_question(
            m._collection_name, normalize_question(question)
        )
        entry_id = stored.id

        original_delete = m._backend.delete

        async def delayed_delete(collection_name, ids):
            # Force both feedback() calls to reach the physical delete before
            # either completes it — the interleaving a race would produce,
            # without depending on asyncio's actual (backend-specific)
            # scheduling order to happen to land there on its own.
            await asyncio.sleep(0)
            return await original_delete(collection_name, ids)

        m._backend.delete = delayed_delete

        results = await asyncio.gather(
            m.feedback(question, correct=False, entry_id=entry_id),
            m.feedback(question, correct=False, entry_id=entry_id),
            return_exceptions=True,
        )

        for r in results:
            assert not isinstance(r, BaseException), f"feedback() raised: {r!r}"
        assert results == [True, True], (
            "expected both calls to cross the threshold and both reach "
            f"delete() under forced interleaving, got {results!r}"
        )

        # The entry was deleted twice (once per call) without either delete
        # raising, and it is gone exactly once.
        remaining, _ = await m._backend.scroll(m._collection_name, limit=10)
        assert remaining == []

        await m.close()


class TestFeedbackCollectionName:
    """feedback() and invalidate() can target a collection other than the
    instance's default one, matching search_batch(collection_name=...)."""

    async def test_feedback_by_question_targets_specified_collection(self, medha_memory):
        other_collection = "other_coll"
        question = "how many orders in region B"
        normalized = normalize_question(question)
        embedding = await medha_memory._embedder.aembed(normalized)
        entry = CacheEntry(
            id=str(uuid.uuid4()),
            vector=embedding,
            original_question=question,
            normalized_question=normalized,
            generated_query="SELECT COUNT(*) FROM orders_region_b",
            query_hash=query_hash("SELECT COUNT(*) FROM orders_region_b"),
        )
        await medha_memory._backend.initialize(
            other_collection, medha_memory._embedder.dimension
        )
        await medha_memory._backend.upsert(other_collection, [entry])

        # The default collection has nothing for this question.
        assert await medha_memory.feedback(question, correct=True) is False

        # Targeting the right collection finds and updates it.
        result = await medha_memory.feedback(
            question, correct=True, collection_name=other_collection
        )
        assert result is True

        updated = await medha_memory._backend.search_by_normalized_question(
            other_collection, normalized
        )
        assert updated.feedback_correct == 1

    async def test_feedback_by_entry_id_targets_specified_collection(self, medha_memory):
        other_collection = "other_coll_2"
        question = "how many refunds in region C"
        normalized = normalize_question(question)
        embedding = await medha_memory._embedder.aembed(normalized)
        entry = CacheEntry(
            id=str(uuid.uuid4()),
            vector=embedding,
            original_question=question,
            normalized_question=normalized,
            generated_query="SELECT COUNT(*) FROM refunds_region_c",
            query_hash=query_hash("SELECT COUNT(*) FROM refunds_region_c"),
        )
        await medha_memory._backend.initialize(
            other_collection, medha_memory._embedder.dimension
        )
        await medha_memory._backend.upsert(other_collection, [entry])

        result = await medha_memory.feedback(
            question, correct=False, entry_id=entry.id, collection_name=other_collection
        )
        assert result is True

        updated = await medha_memory._backend.search_by_normalized_question(
            other_collection, normalized
        )
        assert updated.feedback_incorrect == 1

    async def test_auto_invalidation_respects_collection_name(self, mock_embedder):
        """Same question stored in two collections: auto-invalidation
        triggered against one must not touch the other."""
        from medha.backends.memory import InMemoryBackend
        from medha.core import Medha

        settings = Settings(
            backend_type="memory",
            score_threshold_exact=0.99,
            score_threshold_semantic=0.85,
            feedback_incorrect_threshold=1,
        )
        backend = InMemoryBackend()
        m = Medha("main_coll", mock_embedder, backend, settings)
        await m.start()
        other_collection = "other_coll_3"
        await backend.initialize(other_collection, mock_embedder.dimension)

        question = "how many disputed charges"
        await m.store(question, "SELECT COUNT(*) FROM disputes_main")

        normalized = normalize_question(question)
        embedding = await mock_embedder.aembed(normalized)
        other_entry = CacheEntry(
            id=str(uuid.uuid4()),
            vector=embedding,
            original_question=question,
            normalized_question=normalized,
            generated_query="SELECT COUNT(*) FROM disputes_other",
            query_hash=query_hash("SELECT COUNT(*) FROM disputes_other"),
        )
        await backend.upsert(other_collection, [other_entry])

        result = await m.feedback(question, correct=False, collection_name=other_collection)
        assert result is True  # threshold=1: triggers invalidation in other_collection

        assert await backend.search_by_normalized_question(other_collection, normalized) is None
        main_result = await backend.search_by_normalized_question(
            m._collection_name, normalized
        )
        assert main_result is not None

        await m.close()


class TestFeedbackSettings:
    def test_feedback_incorrect_threshold_none_by_default(self):
        s = Settings()
        assert s.feedback_incorrect_threshold is None

    def test_feedback_incorrect_threshold_accepts_positive_int(self):
        s = Settings(feedback_incorrect_threshold=5)
        assert s.feedback_incorrect_threshold == 5

    def test_feedback_incorrect_threshold_rejects_zero(self):
        with pytest.raises(ValidationError):
            Settings(feedback_incorrect_threshold=0)

    def test_feedback_incorrect_threshold_from_env_var(self, monkeypatch):
        monkeypatch.setenv("MEDHA_FEEDBACK_INCORRECT_THRESHOLD", "3")
        s = Settings()
        assert s.feedback_incorrect_threshold == 3
