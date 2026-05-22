"""Unit tests for the feedback loop (Spec 11 — v0.4.0)."""

import hashlib
import uuid

import pytest
from pydantic import ValidationError

from medha.config import Settings
from medha.types import CacheEntry, CacheResult


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
