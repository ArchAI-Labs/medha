"""End-to-end integration tests for the feedback loop (Spec 11 — v0.4.0)."""

import pytest

from medha.config import Settings
from medha.types import SearchStrategy
from medha.utils.normalization import normalize_question


@pytest.fixture
async def medha_autoinvalidate(mock_embedder):
    """Medha with InMemoryBackend and feedback_incorrect_threshold=2."""
    from medha.backends.memory import InMemoryBackend
    from medha.core import Medha

    settings = Settings(
        backend_type="memory",
        score_threshold_exact=0.99,
        score_threshold_semantic=0.85,
        feedback_incorrect_threshold=2,
    )
    m = Medha("fb_e2e", mock_embedder, InMemoryBackend(), settings)
    await m.start()
    yield m
    await m.close()


async def test_full_feedback_loop_correct(medha_memory):
    question = "How many users are in the system?"
    await medha_memory.store(question, "SELECT COUNT(*) FROM users")

    result = await medha_memory.feedback(question, correct=True)

    assert result is True
    normalized = normalize_question(question)
    entry = await medha_memory._backend.search_by_normalized_question(
        medha_memory._collection_name, normalized
    )
    assert entry is not None
    assert entry.feedback_correct == 1


async def test_full_feedback_loop_incorrect(medha_memory):
    question = "List all inactive accounts?"
    await medha_memory.store(question, "SELECT * FROM accounts WHERE active = 0")

    result = await medha_memory.feedback(question, correct=False)

    assert result is True
    normalized = normalize_question(question)
    entry = await medha_memory._backend.search_by_normalized_question(
        medha_memory._collection_name, normalized
    )
    # No threshold set → entry still present
    assert entry is not None
    assert entry.feedback_incorrect == 1


async def test_feedback_not_found_returns_false(medha_memory):
    result = await medha_memory.feedback("This question was never cached at all", correct=True)
    assert result is False


async def test_feedback_counters_cumulative(medha_memory):
    question = "Show me the top 10 products by revenue"
    await medha_memory.store(question, "SELECT * FROM products ORDER BY revenue DESC LIMIT 10")

    await medha_memory.feedback(question, correct=True)
    await medha_memory.feedback(question, correct=True)
    await medha_memory.feedback(question, correct=True)
    await medha_memory.feedback(question, correct=False)

    normalized = normalize_question(question)
    entry = await medha_memory._backend.search_by_normalized_question(
        medha_memory._collection_name, normalized
    )
    assert entry is not None
    assert entry.feedback_correct == 3
    assert entry.feedback_incorrect == 1


async def test_auto_invalidation_end_to_end(medha_autoinvalidate):
    question = "Find all orders placed last week"
    await medha_autoinvalidate.store(
        question, "SELECT * FROM orders WHERE created_at >= NOW() - INTERVAL '7 days'"
    )

    await medha_autoinvalidate.feedback(question, correct=False)
    await medha_autoinvalidate.feedback(question, correct=False)  # hits threshold=2

    hit = await medha_autoinvalidate.search(question)
    assert hit.strategy == SearchStrategy.NO_MATCH


async def test_auto_invalidation_with_l1(medha_autoinvalidate):
    question = "Count all pending shipments in the warehouse"
    await medha_autoinvalidate.store(
        question, "SELECT COUNT(*) FROM shipments WHERE status = 'pending'"
    )

    # Populate L1 cache
    first_hit = await medha_autoinvalidate.search(question)
    assert first_hit.strategy != SearchStrategy.NO_MATCH

    # Trigger auto-invalidation (threshold=2)
    await medha_autoinvalidate.feedback(question, correct=False)
    await medha_autoinvalidate.feedback(question, correct=False)

    # L1 must be cleared by invalidate() inside feedback()
    hit = await medha_autoinvalidate.search(question)
    assert hit.strategy == SearchStrategy.NO_MATCH
