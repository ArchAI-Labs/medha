"""Integration tests for bulk operations using InMemoryBackend."""

from __future__ import annotations

import pytest

from medha.backends.memory import InMemoryBackend
from medha.config import Settings

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def medha_bulk(mock_embedder):
    """Medha instance backed by InMemoryBackend, ready for bulk tests."""
    from medha.core import Medha

    backend = InMemoryBackend()
    settings = Settings(backend_type="memory", batch_size=100)
    m = Medha(
        collection_name="bulk_e2e",
        embedder=mock_embedder,
        backend=backend,
        settings=settings,
    )
    await m.start()
    yield m
    await m.close()


# ---------------------------------------------------------------------------
# store_many end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_many_e2e(medha_bulk):
    entries = [
        {"question": f"Question {i}", "generated_query": f"SELECT {i} FROM t"}
        for i in range(10)
    ]
    count = await medha_bulk.store_many(entries)
    assert count == 10

    # Verify backend actually has the entries
    backend_count = await medha_bulk._backend.count("bulk_e2e")
    assert backend_count == 10


@pytest.mark.asyncio
async def test_store_many_e2e_searchable(medha_bulk):
    """Stored entries must be retrievable via search."""
    await medha_bulk.store_many([
        {"question": "How many users are there?", "generated_query": "SELECT COUNT(*) FROM users"},
    ])
    hit = await medha_bulk.search("How many users are there?")
    assert hit.generated_query == "SELECT COUNT(*) FROM users"


# ---------------------------------------------------------------------------
# batch_embed_concurrency=3 produces same results as =1
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_many_concurrency_same_result(mock_embedder):
    """Parallel embedding chunks must store the same data as sequential."""
    from medha.backends.memory import InMemoryBackend
    from medha.core import Medha

    entries = [
        {"question": f"Q{i}", "generated_query": f"SELECT {i}"}
        for i in range(6)
    ]

    async def _run(concurrency: int) -> list[str]:
        backend = InMemoryBackend()
        settings = Settings(
            backend_type="memory",
            batch_size=2,
            batch_embed_concurrency=concurrency,
        )
        m = Medha(
            collection_name="concurrency_test",
            embedder=mock_embedder,
            backend=backend,
            settings=settings,
        )
        await m.start()
        count = await m.store_many(entries)
        assert count == 6
        results, _ = await backend.scroll("concurrency_test", limit=100)
        await m.close()
        return sorted(r.generated_query for r in results)

    queries_seq = await _run(concurrency=1)
    queries_par = await _run(concurrency=3)

    assert queries_seq == queries_par
