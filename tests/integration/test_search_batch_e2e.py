"""Integration tests for Medha.search_batch() — real embedder + InMemoryBackend."""

import pytest

pytest.importorskip("fastembed")

from medha.backends.memory import InMemoryBackend
from medha.config import Settings
from medha.core import Medha
from medha.embeddings.fastembed_adapter import FastEmbedAdapter
from medha.types import SearchStrategy


@pytest.fixture(scope="module")
def embedder():
    return FastEmbedAdapter()


@pytest.fixture
def settings():
    return Settings(
        backend_type="memory",
        score_threshold_exact=0.99,
        score_threshold_semantic=0.85,
    )


@pytest.fixture
async def medha_batch(embedder, settings):
    backend = InMemoryBackend()
    m = Medha(
        collection_name="batch_e2e",
        embedder=embedder,
        backend=backend,
        settings=settings,
    )
    await m.start()
    yield m
    await m.close()


class TestSearchBatchE2E:
    async def test_batch_returns_hits_for_stored_questions(self, medha_batch):
        pairs = [
            ("How many users are there", "SELECT COUNT(*) FROM users"),
            ("List all products", "SELECT * FROM products"),
            ("Show total revenue", "SELECT SUM(revenue) FROM sales"),
        ]
        for q, sq in pairs:
            await medha_batch.store(q, sq)

        results = await medha_batch.search_batch([q for q, _ in pairs])
        assert all(r.strategy != SearchStrategy.NO_MATCH for r in results)

    async def test_batch_miss_for_unknown_questions(self, medha_batch):
        await medha_batch.store("count active sessions", "SELECT COUNT(*) FROM sessions")

        results = await medha_batch.search_batch([
            "xyzzy random nonsense query alpha",
            "foo bar baz unrelated question beta",
        ])
        assert all(r.strategy == SearchStrategy.NO_MATCH for r in results)

    async def test_batch_order_preserved(self, medha_batch):
        await medha_batch.store("get product count", "SELECT COUNT(*) FROM products")
        await medha_batch.store("get user count", "SELECT COUNT(*) FROM users")

        results = await medha_batch.search_batch([
            "get user count",
            "get product count",
        ])
        assert results[0].generated_query == "SELECT COUNT(*) FROM users"
        assert results[1].generated_query == "SELECT COUNT(*) FROM products"

    async def test_batch_empty_collection(self, embedder, settings):
        backend = InMemoryBackend()
        m = Medha(
            collection_name="empty_batch_test",
            embedder=embedder,
            backend=backend,
            settings=settings,
        )
        await m.start()
        try:
            results = await m.search_batch(["anything goes here"])
            assert results[0].strategy == SearchStrategy.NO_MATCH
        finally:
            await m.close()
