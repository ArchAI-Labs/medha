"""Integration tests for Medha async context manager — real embedder + InMemoryBackend."""

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


class TestAsyncContextManagerE2E:
    async def test_full_lifecycle(self, embedder, settings):
        backend = InMemoryBackend()
        async with Medha("ctx_test", embedder, backend, settings) as m:
            await m.store("revenue query", "SELECT SUM(revenue) FROM sales")
            result = await m.search("revenue query")
            assert result.strategy != SearchStrategy.NO_MATCH
        # InMemoryBackend.close() clears _store — count returns 0
        assert await backend.count("ctx_test") == 0

    async def test_exception_in_block_still_closes(self, embedder, settings):
        backend = InMemoryBackend()
        closed = False
        original_close = backend.close

        async def spy_close():
            nonlocal closed
            closed = True
            await original_close()

        backend.close = spy_close

        with pytest.raises(ValueError, match="test error"):
            async with Medha("ctx_exc_test", embedder, backend, settings):
                raise ValueError("test error")

        assert closed
