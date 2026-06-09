"""Unit tests for Medha.search_batch() and search_batch_sync()."""

import pytest

from medha.core import Medha
from medha.types import CacheHit


class TestSearchBatch:

    async def test_empty_list_returns_empty(self, medha_memory):
        results = await medha_memory.search_batch([])
        assert results == []

    async def test_returns_same_count_as_input(self, medha_memory):
        results = await medha_memory.search_batch(["q1", "q2", "q3"])
        assert len(results) == 3

    async def test_all_results_are_cache_hits(self, medha_memory):
        results = await medha_memory.search_batch(["q1", "q2"])
        assert all(isinstance(r, CacheHit) for r in results)

    async def test_order_preserved(self, medha_memory):
        await medha_memory.store("what is the revenue", "SELECT revenue FROM sales")
        await medha_memory.store("show me all users", "SELECT * FROM users")
        results = await medha_memory.search_batch(
            ["what is the revenue", "show me all users"]
        )
        assert results[0].generated_query == "SELECT revenue FROM sales"
        assert results[1].generated_query == "SELECT * FROM users"

    async def test_embedder_called_once_for_batch(self, medha_memory, mock_embedder):
        call_log: list[list[str]] = []
        original = mock_embedder.aembed_batch

        async def tracking_batch(texts, **kwargs):
            call_log.append(list(texts))
            return await original(texts, **kwargs)

        mock_embedder.aembed_batch = tracking_batch
        try:
            await medha_memory.search_batch(["q1", "q2", "q3"])
        finally:
            mock_embedder.aembed_batch = original

        assert len(call_log) == 1, f"aembed_batch called {len(call_log)} times, expected 1"
        assert len(call_log[0]) == 3

    def test_search_batch_sync_exists(self):
        assert hasattr(Medha, "search_batch_sync")
        assert callable(Medha.search_batch_sync)
