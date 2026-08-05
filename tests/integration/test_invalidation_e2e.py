"""Integration tests: after invalidation, search no longer returns the entry."""

import pytest

from medha.backends.memory import InMemoryBackend
from medha.config import Settings
from medha.core import Medha
from medha.types import SearchStrategy

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
async def medha(mock_embedder):
    settings = Settings(
        backend_type="memory",
        score_threshold_exact=0.99,
        score_threshold_semantic=0.50,
        l1_cache_max_size=100,
    )
    backend = InMemoryBackend()
    m = Medha(
        collection_name="e2e_inv",
        embedder=mock_embedder,
        backend=backend,
        settings=settings,
    )
    await m.start()
    yield m
    await m.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInvalidationE2E:
    async def test_after_invalidate_search_returns_no_match(self, medha):
        """After invalidate(), search for the same question returns NO_MATCH."""
        await medha.store("how many users", "SELECT COUNT(*) FROM users")

        hit_before = await medha.search("how many users")
        assert hit_before.strategy != SearchStrategy.NO_MATCH, "Entry must be cached before invalidation"

        deleted = await medha.invalidate("how many users")
        assert deleted is True

        hit_after = await medha.search("how many users")
        assert hit_after.strategy == SearchStrategy.NO_MATCH, "Entry must not be found after invalidation"

    async def test_invalidate_does_not_remove_other_entries(self, medha):
        """Invalidating one entry must leave unrelated entries intact."""
        await medha.store("how many users", "SELECT COUNT(*) FROM users")
        await medha.store("list products", "SELECT * FROM products")

        await medha.invalidate("how many users")

        # The second entry must still be retrievable
        hit = await medha.search("list products")
        assert hit.strategy != SearchStrategy.NO_MATCH
        assert hit.generated_query == "SELECT * FROM products"

    async def test_after_invalidate_by_query_hash_search_returns_no_match(self, medha):
        """After invalidate_by_query_hash(), all entries with that hash return NO_MATCH."""
        import hashlib
        query = "SELECT COUNT(*) FROM orders"
        qhash = hashlib.md5(query.encode()).hexdigest()

        await medha.store("how many orders", query)
        await medha.store("order count", query)

        deleted = await medha.invalidate_by_query_hash(qhash)
        assert deleted == 2

        for q in ["how many orders", "order count"]:
            hit = await medha.search(q)
            assert hit.strategy == SearchStrategy.NO_MATCH, f"'{q}' should be NO_MATCH after invalidation"

    async def test_after_invalidate_by_template_search_returns_no_match(self, medha):
        """After invalidate_by_template(), entries for that template return NO_MATCH."""
        await medha.store(
            "how many employees", "SELECT COUNT(*) FROM employees", template_id="count_tmpl"
        )
        await medha.store(
            "how many managers", "SELECT COUNT(*) FROM managers", template_id="count_tmpl"
        )
        await medha.store(
            "list departments", "SELECT * FROM departments", template_id="list_tmpl"
        )

        deleted = await medha.invalidate_by_template("count_tmpl")
        assert deleted == 2

        for q in ["how many employees", "how many managers"]:
            hit = await medha.search(q)
            assert hit.strategy == SearchStrategy.NO_MATCH

        # list_tmpl entry must remain
        hit = await medha.search("list departments")
        assert hit.strategy != SearchStrategy.NO_MATCH

    async def test_after_invalidate_collection_search_returns_no_match(self, medha):
        """After invalidate_collection(), all entries return NO_MATCH and collection is usable."""
        await medha.store("question alpha", "SELECT alpha")
        await medha.store("question beta", "SELECT beta")

        dropped = await medha.invalidate_collection()
        assert dropped == 2

        for q in ["question alpha", "question beta"]:
            hit = await medha.search(q)
            assert hit.strategy == SearchStrategy.NO_MATCH

        # Collection should still be functional after re-init
        ok = await medha.store("question gamma", "SELECT gamma")
        assert ok is True
        hit = await medha.search("question gamma")
        assert hit.strategy != SearchStrategy.NO_MATCH
