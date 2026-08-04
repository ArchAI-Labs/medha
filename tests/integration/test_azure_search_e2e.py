"""End-to-end integration tests: MockEmbedder + AzureSearchBackend + Medha pipeline.

Mirrors the flow of tests/integration/test_weaviate_e2e.py, but drives a live
Azure AI Search service instead of an in-process backend.

Skipped automatically when neither AZURE_SEARCH_ENDPOINT nor
MEDHA_TEST_AZURE_SEARCH_ENDPOINT is set.

The service needs room for three indexes (cache data, templates, and the stats
sidecar) — exactly what the free tier allows — so point this at an empty
service or index creation inside start() will fail on quota.

Run with:
    AZURE_SEARCH_ENDPOINT=https://<service>.search.windows.net \
    AZURE_SEARCH_API_KEY=<admin-key> \
        pytest tests/integration/test_azure_search_e2e.py -x -q
"""

import asyncio
import os
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any

import pytest

pytest.importorskip("azure.search.documents")

from medha.backends.azure_search import AzureSearchBackend, _az_meta_index_name
from medha.config import Settings
from medha.core import Medha
from medha.types import CacheResult, PersistedStats, QueryTemplate, SearchStrategy
from medha.utils.normalization import normalize_question

COLLECTION = "azure_e2e_test"
DIMENSION = 384

AZURE_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT") or os.environ.get(
    "MEDHA_TEST_AZURE_SEARCH_ENDPOINT"
)
AZURE_API_KEY = os.environ.get("AZURE_SEARCH_API_KEY") or os.environ.get(
    "MEDHA_TEST_AZURE_SEARCH_API_KEY"
)

pytestmark = [
    pytest.mark.skipif(
        not AZURE_ENDPOINT,
        reason="AZURE_SEARCH_ENDPOINT not set",
    ),
    pytest.mark.integration,
    pytest.mark.azure_search,
    pytest.mark.slow,
]

# Azure AI Search indexes asynchronously: a document the API has accepted is
# usually queryable within a second, but nothing guarantees it is visible to
# the *next* request. Every read-after-write below polls instead of asserting
# straight away.
INDEX_LAG_TIMEOUT = float(os.environ.get("MEDHA_TEST_AZURE_INDEX_LAG", "20"))
INDEX_LAG_INTERVAL = 0.5


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = dict(
        backend_type="azure-search",
        azure_search_endpoint=AZURE_ENDPOINT or "",
        azure_search_api_key=AZURE_API_KEY,
        score_threshold_exact=0.99,
        score_threshold_semantic=0.85,
        score_threshold_template=0.80,
        l1_cache_max_size=100,
    )
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


async def _eventually(
    probe: Callable[[], Awaitable[Any]],
    *,
    what: str,
    timeout: float = INDEX_LAG_TIMEOUT,
    interval: float = INDEX_LAG_INTERVAL,
) -> Any:
    """Poll *probe* until it returns something truthy, then return that value.

    Absorbs Azure's indexing lag so the suite fails on real defects rather than
    on the couple of hundred milliseconds between an accepted write and a
    queryable document.
    """
    deadline = time.monotonic() + timeout
    while True:
        result = await probe()
        if result:
            return result
        if time.monotonic() >= deadline:
            pytest.fail(f"timed out after {timeout:.0f}s waiting for {what}")
        await asyncio.sleep(interval)


async def _wait_indexed(medha: Medha, question: str) -> CacheResult:
    """Block until *question* is queryable, returning its stored entry."""
    normalized = normalize_question(question)
    return await _eventually(
        lambda: medha._backend.search_by_normalized_question(COLLECTION, normalized),
        what=f"'{question[:40]}' to become searchable",
    )


async def _wait_gone(medha: Medha, question: str) -> None:
    """Block until *question* is no longer queryable.

    Asserting the miss before the delete has propagated would let search() see
    the stale document, cache it in L1, and turn every later retry into a hit.
    """
    normalized = normalize_question(question)
    await _eventually(
        lambda: _negate(
            medha._backend.search_by_normalized_question(COLLECTION, normalized)
        ),
        what=f"'{question[:40]}' to disappear",
    )


async def _negate(awaitable: Awaitable[Any]) -> bool:
    return not await awaitable


async def _drop_everything(medha: Medha) -> None:
    """Delete every Azure index the collection owns: data, templates, stats.

    Index quota is the scarce resource on a search service (three on the free
    tier), so leaving the sidecar behind would block the next run.
    """
    backend = medha._backend
    for coll in (medha._collection_name, medha._template_collection):
        with suppress(Exception):
            await backend.drop_collection(coll)
    if backend._index_client is not None:
        with suppress(Exception):
            await backend._index_client.delete_index(
                _az_meta_index_name(
                    medha._collection_name, backend._settings.azure_search_index_name
                )
            )


@pytest.fixture
async def azure_medha(mock_embedder):
    settings = _settings()
    # Medha.start() calls backend.connect() itself — connecting here as well
    # would replace the index client and leak the first one.
    backend = AzureSearchBackend(settings)

    m = Medha(
        collection_name=COLLECTION,
        embedder=mock_embedder,
        backend=backend,
        settings=settings,
    )
    await m.start()
    yield m
    await m.invalidate_collection(COLLECTION)
    await _drop_everything(m)
    await m.close()


class TestAzureSearchE2E:
    async def test_store_and_search_semantic(self, azure_medha):
        question = "How many active users are registered?"
        stored = await azure_medha.store(
            question, "SELECT COUNT(*) FROM users WHERE active = 1"
        )
        assert stored is True

        await _wait_indexed(azure_medha, question)
        # Drop L1 + embedding cache so the answer has to come back from Azure.
        await azure_medha.clear_caches()

        hit = await azure_medha.search(question)

        # MockEmbedder is deterministic: identical text → identical vector, so
        # the round-trip through Azure scores 1.0 and lands on the exact tier.
        assert hit.strategy in (
            SearchStrategy.EXACT_MATCH,
            SearchStrategy.SEMANTIC_MATCH,
        )
        assert hit.generated_query == "SELECT COUNT(*) FROM users WHERE active = 1"
        assert hit.confidence > 0.0

    async def test_template_match(self, azure_medha):
        await azure_medha.load_templates(
            [
                QueryTemplate(
                    intent="count_entities",
                    template_text="How many {entity} are there",
                    query_template="SELECT COUNT(*) FROM {entity}",
                    parameters=["entity"],
                    priority=1,
                    parameter_patterns={"entity": r"\b(users|products|orders)\b"},
                )
            ]
        )

        hit = await azure_medha.search("How many orders are there")

        # Template scoring runs in-process over the loaded templates, so the
        # arithmetic is backend-independent: keyword overlap 1.0 * 0.5 +
        # params 1.0 * 0.3 + priority bonus 0.08 = 0.88, comfortably above the
        # 0.80 template threshold set in _settings().
        assert hit.strategy == SearchStrategy.TEMPLATE_MATCH
        assert hit.template_used == "count_entities"
        assert hit.generated_query == "SELECT COUNT(*) FROM orders"

    async def test_l1_cache_hit(self, azure_medha):
        question = "List all pending invoices"
        await azure_medha.store(question, "SELECT * FROM invoices WHERE status = 'pending'")
        await _wait_indexed(azure_medha, question)

        await azure_medha.search(question)  # populates L1
        hit = await azure_medha.search(question)

        assert hit.strategy == SearchStrategy.L1_CACHE
        assert hit.generated_query == "SELECT * FROM invoices WHERE status = 'pending'"

        stats = await azure_medha.stats()
        assert stats.total_hits >= 1

    async def test_invalidation(self, azure_medha):
        question = "Show me the newest products"
        await azure_medha.store(question, "SELECT * FROM products ORDER BY created_at DESC")
        await _wait_indexed(azure_medha, question)

        assert await azure_medha.invalidate(question) is True

        await _wait_gone(azure_medha, question)
        await azure_medha.clear_caches()

        hit = await azure_medha.search(question)
        assert hit.strategy == SearchStrategy.NO_MATCH

        # get_document_count() trails the delete by its own margin.
        await _eventually(
            lambda: _negate(azure_medha._backend.count(COLLECTION)),
            what="document count to drop to 0",
        )

    async def test_feedback_update(self, azure_medha):
        question = "Total sales by region"
        await azure_medha.store(
            question, "SELECT region, SUM(amount) FROM sales GROUP BY region"
        )
        entry = await _wait_indexed(azure_medha, question)

        assert await azure_medha.feedback(question, correct=True) is True

        backend = azure_medha._backend

        # CacheResult carries feedback counters only for InMemoryBackend; every
        # network backend (Azure included) drops them on read, so the stored
        # value is verified through the counter update_feedback() returns.
        # update_feedback() reads before it writes, so the increment above may
        # not be visible yet — assert growth rather than an exact total.
        assert await backend.update_feedback(COLLECTION, entry.id, True) >= 2
        assert await backend.update_feedback(COLLECTION, entry.id, False) == 1

    async def test_load_save_stats(self, azure_medha):
        """New in 0.5.0: stats round-trip through the Azure meta index."""
        backend = azure_medha._backend

        snapshot = PersistedStats(
            total_requests=25,
            total_hits=18,
            total_misses=6,
            total_errors=1,
            hits_by_strategy={"semantic_match": 12, "l1_cache": 6},
        )
        await backend.save_stats(COLLECTION, snapshot)

        loaded = await _eventually(
            lambda: backend.load_stats(COLLECTION),
            what="stats document to become readable",
        )
        assert loaded.total_requests == 25
        assert loaded.total_hits == 18
        assert loaded.hits_by_strategy["semantic_match"] == 12
        assert loaded.hit_rate == pytest.approx(0.72)

        # Saving again overwrites the single sidecar document rather than
        # adding one — the key is derived from the collection name.
        await backend.save_stats(COLLECTION, PersistedStats(total_requests=99))

        async def _overwritten() -> PersistedStats | None:
            current = await backend.load_stats(COLLECTION)
            return current if current is not None and current.total_requests == 99 else None

        reloaded = await _eventually(_overwritten, what="stats overwrite to land")
        assert reloaded.total_requests == 99
