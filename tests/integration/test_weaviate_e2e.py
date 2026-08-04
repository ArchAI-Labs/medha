"""End-to-end integration tests: MockEmbedder + WeaviateBackend + Medha pipeline.

Mirrors the flow of tests/integration/test_end_to_end.py, but drives a live
Weaviate instance instead of an in-process backend.

Skipped automatically when neither WEAVIATE_TEST_URL nor
MEDHA_TEST_WEAVIATE_HOST is set.

Run with:
    WEAVIATE_TEST_URL=http://localhost:8080 \
        pytest tests/integration/test_weaviate_e2e.py -x -q
"""

import asyncio
import os
from contextlib import suppress
from urllib.parse import urlparse

import pytest

pytest.importorskip("weaviate")

from medha.backends.weaviate import WeaviateBackend, _wv_meta_collection_name
from medha.config import Settings
from medha.core import Medha
from medha.types import PersistedStats, QueryTemplate, SearchStrategy
from medha.utils.normalization import normalize_question

COLLECTION = "weaviate_e2e_test"
DIMENSION = 384

_RAW_HOST = os.environ.get("WEAVIATE_TEST_URL") or os.environ.get("MEDHA_TEST_WEAVIATE_HOST")

pytestmark = [
    pytest.mark.skipif(
        not _RAW_HOST,
        reason="WEAVIATE_TEST_URL not set",
    ),
    pytest.mark.integration,
    pytest.mark.weaviate,
    pytest.mark.slow,
]


def _parse_host(raw: str | None) -> tuple[str, int]:
    """Accept a bare hostname, a ``host:port`` pair, or a full http(s):// URL.

    weaviate-client v4 wants a bare host plus a port, so a URL like
    ``http://localhost:8080`` has to be split before it reaches Settings.
    """
    if not raw:
        return "localhost", 8080
    if "://" in raw:
        parsed = urlparse(raw)
        return parsed.hostname or "localhost", parsed.port or 8080
    if ":" in raw:
        host, _, port = raw.partition(":")
        return host, int(port)
    return raw, 8080


WEAVIATE_HOST, WEAVIATE_HTTP_PORT = _parse_host(_RAW_HOST)
WEAVIATE_GRPC_PORT = int(os.environ.get("MEDHA_TEST_WEAVIATE_GRPC_PORT", "50051"))


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = dict(
        backend_type="weaviate",
        weaviate_mode="local",
        weaviate_host=WEAVIATE_HOST,
        weaviate_http_port=WEAVIATE_HTTP_PORT,
        weaviate_grpc_port=WEAVIATE_GRPC_PORT,
        score_threshold_exact=0.99,
        score_threshold_semantic=0.85,
        score_threshold_template=0.80,
        l1_cache_max_size=100,
    )
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


async def _drop_everything(medha: Medha) -> None:
    """Remove every Weaviate class the collection owns: data, templates, stats.

    Weaviate keeps classes around after their objects are deleted, so a plain
    invalidate_collection() would leave the schema (and any persisted stats
    sidecar) behind for the next run to inherit.
    """
    backend = medha._backend
    client = backend._client
    for coll in (medha._collection_name, medha._template_collection):
        with suppress(Exception):
            await backend.drop_collection(coll)
    if client is not None:
        with suppress(Exception):
            await client.collections.delete(
                _wv_meta_collection_name(
                    backend._settings.weaviate_collection_prefix, medha._collection_name
                )
            )


async def _drain(medha: Medha) -> None:
    """Await the fire-and-forget stats persistence tasks scheduled by search()."""
    if medha._stats_persist_tasks:
        await asyncio.gather(*medha._stats_persist_tasks)


@pytest.fixture
async def weaviate_medha(mock_embedder):
    settings = _settings()
    # Medha.start() calls backend.connect() itself — connecting here as well
    # would leave a second, unclosed client behind.
    backend = WeaviateBackend(settings)

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


class TestWeaviateE2E:
    async def test_store_and_search_semantic(self, weaviate_medha):
        question = "How many active users are registered?"
        stored = await weaviate_medha.store(
            question, "SELECT COUNT(*) FROM users WHERE active = 1"
        )
        assert stored is True

        # Drop L1 + embedding cache so the answer has to come back from Weaviate.
        await weaviate_medha.clear_caches()

        hit = await weaviate_medha.search(question)

        # MockEmbedder is deterministic: identical text → identical vector, so
        # the round-trip through Weaviate scores 1.0 and lands on the exact tier.
        assert hit.strategy in (
            SearchStrategy.EXACT_MATCH,
            SearchStrategy.SEMANTIC_MATCH,
        )
        assert hit.generated_query == "SELECT COUNT(*) FROM users WHERE active = 1"
        assert hit.confidence > 0.0

    async def test_store_and_search_miss(self, weaviate_medha):
        await weaviate_medha.store("How many active users are registered?", "SELECT COUNT(*) FROM users")
        await weaviate_medha.clear_caches()

        hit = await weaviate_medha.search("completely unrelated question about the weather")

        assert hit.strategy == SearchStrategy.NO_MATCH

    async def test_template_match(self, weaviate_medha):
        await weaviate_medha.load_templates(
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

        hit = await weaviate_medha.search("How many orders are there")

        # keyword overlap 1.0 * 0.5 + params 1.0 * 0.3 + priority bonus 0.08 = 0.88,
        # comfortably above the 0.80 template threshold set in _settings().
        assert hit.strategy == SearchStrategy.TEMPLATE_MATCH
        assert hit.template_used == "count_entities"
        assert hit.generated_query == "SELECT COUNT(*) FROM orders"

    async def test_l1_cache_hit(self, weaviate_medha):
        question = "List all pending invoices"
        await weaviate_medha.store(question, "SELECT * FROM invoices WHERE status = 'pending'")

        await weaviate_medha.search(question)  # populates L1
        hit = await weaviate_medha.search(question)

        assert hit.strategy == SearchStrategy.L1_CACHE
        assert hit.generated_query == "SELECT * FROM invoices WHERE status = 'pending'"

        stats = await weaviate_medha.stats()
        assert stats.total_hits >= 1

    async def test_invalidation(self, weaviate_medha):
        question = "Show me the newest products"
        await weaviate_medha.store(question, "SELECT * FROM products ORDER BY created_at DESC")

        assert await weaviate_medha.invalidate(question) is True
        await weaviate_medha.clear_caches()

        hit = await weaviate_medha.search(question)
        assert hit.strategy == SearchStrategy.NO_MATCH
        assert await weaviate_medha._backend.count(COLLECTION) == 0

    async def test_invalidate_collection_clears_every_entry(self, weaviate_medha):
        await weaviate_medha.store("first cached question", "SELECT 1")
        await weaviate_medha.store("second cached question", "SELECT 2")
        assert await weaviate_medha._backend.count(COLLECTION) == 2

        dropped = await weaviate_medha.invalidate_collection(COLLECTION)

        assert dropped == 2
        assert await weaviate_medha._backend.count(COLLECTION) == 0

    async def test_feedback_update(self, weaviate_medha):
        question = "Total sales by region"
        await weaviate_medha.store(
            question, "SELECT region, SUM(amount) FROM sales GROUP BY region"
        )

        assert await weaviate_medha.feedback(question, correct=True) is True

        backend = weaviate_medha._backend
        entry = await backend.search_by_normalized_question(
            COLLECTION, normalize_question(question)
        )
        assert entry is not None

        # CacheResult carries feedback counters only for InMemoryBackend; every
        # network backend (Weaviate included) drops them on read, so the stored
        # value is verified through the counter update_feedback() returns.
        assert await backend.update_feedback(COLLECTION, entry.id, True) == 2
        assert await backend.update_feedback(COLLECTION, entry.id, False) == 1

    async def test_feedback_on_unknown_question_returns_false(self, weaviate_medha):
        assert await weaviate_medha.feedback("never stored anywhere", correct=True) is False

    async def test_load_save_stats(self, weaviate_medha):
        """New in 0.5.0: stats round-trip through the Weaviate meta class."""
        backend = weaviate_medha._backend

        snapshot = PersistedStats(
            total_requests=25,
            total_hits=18,
            total_misses=6,
            total_errors=1,
            hits_by_strategy={"semantic_match": 12, "l1_cache": 6},
        )
        await backend.save_stats(COLLECTION, snapshot)

        loaded = await backend.load_stats(COLLECTION)

        assert loaded is not None
        assert loaded.total_requests == 25
        assert loaded.total_hits == 18
        assert loaded.hits_by_strategy["semantic_match"] == 12
        assert loaded.hit_rate == pytest.approx(0.72)

        # Saving again overwrites the single sidecar object rather than adding one.
        await backend.save_stats(COLLECTION, PersistedStats(total_requests=99))
        reloaded = await backend.load_stats(COLLECTION)
        assert reloaded is not None
        assert reloaded.total_requests == 99

    async def test_load_stats_returns_none_for_fresh_collection(self, weaviate_medha):
        assert await weaviate_medha._backend.load_stats(COLLECTION) is None

    async def test_stats_survive_restart(self, mock_embedder):
        """A new Medha over the same Weaviate picks up the persisted counters."""
        collection = f"{COLLECTION}_restart"
        settings = _settings(stats_persist_interval=1)  # persist on every request

        m1 = Medha(collection, mock_embedder, WeaviateBackend(settings), settings)
        await m1.start()
        try:
            await m1.store("what is revenue?", "SELECT SUM(revenue) FROM sales")
            await m1.search("what is revenue?")  # hit
            await m1.search("something entirely unrelated to revenue")  # miss
            await _drain(m1)
        finally:
            await m1.close()

        m2 = Medha(collection, mock_embedder, WeaviateBackend(settings), settings)
        await m2.start()
        try:
            stats = await m2.stats(collection)
            assert stats.total_requests == 2
            assert stats.total_hits >= 1
        finally:
            await _drop_everything(m2)
            await m2.close()
