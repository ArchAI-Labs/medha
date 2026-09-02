"""End-to-end metadata filtering against the engines that run in-process.

Qdrant (memory mode) and LanceDB (local storage) are the two backends whose
native filtering can be exercised for real here: everything else needs a
service. Between them they cover both kinds of pushdown — Qdrant's, which is
exact and decides the result, and LanceDB's, which only narrows the scan and
leaves the decision to Python.
"""

import uuid

import pytest

from medha.backends.memory import InMemoryBackend
from medha.backends.qdrant import QdrantBackend
from medha.config import Settings
from medha.core import Medha
from medha.types import CacheEntry, SearchStrategy
from medha.utils.normalization import query_hash
from tests.conftest import MockEmbedder

COLLECTION = "test_metadata_filters"
DIMENSION = 384

QUESTION = "total revenue for the period"
QUERY_12 = "SELECT SUM(amount) FROM sales WHERE day = '2026-08-12'"
QUERY_13 = "SELECT SUM(amount) FROM sales WHERE day = '2026-08-13'"


@pytest.fixture
def embedder():
    return MockEmbedder(dimension=DIMENSION)


@pytest.fixture
async def backend():
    b = QdrantBackend(Settings(qdrant_mode="memory"))
    await b.connect()
    await b.initialize(COLLECTION, DIMENSION)
    yield b
    await b.close()


async def _entry(embedder, question, query, metadata=None):
    return CacheEntry(
        id=str(uuid.uuid4()),
        vector=await embedder.aembed(question.lower()),
        original_question=question,
        normalized_question=question.lower(),
        generated_query=query,
        query_hash=query_hash(query),
        metadata=metadata or {},
    )


class TestQdrantMetadataRoundTrip:
    async def test_metadata_survives_upsert_and_search(self, backend, embedder):
        meta = {"resolved_date": "2026-08-12", "hour": 10, "ratio": 0.25, "draft": True}
        await backend.upsert(COLLECTION, [await _entry(embedder, QUESTION, QUERY_12, meta)])

        results = await backend.search(
            COLLECTION, await embedder.aembed(QUESTION.lower()), limit=1
        )

        assert results[0].metadata == meta

    async def test_entry_without_metadata_reads_back_empty(self, backend, embedder):
        await backend.upsert(COLLECTION, [await _entry(embedder, QUESTION, QUERY_12)])

        results = await backend.search(
            COLLECTION, await embedder.aembed(QUESTION.lower()), limit=1
        )

        assert results[0].metadata == {}

    async def test_metadata_survives_scroll(self, backend, embedder):
        await backend.upsert(
            COLLECTION,
            [await _entry(embedder, QUESTION, QUERY_12, {"tenant": "acme"})],
        )

        results, _ = await backend.scroll(COLLECTION, limit=10)

        assert results[0].metadata == {"tenant": "acme"}


class TestQdrantNativeFilter:
    async def test_filter_selects_the_right_scope(self, backend, embedder):
        await backend.upsert(COLLECTION, [
            await _entry(embedder, QUESTION, QUERY_12, {"resolved_date": "2026-08-12"}),
            await _entry(embedder, QUESTION, QUERY_13, {"resolved_date": "2026-08-13"}),
        ])
        vector = await embedder.aembed(QUESTION.lower())

        got_12 = await backend.search_filtered(
            COLLECTION, vector, limit=1, filters={"resolved_date": "2026-08-12"}
        )
        got_13 = await backend.search_filtered(
            COLLECTION, vector, limit=1, filters={"resolved_date": "2026-08-13"}
        )

        assert [r.generated_query for r in got_12] == [QUERY_12]
        assert [r.generated_query for r in got_13] == [QUERY_13]

    async def test_no_match_returns_nothing(self, backend, embedder):
        await backend.upsert(COLLECTION, [
            await _entry(embedder, QUESTION, QUERY_12, {"resolved_date": "2026-08-12"}),
        ])

        results = await backend.search_filtered(
            COLLECTION,
            await embedder.aembed(QUESTION.lower()),
            limit=5,
            filters={"resolved_date": "2026-08-14"},
        )

        assert results == []

    async def test_reaches_a_match_the_post_filter_would_miss(self, backend, embedder):
        """What pushing the filter down buys, stated as a difference.

        The wanted entry is the *least* similar of five. Asked for one result,
        Qdrant applies the filter server-side and returns it. The base class
        post-filter, over-fetching only as far as it was told to, retrieves the
        closest entry instead, discards it, and returns nothing.
        """
        questions = [f"question number {i}" for i in range(5)]
        entries = [
            await _entry(embedder, q, f"SELECT {i}", {"idx": i})
            for i, q in enumerate(questions)
        ]
        await backend.upsert(COLLECTION, entries)

        memory = InMemoryBackend()
        await memory.initialize(COLLECTION, DIMENSION)
        await memory.upsert(COLLECTION, entries)

        vector = await embedder.aembed(questions[0])
        native = await backend.search_filtered(
            COLLECTION, vector, limit=1, filters={"idx": 4}, overfetch=1
        )
        post_filtered = await memory.search_filtered(
            COLLECTION, vector, limit=1, filters={"idx": 4}, overfetch=1
        )

        assert [r.generated_query for r in native] == ["SELECT 4"]
        assert post_filtered == []

    async def test_filters_and_over_fetch_agree_when_the_match_is_reachable(
        self, backend, embedder
    ):
        """With enough over-fetch both paths return the same entry."""
        questions = [f"question number {i}" for i in range(5)]
        entries = [
            await _entry(embedder, q, f"SELECT {i}", {"idx": i})
            for i, q in enumerate(questions)
        ]
        await backend.upsert(COLLECTION, entries)

        memory = InMemoryBackend()
        await memory.initialize(COLLECTION, DIMENSION)
        await memory.upsert(COLLECTION, entries)

        vector = await embedder.aembed(questions[0])
        native = await backend.search_filtered(
            COLLECTION, vector, limit=1, filters={"idx": 4}, overfetch=10
        )
        post_filtered = await memory.search_filtered(
            COLLECTION, vector, limit=1, filters={"idx": 4}, overfetch=10
        )

        assert [r.generated_query for r in native] == ["SELECT 4"]
        assert [r.generated_query for r in post_filtered] == ["SELECT 4"]

    @pytest.mark.parametrize(
        "value",
        ["2026-08-12", 10, 0.25, True, False],
        ids=["str", "int", "float", "true", "false"],
    )
    async def test_every_scalar_type_filters_natively(self, backend, embedder, value):
        """Floats go through a degenerate range; Qdrant's match ignores them."""
        await backend.upsert(COLLECTION, [
            await _entry(embedder, QUESTION, QUERY_12, {"scope": value}),
            await _entry(embedder, "another question", QUERY_13, {"scope": "other"}),
        ])

        results = await backend.search_filtered(
            COLLECTION,
            await embedder.aembed(QUESTION.lower()),
            limit=5,
            filters={"scope": value},
        )

        assert [r.generated_query for r in results] == [QUERY_12]

    async def test_expired_entries_stay_excluded_when_filtering(self, backend, embedder):
        """The TTL condition is not lost when a metadata filter is added."""
        from datetime import datetime, timedelta, timezone

        entry = await _entry(embedder, QUESTION, QUERY_12, {"tenant": "acme"})
        entry.expires_at = datetime.now(timezone.utc) - timedelta(seconds=60)
        await backend.upsert(COLLECTION, [entry])

        results = await backend.search_filtered(
            COLLECTION,
            await embedder.aembed(QUESTION.lower()),
            limit=5,
            filters={"tenant": "acme"},
        )

        assert results == []


class TestMedhaOverQdrant:
    @pytest.fixture
    async def medha(self, embedder):
        m = Medha(
            collection_name="metadata_e2e",
            embedder=embedder,
            backend=QdrantBackend(Settings(qdrant_mode="memory")),
            settings=Settings(
                qdrant_mode="memory",
                score_threshold_exact=0.99,
                score_threshold_semantic=0.50,
            ),
        )
        await m.start()
        yield m
        await m.close()

    async def test_the_issue_scenario(self, medha):
        """Two days, one question, no confusion — and no answer for a third."""
        await medha.store(QUESTION, QUERY_12, metadata={"resolved_date": "2026-08-12"})
        await medha.store(QUESTION, QUERY_13, metadata={"resolved_date": "2026-08-13"})

        assert (
            await medha.search(QUESTION, filters={"resolved_date": "2026-08-12"})
        ).generated_query == QUERY_12
        assert (
            await medha.search(QUESTION, filters={"resolved_date": "2026-08-13"})
        ).generated_query == QUERY_13
        assert (
            await medha.search(QUESTION, filters={"resolved_date": "2026-08-14"})
        ).strategy == SearchStrategy.NO_MATCH

    async def test_unfiltered_search_is_unaffected(self, medha):
        await medha.store(QUESTION, QUERY_12, metadata={"resolved_date": "2026-08-12"})

        hit = await medha.search(QUESTION)

        assert hit.generated_query == QUERY_12
        assert hit.metadata == {"resolved_date": "2026-08-12"}


# ---------------------------------------------------------------------------
# LanceDB: a narrowing pushdown, with Python deciding
# ---------------------------------------------------------------------------

lancedb = pytest.importorskip("lancedb")


@pytest.fixture
async def lance_medha(embedder, tmp_path):
    from medha.backends.lancedb import LanceDBBackend

    settings = Settings(
        backend_type="lancedb",
        lancedb_uri=str(tmp_path / "lancedb_filters"),
        score_threshold_exact=0.99,
        score_threshold_semantic=0.50,
    )
    backend = LanceDBBackend(settings)
    await backend.connect()
    m = Medha(
        collection_name="lance_filters",
        embedder=embedder,
        backend=backend,
        settings=settings,
    )
    await m.start()
    yield m
    await m.close()


class TestLanceDBMetadataFilters:
    async def test_metadata_round_trips_through_the_column(self, lance_medha):
        meta = {"resolved_date": "2026-08-12", "hour": 10, "ratio": 0.25, "draft": True}
        await lance_medha.store(QUESTION, QUERY_12, metadata=meta)

        hit = await lance_medha.search(QUESTION)

        assert hit.metadata == meta

    async def test_the_issue_scenario(self, lance_medha):
        await lance_medha.store(QUESTION, QUERY_12, metadata={"resolved_date": "2026-08-12"})
        await lance_medha.store(QUESTION, QUERY_13, metadata={"resolved_date": "2026-08-13"})

        assert (
            await lance_medha.search(QUESTION, filters={"resolved_date": "2026-08-12"})
        ).generated_query == QUERY_12
        assert (
            await lance_medha.search(QUESTION, filters={"resolved_date": "2026-08-13"})
        ).generated_query == QUERY_13
        assert (
            await lance_medha.search(QUESTION, filters={"resolved_date": "2026-08-14"})
        ).strategy == SearchStrategy.NO_MATCH

    async def test_the_like_clause_actually_runs(self, lance_medha):
        """DataFusion has to accept the expression, not just look plausible."""
        results = await lance_medha._backend.search_filtered(
            "lance_filters",
            await lance_medha._embedder.aembed(QUESTION),
            limit=5,
            filters={"tenant": "acme"},
        )

        assert results == []

    async def test_a_wildcard_in_a_value_cannot_produce_a_wrong_answer(self, lance_medha):
        """`%` is a LIKE wildcard: it widens the scan, never the result."""
        await lance_medha.store("q one", "SELECT 1", metadata={"tenant": "a%c"})
        await lance_medha.store("q two", "SELECT 2", metadata={"tenant": "abc"})
        await lance_medha._l1_backend.clear()

        hit = await lance_medha.search("q two", filters={"tenant": "a%c"})

        # 'abc' is what LIKE 'a%c' matches; only Python rejects it.
        assert hit.strategy == SearchStrategy.NO_MATCH

    async def test_a_quote_in_a_value_is_escaped(self, lance_medha):
        await lance_medha.store(QUESTION, QUERY_12, metadata={"tenant": "o'brien"})
        await lance_medha._l1_backend.clear()

        hit = await lance_medha.search(QUESTION, filters={"tenant": "o'brien"})

        assert hit.generated_query == QUERY_12

    async def test_numbers_ride_the_residual(self, lance_medha):
        await lance_medha.store(QUESTION, QUERY_12, metadata={"hour": 10})
        await lance_medha._l1_backend.clear()

        assert (
            await lance_medha.search(QUESTION, filters={"hour": 10})
        ).generated_query == QUERY_12
        assert (
            await lance_medha.search(QUESTION, filters={"hour": 11})
        ).strategy == SearchStrategy.NO_MATCH

    async def test_expired_entries_stay_excluded_when_filtering(self, lance_medha):
        """The parenthesised TTL clause, against a real query planner."""
        await lance_medha.store(
            QUESTION, QUERY_12, metadata={"tenant": "acme"}, ttl=-60
        )
        await lance_medha._l1_backend.clear()

        hit = await lance_medha.search(QUESTION, filters={"tenant": "acme"})

        assert hit.strategy == SearchStrategy.NO_MATCH
