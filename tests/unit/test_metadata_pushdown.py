"""What each backend actually sends when a filtered search is pushed down.

Only Qdrant and the in-memory backend can be exercised against a real engine
here — Chroma and LanceDB have no driver installed, and pgvector, VectorChord
and Elasticsearch need a server that neither this environment nor CI provides.
So the query each dialect builds is asserted directly. That catches a
malformed clause, a parameter numbered wrong, a predicate that binds the wrong
way; it cannot catch an engine disagreeing about what the clause means, which
is why `split_filters` only ever pushes down string equality.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from medha.config import Settings
from medha.types import CacheEntry, CacheResult
from medha.utils.metadata import (
    MAX_FILTER_FETCH,
    filter_fetch_size,
    split_filters,
    verify_filters,
)

COLL = "pushdown_test"
DIM = 8


def _entry(metadata) -> CacheEntry:
    return CacheEntry(
        id="00000000-0000-0000-0000-000000000001",
        vector=[0.1] * DIM,
        original_question="q",
        normalized_question="q",
        generated_query="SELECT SUM(amount) FROM sales",
        query_hash="h",
        metadata=metadata,
    )


def _result(id_: str, metadata=None, score: float = 1.0) -> CacheResult:
    return CacheResult(
        id=id_,
        score=score,
        original_question="q",
        normalized_question="q",
        generated_query=f"SELECT {id_}",
        query_hash="h",
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# The rules every backend shares
# ---------------------------------------------------------------------------

class TestSplitFilters:
    def test_strings_go_down_numbers_stay(self):
        pushable, residual = split_filters(
            {"date": "2026-08-12", "hour": 10, "ratio": 0.5, "draft": True}
        )

        assert pushable == {"date": "2026-08-12"}
        assert residual == {"hour": 10, "ratio": 0.5, "draft": True}

    def test_empty_filters_split_into_nothing(self):
        assert split_filters(None) == ({}, {})
        assert split_filters({}) == ({}, {})

    def test_a_backend_can_widen_what_it_accepts(self):
        pushable, residual = split_filters(
            {"date": "2026-08-12", "draft": True, "hour": 10}, pushable=(str, bool)
        )

        assert pushable == {"date": "2026-08-12", "draft": True}
        assert residual == {"hour": 10}

    def test_bool_is_not_swept_up_as_a_number(self):
        pushable, residual = split_filters({"draft": True}, pushable=(str, int))

        # bool is an int subclass, so a backend widening to int gets bools too —
        # which is what isinstance means and what the dialects then see.
        assert pushable == {"draft": True}
        assert residual == {}


class TestFilterFetchSize:
    def test_exact_when_the_engine_decided_everything(self):
        assert filter_fetch_size(5, {}, 10) == 5

    def test_widened_when_python_still_has_to_check(self):
        assert filter_fetch_size(5, {"hour": 10}, 10) == 50

    def test_capped(self):
        assert filter_fetch_size(500, {"hour": 10}, 10) == MAX_FILTER_FETCH

    def test_never_below_the_limit(self):
        assert filter_fetch_size(5, {"hour": 10}, 1) == 5


class TestVerifyFilters:
    def test_drops_mismatches_and_trims(self):
        results = [
            _result("a", {"tenant": "acme"}),
            _result("b", {"tenant": "globex"}),
            _result("c", {"tenant": "acme"}),
        ]

        kept = verify_filters(results, {"tenant": "acme"}, limit=1)

        assert [r.id for r in kept] == ["a"]

    def test_without_filters_only_trims(self):
        results = [_result("a"), _result("b")]

        assert [r.id for r in verify_filters(results, None, limit=1)] == ["a"]

    def test_corrects_an_engine_that_was_too_permissive(self):
        """The point of running it even after a native filter."""
        as_if_engine_ignored_the_filter = [_result("a", {"tenant": "globex"})]

        assert verify_filters(as_if_engine_ignored_the_filter, {"tenant": "acme"}, 5) == []


# ---------------------------------------------------------------------------
# PostgreSQL (pgvector and VectorChord share the statement)
# ---------------------------------------------------------------------------
#
# The driver checks live inside the fixtures, not at module level: an
# importorskip out here would skip the whole file, and CI installs only
# [dev,qdrant,fuzzy] — so the dialect tests that need no driver at all would
# be the ones silently lost.


@pytest.fixture
def mock_pool():
    pool = MagicMock()
    conn = AsyncMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire = MagicMock(return_value=ctx)
    pool.close = AsyncMock()
    conn.fetch.return_value = []
    return pool, conn


@pytest.fixture
async def pg_backend(mock_pool):
    pytest.importorskip("asyncpg")
    pytest.importorskip("pgvector")
    from medha.backends.pgvector import PgVectorBackend

    pool, conn = mock_pool
    with patch(
        "medha.backends.pgvector.asyncpg.create_pool", new=AsyncMock(return_value=pool)
    ):
        b = PgVectorBackend(Settings(backend_type="pgvector"))
        await b.connect()
        yield b, conn
    await b.close()


class TestPostgresPushdown:
    async def test_unfiltered_statement_is_unchanged(self, pg_backend):
        b, conn = pg_backend

        await b.search(COLL, [0.1] * DIM, limit=5, score_threshold=0.5)

        sql, *params = conn.fetch.call_args.args
        assert "metadata @>" not in sql
        assert "LIMIT $3" in sql
        assert params == [[0.1] * DIM, 0.5, 5]

    async def test_string_filter_becomes_jsonb_containment(self, pg_backend):
        b, conn = pg_backend

        await b.search_filtered(
            COLL, [0.1] * DIM, limit=5, filters={"resolved_date": "2026-08-12"}
        )

        sql, *params = conn.fetch.call_args.args
        assert "AND metadata @> $3::jsonb" in sql
        assert "LIMIT $4" in sql
        # Bound, never interpolated.
        assert params[2] == '{"resolved_date":"2026-08-12"}'
        assert params[3] == 5  # nothing left for Python, so no over-fetch

    async def test_numeric_filter_stays_in_python_and_widens_the_fetch(self, pg_backend):
        b, conn = pg_backend

        await b.search_filtered(COLL, [0.1] * DIM, limit=5, filters={"hour": 10})

        sql, *params = conn.fetch.call_args.args
        assert "metadata @>" not in sql
        assert params[-1] == 50

    async def test_mixed_filter_pushes_the_string_and_widens(self, pg_backend):
        b, conn = pg_backend

        await b.search_filtered(
            COLL, [0.1] * DIM, limit=2, filters={"date": "2026-08-12", "hour": 10}
        )

        sql, *params = conn.fetch.call_args.args
        assert "AND metadata @> $3::jsonb" in sql
        assert params[2] == '{"date":"2026-08-12"}'
        assert params[3] == 20

    async def test_results_are_verified_against_the_whole_filter(self, pg_backend):
        """A row the SQL let through still has to satisfy the residual."""
        b, conn = pg_backend
        conn.fetch.return_value = [
            {
                "id": "a", "original_question": "q", "normalized_question": "q",
                "generated_query": "SELECT 1", "query_hash": "h", "score": 1.0,
                "metadata": '{"date":"2026-08-12","hour":11}',
            }
        ]

        results = await b.search_filtered(
            COLL, [0.1] * DIM, limit=5, filters={"date": "2026-08-12", "hour": 10}
        )

        assert results == []

    async def test_upgrading_adds_a_column_and_builds_no_index(self, pg_backend):
        """An upgrade must not stall writes on an existing deployment.

        A plain CREATE INDEX locks the table for the whole build, and on every
        row written before the upgrade the column it would index is `{}`.
        """
        b, conn = pg_backend

        await b.initialize(COLL, DIM)

        statements = [call.args[0] for call in conn.execute.call_args_list]
        metadata_ddl = [s for s in statements if "metadata" in s]
        assert len(metadata_ddl) == 1
        assert "ADD COLUMN IF NOT EXISTS metadata JSONB" in metadata_ddl[0]
        # "gin" alone would match ori-gin-al_question.
        assert not any("using gin" in s.lower() for s in statements)

    async def test_the_index_is_handed_over_instead(self, pg_backend):
        b, _ = pg_backend

        sql = b.metadata_index_sql("my_cache")

        assert "CREATE INDEX CONCURRENTLY IF NOT EXISTS" in sql
        assert "USING gin (metadata)" in sql
        assert b._table_name("my_cache") in sql

    async def test_vectorchord_runs_the_same_statement(self, mock_pool):
        pytest.importorskip("asyncpg")
        from medha.backends.vectorchord import VectorChordBackend

        pool, conn = mock_pool
        with patch(
            "medha.backends.vectorchord.asyncpg.create_pool",
            new=AsyncMock(return_value=pool),
        ):
            b = VectorChordBackend(Settings(backend_type="vectorchord"))
            await b.connect()
            await b.search_filtered(COLL, [0.1] * DIM, limit=5, filters={"t": "acme"})
            await b.close()

        sql, *params = conn.fetch.call_args.args
        assert "AND metadata @> $3::jsonb" in sql
        assert params[2] == '{"t":"acme"}'


# ---------------------------------------------------------------------------
# Elasticsearch
# ---------------------------------------------------------------------------

@pytest.fixture
async def es_backend():
    pytest.importorskip("elasticsearch")
    from medha.backends.elasticsearch import ElasticsearchBackend

    client = AsyncMock()
    client.info = AsyncMock(return_value={"version": {"number": "8.0.0"}})
    client.search = AsyncMock(return_value={"hits": {"hits": []}})
    client.close = AsyncMock()
    with patch(
        "medha.backends.elasticsearch.AsyncElasticsearch", return_value=client
    ):
        b = ElasticsearchBackend(Settings(backend_type="elasticsearch"))
        await b.connect()
        yield b, client
    await b.close()


class TestElasticsearchPushdown:
    async def test_unfiltered_query_carries_only_the_ttl_filter(self, es_backend):
        b, client = es_backend

        await b.search(COLL, [0.1] * DIM, limit=5)

        knn = client.search.call_args.kwargs["body"]["knn"]
        assert "should" in knn["filter"]["bool"]  # the ttl clause, unwrapped
        assert knn["k"] == 5

    async def test_string_filter_becomes_a_term_on_the_flattened_field(self, es_backend):
        b, client = es_backend

        await b.search_filtered(
            COLL, [0.1] * DIM, limit=5, filters={"resolved_date": "2026-08-12"}
        )

        clauses = client.search.call_args.kwargs["body"]["knn"]["filter"]["bool"]["filter"]
        assert {"term": {"metadata.resolved_date": "2026-08-12"}} in clauses
        assert len(clauses) == 2  # ttl + one term

    async def test_numbers_are_not_pushed_into_a_term(self, es_backend):
        """A flattened field indexes 10 as "10"; the term would be a guess."""
        b, client = es_backend

        await b.search_filtered(COLL, [0.1] * DIM, limit=5, filters={"hour": 10})

        body = client.search.call_args.kwargs["body"]
        assert "should" in body["knn"]["filter"]["bool"]
        assert body["knn"]["k"] == 50  # widened instead

    async def test_num_candidates_keeps_up_with_the_fetch(self, es_backend):
        b, client = es_backend

        await b.search_filtered(COLL, [0.1] * DIM, limit=100, filters={"hour": 10})

        knn = client.search.call_args.kwargs["body"]["knn"]
        assert knn["num_candidates"] >= knn["k"]


# ---------------------------------------------------------------------------
# Chroma (driver not installed here: the clause builders are pure)
# ---------------------------------------------------------------------------

class TestChromaPushdown:
    def test_unfiltered_where_is_the_ttl_clause_alone(self):
        from medha.backends.chroma import ChromaBackend

        where = ChromaBackend._build_where({})

        assert set(where) == {"$or"}

    def test_filters_are_anded_with_the_ttl_clause(self):
        from medha.backends.chroma import ChromaBackend

        where = ChromaBackend._build_where({"resolved_date": "2026-08-12"})

        assert {"md.resolved_date": {"$eq": "2026-08-12"}} in where["$and"]
        assert len(where["$and"]) == 2

    def test_metadata_keys_are_mirrored_for_filtering(self):
        from medha.backends.chroma import _entry_to_metadata

        stored = _entry_to_metadata(_entry({"resolved_date": "2026-08-12", "hour": 10}))

        assert stored["md.resolved_date"] == "2026-08-12"
        assert stored["md.hour"] == 10
        # and the blob is still there, as the source of truth on read
        assert stored["metadata_json"] == '{"hour":10,"resolved_date":"2026-08-12"}'

    def test_mirrored_keys_cannot_collide_with_the_fixed_ones(self):
        from medha.backends.chroma import _MD_PREFIX, _entry_to_metadata

        fixed = set(_entry_to_metadata(_entry({}))) - {"metadata_json"}

        assert not any(name.startswith(_MD_PREFIX) for name in fixed)

    def test_a_metadata_key_cannot_overwrite_a_fixed_one(self):
        """Even a key named after a column lands in its own namespace."""
        from medha.backends.chroma import _entry_to_metadata

        stored = _entry_to_metadata(_entry({"generated_query": "evil"}))

        assert stored["generated_query"] == "SELECT SUM(amount) FROM sales"
        assert stored["md.generated_query"] == "evil"


# ---------------------------------------------------------------------------
# LanceDB (driver not installed here: the clause builder is pure)
# ---------------------------------------------------------------------------

class TestLanceDBPushdown:
    def test_ttl_clause_is_parenthesised(self):
        """An unparenthesised OR would let entries with no expiry skip the AND."""
        from medha.backends.lancedb import LanceDBBackend

        where = LanceDBBackend._build_where({"tenant": "acme"})

        assert where.startswith("(expires_at = '' OR expires_at > '")
        assert ") AND metadata_json LIKE " in where

    def test_pair_is_matched_as_canonical_json(self):
        from medha.backends.lancedb import LanceDBBackend

        where = LanceDBBackend._build_where({"resolved_date": "2026-08-12"})

        assert """metadata_json LIKE '%"resolved_date":"2026-08-12"%'""" in where

    def test_the_needle_is_a_substring_of_what_was_stored(self):
        """Soundness: a matching row must contain the pair verbatim."""
        from medha.backends.lancedb import LanceDBBackend, _entry_to_row

        stored = _entry_to_row(_entry({"resolved_date": "2026-08-12", "hour": 10}))
        where = LanceDBBackend._build_where({"resolved_date": "2026-08-12"})
        needle = where.split("LIKE '%")[1].split("%'")[0]

        assert needle in stored["metadata_json"]

    def test_quotes_in_a_value_are_escaped(self):
        from medha.backends.lancedb import LanceDBBackend

        where = LanceDBBackend._build_where({"tenant": "o'brien"})

        assert "o''brien" in where

    def test_several_filters_are_anded(self):
        from medha.backends.lancedb import LanceDBBackend

        where = LanceDBBackend._build_where({"a": "1", "b": "2"})

        assert where.count("metadata_json LIKE") == 2

    def test_no_filters_leaves_the_clause_alone(self):
        from medha.backends.lancedb import LanceDBBackend

        assert "metadata_json" not in LanceDBBackend._build_where({})
