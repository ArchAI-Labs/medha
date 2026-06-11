"""Unit tests for VectorChordBackend (mocked asyncpg — no real PostgreSQL required)."""

import hashlib
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

asyncpg = pytest.importorskip("asyncpg")

from medha.config import Settings
from medha.exceptions import ConfigurationError, StorageError
from medha.types import CacheEntry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COLL = "test_collection"
DIM = 8


def _make_entry(
    id: str | None = None,
    vector: list[float] | None = None,
    question: str = "test question",
    query: str = "SELECT 1",
    dim: int = DIM,
) -> CacheEntry:
    vec = vector or [0.1] * dim
    return CacheEntry(
        id=id or str(uuid.uuid4()),
        vector=vec,
        original_question=question,
        normalized_question=question.lower(),
        generated_query=query,
        query_hash=hashlib.md5(query.encode()).hexdigest(),
    )


def _make_pg_error(msg: str = "simulated postgres error") -> asyncpg.PostgresError:
    err = asyncpg.PostgresError.__new__(asyncpg.PostgresError)
    Exception.__init__(err, msg)
    return err


def _vc_settings(**overrides) -> Settings:
    return Settings(backend_type="vectorchord", **overrides)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
    conn.fetchrow.return_value = [0]
    conn.execute.return_value = "UPDATE 0"
    conn.executemany.return_value = None
    return pool, conn


@pytest.fixture
async def vc_backend(mock_pool):
    from medha.backends.vectorchord import VectorChordBackend

    pool, conn = mock_pool
    with patch(
        "medha.backends.vectorchord.asyncpg.create_pool",
        new=AsyncMock(return_value=pool),
    ):
        b = VectorChordBackend(_vc_settings())
        await b.connect()
        yield b, conn
    await b.close()


# ---------------------------------------------------------------------------
# connect
# ---------------------------------------------------------------------------


async def test_connect_creates_pool(mock_pool):
    from medha.backends.vectorchord import VectorChordBackend

    pool, _ = mock_pool
    settings = _vc_settings(
        pg_host="db-host",
        pg_port=5432,
        pg_database="mydb",
        pg_user="alice",
        pg_password="secret",
    )
    with patch(
        "medha.backends.vectorchord.asyncpg.create_pool",
        new=AsyncMock(return_value=pool),
    ) as mock_cp:
        b = VectorChordBackend(settings)
        await b.connect()
        await b.close()

    mock_cp.assert_awaited_once()
    kwargs = mock_cp.call_args.kwargs
    assert kwargs["host"] == "db-host"
    assert kwargs["database"] == "mydb"
    assert kwargs["user"] == "alice"
    assert "dsn" not in kwargs


async def test_connect_uses_dsn_when_set(mock_pool):
    from medha.backends.vectorchord import VectorChordBackend

    pool, _ = mock_pool
    dsn = "postgresql://user:pw@host:5432/db"
    settings = _vc_settings(pg_dsn=dsn)
    with patch(
        "medha.backends.vectorchord.asyncpg.create_pool",
        new=AsyncMock(return_value=pool),
    ) as mock_cp:
        b = VectorChordBackend(settings)
        await b.connect()
        await b.close()

    kwargs = mock_cp.call_args.kwargs
    assert kwargs["dsn"] == dsn
    assert "host" not in kwargs


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


async def test_initialize_executes_ddl(vc_backend):
    b, conn = vc_backend
    await b.initialize(COLL, DIM)

    all_sql = " ".join(str(c.args[0]) for c in conn.execute.call_args_list)
    assert "CREATE EXTENSION" in all_sql
    assert "vectorchord" in all_sql.lower()
    assert "CREATE TABLE" in all_sql
    assert "CREATE INDEX" in all_sql
    assert "vchordrq" in all_sql
    assert "query_hash" in all_sql


async def test_initialize_idempotent(vc_backend):
    b, conn = vc_backend
    await b.initialize(COLL, DIM)
    conn.execute.reset_mock()
    await b.initialize(COLL, DIM)

    assert conn.execute.call_count == 0


# ---------------------------------------------------------------------------
# _validate_vc_lists (vc_lists is interpolated into the vchordrq index DDL)
# ---------------------------------------------------------------------------


def test_validate_vc_lists_accepts_list_of_ints():
    from medha.backends.vectorchord import _validate_vc_lists

    assert _validate_vc_lists([1000]) == [1000]
    assert _validate_vc_lists([500, 1000]) == [500, 1000]


def test_validate_vc_lists_normalises_tuple_to_list():
    from medha.backends.vectorchord import _validate_vc_lists

    result = _validate_vc_lists((100, 200))
    assert result == [100, 200]
    assert isinstance(result, list)


def test_validate_vc_lists_allows_empty():
    # vchordrq accepts an empty lists config — we must not change that behaviour.
    from medha.backends.vectorchord import _validate_vc_lists

    assert _validate_vc_lists([]) == []


@pytest.mark.parametrize(
    "bad",
    [
        "'); DROP TABLE x; --",  # SQL-injection-style string (the original concern)
        ["a", "b"],              # non-int elements
        [1000, "2000"],          # mixed
        [True, False],           # bool is an int subclass but not a valid count
        [1.5],                   # float
        None,                    # not a sequence
        {"lists": 1000},         # mapping
    ],
)
def test_validate_vc_lists_rejects_invalid(bad):
    from medha.backends.vectorchord import _validate_vc_lists
    from medha.exceptions import StorageInitializationError

    with pytest.raises(StorageInitializationError):
        _validate_vc_lists(bad)


async def test_initialize_rejects_malicious_vc_lists_kwarg(vc_backend):
    """A non-int vc_lists passed via kwargs is rejected before any SQL runs."""
    from medha.exceptions import StorageInitializationError

    b, conn = vc_backend
    with pytest.raises(StorageInitializationError):
        await b.initialize(COLL, DIM, vc_lists="'); DROP TABLE x; --")
    conn.execute.assert_not_called()


# ---------------------------------------------------------------------------
# upsert
# ---------------------------------------------------------------------------


async def test_upsert_calls_executemany(vc_backend):
    b, conn = vc_backend
    await b.initialize(COLL, DIM)

    entries = [_make_entry() for _ in range(3)]
    await b.upsert(COLL, entries)

    conn.executemany.assert_awaited_once()
    _, rows = conn.executemany.call_args.args
    assert len(rows) == 3


async def test_upsert_empty_list_no_call(vc_backend):
    b, conn = vc_backend
    await b.initialize(COLL, DIM)
    await b.upsert(COLL, [])
    conn.executemany.assert_not_awaited()


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


async def test_search_executes_correct_sql(vc_backend):
    b, conn = vc_backend
    await b.initialize(COLL, DIM)
    conn.fetch.return_value = []

    await b.search(COLL, [0.1] * DIM, limit=5, score_threshold=0.8)

    conn.fetch.assert_awaited_once()
    sql = conn.fetch.call_args.args[0]
    assert "<=>" in sql
    assert "ORDER BY" in sql
    assert "LIMIT" in sql
    assert "WHERE" in sql


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------


async def test_count_executes_count_sql(vc_backend):
    b, conn = vc_backend
    await b.initialize(COLL, DIM)
    conn.fetchrow.return_value = [42]

    result = await b.count(COLL)

    conn.fetchrow.assert_awaited_once()
    sql = conn.fetchrow.call_args.args[0]
    assert "COUNT(*)" in sql
    assert result == 42


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


async def test_delete_executes_delete_sql(vc_backend):
    b, conn = vc_backend
    await b.initialize(COLL, DIM)
    ids = [str(uuid.uuid4()), str(uuid.uuid4())]

    await b.delete(COLL, ids)

    conn.execute.assert_awaited()
    delete_calls = [
        c for c in conn.execute.call_args_list if "DELETE" in str(c.args[0])
    ]
    assert len(delete_calls) == 1
    assert "ANY" in delete_calls[0].args[0]


async def test_delete_empty_list_no_call(vc_backend):
    b, conn = vc_backend
    await b.initialize(COLL, DIM)
    conn.execute.reset_mock()

    await b.delete(COLL, [])
    conn.execute.assert_not_awaited()


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


async def test_close_closes_pool(mock_pool):
    from medha.backends.vectorchord import VectorChordBackend

    pool, _ = mock_pool
    with patch(
        "medha.backends.vectorchord.asyncpg.create_pool",
        new=AsyncMock(return_value=pool),
    ):
        b = VectorChordBackend(_vc_settings())
        await b.connect()
        assert b._pool is not None
        await b.close()

    pool.close.assert_awaited_once()
    assert b._pool is None
    assert len(b._initialized_tables) == 0


# ---------------------------------------------------------------------------
# not connected guard
# ---------------------------------------------------------------------------


async def test_not_connected_raises():
    from medha.backends.vectorchord import VectorChordBackend

    b = VectorChordBackend.__new__(VectorChordBackend)
    b._pool = None
    b._initialized_tables = set()
    b._settings = _vc_settings()

    with pytest.raises(StorageError, match="connect()"):
        await b.initialize(COLL, DIM)

    with pytest.raises(StorageError, match="connect()"):
        await b.search(COLL, [0.1] * DIM)

    with pytest.raises(StorageError, match="connect()"):
        await b.upsert(COLL, [_make_entry()])

    with pytest.raises(StorageError, match="connect()"):
        await b.count(COLL)

    with pytest.raises(StorageError, match="connect()"):
        await b.delete(COLL, ["some-id"])

    with pytest.raises(StorageError, match="connect()"):
        await b.search_by_query_hash(COLL, "hash")

    with pytest.raises(StorageError, match="connect()"):
        await b.update_usage_count(COLL, str(uuid.uuid4()))


# ---------------------------------------------------------------------------
# missing deps
# ---------------------------------------------------------------------------


async def test_missing_deps_raises():
    with patch("medha.backends.vectorchord.HAS_VECTORCHORD", False):
        from medha.backends.vectorchord import VectorChordBackend

        with pytest.raises(ConfigurationError, match="pip install medha-archai"):
            VectorChordBackend()


# ---------------------------------------------------------------------------
# table name sanitization
# ---------------------------------------------------------------------------


def test_table_name_sanitization():
    from medha.backends.vectorchord import VectorChordBackend

    b = VectorChordBackend.__new__(VectorChordBackend)
    b._settings = _vc_settings(pg_table_prefix="medha")

    assert b._table_name("my_cache") == "medha_my_cache"
    assert b._table_name("my-cache.v2") == "medha_my_cache_v2"


# ---------------------------------------------------------------------------
# postgres error wrapping
# ---------------------------------------------------------------------------


async def test_postgres_error_wrapped(vc_backend):
    b, conn = vc_backend
    await b.initialize(COLL, DIM)

    conn.fetch.side_effect = _make_pg_error("relation does not exist")

    with pytest.raises(StorageError, match="VectorChord operation failed"):
        await b.search(COLL, [0.1] * DIM)


async def test_postgres_error_on_upsert_wrapped(vc_backend):
    b, conn = vc_backend
    await b.initialize(COLL, DIM)

    conn.executemany.side_effect = _make_pg_error("unique violation")

    with pytest.raises(StorageError, match="VectorChord operation failed"):
        await b.upsert(COLL, [_make_entry()])


# ---------------------------------------------------------------------------
# scroll (from _AsyncpgBackendMixin)
# ---------------------------------------------------------------------------


async def test_scroll_executes_sql(vc_backend):
    b, conn = vc_backend
    await b.initialize(COLL, DIM)
    conn.fetch.return_value = []

    results, next_offset = await b.scroll(COLL, limit=10)

    conn.fetch.assert_awaited()
    sql = conn.fetch.call_args.args[0]
    assert "ORDER BY" in sql
    assert "LIMIT" in sql
    assert results == []
    assert next_offset is None


# ---------------------------------------------------------------------------
# search_by_query_hash (from _AsyncpgBackendMixin)
# ---------------------------------------------------------------------------


async def test_search_by_query_hash_found(vc_backend):
    b, conn = vc_backend
    await b.initialize(COLL, DIM)

    row = {
        "id": str(uuid.uuid4()),
        "original_question": "test",
        "normalized_question": "test",
        "generated_query": "SELECT 42",
        "query_hash": "abc123",
        "response_summary": None,
        "template_id": None,
        "usage_count": 1,
        "created_at": None,
    }
    conn.fetchrow.return_value = row

    result = await b.search_by_query_hash(COLL, "abc123")

    conn.fetchrow.assert_awaited()
    assert result is not None
    assert result.generated_query == "SELECT 42"


async def test_search_by_query_hash_not_found(vc_backend):
    b, conn = vc_backend
    await b.initialize(COLL, DIM)
    conn.fetchrow.return_value = None

    result = await b.search_by_query_hash(COLL, "nonexistent")

    assert result is None


# ---------------------------------------------------------------------------
# update_usage_count (from _AsyncpgBackendMixin)
# ---------------------------------------------------------------------------


async def test_update_usage_count_executes_update(vc_backend):
    b, conn = vc_backend
    await b.initialize(COLL, DIM)
    conn.execute.return_value = "UPDATE 1"

    await b.update_usage_count(COLL, str(uuid.uuid4()))

    update_calls = [c for c in conn.execute.call_args_list if "UPDATE" in str(c.args[0])]
    assert any("usage_count" in str(c.args[0]) for c in update_calls)


# ---------------------------------------------------------------------------
# find_expired (VectorChord-specific)
# ---------------------------------------------------------------------------


async def test_find_expired_executes_sql(vc_backend):
    b, conn = vc_backend
    await b.initialize(COLL, DIM)
    conn.fetch.return_value = []

    expired_ids = await b.find_expired(COLL)

    conn.fetch.assert_awaited()
    sql = conn.fetch.call_args.args[0]
    assert "expires_at" in sql
    assert expired_ids == []


# ---------------------------------------------------------------------------
# drop_collection (from _AsyncpgBackendMixin)
# ---------------------------------------------------------------------------


async def test_drop_collection_executes_drop(vc_backend):
    b, conn = vc_backend
    await b.initialize(COLL, DIM)
    conn.execute.reset_mock()

    await b.drop_collection(COLL)

    drop_calls = [c for c in conn.execute.call_args_list if "DROP TABLE" in str(c.args[0])]
    assert len(drop_calls) == 1


# ---------------------------------------------------------------------------
# find_by_query_hash (from _AsyncpgBackendMixin)
# ---------------------------------------------------------------------------


async def test_find_by_query_hash(vc_backend):
    b, conn = vc_backend
    await b.initialize(COLL, DIM)
    eid = str(uuid.uuid4())
    conn.fetch.return_value = [{"id": eid}]

    ids = await b.find_by_query_hash(COLL, "abc123")

    assert ids == [eid]


# ---------------------------------------------------------------------------
# find_by_template_id (from _AsyncpgBackendMixin)
# ---------------------------------------------------------------------------


async def test_find_by_template_id(vc_backend):
    b, conn = vc_backend
    await b.initialize(COLL, DIM)
    eids = [str(uuid.uuid4()), str(uuid.uuid4())]
    conn.fetch.return_value = [{"id": e} for e in eids]

    ids = await b.find_by_template_id(COLL, "my_template")

    assert set(ids) == set(eids)


# ---------------------------------------------------------------------------
# search_by_normalized_question (from _AsyncpgBackendMixin)
# ---------------------------------------------------------------------------


async def test_search_by_normalized_question_found(vc_backend):
    b, conn = vc_backend
    await b.initialize(COLL, DIM)
    row = {
        "id": str(uuid.uuid4()),
        "original_question": "test",
        "normalized_question": "test",
        "generated_query": "SELECT 1",
        "query_hash": "abc",
        "response_summary": None,
        "template_id": None,
        "usage_count": 1,
        "created_at": None,
        "expires_at": None,
    }
    conn.fetchrow.return_value = row

    result = await b.search_by_normalized_question(COLL, "test")

    assert result is not None
    assert result.generated_query == "SELECT 1"


async def test_search_by_normalized_question_not_found(vc_backend):
    b, conn = vc_backend
    await b.initialize(COLL, DIM)
    conn.fetchrow.return_value = None

    result = await b.search_by_normalized_question(COLL, "nothing here")

    assert result is None
