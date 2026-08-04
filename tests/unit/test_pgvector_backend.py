"""Unit tests for PgVectorBackend (mocked asyncpg — no real PostgreSQL required)."""

import hashlib
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

asyncpg = pytest.importorskip("asyncpg")
pytest.importorskip("pgvector")

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
    """Create an asyncpg.PostgresError without triggering its custom __init__."""
    err = asyncpg.PostgresError.__new__(asyncpg.PostgresError)
    Exception.__init__(err, msg)
    return err


def _make_undefined_table_error() -> asyncpg.PostgresError:
    """An error carrying SQLSTATE 42P01 (undefined_table)."""
    cls = asyncpg.exceptions.UndefinedTableError
    err = cls.__new__(cls)
    Exception.__init__(err, "relation does not exist")
    return err


def _pg_settings(**overrides) -> Settings:
    return Settings(backend_type="pgvector", **overrides)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pool():
    """Mocked asyncpg pool + connection."""
    pool = MagicMock()
    conn = AsyncMock()

    # asyncpg uses `async with pool.acquire() as conn` — acquire() is NOT awaited,
    # it returns a context manager whose __aenter__ / __aexit__ are awaited.
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire = MagicMock(return_value=ctx)
    pool.close = AsyncMock()

    # Default return values that won't crash callers
    conn.fetch.return_value = []
    conn.fetchrow.return_value = [0]
    conn.execute.return_value = "UPDATE 0"
    conn.executemany.return_value = None
    return pool, conn


@pytest.fixture
async def pg_backend(mock_pool):
    """PgVectorBackend with fully mocked asyncpg pool."""
    from medha.backends.pgvector import PgVectorBackend

    pool, conn = mock_pool
    with patch(
        "medha.backends.pgvector.asyncpg.create_pool",
        new=AsyncMock(return_value=pool),
    ):
        b = PgVectorBackend(_pg_settings())
        await b.connect()
        yield b, conn
    await b.close()


# ---------------------------------------------------------------------------
# connect
# ---------------------------------------------------------------------------


async def test_connect_creates_pool(mock_pool):
    """asyncpg.create_pool is called with individual connection fields."""
    from medha.backends.pgvector import PgVectorBackend

    pool, _ = mock_pool
    settings = _pg_settings(
        pg_host="db-host",
        pg_port=5432,
        pg_database="mydb",
        pg_user="alice",
        pg_password="secret",
    )
    with patch(
        "medha.backends.pgvector.asyncpg.create_pool",
        new=AsyncMock(return_value=pool),
    ) as mock_cp:
        b = PgVectorBackend(settings)
        await b.connect()
        await b.close()

    mock_cp.assert_awaited_once()
    kwargs = mock_cp.call_args.kwargs
    assert kwargs["host"] == "db-host"
    assert kwargs["database"] == "mydb"
    assert kwargs["user"] == "alice"
    assert "dsn" not in kwargs


async def test_connect_uses_dsn_when_set(mock_pool):
    """When pg_dsn is set, passes dsn= instead of individual fields."""
    from medha.backends.pgvector import PgVectorBackend

    pool, _ = mock_pool
    dsn = "postgresql://user:pw@host:5432/db"
    settings = _pg_settings(pg_dsn=dsn)
    with patch(
        "medha.backends.pgvector.asyncpg.create_pool",
        new=AsyncMock(return_value=pool),
    ) as mock_cp:
        b = PgVectorBackend(settings)
        await b.connect()
        await b.close()

    kwargs = mock_cp.call_args.kwargs
    assert kwargs["dsn"] == dsn
    assert "host" not in kwargs
    assert "database" not in kwargs


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


async def test_initialize_executes_ddl(pg_backend):
    """initialize() calls CREATE EXTENSION, CREATE TABLE, and CREATE INDEX."""
    b, conn = pg_backend
    await b.initialize(COLL, DIM)

    all_sql = " ".join(str(c.args[0]) for c in conn.execute.call_args_list)
    assert "CREATE EXTENSION" in all_sql
    assert "CREATE TABLE" in all_sql
    assert "CREATE INDEX" in all_sql
    assert "hnsw" in all_sql
    assert "query_hash" in all_sql
    assert "template_id" in all_sql


async def test_initialize_idempotent(pg_backend):
    """Second initialize() call on same collection skips DDL."""
    b, conn = pg_backend
    await b.initialize(COLL, DIM)
    conn.execute.reset_mock()
    await b.initialize(COLL, DIM)  # should be a no-op

    assert conn.execute.call_count == 0


# ---------------------------------------------------------------------------
# upsert
# ---------------------------------------------------------------------------


async def test_upsert_calls_executemany(pg_backend):
    """upsert() calls conn.executemany with the correct number of tuples."""
    b, conn = pg_backend
    await b.initialize(COLL, DIM)

    entries = [_make_entry() for _ in range(3)]
    await b.upsert(COLL, entries)

    conn.executemany.assert_awaited_once()
    _, rows = conn.executemany.call_args.args
    assert len(rows) == 3


async def test_upsert_empty_list_no_call(pg_backend):
    """upsert() with empty list never calls executemany."""
    b, conn = pg_backend
    await b.initialize(COLL, DIM)
    await b.upsert(COLL, [])
    conn.executemany.assert_not_awaited()


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


async def test_search_executes_correct_sql(pg_backend):
    """search() passes SQL with <=> operator, WHERE, ORDER BY, LIMIT."""
    b, conn = pg_backend
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


async def test_count_executes_count_sql(pg_backend):
    """count() executes SELECT COUNT(*) and returns the integer value."""
    b, conn = pg_backend
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


async def test_delete_executes_delete_sql(pg_backend):
    """delete() executes DELETE ... WHERE id = ANY(...)."""
    b, conn = pg_backend
    await b.initialize(COLL, DIM)
    ids = [str(uuid.uuid4()), str(uuid.uuid4())]

    await b.delete(COLL, ids)

    conn.execute.assert_awaited()
    # Find the DELETE call among all execute calls
    delete_calls = [
        c for c in conn.execute.call_args_list if "DELETE" in str(c.args[0])
    ]
    assert len(delete_calls) == 1
    assert "ANY" in delete_calls[0].args[0]


async def test_delete_empty_list_no_call(pg_backend):
    """delete() with empty list is a no-op."""
    b, conn = pg_backend
    await b.initialize(COLL, DIM)
    conn.execute.reset_mock()

    await b.delete(COLL, [])
    conn.execute.assert_not_awaited()


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


async def test_close_closes_pool(mock_pool):
    """close() calls pool.close() and sets pool to None."""
    from medha.backends.pgvector import PgVectorBackend

    pool, _ = mock_pool
    with patch(
        "medha.backends.pgvector.asyncpg.create_pool",
        new=AsyncMock(return_value=pool),
    ):
        b = PgVectorBackend(_pg_settings())
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
    """All data methods raise StorageError when pool is None."""
    from medha.backends.pgvector import PgVectorBackend

    b = PgVectorBackend.__new__(PgVectorBackend)
    b._pool = None
    b._initialized_tables = set()
    b._settings = _pg_settings()

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
    """ConfigurationError is raised when HAS_PGVECTOR is False."""
    with patch("medha.backends.pgvector.HAS_PGVECTOR", False):
        from medha.backends.pgvector import PgVectorBackend

        with pytest.raises(ConfigurationError, match="pip install medha-archai"):
            PgVectorBackend()


# ---------------------------------------------------------------------------
# table name sanitization
# ---------------------------------------------------------------------------


def test_table_name_sanitization():
    """Special chars in collection name are replaced with underscores."""
    from medha.backends.pgvector import PgVectorBackend

    # Create instance bypassing __init__ checks
    b = PgVectorBackend.__new__(PgVectorBackend)
    b._settings = _pg_settings(pg_table_prefix="medha")

    assert b._table_name("my_cache") == "medha_my_cache"
    # __medha_templates_x: leading __ stays as __ (underscores are allowed)
    result = b._table_name("__medha_templates_my_cache")
    assert result == "medha___medha_templates_my_cache"
    # Hyphens and dots replaced
    assert b._table_name("my-cache.v2") == "medha_my_cache_v2"


# ---------------------------------------------------------------------------
# postgres error wrapping
# ---------------------------------------------------------------------------


async def test_postgres_error_wrapped(pg_backend):
    """asyncpg.PostgresError is wrapped in StorageError."""
    b, conn = pg_backend
    await b.initialize(COLL, DIM)

    conn.fetch.side_effect = _make_pg_error("relation does not exist")

    with pytest.raises(StorageError, match="PgVector operation failed"):
        await b.search(COLL, [0.1] * DIM)


async def test_postgres_error_on_upsert_wrapped(pg_backend):
    """asyncpg.PostgresError during upsert is wrapped in StorageError."""
    b, conn = pg_backend
    await b.initialize(COLL, DIM)

    conn.executemany.side_effect = _make_pg_error("unique violation")

    with pytest.raises(StorageError, match="PgVector operation failed"):
        await b.upsert(COLL, [_make_entry()])


# ---------------------------------------------------------------------------
# scroll (from _AsyncpgBackendMixin)
# ---------------------------------------------------------------------------


async def test_scroll_executes_sql(pg_backend):
    b, conn = pg_backend
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


async def test_search_by_query_hash_found(pg_backend):
    b, conn = pg_backend
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


async def test_search_by_query_hash_not_found(pg_backend):
    b, conn = pg_backend
    await b.initialize(COLL, DIM)
    conn.fetchrow.return_value = None

    result = await b.search_by_query_hash(COLL, "nonexistent")

    assert result is None


# ---------------------------------------------------------------------------
# update_usage_count (from _AsyncpgBackendMixin)
# ---------------------------------------------------------------------------


async def test_update_usage_count_executes_update(pg_backend):
    b, conn = pg_backend
    await b.initialize(COLL, DIM)
    conn.execute.return_value = "UPDATE 1"

    await b.update_usage_count(COLL, str(uuid.uuid4()))

    update_calls = [c for c in conn.execute.call_args_list if "UPDATE" in str(c.args[0])]
    assert any("usage_count" in str(c.args[0]) for c in update_calls)


# ---------------------------------------------------------------------------
# find_expired (PgVector-specific)
# ---------------------------------------------------------------------------


async def test_find_expired_executes_sql(pg_backend):
    b, conn = pg_backend
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


async def test_drop_collection_executes_drop(pg_backend):
    b, conn = pg_backend
    await b.initialize(COLL, DIM)
    conn.execute.reset_mock()

    await b.drop_collection(COLL)

    drop_calls = [c for c in conn.execute.call_args_list if "DROP TABLE" in str(c.args[0])]
    assert len(drop_calls) == 1


# ---------------------------------------------------------------------------
# find_by_query_hash (from _AsyncpgBackendMixin)
# ---------------------------------------------------------------------------


async def test_find_by_query_hash(pg_backend):
    b, conn = pg_backend
    await b.initialize(COLL, DIM)
    eid = str(uuid.uuid4())
    conn.fetch.return_value = [{"id": eid}]

    ids = await b.find_by_query_hash(COLL, "abc123")

    assert ids == [eid]


# ---------------------------------------------------------------------------
# find_by_template_id (from _AsyncpgBackendMixin)
# ---------------------------------------------------------------------------


async def test_find_by_template_id(pg_backend):
    b, conn = pg_backend
    await b.initialize(COLL, DIM)
    eids = [str(uuid.uuid4()), str(uuid.uuid4())]
    conn.fetch.return_value = [{"id": e} for e in eids]

    ids = await b.find_by_template_id(COLL, "my_template")

    assert set(ids) == set(eids)


# ---------------------------------------------------------------------------
# search_by_normalized_question (from _AsyncpgBackendMixin)
# ---------------------------------------------------------------------------


async def test_search_by_normalized_question_found(pg_backend):
    b, conn = pg_backend
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


async def test_search_by_normalized_question_not_found(pg_backend):
    b, conn = pg_backend
    await b.initialize(COLL, DIM)
    conn.fetchrow.return_value = None

    result = await b.search_by_normalized_question(COLL, "nothing here")

    assert result is None


# ---------------------------------------------------------------------------
# load_stats / save_stats (from _AsyncpgBackendMixin)
# ---------------------------------------------------------------------------


async def test_initialize_creates_stats_table(pg_backend):
    """initialize() provisions the shared _medha_stats table."""
    b, conn = pg_backend
    await b.initialize(COLL, DIM)

    stats_ddl = [
        c for c in conn.execute.call_args_list
        if "_medha_stats" in str(c.args[0])
    ]
    assert len(stats_ddl) == 1
    sql = stats_ddl[0].args[0]
    assert "CREATE TABLE IF NOT EXISTS" in sql
    assert "public._medha_stats" in sql
    assert "collection_name TEXT PRIMARY KEY" in sql


async def test_save_stats_upserts_with_bind_params(pg_backend):
    """save_stats() upserts; collection_name and payload travel as bind params."""
    from medha.types import PersistedStats

    b, conn = pg_backend
    await b.initialize(COLL, DIM)
    conn.execute.reset_mock()

    await b.save_stats(COLL, PersistedStats(total_requests=7, total_hits=3))

    conn.execute.assert_awaited_once()
    sql, collection, payload = conn.execute.call_args.args
    assert "INSERT INTO public._medha_stats" in sql
    assert "ON CONFLICT (collection_name) DO UPDATE" in sql
    assert collection == COLL
    assert PersistedStats.model_validate_json(payload).total_requests == 7


async def test_load_stats_returns_parsed_snapshot(pg_backend):
    from medha.types import PersistedStats

    b, conn = pg_backend
    await b.initialize(COLL, DIM)
    stats = PersistedStats(total_requests=5, total_hits=2, hits_by_strategy={"exact": 2})
    conn.fetchrow.return_value = {"stats_json": stats.model_dump_json()}

    loaded = await b.load_stats(COLL)

    assert loaded is not None
    assert loaded.total_requests == 5
    assert loaded.hits_by_strategy == {"exact": 2}
    # collection_name is bound, never interpolated into the SQL text
    sql, bound = conn.fetchrow.call_args.args
    assert bound == COLL
    assert COLL not in sql


async def test_load_stats_returns_none_when_absent(pg_backend):
    b, conn = pg_backend
    await b.initialize(COLL, DIM)
    conn.fetchrow.return_value = None

    assert await b.load_stats(COLL) is None


async def test_load_stats_returns_none_when_table_missing(pg_backend):
    """A never-provisioned stats table means 'never saved', not an error."""
    b, conn = pg_backend
    await b.initialize(COLL, DIM)
    conn.fetchrow.side_effect = _make_undefined_table_error()

    assert await b.load_stats(COLL) is None


async def test_load_stats_wraps_other_errors(pg_backend):
    b, conn = pg_backend
    await b.initialize(COLL, DIM)
    conn.fetchrow.side_effect = _make_pg_error("connection reset")

    with pytest.raises(StorageError, match="load_stats failed"):
        await b.load_stats(COLL)


async def test_save_stats_wraps_errors(pg_backend):
    from medha.types import PersistedStats

    b, conn = pg_backend
    await b.initialize(COLL, DIM)
    conn.execute.side_effect = _make_pg_error("disk full")

    with pytest.raises(StorageError, match="save_stats failed"):
        await b.save_stats(COLL, PersistedStats())


async def test_stats_methods_require_connection():
    from medha.backends.pgvector import PgVectorBackend
    from medha.types import PersistedStats

    b = PgVectorBackend.__new__(PgVectorBackend)
    b._pool = None
    b._initialized_tables = set()
    b._settings = _pg_settings()

    with pytest.raises(StorageError, match="connect()"):
        await b.load_stats(COLL)

    with pytest.raises(StorageError, match="connect()"):
        await b.save_stats(COLL, PersistedStats())
