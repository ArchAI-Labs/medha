"""Unit tests for LanceDBBackend (mocked lancedb — no real storage required)."""

import hashlib
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

lancedb = pytest.importorskip("lancedb")

from medha.config import Settings
from medha.exceptions import ConfigurationError, StorageError

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
):
    from medha.types import CacheEntry
    vec = vector or [0.1] * dim
    return CacheEntry(
        id=id or str(uuid.uuid4()),
        vector=vec,
        original_question=question,
        normalized_question=question.lower(),
        generated_query=query,
        query_hash=hashlib.md5(query.encode()).hexdigest(),
    )


def _make_row(
    id: str | None = None,
    question: str = "test question",
    query: str = "SELECT 1",
    distance: float = 0.1,
) -> dict[str, Any]:
    return {
        "id": id or str(uuid.uuid4()),
        "_distance": distance,
        "original_question": question,
        "normalized_question": question.lower(),
        "generated_query": query,
        "query_hash": hashlib.md5(query.encode()).hexdigest(),
        "response_summary": "",
        "template_id": "",
        "usage_count": 1,
        "created_at": "",
        "expires_at": "",
    }


def _lancedb_settings(**overrides) -> Settings:
    return Settings(backend_type="lancedb", lancedb_uri="/tmp/test_lancedb", **overrides)


def _make_search_chain(rows: list[dict[str, Any]] | None = None) -> MagicMock:
    # Represents VectorQuery returned by table.vector_search(vector) (sync builder, async to_list)
    chain = MagicMock()
    chain.distance_type.return_value = chain
    chain.where.return_value = chain
    chain.limit.return_value = chain
    chain.to_list = AsyncMock(return_value=rows or [])
    return chain


def _make_query_chain(rows: list[dict[str, Any]] | None = None) -> MagicMock:
    chain = MagicMock()
    chain.where.return_value = chain
    chain.limit.return_value = chain
    chain.offset.return_value = chain
    chain.select.return_value = chain
    chain.to_list = AsyncMock(return_value=rows or [])
    return chain


def _make_merge_chain() -> MagicMock:
    chain = MagicMock()
    chain.when_matched_update_all.return_value = chain
    chain.when_not_matched_insert_all.return_value = chain
    chain.execute = AsyncMock(return_value=None)
    return chain


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_lancedb_table():
    table = MagicMock()
    table.vector_search = MagicMock(return_value=_make_search_chain())
    table.query.return_value = _make_query_chain()
    table.merge_insert.return_value = _make_merge_chain()
    table.count_rows = AsyncMock(return_value=0)
    table.delete = AsyncMock(return_value=None)
    return table


@pytest.fixture
def mock_lancedb_db(mock_lancedb_table):
    db = AsyncMock()
    db.list_tables.return_value = []
    db.create_table.return_value = mock_lancedb_table
    db.open_table.return_value = mock_lancedb_table
    db.drop_table = AsyncMock(return_value=None)
    return db


@pytest.fixture
async def lancedb_backend(mock_lancedb_db, mock_lancedb_table):
    from medha.backends.lancedb import LanceDBBackend

    with patch("lancedb.connect_async", new=AsyncMock(return_value=mock_lancedb_db)):
        b = LanceDBBackend(_lancedb_settings())
        await b.connect()
        yield b, mock_lancedb_table


# ---------------------------------------------------------------------------
# connect
# ---------------------------------------------------------------------------


async def test_connect_stores_db(mock_lancedb_db):
    from medha.backends.lancedb import LanceDBBackend

    with patch("lancedb.connect_async", new=AsyncMock(return_value=mock_lancedb_db)):
        b = LanceDBBackend(_lancedb_settings())
        await b.connect()

    assert b._db is mock_lancedb_db

    await b.close()
    assert b._db is None


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


async def test_initialize_creates_table_when_not_exists(lancedb_backend, mock_lancedb_db):
    b, table = lancedb_backend
    mock_lancedb_db.list_tables.return_value = []

    await b.initialize(COLL, DIM)

    mock_lancedb_db.create_table.assert_called_once()
    assert COLL in b._tables


async def test_initialize_opens_existing_table(lancedb_backend, mock_lancedb_db):
    b, table = lancedb_backend
    table_name = b._table_name(COLL)
    mock_lancedb_db.list_tables.return_value = [table_name]

    await b.initialize(COLL, DIM)

    # Backend uses create_table(exist_ok=True) for both new and existing tables
    mock_lancedb_db.create_table.assert_called_once()
    call_kwargs = mock_lancedb_db.create_table.call_args.kwargs
    assert call_kwargs.get("exist_ok") is True
    mock_lancedb_db.open_table.assert_not_called()


async def test_initialize_idempotent(lancedb_backend, mock_lancedb_db):
    b, _ = lancedb_backend
    mock_lancedb_db.list_tables.return_value = []

    await b.initialize(COLL, DIM)
    calls_after_first = mock_lancedb_db.create_table.call_count
    await b.initialize(COLL, DIM)

    assert mock_lancedb_db.create_table.call_count == calls_after_first


async def test_initialize_without_connect_raises(mock_lancedb_db):
    from medha.backends.lancedb import LanceDBBackend

    b = LanceDBBackend(_lancedb_settings())
    with pytest.raises(StorageError, match="connect"):
        await b.initialize(COLL, DIM)


# ---------------------------------------------------------------------------
# upsert
# ---------------------------------------------------------------------------


async def test_upsert_calls_merge_insert(lancedb_backend, mock_lancedb_db):
    b, table = lancedb_backend
    mock_lancedb_db.list_tables.return_value = []
    await b.initialize(COLL, DIM)

    merge = _make_merge_chain()
    table.merge_insert.return_value = merge
    entries = [_make_entry() for _ in range(3)]

    await b.upsert(COLL, entries)

    table.merge_insert.assert_called_once_with("id")
    merge.when_matched_update_all.assert_called_once()
    merge.when_not_matched_insert_all.assert_called_once()
    merge.execute.assert_called_once()
    rows_arg = merge.execute.call_args.args[0]
    assert len(rows_arg) == 3


async def test_upsert_empty_list_no_call(lancedb_backend, mock_lancedb_db):
    b, table = lancedb_backend
    mock_lancedb_db.list_tables.return_value = []
    await b.initialize(COLL, DIM)

    await b.upsert(COLL, [])

    table.merge_insert.assert_not_called()


async def test_upsert_uninitialized_raises(lancedb_backend):
    b, _ = lancedb_backend
    with pytest.raises(StorageError, match="not initialized"):
        await b.upsert("nonexistent", [_make_entry()])


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


async def test_search_returns_results(lancedb_backend, mock_lancedb_db):
    b, table = lancedb_backend
    mock_lancedb_db.list_tables.return_value = []
    await b.initialize(COLL, DIM)

    row = _make_row(distance=0.1)
    table.vector_search = MagicMock(return_value=_make_search_chain([row]))

    results = await b.search(COLL, [0.1] * DIM, limit=5, score_threshold=0.0)

    assert len(results) == 1
    assert results[0].score == pytest.approx(0.9, abs=1e-5)  # 1.0 - 0.1
    assert results[0].generated_query == "SELECT 1"


async def test_search_applies_score_threshold(lancedb_backend, mock_lancedb_db):
    b, table = lancedb_backend
    mock_lancedb_db.list_tables.return_value = []
    await b.initialize(COLL, DIM)

    row = _make_row(distance=0.9)  # score = 0.1
    table.vector_search = MagicMock(return_value=_make_search_chain([row]))

    results = await b.search(COLL, [0.1] * DIM, score_threshold=0.5)

    assert results == []


async def test_search_returns_empty_when_no_rows(lancedb_backend, mock_lancedb_db):
    b, table = lancedb_backend
    mock_lancedb_db.list_tables.return_value = []
    await b.initialize(COLL, DIM)

    table.vector_search.return_value = _make_search_chain([])

    results = await b.search(COLL, [0.1] * DIM)

    assert results == []


async def test_search_uses_configured_metric(lancedb_backend, mock_lancedb_db):
    b, table = lancedb_backend
    b._settings = _lancedb_settings(lancedb_metric="l2")
    mock_lancedb_db.list_tables.return_value = []
    await b.initialize(COLL, DIM)

    chain = _make_search_chain([])
    table.vector_search = MagicMock(return_value=chain)

    await b.search(COLL, [0.1] * DIM)

    chain.distance_type.assert_called_once_with("l2")


async def test_search_uninitialized_raises(lancedb_backend):
    b, _ = lancedb_backend
    with pytest.raises(StorageError, match="not initialized"):
        await b.search("nonexistent", [0.1] * DIM)


# ---------------------------------------------------------------------------
# scroll
# ---------------------------------------------------------------------------


async def test_scroll_returns_entries(lancedb_backend, mock_lancedb_db):
    b, table = lancedb_backend
    mock_lancedb_db.list_tables.return_value = []
    await b.initialize(COLL, DIM)

    rows = [_make_row() for _ in range(3)]
    table.query.return_value = _make_query_chain(rows)

    results, next_offset = await b.scroll(COLL, limit=10)

    assert len(results) == 3
    assert next_offset is None  # 3 < limit=10


async def test_scroll_pagination_next_offset(lancedb_backend, mock_lancedb_db):
    b, table = lancedb_backend
    mock_lancedb_db.list_tables.return_value = []
    await b.initialize(COLL, DIM)

    rows = [_make_row() for _ in range(2)]
    table.query.return_value = _make_query_chain(rows)

    results, next_offset = await b.scroll(COLL, limit=2)

    assert len(results) == 2
    assert next_offset == "2"


async def test_scroll_with_offset_uses_native_offset(lancedb_backend, mock_lancedb_db):
    b, table = lancedb_backend
    mock_lancedb_db.list_tables.return_value = []
    await b.initialize(COLL, DIM)

    # With native offset, lancedb returns only the page rows (no client-side slicing needed)
    rows = [_make_row() for _ in range(2)]
    query_chain = _make_query_chain(rows)
    table.query.return_value = query_chain

    results, next_offset = await b.scroll(COLL, limit=10, offset="3")

    query_chain.offset.assert_called_once_with(3)
    assert len(results) == 2
    assert next_offset is None


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------


async def test_count_returns_value(lancedb_backend, mock_lancedb_db):
    b, table = lancedb_backend
    mock_lancedb_db.list_tables.return_value = []
    await b.initialize(COLL, DIM)
    table.count_rows.return_value = 42

    result = await b.count(COLL)

    assert result == 42


async def test_count_uninitialized_raises(lancedb_backend):
    b, _ = lancedb_backend
    with pytest.raises(StorageError, match="not initialized"):
        await b.count("nonexistent")


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


async def test_delete_calls_table_delete(lancedb_backend, mock_lancedb_db):
    b, table = lancedb_backend
    mock_lancedb_db.list_tables.return_value = []
    await b.initialize(COLL, DIM)
    ids = [str(uuid.uuid4()), str(uuid.uuid4())]

    await b.delete(COLL, ids)

    table.delete.assert_called_once()
    predicate = table.delete.call_args.args[0]
    assert ids[0] in predicate
    assert ids[1] in predicate


async def test_delete_empty_list_no_call(lancedb_backend, mock_lancedb_db):
    b, table = lancedb_backend
    mock_lancedb_db.list_tables.return_value = []
    await b.initialize(COLL, DIM)

    await b.delete(COLL, [])

    table.delete.assert_not_called()


# ---------------------------------------------------------------------------
# find_expired
# ---------------------------------------------------------------------------


async def test_find_expired_returns_ids(lancedb_backend, mock_lancedb_db):
    b, table = lancedb_backend
    mock_lancedb_db.list_tables.return_value = []
    await b.initialize(COLL, DIM)

    eid = str(uuid.uuid4())
    table.query.return_value = _make_query_chain([{"id": eid}])

    expired_ids = await b.find_expired(COLL)

    assert expired_ids == [eid]


async def test_find_expired_empty(lancedb_backend, mock_lancedb_db):
    b, table = lancedb_backend
    mock_lancedb_db.list_tables.return_value = []
    await b.initialize(COLL, DIM)

    table.query.return_value = _make_query_chain([])

    expired_ids = await b.find_expired(COLL)

    assert expired_ids == []


# ---------------------------------------------------------------------------
# search_by_normalized_question
# ---------------------------------------------------------------------------


async def test_search_by_normalized_question_found(lancedb_backend, mock_lancedb_db):
    b, table = lancedb_backend
    mock_lancedb_db.list_tables.return_value = []
    await b.initialize(COLL, DIM)

    eid = str(uuid.uuid4())
    row = _make_row(id=eid, question="how many users", query="SELECT COUNT(*) FROM users")
    table.query.return_value = _make_query_chain([row])

    result = await b.search_by_normalized_question(COLL, "how many users")

    assert result is not None
    assert result.id == eid
    assert result.generated_query == "SELECT COUNT(*) FROM users"


async def test_search_by_normalized_question_not_found(lancedb_backend, mock_lancedb_db):
    b, table = lancedb_backend
    mock_lancedb_db.list_tables.return_value = []
    await b.initialize(COLL, DIM)

    table.query.return_value = _make_query_chain([])

    result = await b.search_by_normalized_question(COLL, "nothing")

    assert result is None


# ---------------------------------------------------------------------------
# find_by_query_hash
# ---------------------------------------------------------------------------


async def test_find_by_query_hash(lancedb_backend, mock_lancedb_db):
    b, table = lancedb_backend
    mock_lancedb_db.list_tables.return_value = []
    await b.initialize(COLL, DIM)

    eid = str(uuid.uuid4())
    table.query.return_value = _make_query_chain([{"id": eid}])

    ids = await b.find_by_query_hash(COLL, "abc123")

    assert ids == [eid]


async def test_find_by_query_hash_empty(lancedb_backend, mock_lancedb_db):
    b, table = lancedb_backend
    mock_lancedb_db.list_tables.return_value = []
    await b.initialize(COLL, DIM)

    table.query.return_value = _make_query_chain([])

    ids = await b.find_by_query_hash(COLL, "nonexistent")

    assert ids == []


# ---------------------------------------------------------------------------
# find_by_template_id
# ---------------------------------------------------------------------------


async def test_find_by_template_id(lancedb_backend, mock_lancedb_db):
    b, table = lancedb_backend
    mock_lancedb_db.list_tables.return_value = []
    await b.initialize(COLL, DIM)

    eids = [str(uuid.uuid4()), str(uuid.uuid4())]
    table.query.return_value = _make_query_chain([{"id": e} for e in eids])

    ids = await b.find_by_template_id(COLL, "tmpl1")

    assert set(ids) == set(eids)


# ---------------------------------------------------------------------------
# drop_collection
# ---------------------------------------------------------------------------


async def test_drop_collection_calls_drop_table(lancedb_backend, mock_lancedb_db):
    b, _ = lancedb_backend
    mock_lancedb_db.list_tables.return_value = []
    await b.initialize(COLL, DIM)

    await b.drop_collection(COLL)

    mock_lancedb_db.drop_table.assert_called_once_with(b._table_name(COLL))
    assert COLL not in b._tables


async def test_drop_collection_unconnected_raises(lancedb_backend):
    b, _ = lancedb_backend
    b._db = None
    with pytest.raises(StorageError, match="connect"):
        await b.drop_collection(COLL)


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


async def test_close_clears_state(mock_lancedb_db, mock_lancedb_table):
    from medha.backends.lancedb import LanceDBBackend

    with patch("lancedb.connect_async", new=AsyncMock(return_value=mock_lancedb_db)):
        mock_lancedb_db.list_tables.return_value = []
        b = LanceDBBackend(_lancedb_settings())
        await b.connect()
        await b.initialize(COLL, DIM)

    assert b._db is not None
    assert COLL in b._tables

    await b.close()

    assert b._db is None
    assert b._tables == {}
    assert b._dimensions == {}


# ---------------------------------------------------------------------------
# missing deps
# ---------------------------------------------------------------------------


async def test_missing_deps_raises():
    with patch("medha.backends.lancedb.HAS_LANCEDB", False):
        from medha.backends.lancedb import LanceDBBackend

        with pytest.raises(ConfigurationError, match="pip install medha-archai"):
            LanceDBBackend()


# ---------------------------------------------------------------------------
# table_name helper
# ---------------------------------------------------------------------------


def test_table_name_with_prefix():
    from medha.backends.lancedb import LanceDBBackend

    b = LanceDBBackend.__new__(LanceDBBackend)
    b._settings = _lancedb_settings(lancedb_table_prefix="myapp")
    assert b._table_name("cache") == "myapp_cache"


def test_table_name_without_prefix():
    from medha.backends.lancedb import LanceDBBackend

    b = LanceDBBackend.__new__(LanceDBBackend)
    b._settings = _lancedb_settings(lancedb_table_prefix="")
    assert b._table_name("cache") == "cache"
