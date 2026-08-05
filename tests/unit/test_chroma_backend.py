"""Unit tests for ChromaBackend (mocked chromadb sync client — no real Chroma server required)."""

import hashlib
import uuid
from unittest.mock import MagicMock, patch

import pytest

chromadb = pytest.importorskip("chromadb")

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


def _chroma_settings(**overrides) -> Settings:
    return Settings(backend_type="chroma", chroma_mode="ephemeral", **overrides)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_chroma_collection():
    col = MagicMock()
    col.count.return_value = 0
    col.query.return_value = {"ids": [[]], "distances": [[]], "metadatas": [[]]}
    col.get.return_value = {"ids": [], "metadatas": []}
    col.upsert.return_value = None
    col.delete.return_value = None
    return col


@pytest.fixture
def mock_chroma_client(mock_chroma_collection):
    client = MagicMock()
    client.get_or_create_collection.return_value = mock_chroma_collection
    client.delete_collection.return_value = None
    return client


@pytest.fixture
async def chroma_backend(mock_chroma_client, mock_chroma_collection):
    from medha.backends.chroma import ChromaBackend

    with patch("chromadb.EphemeralClient", return_value=mock_chroma_client):
        b = ChromaBackend(_chroma_settings())
        await b.connect()
        yield b, mock_chroma_collection


# ---------------------------------------------------------------------------
# connect
# ---------------------------------------------------------------------------


async def test_connect_ephemeral(mock_chroma_client):
    from medha.backends.chroma import ChromaBackend

    with patch("chromadb.EphemeralClient", return_value=mock_chroma_client) as mock_cls:
        b = ChromaBackend(_chroma_settings())
        await b.connect()
        await b.close()

    mock_cls.assert_called_once()
    assert b._client is None  # cleared by close()


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


async def test_initialize_calls_get_or_create(chroma_backend):
    b, col = chroma_backend
    await b.initialize(COLL, DIM)

    client = b._client
    client.get_or_create_collection.assert_called_once()
    call_kwargs = client.get_or_create_collection.call_args.kwargs
    assert call_kwargs["metadata"] == {"hnsw:space": "cosine"}


async def test_initialize_idempotent(chroma_backend):
    b, col = chroma_backend
    await b.initialize(COLL, DIM)
    call_count_after_first = b._client.get_or_create_collection.call_count
    await b.initialize(COLL, DIM)
    # No new call made
    assert b._client.get_or_create_collection.call_count == call_count_after_first


async def test_initialize_stores_collection(chroma_backend):
    b, col = chroma_backend
    await b.initialize(COLL, DIM)
    assert COLL in b._collections


# ---------------------------------------------------------------------------
# upsert
# ---------------------------------------------------------------------------


async def test_upsert_calls_collection_upsert(chroma_backend):
    b, col = chroma_backend
    await b.initialize(COLL, DIM)
    entries = [_make_entry() for _ in range(3)]

    await b.upsert(COLL, entries)

    col.upsert.assert_called_once()
    call_kwargs = col.upsert.call_args.kwargs
    assert len(call_kwargs["ids"]) == 3
    assert len(call_kwargs["embeddings"]) == 3
    assert len(call_kwargs["metadatas"]) == 3


async def test_upsert_empty_list_no_call(chroma_backend):
    b, col = chroma_backend
    await b.initialize(COLL, DIM)

    await b.upsert(COLL, [])

    col.upsert.assert_not_called()


async def test_upsert_uninitialized_raises(chroma_backend):
    b, _ = chroma_backend
    with pytest.raises(StorageError, match="not initialized"):
        await b.upsert("nonexistent", [_make_entry()])


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


async def test_search_calls_query(chroma_backend):
    b, col = chroma_backend
    await b.initialize(COLL, DIM)
    col.count.return_value = 1
    col.query.return_value = {
        "ids": [["abc"]],
        "distances": [[0.1]],
        "metadatas": [[{
            "original_question": "test",
            "normalized_question": "test",
            "generated_query": "SELECT 1",
            "query_hash": "abc",
            "usage_count": 1,
            "created_at": "",
            "expires_at": "",
        }]],
    }

    results = await b.search(COLL, [0.1] * DIM, limit=5, score_threshold=0.0)

    col.query.assert_called_once()
    assert len(results) == 1
    assert results[0].score == pytest.approx(0.9, abs=1e-5)  # 1.0 - 0.1


async def test_search_empty_collection_returns_empty(chroma_backend):
    b, col = chroma_backend
    await b.initialize(COLL, DIM)
    col.count.return_value = 0

    results = await b.search(COLL, [0.1] * DIM)

    assert results == []
    col.query.assert_not_called()


async def test_search_score_threshold(chroma_backend):
    b, col = chroma_backend
    await b.initialize(COLL, DIM)
    col.count.return_value = 1
    col.query.return_value = {
        "ids": [["abc"]],
        "distances": [[0.8]],  # score = 1 - 0.8 = 0.2
        "metadatas": [[{
            "original_question": "test",
            "normalized_question": "test",
            "generated_query": "SELECT 1",
            "query_hash": "abc",
            "usage_count": 1,
            "created_at": "",
            "expires_at": "",
        }]],
    }

    results = await b.search(COLL, [0.1] * DIM, score_threshold=0.5)

    assert results == []


async def test_search_uninitialized_raises(chroma_backend):
    b, _ = chroma_backend
    with pytest.raises(StorageError, match="not initialized"):
        await b.search("nonexistent", [0.1] * DIM)


# ---------------------------------------------------------------------------
# scroll
# ---------------------------------------------------------------------------


async def test_scroll_returns_entries(chroma_backend):
    b, col = chroma_backend
    await b.initialize(COLL, DIM)
    meta = {
        "original_question": "test",
        "normalized_question": "test",
        "generated_query": "SELECT 1",
        "query_hash": "abc",
        "usage_count": 1,
        "created_at": "",
        "expires_at": "",
    }
    col.get.return_value = {"ids": ["a", "b", "c"], "metadatas": [meta, meta, meta]}

    results, next_offset = await b.scroll(COLL, limit=10)

    assert len(results) == 3
    assert next_offset is None  # 3 < limit=10


async def test_scroll_pagination(chroma_backend):
    b, col = chroma_backend
    await b.initialize(COLL, DIM)
    meta = {
        "original_question": "test",
        "normalized_question": "test",
        "generated_query": "SELECT 1",
        "query_hash": "abc",
        "usage_count": 1,
        "created_at": "",
        "expires_at": "",
    }
    col.get.return_value = {"ids": ["a", "b"], "metadatas": [meta, meta]}

    results, next_offset = await b.scroll(COLL, limit=2)

    assert len(results) == 2
    assert next_offset == "2"


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------


async def test_count_returns_value(chroma_backend):
    b, col = chroma_backend
    await b.initialize(COLL, DIM)
    col.count.return_value = 7

    result = await b.count(COLL)

    assert result == 7


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


async def test_delete_calls_collection_delete(chroma_backend):
    b, col = chroma_backend
    await b.initialize(COLL, DIM)
    ids = [str(uuid.uuid4()), str(uuid.uuid4())]

    await b.delete(COLL, ids)

    col.delete.assert_called_once_with(ids=ids)


async def test_delete_empty_list_no_call(chroma_backend):
    b, col = chroma_backend
    await b.initialize(COLL, DIM)

    await b.delete(COLL, [])

    col.delete.assert_not_called()


# ---------------------------------------------------------------------------
# search_by_query_hash
# ---------------------------------------------------------------------------


async def test_search_by_query_hash_found(chroma_backend):
    b, col = chroma_backend
    await b.initialize(COLL, DIM)
    eid = str(uuid.uuid4())
    meta = {
        "original_question": "test",
        "normalized_question": "test",
        "generated_query": "SELECT 42",
        "query_hash": "abc123",
        "usage_count": 1,
        "created_at": "",
        "expires_at": "",
    }
    col.get.return_value = {"ids": [eid], "metadatas": [meta]}

    result = await b.search_by_query_hash(COLL, "abc123")

    assert result is not None
    assert result.id == eid
    assert result.generated_query == "SELECT 42"


async def test_search_by_query_hash_not_found(chroma_backend):
    b, col = chroma_backend
    await b.initialize(COLL, DIM)
    col.get.return_value = {"ids": [], "metadatas": []}

    result = await b.search_by_query_hash(COLL, "nonexistent")

    assert result is None


# ---------------------------------------------------------------------------
# update_usage_count
# ---------------------------------------------------------------------------


async def test_update_usage_count_increments(chroma_backend):
    b, col = chroma_backend
    await b.initialize(COLL, DIM)
    eid = str(uuid.uuid4())
    meta = {
        "original_question": "test",
        "normalized_question": "test",
        "generated_query": "SELECT 1",
        "query_hash": "abc",
        "usage_count": 3,
        "created_at": "",
        "expires_at": "",
    }
    col.get.return_value = {"ids": [eid], "metadatas": [meta]}

    await b.update_usage_count(COLL, eid)

    col.upsert.assert_called_once()
    upserted_meta = col.upsert.call_args.kwargs["metadatas"][0]
    assert upserted_meta["usage_count"] == 4


async def test_update_usage_count_unknown_id(chroma_backend):
    b, col = chroma_backend
    await b.initialize(COLL, DIM)
    col.get.return_value = {"ids": [], "metadatas": []}

    await b.update_usage_count(COLL, "nonexistent")  # must not raise


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


async def test_close_clears_client_and_collections(mock_chroma_client, mock_chroma_collection):
    from medha.backends.chroma import ChromaBackend

    with patch("chromadb.EphemeralClient", return_value=mock_chroma_client):
        b = ChromaBackend(_chroma_settings())
        await b.connect()
        await b.initialize(COLL, DIM)

    assert b._client is not None
    assert COLL in b._collections

    await b.close()

    assert b._client is None
    assert b._collections == {}


# ---------------------------------------------------------------------------
# missing deps
# ---------------------------------------------------------------------------


async def test_missing_deps_raises():
    with patch("medha.backends.chroma.HAS_CHROMA", False):
        from medha.backends.chroma import ChromaBackend

        with pytest.raises(ConfigurationError, match="pip install medha-archai"):
            ChromaBackend()


# ---------------------------------------------------------------------------
# find_expired
# ---------------------------------------------------------------------------


async def test_find_expired_returns_ids(chroma_backend):
    b, col = chroma_backend
    await b.initialize(COLL, DIM)
    eid = str(uuid.uuid4())
    col.get.return_value = {"ids": [eid]}

    expired_ids = await b.find_expired(COLL)

    col.get.assert_called()
    assert expired_ids == [eid]


async def test_find_expired_empty(chroma_backend):
    b, col = chroma_backend
    await b.initialize(COLL, DIM)
    col.get.return_value = {"ids": []}

    expired_ids = await b.find_expired(COLL)

    assert expired_ids == []


# ---------------------------------------------------------------------------
# drop_collection
# ---------------------------------------------------------------------------


async def test_drop_collection_calls_delete_collection(chroma_backend):
    b, col = chroma_backend
    await b.initialize(COLL, DIM)

    await b.drop_collection(COLL)

    b._client.delete_collection.assert_called_once()
    assert COLL not in b._collections


async def test_drop_collection_unconnected_raises(chroma_backend):
    b, _ = chroma_backend
    b._client = None  # simulate not connected
    with pytest.raises(StorageError):
        await b.drop_collection(COLL)


# ---------------------------------------------------------------------------
# find_by_query_hash
# ---------------------------------------------------------------------------


async def test_find_by_query_hash(chroma_backend):
    b, col = chroma_backend
    await b.initialize(COLL, DIM)
    eid = str(uuid.uuid4())
    col.get.return_value = {"ids": [eid], "metadatas": [{"query_hash": "abc123"}]}

    ids = await b.find_by_query_hash(COLL, "abc123")

    assert ids == [eid]


async def test_find_by_query_hash_empty(chroma_backend):
    b, col = chroma_backend
    await b.initialize(COLL, DIM)
    col.get.return_value = {"ids": [], "metadatas": []}

    ids = await b.find_by_query_hash(COLL, "nonexistent")

    assert ids == []


# ---------------------------------------------------------------------------
# find_by_template_id
# ---------------------------------------------------------------------------


async def test_find_by_template_id(chroma_backend):
    b, col = chroma_backend
    await b.initialize(COLL, DIM)
    eids = [str(uuid.uuid4()), str(uuid.uuid4())]
    col.get.return_value = {"ids": eids, "metadatas": [{}, {}]}

    ids = await b.find_by_template_id(COLL, "tmpl1")

    assert set(ids) == set(eids)


# ---------------------------------------------------------------------------
# search_by_normalized_question
# ---------------------------------------------------------------------------


async def test_search_by_normalized_question_found(chroma_backend):
    b, col = chroma_backend
    await b.initialize(COLL, DIM)
    eid = str(uuid.uuid4())
    meta = {
        "original_question": "how many users",
        "normalized_question": "how many users",
        "generated_query": "SELECT COUNT(*) FROM users",
        "query_hash": "abc",
        "usage_count": 1,
        "created_at": "",
        "expires_at": "",
    }
    col.get.return_value = {"ids": [eid], "metadatas": [meta]}

    result = await b.search_by_normalized_question(COLL, "how many users")

    assert result is not None
    assert result.id == eid
    assert result.generated_query == "SELECT COUNT(*) FROM users"


async def test_search_by_normalized_question_not_found(chroma_backend):
    b, col = chroma_backend
    await b.initialize(COLL, DIM)
    col.get.return_value = {"ids": [], "metadatas": []}

    result = await b.search_by_normalized_question(COLL, "nothing")

    assert result is None


# ---------------------------------------------------------------------------
# load_stats / save_stats
# ---------------------------------------------------------------------------


def test_meta_collection_name_is_sanitised():
    """The stats collection reuses the data sanitiser, plus a suffix."""
    from medha.backends.chroma import _chroma_meta_collection_name

    assert _chroma_meta_collection_name("my_cache") == "my_cache__meta"
    assert _chroma_meta_collection_name("My-Cache.V2") == "my-cache_v2__meta"
    # Long names stay inside Chroma's 63-character limit, suffix intact
    long_name = _chroma_meta_collection_name("x" * 200)
    assert len(long_name) <= 63
    assert long_name.endswith("__meta")


async def test_save_stats_upserts_into_meta_collection(chroma_backend, mock_chroma_client):
    from medha.backends.chroma import _chroma_meta_collection_name
    from medha.types import PersistedStats

    b, col = chroma_backend
    await b.initialize(COLL, DIM)
    col.upsert.reset_mock()

    await b.save_stats(COLL, PersistedStats(total_requests=8, total_hits=5))

    meta_names = [
        c.kwargs.get("name")
        for c in mock_chroma_client.get_or_create_collection.call_args_list
    ]
    assert _chroma_meta_collection_name(COLL) in meta_names

    col.upsert.assert_called_once()
    kwargs = col.upsert.call_args.kwargs
    assert kwargs["ids"] == ["_stats"]
    parsed = PersistedStats.model_validate_json(kwargs["metadatas"][0]["stats_json"])
    assert parsed.total_requests == 8


async def test_load_stats_returns_parsed_snapshot(chroma_backend):
    from medha.types import PersistedStats

    b, col = chroma_backend
    await b.initialize(COLL, DIM)
    stats = PersistedStats(total_requests=3, hits_by_strategy={"template": 2})
    col.get.return_value = {
        "ids": ["_stats"],
        "metadatas": [{"stats_json": stats.model_dump_json()}],
    }

    loaded = await b.load_stats(COLL)

    assert loaded is not None
    assert loaded.total_requests == 3
    assert loaded.hits_by_strategy == {"template": 2}


async def test_load_stats_returns_none_when_absent(chroma_backend):
    b, col = chroma_backend
    await b.initialize(COLL, DIM)
    col.get.return_value = {"ids": [], "metadatas": []}

    assert await b.load_stats(COLL) is None


async def test_load_stats_wraps_errors(chroma_backend):
    b, col = chroma_backend
    await b.initialize(COLL, DIM)
    col.get.side_effect = RuntimeError("boom")

    with pytest.raises(StorageError, match="load_stats failed"):
        await b.load_stats(COLL)


async def test_save_stats_wraps_errors(chroma_backend):
    from medha.types import PersistedStats

    b, col = chroma_backend
    await b.initialize(COLL, DIM)
    col.upsert.side_effect = RuntimeError("boom")

    with pytest.raises(StorageError, match="save_stats failed"):
        await b.save_stats(COLL, PersistedStats())


async def test_stats_methods_require_connection():
    from medha.backends.chroma import ChromaBackend
    from medha.types import PersistedStats

    b = ChromaBackend.__new__(ChromaBackend)
    b._client = None
    b._is_async = False
    b._collections = {}
    b._meta_collections = {}
    b._settings = _chroma_settings()

    with pytest.raises(StorageError, match="connect()"):
        await b.load_stats(COLL)

    with pytest.raises(StorageError, match="connect()"):
        await b.save_stats(COLL, PersistedStats())
