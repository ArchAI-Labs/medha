"""Unit tests for WeaviateBackend (mocked weaviate async client — no real server required)."""

import hashlib
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

weaviate = pytest.importorskip("weaviate")

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


def _weaviate_settings(**overrides) -> Settings:
    return Settings(backend_type="weaviate", **overrides)


def _make_wv_obj(id_: str, **props) -> MagicMock:
    obj = MagicMock()
    obj.uuid = uuid.UUID(id_)
    default_props = {
        "original_question": "test",
        "normalized_question": "test",
        "generated_query": "SELECT 1",
        "query_hash": "abc",
        "usage_count": 1,
        "created_at": None,
        "expires_at": None,
        "response_summary": "",
        "template_id": "",
    }
    default_props.update(props)
    obj.properties = default_props
    obj.metadata = MagicMock()
    obj.metadata.distance = 0.05
    return obj


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_wv_collection():
    col = AsyncMock()

    near_vector_result = MagicMock()
    near_vector_result.objects = []
    col.query = AsyncMock()
    col.query.near_vector = AsyncMock(return_value=near_vector_result)
    col.query.fetch_objects = AsyncMock(return_value=MagicMock(objects=[]))
    col.query.fetch_object_by_id = AsyncMock(return_value=None)

    insert_result = MagicMock()
    insert_result.has_errors = False
    col.data = AsyncMock()
    col.data.insert_many = AsyncMock(return_value=insert_result)
    col.data.delete_by_id = AsyncMock()
    col.data.delete_many = AsyncMock()
    col.data.update = AsyncMock()

    agg_result = MagicMock()
    agg_result.total_count = 0
    col.aggregate = AsyncMock()
    col.aggregate.over_all = AsyncMock(return_value=agg_result)

    return col


@pytest.fixture
def mock_wv_client(mock_wv_collection):
    client = AsyncMock()
    client.connect = AsyncMock()
    client.close = AsyncMock()

    collections_ns = AsyncMock()
    collections_ns.exists = AsyncMock(return_value=False)
    collections_ns.create = AsyncMock()
    collections_ns.delete = AsyncMock()
    collections_ns.get = MagicMock(return_value=mock_wv_collection)  # sync call

    client.collections = collections_ns
    return client


@pytest.fixture
async def wv_backend(mock_wv_client, mock_wv_collection):
    from medha.backends.weaviate import WeaviateBackend

    with patch("medha.backends.weaviate.weaviate.use_async_with_local", return_value=mock_wv_client):
        b = WeaviateBackend(_weaviate_settings())
        await b.connect()
        yield b, mock_wv_collection, mock_wv_client


# ---------------------------------------------------------------------------
# connect
# ---------------------------------------------------------------------------


async def test_connect_calls_client_connect(mock_wv_client):
    from medha.backends.weaviate import WeaviateBackend

    with patch("medha.backends.weaviate.weaviate.use_async_with_local", return_value=mock_wv_client):
        b = WeaviateBackend(_weaviate_settings())
        await b.connect()

    mock_wv_client.connect.assert_awaited_once()


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


async def test_initialize_creates_collection(wv_backend):
    b, col, client = wv_backend
    await b.initialize(COLL, DIM)

    client.collections.exists.assert_awaited_once()
    client.collections.create.assert_awaited_once()
    assert COLL in b._collections


async def test_initialize_skips_when_exists(wv_backend):
    b, col, client = wv_backend
    client.collections.exists.return_value = True

    await b.initialize(COLL, DIM)

    client.collections.create.assert_not_awaited()


async def test_initialize_idempotent(wv_backend):
    b, col, client = wv_backend
    await b.initialize(COLL, DIM)
    client.collections.exists.reset_mock()
    client.collections.create.reset_mock()

    await b.initialize(COLL, DIM)

    client.collections.exists.assert_not_awaited()
    client.collections.create.assert_not_awaited()


# ---------------------------------------------------------------------------
# upsert
# ---------------------------------------------------------------------------


async def test_upsert_calls_insert_many(wv_backend):
    b, col, client = wv_backend
    await b.initialize(COLL, DIM)
    entries = [_make_entry() for _ in range(3)]

    await b.upsert(COLL, entries)

    col.data.insert_many.assert_awaited_once()
    call_args = col.data.insert_many.call_args.args[0]
    assert len(call_args) == 3


async def test_upsert_empty_list_no_call(wv_backend):
    b, col, client = wv_backend
    await b.initialize(COLL, DIM)

    await b.upsert(COLL, [])

    col.data.insert_many.assert_not_awaited()


async def test_upsert_uninitialized_raises(wv_backend):
    b, _, _ = wv_backend
    with pytest.raises(StorageError, match="not initialized"):
        await b.upsert("nonexistent", [_make_entry()])


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


async def test_search_calls_near_vector(wv_backend):
    b, col, _ = wv_backend
    await b.initialize(COLL, DIM)
    eid = str(uuid.uuid4())
    obj = _make_wv_obj(eid)
    col.query.near_vector.return_value = MagicMock(objects=[obj])

    results = await b.search(COLL, [0.1] * DIM, limit=5, score_threshold=0.0)

    col.query.near_vector.assert_awaited_once()
    assert len(results) == 1
    assert results[0].score == pytest.approx(0.95, abs=1e-5)  # 1.0 - 0.05


async def test_search_score_threshold(wv_backend):
    b, col, _ = wv_backend
    await b.initialize(COLL, DIM)
    eid = str(uuid.uuid4())
    obj = _make_wv_obj(eid)
    obj.metadata.distance = 0.9  # score = 1 - 0.9 = 0.1
    col.query.near_vector.return_value = MagicMock(objects=[obj])

    results = await b.search(COLL, [0.1] * DIM, score_threshold=0.5)

    assert results == []


async def test_search_uninitialized_raises(wv_backend):
    b, _, _ = wv_backend
    with pytest.raises(StorageError, match="not initialized"):
        await b.search("nonexistent", [0.1] * DIM)


# ---------------------------------------------------------------------------
# scroll
# ---------------------------------------------------------------------------


async def test_scroll_returns_entries(wv_backend):
    b, col, _ = wv_backend
    await b.initialize(COLL, DIM)
    objects = [_make_wv_obj(str(uuid.uuid4())) for _ in range(3)]
    col.query.fetch_objects.return_value = MagicMock(objects=objects)

    results, next_offset = await b.scroll(COLL, limit=10)

    assert len(results) == 3
    assert next_offset is None  # 3 < limit=10


async def test_scroll_pagination(wv_backend):
    b, col, _ = wv_backend
    await b.initialize(COLL, DIM)
    objects = [_make_wv_obj(str(uuid.uuid4())) for _ in range(2)]
    col.query.fetch_objects.return_value = MagicMock(objects=objects)

    results, next_offset = await b.scroll(COLL, limit=2)

    assert len(results) == 2
    assert next_offset == str(objects[-1].uuid)


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------


async def test_count_returns_value(wv_backend):
    b, col, _ = wv_backend
    await b.initialize(COLL, DIM)
    col.aggregate.over_all.return_value = MagicMock(total_count=5)

    result = await b.count(COLL)

    assert result == 5


async def test_count_uninitialized_raises(wv_backend):
    b, _, _ = wv_backend
    with pytest.raises(StorageError, match="not initialized"):
        await b.count("nonexistent")


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


async def test_delete_few_ids_uses_delete_by_id(wv_backend):
    b, col, _ = wv_backend
    await b.initialize(COLL, DIM)
    ids = [str(uuid.uuid4()) for _ in range(3)]

    await b.delete(COLL, ids)

    assert col.data.delete_by_id.await_count == 3
    col.data.delete_many.assert_not_awaited()


async def test_delete_many_ids_uses_delete_many(wv_backend):
    b, col, _ = wv_backend
    await b.initialize(COLL, DIM)
    ids = [str(uuid.uuid4()) for _ in range(15)]

    await b.delete(COLL, ids)

    col.data.delete_many.assert_awaited_once()
    col.data.delete_by_id.assert_not_awaited()


async def test_delete_empty_list_no_call(wv_backend):
    b, col, _ = wv_backend
    await b.initialize(COLL, DIM)

    await b.delete(COLL, [])

    col.data.delete_by_id.assert_not_awaited()
    col.data.delete_many.assert_not_awaited()


# ---------------------------------------------------------------------------
# search_by_query_hash
# ---------------------------------------------------------------------------


async def test_search_by_query_hash_found(wv_backend):
    b, col, _ = wv_backend
    await b.initialize(COLL, DIM)
    eid = str(uuid.uuid4())
    obj = _make_wv_obj(eid, generated_query="SELECT 42")
    col.query.fetch_objects.return_value = MagicMock(objects=[obj])

    result = await b.search_by_query_hash(COLL, "abc123")

    assert result is not None
    assert result.generated_query == "SELECT 42"


async def test_search_by_query_hash_not_found(wv_backend):
    b, col, _ = wv_backend
    await b.initialize(COLL, DIM)
    col.query.fetch_objects.return_value = MagicMock(objects=[])

    result = await b.search_by_query_hash(COLL, "nonexistent")

    assert result is None


# ---------------------------------------------------------------------------
# update_usage_count
# ---------------------------------------------------------------------------


async def test_update_usage_count_increments(wv_backend):
    b, col, _ = wv_backend
    await b.initialize(COLL, DIM)
    eid = str(uuid.uuid4())
    obj = _make_wv_obj(eid, usage_count=2)
    col.query.fetch_object_by_id.return_value = obj

    await b.update_usage_count(COLL, eid)

    col.data.update.assert_awaited_once()
    update_kwargs = col.data.update.call_args.kwargs
    assert update_kwargs["properties"]["usage_count"] == 3


async def test_update_usage_count_unknown_id(wv_backend):
    b, col, _ = wv_backend
    await b.initialize(COLL, DIM)
    col.query.fetch_object_by_id.return_value = None

    await b.update_usage_count(COLL, "nonexistent")  # must not raise

    col.data.update.assert_not_awaited()


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


async def test_close_calls_client_close(mock_wv_client, mock_wv_collection):
    from medha.backends.weaviate import WeaviateBackend

    with patch("medha.backends.weaviate.weaviate.use_async_with_local", return_value=mock_wv_client):
        b = WeaviateBackend(_weaviate_settings())
        await b.connect()
        await b.close()

    mock_wv_client.close.assert_awaited_once()
    assert b._client is None
    assert b._collections == {}


# ---------------------------------------------------------------------------
# missing deps
# ---------------------------------------------------------------------------


async def test_missing_deps_raises():
    with patch("medha.backends.weaviate.HAS_WEAVIATE", False):
        from medha.backends.weaviate import WeaviateBackend

        with pytest.raises(ConfigurationError, match="pip install medha-archai"):
            WeaviateBackend()


# ---------------------------------------------------------------------------
# find_expired
# ---------------------------------------------------------------------------


async def test_find_expired_returns_ids(wv_backend):
    b, col, _ = wv_backend
    await b.initialize(COLL, DIM)
    eid = str(uuid.uuid4())
    obj = _make_wv_obj(eid)
    col.query.fetch_objects.return_value = MagicMock(objects=[obj])

    expired_ids = await b.find_expired(COLL)

    col.query.fetch_objects.assert_awaited()
    assert expired_ids == [str(obj.uuid)]


async def test_find_expired_empty(wv_backend):
    b, col, _ = wv_backend
    await b.initialize(COLL, DIM)
    col.query.fetch_objects.return_value = MagicMock(objects=[])

    expired_ids = await b.find_expired(COLL)

    assert expired_ids == []


# ---------------------------------------------------------------------------
# drop_collection
# ---------------------------------------------------------------------------


async def test_drop_collection(wv_backend):
    b, col, client = wv_backend
    await b.initialize(COLL, DIM)

    await b.drop_collection(COLL)

    client.collections.delete.assert_awaited_once()
    assert COLL not in b._collections


async def test_drop_collection_unconnected_raises(wv_backend):
    b, col, _ = wv_backend
    b._client = None
    with pytest.raises(StorageError):
        await b.drop_collection(COLL)


# ---------------------------------------------------------------------------
# find_by_query_hash
# ---------------------------------------------------------------------------


async def test_find_by_query_hash(wv_backend):
    b, col, _ = wv_backend
    await b.initialize(COLL, DIM)
    eid = str(uuid.uuid4())
    obj = _make_wv_obj(eid)
    col.query.fetch_objects.return_value = MagicMock(objects=[obj])

    ids = await b.find_by_query_hash(COLL, "abc123")

    assert ids == [str(obj.uuid)]


async def test_find_by_query_hash_empty(wv_backend):
    b, col, _ = wv_backend
    await b.initialize(COLL, DIM)
    col.query.fetch_objects.return_value = MagicMock(objects=[])

    ids = await b.find_by_query_hash(COLL, "nonexistent")

    assert ids == []


# ---------------------------------------------------------------------------
# find_by_template_id
# ---------------------------------------------------------------------------


async def test_find_by_template_id(wv_backend):
    b, col, _ = wv_backend
    await b.initialize(COLL, DIM)
    eids = [str(uuid.uuid4()), str(uuid.uuid4())]
    objects = [_make_wv_obj(e) for e in eids]
    col.query.fetch_objects.return_value = MagicMock(objects=objects)

    ids = await b.find_by_template_id(COLL, "tmpl1")

    assert set(ids) == {str(obj.uuid) for obj in objects}


# ---------------------------------------------------------------------------
# search_by_normalized_question
# ---------------------------------------------------------------------------


async def test_search_by_normalized_question_found(wv_backend):
    b, col, _ = wv_backend
    await b.initialize(COLL, DIM)
    eid = str(uuid.uuid4())
    obj = _make_wv_obj(eid, generated_query="SELECT COUNT(*) FROM users")
    col.query.fetch_objects.return_value = MagicMock(objects=[obj])

    result = await b.search_by_normalized_question(COLL, "how many users")

    assert result is not None
    assert result.generated_query == "SELECT COUNT(*) FROM users"


async def test_search_by_normalized_question_not_found(wv_backend):
    b, col, _ = wv_backend
    await b.initialize(COLL, DIM)
    col.query.fetch_objects.return_value = MagicMock(objects=[])

    result = await b.search_by_normalized_question(COLL, "nothing")

    assert result is None


# ---------------------------------------------------------------------------
# load_stats / save_stats
# ---------------------------------------------------------------------------


def test_meta_collection_name_and_id_are_derived():
    """The stats class reuses the data class name; the id is a uuid5."""
    from medha.backends.weaviate import (
        _wv_collection_name,
        _wv_meta_collection_name,
        _wv_meta_id,
    )

    assert _wv_meta_collection_name("Medha", "my_cache") == "MedhaMyCacheMeta"
    assert _wv_meta_collection_name("Medha", "my_cache") == (
        _wv_collection_name("Medha", "my_cache") + "Meta"
    )

    # Deterministic, and a valid UUID whatever the collection is called
    weird = "'; DROP TABLE x; --"
    assert _wv_meta_id(weird) == _wv_meta_id(weird)
    assert uuid.UUID(_wv_meta_id(weird))
    assert _wv_meta_id("a") != _wv_meta_id("b")


async def test_save_stats_creates_meta_class_and_upserts(wv_backend):
    from medha.backends.weaviate import _wv_meta_id
    from medha.types import PersistedStats

    b, col, client = wv_backend
    await b.initialize(COLL, DIM)
    client.collections.create.reset_mock()

    await b.save_stats(COLL, PersistedStats(total_requests=6, total_hits=2))

    created_name = client.collections.create.call_args.kwargs["name"]
    assert created_name.endswith("Meta")

    col.data.insert_many.assert_awaited()
    objects = col.data.insert_many.call_args.args[0]
    assert len(objects) == 1
    assert str(objects[0].uuid) == _wv_meta_id(COLL)
    parsed = PersistedStats.model_validate_json(objects[0].properties["statsJson"])
    assert parsed.total_requests == 6


async def test_save_stats_skips_create_when_meta_class_exists(wv_backend):
    from medha.types import PersistedStats

    b, col, client = wv_backend
    await b.initialize(COLL, DIM)
    client.collections.create.reset_mock()
    client.collections.exists.return_value = True

    await b.save_stats(COLL, PersistedStats())

    client.collections.create.assert_not_awaited()
    col.data.insert_many.assert_awaited()


async def test_load_stats_returns_parsed_snapshot(wv_backend):
    from medha.types import PersistedStats

    b, col, client = wv_backend
    await b.initialize(COLL, DIM)
    client.collections.exists.return_value = True
    stats = PersistedStats(total_requests=2, hits_by_strategy={"semantic": 1})
    obj = MagicMock()
    obj.properties = {"statsJson": stats.model_dump_json()}
    col.query.fetch_object_by_id.return_value = obj

    loaded = await b.load_stats(COLL)

    assert loaded is not None
    assert loaded.total_requests == 2
    assert loaded.hits_by_strategy == {"semantic": 1}


async def test_load_stats_returns_none_when_meta_class_missing(wv_backend):
    b, col, client = wv_backend
    await b.initialize(COLL, DIM)
    client.collections.exists.return_value = False

    assert await b.load_stats(COLL) is None


async def test_load_stats_returns_none_when_object_missing(wv_backend):
    b, col, client = wv_backend
    await b.initialize(COLL, DIM)
    client.collections.exists.return_value = True
    col.query.fetch_object_by_id.return_value = None

    assert await b.load_stats(COLL) is None


async def test_load_stats_wraps_errors(wv_backend):
    b, col, client = wv_backend
    await b.initialize(COLL, DIM)
    client.collections.exists.return_value = True
    col.query.fetch_object_by_id.side_effect = RuntimeError("boom")

    with pytest.raises(StorageError, match="load_stats failed"):
        await b.load_stats(COLL)


async def test_save_stats_wraps_errors(wv_backend):
    from medha.types import PersistedStats

    b, col, client = wv_backend
    await b.initialize(COLL, DIM)
    col.data.insert_many.side_effect = RuntimeError("boom")

    with pytest.raises(StorageError, match="save_stats failed"):
        await b.save_stats(COLL, PersistedStats())


async def test_stats_methods_require_connection():
    from medha.backends.weaviate import WeaviateBackend
    from medha.types import PersistedStats

    b = WeaviateBackend.__new__(WeaviateBackend)
    b._client = None
    b._collections = {}
    b._settings = _weaviate_settings()

    with pytest.raises(StorageError, match="connect()"):
        await b.load_stats(COLL)

    with pytest.raises(StorageError, match="connect()"):
        await b.save_stats(COLL, PersistedStats())
