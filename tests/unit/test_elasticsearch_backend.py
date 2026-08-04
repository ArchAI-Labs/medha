"""Unit tests for ElasticsearchBackend (mocked AsyncElasticsearch — no real server required)."""

import hashlib
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

elasticsearch = pytest.importorskip("elasticsearch")

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


def _es_settings(**overrides) -> Settings:
    return Settings(backend_type="elasticsearch", **overrides)


def _empty_search_response() -> dict:
    return {"hits": {"hits": []}}


def _search_response_with_hit(doc_id: str, score: float = 1.0, **source_fields) -> dict:
    src = {
        "original_question": "test",
        "normalized_question": "test",
        "generated_query": "SELECT 1",
        "query_hash": "abc",
        "usage_count": 1,
        "created_at": None,
        **source_fields,
    }
    return {
        "hits": {
            "hits": [
                {
                    "_id": doc_id,
                    "_score": score,
                    "_source": src,
                    "sort": [0, doc_id],
                }
            ]
        }
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_es_client():
    client = AsyncMock()
    client.info = AsyncMock(return_value={"version": {"number": "8.0.0"}})
    client.indices = AsyncMock()
    client.indices.exists = AsyncMock(return_value=False)
    client.indices.create = AsyncMock()
    client.search = AsyncMock(return_value=_empty_search_response())
    client.count = AsyncMock(return_value={"count": 0})
    client.update = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
async def es_backend(mock_es_client):
    from medha.backends.elasticsearch import ElasticsearchBackend

    with patch(
        "medha.backends.elasticsearch.AsyncElasticsearch",
        return_value=mock_es_client,
    ):
        b = ElasticsearchBackend(_es_settings())
        await b.connect()
        yield b, mock_es_client
    await b.close()


# ---------------------------------------------------------------------------
# connect
# ---------------------------------------------------------------------------


async def test_connect_creates_client(mock_es_client):
    from medha.backends.elasticsearch import ElasticsearchBackend

    with patch(
        "medha.backends.elasticsearch.AsyncElasticsearch",
        return_value=mock_es_client,
    ) as mock_cls:
        b = ElasticsearchBackend(_es_settings())
        await b.connect()
        await b.close()

    mock_cls.assert_called_once()
    mock_es_client.info.assert_awaited_once()


async def test_connect_uses_api_key(mock_es_client):
    from medha.backends.elasticsearch import ElasticsearchBackend

    settings = _es_settings(es_api_key="mykey")
    with patch(
        "medha.backends.elasticsearch.AsyncElasticsearch",
        return_value=mock_es_client,
    ) as mock_cls:
        b = ElasticsearchBackend(settings)
        await b.connect()
        await b.close()

    kwargs = mock_cls.call_args.kwargs
    assert kwargs["api_key"] == "mykey"
    assert "basic_auth" not in kwargs


async def test_connect_uses_basic_auth(mock_es_client):
    from medha.backends.elasticsearch import ElasticsearchBackend

    settings = _es_settings(es_username="alice", es_password="secret")
    with patch(
        "medha.backends.elasticsearch.AsyncElasticsearch",
        return_value=mock_es_client,
    ) as mock_cls:
        b = ElasticsearchBackend(settings)
        await b.connect()
        await b.close()

    kwargs = mock_cls.call_args.kwargs
    assert kwargs["basic_auth"] == ("alice", "secret")


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


async def test_initialize_creates_index_when_not_exists(es_backend):
    b, client = es_backend
    client.indices.exists.return_value = False

    await b.initialize(COLL, DIM)

    client.indices.create.assert_awaited_once()
    call_kwargs = client.indices.create.call_args.kwargs
    body = call_kwargs["body"]
    assert "dense_vector" in str(body)
    assert "query_hash" in str(body)


async def test_initialize_skips_when_index_exists(es_backend):
    b, client = es_backend
    client.indices.exists.return_value = True

    await b.initialize(COLL, DIM)

    client.indices.create.assert_not_awaited()


async def test_initialize_idempotent(es_backend):
    """Second initialize on same collection does not recreate the index."""
    b, client = es_backend
    client.indices.exists.return_value = False

    await b.initialize(COLL, DIM)
    client.indices.create.reset_mock()

    # Index now exists → second call must skip creation
    client.indices.exists.return_value = True
    await b.initialize(COLL, DIM)

    client.indices.create.assert_not_awaited()


# ---------------------------------------------------------------------------
# upsert
# ---------------------------------------------------------------------------


async def test_upsert_calls_async_bulk(es_backend):
    b, client = es_backend
    await b.initialize(COLL, DIM)

    entries = [_make_entry() for _ in range(3)]
    with patch("medha.backends.elasticsearch.async_bulk", new=AsyncMock()) as mock_bulk:
        await b.upsert(COLL, entries)

    mock_bulk.assert_awaited_once()
    bulk_client, actions = mock_bulk.call_args.args
    actions_list = list(actions)
    assert len(actions_list) == 3
    assert all(a["_op_type"] == "index" for a in actions_list)


async def test_upsert_empty_list_no_bulk_call(es_backend):
    b, client = es_backend
    await b.initialize(COLL, DIM)

    with patch("medha.backends.elasticsearch.async_bulk", new=AsyncMock()) as mock_bulk:
        await b.upsert(COLL, [])

    mock_bulk.assert_not_awaited()


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


async def test_search_calls_knn_query(es_backend):
    b, client = es_backend
    await b.initialize(COLL, DIM)
    client.search.return_value = _empty_search_response()

    results = await b.search(COLL, [0.1] * DIM, limit=5, score_threshold=0.0)

    client.search.assert_awaited()
    call_body = client.search.call_args.kwargs["body"]
    assert "knn" in call_body
    assert results == []


async def test_search_converts_score(es_backend):
    b, client = es_backend
    await b.initialize(COLL, DIM)
    eid = str(uuid.uuid4())
    # ES cosine score is in [0, 1] → converted: (score * 2) - 1
    # ES score 1.0 → converted 1.0; ES score 0.5 → converted 0.0
    client.search.return_value = _search_response_with_hit(eid, score=1.0)

    results = await b.search(COLL, [0.1] * DIM, limit=5, score_threshold=0.0)

    assert len(results) == 1
    assert results[0].score == pytest.approx(1.0, abs=1e-5)


async def test_search_score_threshold_filters(es_backend):
    b, client = es_backend
    await b.initialize(COLL, DIM)
    eid = str(uuid.uuid4())
    # ES score 0.6 → converted (0.6*2)-1 = 0.2
    client.search.return_value = _search_response_with_hit(eid, score=0.6)

    results = await b.search(COLL, [0.1] * DIM, limit=5, score_threshold=0.5)

    assert results == []


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------


async def test_count_returns_value(es_backend):
    b, client = es_backend
    await b.initialize(COLL, DIM)
    client.count.return_value = {"count": 42}

    result = await b.count(COLL)

    assert result == 42
    client.count.assert_awaited_once()


async def test_count_returns_zero_on_not_found(es_backend):
    from elasticsearch import NotFoundError

    b, client = es_backend
    await b.initialize(COLL, DIM)
    # NotFoundError requires specific constructor args; patch it
    client.count.side_effect = NotFoundError(404, {}, {"error": "index_not_found_exception"})

    result = await b.count(COLL)

    assert result == 0


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


async def test_delete_calls_async_bulk(es_backend):
    b, client = es_backend
    await b.initialize(COLL, DIM)
    ids = [str(uuid.uuid4()), str(uuid.uuid4())]

    with patch("medha.backends.elasticsearch.async_bulk", new=AsyncMock()) as mock_bulk:
        await b.delete(COLL, ids)

    mock_bulk.assert_awaited_once()
    _, actions = mock_bulk.call_args.args
    actions_list = list(actions)
    assert len(actions_list) == 2
    assert all(a["_op_type"] == "delete" for a in actions_list)


async def test_delete_empty_list_no_bulk(es_backend):
    b, client = es_backend
    await b.initialize(COLL, DIM)

    with patch("medha.backends.elasticsearch.async_bulk", new=AsyncMock()) as mock_bulk:
        await b.delete(COLL, [])

    mock_bulk.assert_not_awaited()


# ---------------------------------------------------------------------------
# search_by_query_hash
# ---------------------------------------------------------------------------


async def test_search_by_query_hash_found(es_backend):
    b, client = es_backend
    await b.initialize(COLL, DIM)
    eid = str(uuid.uuid4())
    client.search.return_value = _search_response_with_hit(eid, score=1.0, query_hash="abc123")

    result = await b.search_by_query_hash(COLL, "abc123")

    assert result is not None
    assert result.id == eid


async def test_search_by_query_hash_not_found(es_backend):
    b, client = es_backend
    await b.initialize(COLL, DIM)
    client.search.return_value = _empty_search_response()

    result = await b.search_by_query_hash(COLL, "nonexistent")

    assert result is None


# ---------------------------------------------------------------------------
# update_usage_count
# ---------------------------------------------------------------------------


async def test_update_usage_count_calls_update(es_backend):
    b, client = es_backend
    await b.initialize(COLL, DIM)
    eid = str(uuid.uuid4())

    await b.update_usage_count(COLL, eid)

    client.update.assert_awaited_once()
    call_kwargs = client.update.call_args.kwargs
    assert call_kwargs["id"] == eid
    assert "painless" in str(call_kwargs["body"])


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


async def test_close_closes_client(mock_es_client):
    from medha.backends.elasticsearch import ElasticsearchBackend

    with patch(
        "medha.backends.elasticsearch.AsyncElasticsearch",
        return_value=mock_es_client,
    ):
        b = ElasticsearchBackend(_es_settings())
        await b.connect()
        assert b._client is not None
        await b.close()

    mock_es_client.close.assert_awaited_once()
    assert b._client is None


# ---------------------------------------------------------------------------
# not connected guard
# ---------------------------------------------------------------------------


async def test_not_connected_raises():
    from medha.backends.elasticsearch import ElasticsearchBackend

    b = ElasticsearchBackend.__new__(ElasticsearchBackend)
    b._client = None
    b._settings = _es_settings()

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
    with patch("medha.backends.elasticsearch.HAS_ELASTICSEARCH", False):
        from medha.backends.elasticsearch import ElasticsearchBackend

        with pytest.raises(ConfigurationError, match="pip install medha-archai"):
            ElasticsearchBackend()


# ---------------------------------------------------------------------------
# index name sanitization
# ---------------------------------------------------------------------------


def test_index_name_sanitization():
    from medha.backends.elasticsearch import ElasticsearchBackend

    b = ElasticsearchBackend.__new__(ElasticsearchBackend)
    b._settings = _es_settings(es_index_prefix="medha")

    assert b._index_name("my_cache") == "medha_my_cache"
    # Hyphens are valid ES index chars and are preserved; dots are replaced
    assert b._index_name("My-Cache.V2") == "medha_my-cache_v2"
    assert b._index_name("My Cache") == "medha_my_cache"


# ---------------------------------------------------------------------------
# transport error wrapping
# ---------------------------------------------------------------------------


async def test_transport_error_wrapped_in_search(es_backend):
    from elasticsearch import TransportError

    b, client = es_backend
    await b.initialize(COLL, DIM)

    client.search.side_effect = TransportError("network error")

    with pytest.raises(StorageError):
        await b.search(COLL, [0.1] * DIM)


# ---------------------------------------------------------------------------
# find_expired
# ---------------------------------------------------------------------------


async def test_find_expired_queries_range(es_backend):
    b, client = es_backend
    await b.initialize(COLL, DIM)
    client.search.return_value = _empty_search_response()

    expired_ids = await b.find_expired(COLL)

    client.search.assert_awaited()
    call_body = client.search.call_args.kwargs["body"]
    assert "expires_at" in str(call_body)
    assert expired_ids == []


async def test_find_expired_returns_ids(es_backend):
    b, client = es_backend
    await b.initialize(COLL, DIM)
    eid = str(uuid.uuid4())
    client.search.return_value = {"hits": {"hits": [{"_id": eid}]}}

    expired_ids = await b.find_expired(COLL)

    assert expired_ids == [eid]


# ---------------------------------------------------------------------------
# drop_collection
# ---------------------------------------------------------------------------


async def test_drop_collection_calls_delete(es_backend):
    b, client = es_backend
    await b.initialize(COLL, DIM)
    client.indices.delete = AsyncMock()

    await b.drop_collection(COLL)

    client.indices.delete.assert_awaited_once()


# ---------------------------------------------------------------------------
# find_by_query_hash
# ---------------------------------------------------------------------------


async def test_find_by_query_hash(es_backend):
    b, client = es_backend
    await b.initialize(COLL, DIM)
    eid = str(uuid.uuid4())
    client.search.return_value = {"hits": {"hits": [{"_id": eid}]}}

    ids = await b.find_by_query_hash(COLL, "abc123")

    assert ids == [eid]


async def test_find_by_query_hash_empty(es_backend):
    b, client = es_backend
    await b.initialize(COLL, DIM)
    client.search.return_value = _empty_search_response()

    ids = await b.find_by_query_hash(COLL, "nonexistent")

    assert ids == []


# ---------------------------------------------------------------------------
# find_by_template_id
# ---------------------------------------------------------------------------


async def test_find_by_template_id(es_backend):
    b, client = es_backend
    await b.initialize(COLL, DIM)
    eids = [str(uuid.uuid4()), str(uuid.uuid4())]
    client.search.return_value = {"hits": {"hits": [{"_id": e} for e in eids]}}

    ids = await b.find_by_template_id(COLL, "my_template")

    assert set(ids) == set(eids)


# ---------------------------------------------------------------------------
# search_by_normalized_question
# ---------------------------------------------------------------------------


async def test_search_by_normalized_question_found(es_backend):
    b, client = es_backend
    await b.initialize(COLL, DIM)
    eid = str(uuid.uuid4())
    client.search.return_value = _search_response_with_hit(eid, score=1.0)

    result = await b.search_by_normalized_question(COLL, "how many users")

    assert result is not None
    assert result.id == eid


async def test_search_by_normalized_question_not_found(es_backend):
    b, client = es_backend
    await b.initialize(COLL, DIM)
    client.search.return_value = _empty_search_response()

    result = await b.search_by_normalized_question(COLL, "nothing")

    assert result is None


# ---------------------------------------------------------------------------
# load_stats / save_stats
# ---------------------------------------------------------------------------


def test_meta_index_name_is_derived_from_sanitised_index_name():
    """The stats index reuses the data index sanitiser, plus a suffix."""
    from medha.backends.elasticsearch import ElasticsearchBackend

    b = ElasticsearchBackend.__new__(ElasticsearchBackend)
    b._settings = _es_settings(es_index_prefix="medha")

    assert b._meta_index_name("my_cache") == "medha_my_cache__meta"
    # Uppercase and unsafe characters are folded exactly as for the data index
    assert b._meta_index_name("My-Cache.V2") == "medha_my-cache_v2__meta"
    # Long names stay inside the 255-byte limit without losing the suffix
    long_name = b._meta_index_name("x" * 400)
    assert len(long_name) <= 255
    assert long_name.endswith("__meta")


async def test_save_stats_indexes_stats_document(es_backend):
    from medha.types import PersistedStats

    b, client = es_backend
    await b.initialize(COLL, DIM)

    await b.save_stats(COLL, PersistedStats(total_requests=11, total_hits=6))

    client.index.assert_awaited_once()
    kwargs = client.index.call_args.kwargs
    assert kwargs["index"] == b._meta_index_name(COLL)
    assert kwargs["index"] != b._index_name(COLL)
    assert kwargs["id"] == "_stats"
    parsed = PersistedStats.model_validate_json(kwargs["document"]["stats_json"])
    assert parsed.total_requests == 11


async def test_load_stats_returns_parsed_snapshot(es_backend):
    from medha.types import PersistedStats

    b, client = es_backend
    await b.initialize(COLL, DIM)
    stats = PersistedStats(total_requests=4, hits_by_strategy={"fuzzy": 1})
    client.get = AsyncMock(return_value={"_source": {"stats_json": stats.model_dump_json()}})

    loaded = await b.load_stats(COLL)

    assert loaded is not None
    assert loaded.total_requests == 4
    assert loaded.hits_by_strategy == {"fuzzy": 1}
    assert client.get.call_args.kwargs["id"] == "_stats"


async def test_load_stats_returns_none_when_absent(es_backend):
    from elasticsearch import NotFoundError

    b, client = es_backend
    await b.initialize(COLL, DIM)
    client.get = AsyncMock(
        side_effect=NotFoundError(404, {}, {"error": "index_not_found_exception"})
    )

    assert await b.load_stats(COLL) is None


async def test_load_stats_wraps_transport_errors(es_backend):
    from elasticsearch import TransportError

    b, client = es_backend
    await b.initialize(COLL, DIM)
    client.get = AsyncMock(side_effect=TransportError("boom"))

    with pytest.raises(StorageError, match="load_stats failed"):
        await b.load_stats(COLL)


async def test_save_stats_wraps_transport_errors(es_backend):
    from elasticsearch import TransportError

    from medha.types import PersistedStats

    b, client = es_backend
    await b.initialize(COLL, DIM)
    client.index = AsyncMock(side_effect=TransportError("boom"))

    with pytest.raises(StorageError, match="save_stats failed"):
        await b.save_stats(COLL, PersistedStats())


async def test_stats_methods_require_connection():
    from medha.backends.elasticsearch import ElasticsearchBackend
    from medha.types import PersistedStats

    b = ElasticsearchBackend.__new__(ElasticsearchBackend)
    b._client = None
    b._settings = _es_settings()

    with pytest.raises(StorageError, match="connect()"):
        await b.load_stats(COLL)

    with pytest.raises(StorageError, match="connect()"):
        await b.save_stats(COLL, PersistedStats())
