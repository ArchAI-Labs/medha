"""Unit tests for AzureSearchBackend (mocked azure-search-documents — no real service required)."""

import hashlib
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

azure_search = pytest.importorskip("azure.search.documents")

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


def _az_settings(**overrides) -> Settings:
    return Settings(
        backend_type="azure-search",
        azure_search_endpoint="https://my-service.search.windows.net",
        azure_search_api_key="test-api-key",
        **overrides,
    )


class _AsyncIter:
    """Simple async iterable over a fixed list of items."""

    def __init__(self, items: list):
        self._items = items

    def __aiter__(self):
        return self._aiter()

    async def _aiter(self):
        for item in self._items:
            yield item


def _make_search_result(**fields) -> dict:
    defaults = {
        "id": str(uuid.uuid4()),
        "original_question": "test",
        "normalized_question": "test",
        "generated_query": "SELECT 1",
        "query_hash": "abc",
        "usage_count": 1,
        "created_at": None,
        "expires_at": None,
        "@search.score": 0.9,
    }
    defaults.update(fields)
    return defaults


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_search_client():
    client = AsyncMock()
    client.search = AsyncMock(return_value=_AsyncIter([]))
    client.merge_or_upload_documents = AsyncMock()
    client.delete_documents = AsyncMock()
    client.get_document_count = AsyncMock(return_value=0)
    client.get_document = AsyncMock()
    client.merge_documents = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_index_client():
    client = AsyncMock()
    client.list_index_names = MagicMock(return_value=_AsyncIter([]))
    client.get_index = AsyncMock()
    client.create_index = AsyncMock()
    client.delete_index = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
async def az_backend(mock_search_client, mock_index_client):
    from azure.core.exceptions import ResourceNotFoundError
    from medha.backends.azure_search import AzureSearchBackend

    # Simulate index not found → triggers create
    mock_index_client.get_index.side_effect = ResourceNotFoundError()

    with (
        patch(
            "medha.backends.azure_search.AsyncSearchIndexClient",
            return_value=mock_index_client,
        ),
        patch(
            "medha.backends.azure_search.AsyncSearchClient",
            return_value=mock_search_client,
        ),
    ):
        b = AzureSearchBackend(_az_settings())
        await b.connect()
        await b.initialize(COLL, DIM)
        yield b, mock_search_client, mock_index_client


# ---------------------------------------------------------------------------
# connect
# ---------------------------------------------------------------------------


async def test_connect_creates_index_client(mock_index_client):
    from medha.backends.azure_search import AzureSearchBackend

    with patch(
        "medha.backends.azure_search.AsyncSearchIndexClient",
        return_value=mock_index_client,
    ) as mock_cls:
        b = AzureSearchBackend(_az_settings())
        await b.connect()
        await b.close()

    mock_cls.assert_called_once()
    mock_index_client.list_index_names.assert_called_once()


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


async def test_initialize_creates_index_when_not_found(mock_index_client, mock_search_client):
    from azure.core.exceptions import ResourceNotFoundError
    from medha.backends.azure_search import AzureSearchBackend

    mock_index_client.get_index.side_effect = ResourceNotFoundError()

    with (
        patch("medha.backends.azure_search.AsyncSearchIndexClient", return_value=mock_index_client),
        patch("medha.backends.azure_search.AsyncSearchClient", return_value=mock_search_client),
    ):
        b = AzureSearchBackend(_az_settings())
        await b.connect()
        await b.initialize(COLL, DIM)

    mock_index_client.create_index.assert_awaited_once()
    assert COLL in b._search_clients


async def test_initialize_skips_when_index_exists(mock_index_client, mock_search_client):
    from medha.backends.azure_search import AzureSearchBackend

    mock_index_client.get_index.side_effect = None
    mock_index_client.get_index.return_value = MagicMock(name="existing-index")

    with (
        patch("medha.backends.azure_search.AsyncSearchIndexClient", return_value=mock_index_client),
        patch("medha.backends.azure_search.AsyncSearchClient", return_value=mock_search_client),
    ):
        b = AzureSearchBackend(_az_settings())
        await b.connect()
        await b.initialize(COLL, DIM)

    mock_index_client.create_index.assert_not_awaited()


async def test_initialize_idempotent(az_backend):
    """Second initialize call on an existing index skips creation."""
    b, search_client, index_client = az_backend
    # Simulate index now exists (backend already added COLL to _search_clients)
    index_client.get_index.side_effect = None
    index_client.get_index.return_value = MagicMock(name="existing-index")
    index_client.create_index.reset_mock()

    await b.initialize(COLL, DIM)

    index_client.create_index.assert_not_awaited()


# ---------------------------------------------------------------------------
# upsert
# ---------------------------------------------------------------------------


async def test_upsert_calls_merge_or_upload(az_backend):
    b, search_client, _ = az_backend
    entries = [_make_entry() for _ in range(3)]

    await b.upsert(COLL, entries)

    search_client.merge_or_upload_documents.assert_awaited_once()
    docs = search_client.merge_or_upload_documents.call_args.args[0]
    assert len(docs) == 3
    assert all("id" in d for d in docs)
    assert all("vector" in d for d in docs)


async def test_upsert_empty_list_no_call(az_backend):
    b, search_client, _ = az_backend

    await b.upsert(COLL, [])

    search_client.merge_or_upload_documents.assert_not_awaited()


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


async def test_search_calls_search_with_vector_query(az_backend):
    b, search_client, _ = az_backend
    doc = _make_search_result(**{"@search.score": 0.85})
    search_client.search = AsyncMock(return_value=_AsyncIter([doc]))

    results = await b.search(COLL, [0.1] * DIM, limit=5, score_threshold=0.0)

    search_client.search.assert_awaited_once()
    assert len(results) == 1
    assert results[0].score == pytest.approx(0.85, abs=1e-5)


async def test_search_score_threshold(az_backend):
    b, search_client, _ = az_backend
    doc = _make_search_result(**{"@search.score": 0.3})
    search_client.search = AsyncMock(return_value=_AsyncIter([doc]))

    results = await b.search(COLL, [0.1] * DIM, limit=5, score_threshold=0.5)

    assert results == []


async def test_search_empty_result(az_backend):
    b, search_client, _ = az_backend
    search_client.search = AsyncMock(return_value=_AsyncIter([]))

    results = await b.search(COLL, [0.1] * DIM)

    assert results == []


# ---------------------------------------------------------------------------
# scroll
# ---------------------------------------------------------------------------


async def test_scroll_returns_entries(az_backend):
    b, search_client, _ = az_backend
    docs = [_make_search_result() for _ in range(3)]
    search_client.search = AsyncMock(return_value=_AsyncIter(docs))

    results, next_offset = await b.scroll(COLL, limit=10)

    assert len(results) == 3
    assert next_offset is None


async def test_scroll_pagination(az_backend):
    b, search_client, _ = az_backend
    docs = [_make_search_result() for _ in range(2)]
    search_client.search = AsyncMock(return_value=_AsyncIter(docs))

    results, next_offset = await b.scroll(COLL, limit=2)

    assert len(results) == 2
    assert next_offset == "2"


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------


async def test_count_returns_value(az_backend):
    b, search_client, _ = az_backend
    search_client.get_document_count.return_value = 42

    result = await b.count(COLL)

    assert result == 42


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


async def test_delete_calls_delete_documents(az_backend):
    b, search_client, _ = az_backend
    ids = [str(uuid.uuid4()), str(uuid.uuid4())]

    await b.delete(COLL, ids)

    search_client.delete_documents.assert_awaited_once()
    docs_to_delete = search_client.delete_documents.call_args.args[0]
    assert len(docs_to_delete) == 2
    assert all(d["id"] in ids for d in docs_to_delete)


async def test_delete_empty_list_no_call(az_backend):
    b, search_client, _ = az_backend

    await b.delete(COLL, [])

    search_client.delete_documents.assert_not_awaited()


# ---------------------------------------------------------------------------
# search_by_query_hash
# ---------------------------------------------------------------------------


async def test_search_by_query_hash_found(az_backend):
    b, search_client, _ = az_backend
    eid = str(uuid.uuid4())
    doc = _make_search_result(id=eid, generated_query="SELECT 42")
    search_client.search = AsyncMock(return_value=_AsyncIter([doc]))

    result = await b.search_by_query_hash(COLL, "abc123")

    assert result is not None
    assert result.id == eid
    assert result.generated_query == "SELECT 42"


async def test_search_by_query_hash_not_found(az_backend):
    b, search_client, _ = az_backend
    search_client.search = AsyncMock(return_value=_AsyncIter([]))

    result = await b.search_by_query_hash(COLL, "nonexistent")

    assert result is None


# ---------------------------------------------------------------------------
# update_usage_count
# ---------------------------------------------------------------------------


async def test_update_usage_count_increments(az_backend):
    b, search_client, _ = az_backend
    eid = str(uuid.uuid4())
    search_client.get_document.return_value = {"id": eid, "usage_count": 3}

    await b.update_usage_count(COLL, eid)

    search_client.merge_documents.assert_awaited_once()
    merge_args = search_client.merge_documents.call_args.args[0]
    assert merge_args[0]["usage_count"] == 4


async def test_update_usage_count_unknown_id(az_backend):
    from azure.core.exceptions import ResourceNotFoundError

    b, search_client, _ = az_backend
    search_client.get_document.side_effect = ResourceNotFoundError()

    await b.update_usage_count(COLL, "nonexistent")  # must not raise

    search_client.merge_documents.assert_not_awaited()


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


async def test_close_closes_all_clients(mock_search_client, mock_index_client):
    from azure.core.exceptions import ResourceNotFoundError
    from medha.backends.azure_search import AzureSearchBackend

    mock_index_client.get_index.side_effect = ResourceNotFoundError()

    with (
        patch("medha.backends.azure_search.AsyncSearchIndexClient", return_value=mock_index_client),
        patch("medha.backends.azure_search.AsyncSearchClient", return_value=mock_search_client),
    ):
        b = AzureSearchBackend(_az_settings())
        await b.connect()
        await b.initialize(COLL, DIM)
        await b.close()

    mock_search_client.close.assert_awaited()
    mock_index_client.close.assert_awaited()
    assert b._index_client is None
    assert b._search_clients == {}


# ---------------------------------------------------------------------------
# not connected guard
# ---------------------------------------------------------------------------


async def test_not_connected_raises():
    from medha.backends.azure_search import AzureSearchBackend

    b = AzureSearchBackend.__new__(AzureSearchBackend)
    b._index_client = None
    b._search_clients = {}
    b._credential = None
    b._settings = _az_settings()

    with pytest.raises(StorageError, match="connect"):
        await b.initialize(COLL, DIM)


# ---------------------------------------------------------------------------
# missing deps
# ---------------------------------------------------------------------------


async def test_missing_deps_raises():
    with patch("medha.backends.azure_search.HAS_AZURE_SEARCH", False):
        from medha.backends.azure_search import AzureSearchBackend

        with pytest.raises(ConfigurationError, match="pip install medha-archai"):
            AzureSearchBackend()


# ---------------------------------------------------------------------------
# index name sanitization
# ---------------------------------------------------------------------------


def test_az_index_name_sanitization():
    from medha.backends.azure_search import _az_index_name

    assert _az_index_name("my_cache", "medha") == "medha-my-cache"
    assert _az_index_name("My Cache V2", "medha") == "medha-my-cache-v2"


# ---------------------------------------------------------------------------
# find_expired
# ---------------------------------------------------------------------------


async def test_find_expired_returns_ids(az_backend):
    b, search_client, _ = az_backend
    eid = str(uuid.uuid4())
    search_client.search = AsyncMock(return_value=_AsyncIter([{"id": eid}]))

    expired_ids = await b.find_expired(COLL)

    search_client.search.assert_awaited_once()
    call_kwargs = search_client.search.call_args.kwargs
    assert "expires_at" in call_kwargs.get("filter", "")
    assert expired_ids == [eid]


async def test_find_expired_empty(az_backend):
    b, search_client, _ = az_backend
    search_client.search = AsyncMock(return_value=_AsyncIter([]))

    expired_ids = await b.find_expired(COLL)

    assert expired_ids == []


# ---------------------------------------------------------------------------
# drop_collection
# ---------------------------------------------------------------------------


async def test_drop_collection_deletes_index(az_backend):
    b, search_client, index_client = az_backend

    await b.drop_collection(COLL)

    index_client.delete_index.assert_awaited_once()
    assert COLL not in b._search_clients


async def test_drop_collection_not_connected_raises():
    from medha.backends.azure_search import AzureSearchBackend

    b = AzureSearchBackend.__new__(AzureSearchBackend)
    b._index_client = None
    b._search_clients = {}
    b._credential = None
    b._settings = _az_settings()

    with pytest.raises(StorageError):
        await b.drop_collection(COLL)


# ---------------------------------------------------------------------------
# find_by_query_hash
# ---------------------------------------------------------------------------


async def test_find_by_query_hash(az_backend):
    b, search_client, _ = az_backend
    eid = str(uuid.uuid4())
    search_client.search = AsyncMock(return_value=_AsyncIter([{"id": eid}]))

    ids = await b.find_by_query_hash(COLL, "abc123")

    assert ids == [eid]


async def test_find_by_query_hash_empty(az_backend):
    b, search_client, _ = az_backend
    search_client.search = AsyncMock(return_value=_AsyncIter([]))

    ids = await b.find_by_query_hash(COLL, "nonexistent")

    assert ids == []


# ---------------------------------------------------------------------------
# find_by_template_id
# ---------------------------------------------------------------------------


async def test_find_by_template_id(az_backend):
    b, search_client, _ = az_backend
    eids = [str(uuid.uuid4()), str(uuid.uuid4())]
    search_client.search = AsyncMock(return_value=_AsyncIter([{"id": e} for e in eids]))

    ids = await b.find_by_template_id(COLL, "tmpl1")

    assert set(ids) == set(eids)


# ---------------------------------------------------------------------------
# search_by_normalized_question
# ---------------------------------------------------------------------------


async def test_search_by_normalized_question_found(az_backend):
    b, search_client, _ = az_backend
    eid = str(uuid.uuid4())
    doc = _make_search_result(id=eid, generated_query="SELECT COUNT(*) FROM users")
    search_client.search = AsyncMock(return_value=_AsyncIter([doc]))

    result = await b.search_by_normalized_question(COLL, "how many users")

    assert result is not None
    assert result.generated_query == "SELECT COUNT(*) FROM users"


async def test_search_by_normalized_question_not_found(az_backend):
    b, search_client, _ = az_backend
    search_client.search = AsyncMock(return_value=_AsyncIter([]))

    result = await b.search_by_normalized_question(COLL, "nothing")

    assert result is None


# ---------------------------------------------------------------------------
# load_stats / save_stats
# ---------------------------------------------------------------------------


def test_meta_index_name_and_stats_key_are_sanitised():
    from medha.backends.azure_search import (
        _az_index_name,
        _az_meta_index_name,
        _az_stats_key,
    )

    assert _az_meta_index_name("my_cache", "medha") == "medha-my-cache-meta"
    assert _az_meta_index_name("my_cache", "medha") != _az_index_name("my_cache", "medha")
    # Long names stay inside Azure's 128-character limit, suffix intact
    long_name = _az_meta_index_name("x" * 400, "medha")
    assert len(long_name) <= 128
    assert long_name.endswith("-meta")

    # Document keys accept only [A-Za-z0-9_=-]; everything else is folded
    assert _az_stats_key("my_cache") == "stats_my_cache"
    assert _az_stats_key("my cache'; DROP") == "stats_my_cache___DROP"


async def test_save_stats_creates_meta_index_and_uploads(az_backend):
    from medha.backends.azure_search import _az_meta_index_name, _az_stats_key
    from medha.types import PersistedStats

    b, search_client, index_client = az_backend
    index_client.create_index.reset_mock()
    search_client.merge_or_upload_documents.reset_mock()

    await b.save_stats(COLL, PersistedStats(total_requests=13, total_hits=8))

    created = index_client.create_index.call_args.args[0]
    assert created.name == _az_meta_index_name(COLL, b._settings.azure_search_index_name)
    assert {f.name for f in created.fields} == {"id", "stats_json"}

    search_client.merge_or_upload_documents.assert_awaited_once()
    docs = search_client.merge_or_upload_documents.call_args.args[0]
    assert docs[0]["id"] == _az_stats_key(COLL)
    assert PersistedStats.model_validate_json(docs[0]["stats_json"]).total_requests == 13


async def test_save_stats_skips_create_when_meta_index_exists(az_backend):
    from medha.types import PersistedStats

    b, search_client, index_client = az_backend
    index_client.get_index.side_effect = None
    index_client.get_index.return_value = MagicMock()
    index_client.create_index.reset_mock()

    await b.save_stats(COLL, PersistedStats())

    index_client.create_index.assert_not_awaited()
    search_client.merge_or_upload_documents.assert_awaited()


async def test_load_stats_returns_parsed_snapshot(az_backend):
    from medha.backends.azure_search import _az_stats_key
    from medha.types import PersistedStats

    b, search_client, _ = az_backend
    stats = PersistedStats(total_requests=5, hits_by_strategy={"exact": 5})
    search_client.get_document = AsyncMock(
        return_value={"id": _az_stats_key(COLL), "stats_json": stats.model_dump_json()}
    )

    loaded = await b.load_stats(COLL)

    assert loaded is not None
    assert loaded.total_requests == 5
    assert loaded.hits_by_strategy == {"exact": 5}
    assert search_client.get_document.call_args.kwargs["key"] == _az_stats_key(COLL)


async def test_load_stats_returns_none_when_absent(az_backend):
    from azure.core.exceptions import ResourceNotFoundError

    b, search_client, _ = az_backend
    search_client.get_document = AsyncMock(side_effect=ResourceNotFoundError())

    assert await b.load_stats(COLL) is None


async def test_load_stats_wraps_http_errors(az_backend):
    from azure.core.exceptions import HttpResponseError

    b, search_client, _ = az_backend
    search_client.get_document = AsyncMock(side_effect=HttpResponseError("boom"))

    with pytest.raises(StorageError, match="load_stats failed"):
        await b.load_stats(COLL)


async def test_save_stats_wraps_http_errors(az_backend):
    from azure.core.exceptions import HttpResponseError

    from medha.types import PersistedStats

    b, search_client, _ = az_backend
    search_client.merge_or_upload_documents = AsyncMock(
        side_effect=HttpResponseError("boom")
    )

    with pytest.raises(StorageError, match="save_stats failed"):
        await b.save_stats(COLL, PersistedStats())


async def test_close_closes_meta_clients(az_backend):
    from medha.types import PersistedStats

    b, search_client, _ = az_backend
    await b.save_stats(COLL, PersistedStats())
    assert b._meta_clients

    await b.close()

    assert not b._meta_clients
    search_client.close.assert_awaited()


async def test_stats_methods_require_connection():
    from medha.backends.azure_search import AzureSearchBackend
    from medha.types import PersistedStats

    b = AzureSearchBackend.__new__(AzureSearchBackend)
    b._search_clients = {}
    b._meta_clients = {}
    b._index_client = None
    b._credential = None
    b._settings = _az_settings()

    with pytest.raises(StorageError, match="connect"):
        await b.load_stats(COLL)

    with pytest.raises(StorageError, match="connect"):
        await b.save_stats(COLL, PersistedStats())
