"""Unit tests for store_many, warm_from_dataframe, export_to_dataframe, dedup_collection."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from medha.config import Settings
from medha.exceptions import ConfigurationError
from medha.types import CacheResult

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_result(
    id: str | None = None,
    query: str = "SELECT 1",
    created_at: datetime | None = None,
) -> CacheResult:
    q = query
    return CacheResult(
        id=id or str(uuid.uuid4()),
        score=1.0,
        original_question="q",
        normalized_question="q",
        generated_query=q,
        query_hash=hashlib.md5(q.encode()).hexdigest(),
        created_at=created_at or datetime.now(timezone.utc),
    )


def _make_medha(settings: Settings | None = None, embedder=None, backend=None):
    """Build a Medha instance with all externals mocked."""
    from medha.core import Medha

    if embedder is None:
        embedder = MagicMock()
        embedder.dimension = 4
        embedder.aembed_batch = AsyncMock(return_value=[[0.1, 0.2, 0.3, 0.4]])

    if backend is None:
        backend = AsyncMock()
        backend.upsert = AsyncMock()
        backend.scroll = AsyncMock(return_value=([], None))
        backend.delete = AsyncMock()

    s = settings or Settings(backend_type="memory")
    m = Medha.__new__(Medha)
    Medha.__init__(
        m,
        collection_name="test",
        embedder=embedder,
        backend=backend,
        settings=s,
    )
    return m, embedder, backend


# ---------------------------------------------------------------------------
# store_many: basic insertion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_many_inserts_all_entries():
    entries = [
        {"question": "How many users?", "generated_query": "SELECT COUNT(*) FROM users"},
        {"question": "List orders", "generated_query": "SELECT * FROM orders"},
    ]

    m, embedder, backend = _make_medha()
    embedder.aembed_batch = AsyncMock(
        side_effect=[
            [[0.1] * 4],
            [[0.2] * 4],
        ]
    )
    # batch_size=1 so two chunks, two upsert calls
    count = await m.store_many(entries, batch_size=1)

    assert count == 2
    assert backend.upsert.call_count == 2


@pytest.mark.asyncio
async def test_store_many_single_chunk():
    entries = [
        {"question": "Q1", "generated_query": "SELECT 1"},
        {"question": "Q2", "generated_query": "SELECT 2"},
    ]

    m, embedder, backend = _make_medha()
    embedder.aembed_batch = AsyncMock(return_value=[[0.1] * 4, [0.2] * 4])

    count = await m.store_many(entries, batch_size=10)

    assert count == 2
    assert backend.upsert.call_count == 1


# ---------------------------------------------------------------------------
# store_many: chunking with batch_size=2
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_many_chunks_correctly():
    """With batch_size=2 and 5 entries, embedder is called 3 times."""
    entries = [{"question": f"Q{i}", "generated_query": f"SELECT {i}"} for i in range(5)]

    m, embedder, backend = _make_medha()

    call_sizes: list[int] = []

    async def fake_embed(texts, **kwargs):
        call_sizes.append(len(texts))
        return [[0.1] * 4 for _ in texts]

    embedder.aembed_batch = fake_embed

    count = await m.store_many(entries, batch_size=2)

    assert count == 5
    assert call_sizes == [2, 2, 1]
    assert backend.upsert.call_count == 3


# ---------------------------------------------------------------------------
# store_many: on_progress callback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_many_on_progress_called_correctly():
    entries = [{"question": f"Q{i}", "generated_query": f"SELECT {i}"} for i in range(4)]

    m, embedder, backend = _make_medha()

    async def fake_embed(texts, **kwargs):
        return [[0.1] * 4 for _ in texts]

    embedder.aembed_batch = fake_embed

    progress_calls: list[tuple[int, int]] = []

    def on_progress(done, total):
        progress_calls.append((done, total))

    await m.store_many(entries, batch_size=2, on_progress=on_progress)

    assert progress_calls == [(2, 4), (4, 4)]


# ---------------------------------------------------------------------------
# store_many: fail-fast validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_many_raises_on_missing_question():
    entries = [
        {"question": "Q1", "generated_query": "SELECT 1"},
        {"generated_query": "SELECT 2"},  # missing question
    ]
    m, _, _ = _make_medha()
    with pytest.raises(ValueError, match="missing or empty 'question'"):
        await m.store_many(entries)


@pytest.mark.asyncio
async def test_store_many_raises_on_missing_generated_query():
    entries = [
        {"question": "Q1"},  # missing generated_query
    ]
    m, _, _ = _make_medha()
    with pytest.raises(ValueError, match="missing or empty 'generated_query'"):
        await m.store_many(entries)


@pytest.mark.asyncio
async def test_store_many_raises_before_any_embedding():
    """ValueError must fire before any embedding call."""
    entries = [
        {"question": "Q1", "generated_query": "SELECT 1"},
        {"question": "", "generated_query": "SELECT 2"},  # empty question
    ]
    m, embedder, _ = _make_medha()
    embedder.aembed_batch = AsyncMock()

    with pytest.raises(ValueError):
        await m.store_many(entries)

    embedder.aembed_batch.assert_not_called()


# ---------------------------------------------------------------------------
# warm_from_dataframe
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_warm_from_dataframe_with_custom_columns():
    pd = pytest.importorskip("pandas")

    df = pd.DataFrame({
        "nl_question": ["How many users?", "List orders"],
        "sql_query": ["SELECT COUNT(*) FROM users", "SELECT * FROM orders"],
    })

    m, embedder, backend = _make_medha()

    async def fake_embed(texts, **kwargs):
        return [[0.1] * 4 for _ in texts]

    embedder.aembed_batch = fake_embed

    count = await m.warm_from_dataframe(
        df,
        question_col="nl_question",
        query_col="sql_query",
    )

    assert count == 2
    assert backend.upsert.call_count >= 1


@pytest.mark.asyncio
async def test_warm_from_dataframe_without_pandas_raises():
    m, _, _ = _make_medha()

    with patch.dict("sys.modules", {"pandas": None}), pytest.raises(ConfigurationError, match="pandas"):
        await m.warm_from_dataframe(MagicMock())


# ---------------------------------------------------------------------------
# export_to_dataframe
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_export_to_dataframe_returns_all_entries():
    pd = pytest.importorskip("pandas")

    results = [
        _make_result(id="1", query="SELECT 1"),
        _make_result(id="2", query="SELECT 2"),
    ]

    m, embedder, backend = _make_medha()
    # First call returns results + next_offset=None → single page
    backend.scroll = AsyncMock(return_value=(results, None))

    df = await m.export_to_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert set(df["generated_query"]) == {"SELECT 1", "SELECT 2"}


@pytest.mark.asyncio
async def test_export_to_dataframe_without_pandas_raises():
    m, _, _ = _make_medha()

    with patch.dict("sys.modules", {"pandas": None}), pytest.raises(ConfigurationError, match="pandas"):
        await m.export_to_dataframe()


# ---------------------------------------------------------------------------
# dedup_collection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_keep_latest_removes_older_duplicate():
    older = _make_result(id="old", query="SELECT 1", created_at=datetime(2024, 1, 1, tzinfo=timezone.utc))
    newer = _make_result(id="new", query="SELECT 1", created_at=datetime(2024, 6, 1, tzinfo=timezone.utc))

    m, embedder, backend = _make_medha()
    backend.scroll = AsyncMock(return_value=([older, newer], None))
    backend.delete = AsyncMock()

    deleted = await m.dedup_collection(strategy="keep_latest")

    assert deleted == 1
    backend.delete.assert_called_once()
    deleted_ids = backend.delete.call_args[0][1]
    assert "old" in deleted_ids


@pytest.mark.asyncio
async def test_dedup_keep_first_removes_newer_duplicate():
    older = _make_result(id="old", query="SELECT 1", created_at=datetime(2024, 1, 1, tzinfo=timezone.utc))
    newer = _make_result(id="new", query="SELECT 1", created_at=datetime(2024, 6, 1, tzinfo=timezone.utc))

    m, embedder, backend = _make_medha()
    backend.scroll = AsyncMock(return_value=([older, newer], None))
    backend.delete = AsyncMock()

    deleted = await m.dedup_collection(strategy="keep_first")

    assert deleted == 1
    deleted_ids = backend.delete.call_args[0][1]
    assert "new" in deleted_ids


@pytest.mark.asyncio
async def test_dedup_no_duplicates_returns_zero():
    r1 = _make_result(id="1", query="SELECT 1")
    r2 = _make_result(id="2", query="SELECT 2")

    m, embedder, backend = _make_medha()
    backend.scroll = AsyncMock(return_value=([r1, r2], None))
    backend.delete = AsyncMock()

    deleted = await m.dedup_collection()

    assert deleted == 0
    backend.delete.assert_not_called()
