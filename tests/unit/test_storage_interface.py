"""Unit tests for medha.interfaces.storage.VectorStorageBackend ABC."""

import hashlib
import uuid

import pytest

from medha.interfaces.storage import VectorStorageBackend
from medha.types import CacheEntry, CacheResult


def _make_entry(
    question: str = "test question",
    query: str = "SELECT 1",
    dim: int = 8,
) -> CacheEntry:
    vec = [0.1] * dim
    vec[0] = abs(hash(question) % 100) / 100.0 + 0.01
    mag = sum(v ** 2 for v in vec) ** 0.5
    vec = [v / mag for v in vec]
    return CacheEntry(
        id=str(uuid.uuid4()),
        vector=vec,
        original_question=question,
        normalized_question=question.lower(),
        generated_query=query,
        query_hash=hashlib.md5(query.encode()).hexdigest(),
    )


# Detect available backends for parametrized contract tests
def _available_no_service_params() -> list[str]:
    params = ["memory"]
    try:
        import chromadb  # noqa: F401
        params.append("chroma")
    except ImportError:
        pass
    return params


_CONTRACT_BACKENDS = _available_no_service_params()


class TestVectorStorageBackendABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            VectorStorageBackend()

    def test_partial_implementation_fails(self):
        class PartialBackend(VectorStorageBackend):
            async def initialize(self, collection_name, dimension, **kwargs):
                pass

            # Missing search, upsert, scroll, count, delete, close

        with pytest.raises(TypeError):
            PartialBackend()


# ---------------------------------------------------------------------------
# Cross-backend contract tests (CRUD)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("any_backend", _CONTRACT_BACKENDS, indirect=True)
class TestBackendContract:
    """These tests must pass on ALL backends that need no external service."""

    async def test_initialize_is_idempotent(self, any_backend):
        await any_backend.initialize("contract_test", 8)
        await any_backend.initialize("contract_test", 8)  # no exception

    async def test_upsert_and_count(self, any_backend, make_entry_fixture):
        await any_backend.initialize("contract_test", 8)
        await any_backend.upsert("contract_test", [make_entry_fixture()])
        assert await any_backend.count("contract_test") == 1

    async def test_search_returns_cache_result(self, any_backend, make_entry_fixture):
        vec = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        entry = make_entry_fixture(vector=vec, dim=8)
        await any_backend.initialize("contract_test", 8)
        await any_backend.upsert("contract_test", [entry])
        results = await any_backend.search("contract_test", vec, limit=1)
        assert len(results) == 1
        assert isinstance(results[0], CacheResult)
        assert results[0].score > 0.9

    async def test_delete_removes_entry(self, any_backend, make_entry_fixture):
        entry = make_entry_fixture()
        await any_backend.initialize("contract_test", 8)
        await any_backend.upsert("contract_test", [entry])
        await any_backend.delete("contract_test", [entry.id])
        assert await any_backend.count("contract_test") == 0

    async def test_scroll_returns_all(self, any_backend, make_entry_fixture):
        entries = [make_entry_fixture() for _ in range(5)]
        await any_backend.initialize("contract_test", 8)
        await any_backend.upsert("contract_test", entries)
        results, next_offset = await any_backend.scroll("contract_test", limit=10)
        assert len(results) == 5
        assert next_offset is None

    async def test_upsert_same_id_overwrites(self, any_backend):
        eid = str(uuid.uuid4())
        e1 = _make_entry(question="q1", query="SELECT 1")
        e1 = CacheEntry(
            id=eid, vector=e1.vector, original_question="q1",
            normalized_question="q1", generated_query="SELECT 1",
            query_hash=hashlib.md5(b"SELECT 1").hexdigest(),
        )
        e2 = CacheEntry(
            id=eid, vector=e1.vector, original_question="q1",
            normalized_question="q1", generated_query="SELECT 2",
            query_hash=hashlib.md5(b"SELECT 2").hexdigest(),
        )
        await any_backend.initialize("contract_test", 8)
        await any_backend.upsert("contract_test", [e1])
        await any_backend.upsert("contract_test", [e2])
        assert await any_backend.count("contract_test") == 1

    async def test_search_by_query_hash(self, any_backend):
        entry = _make_entry(question="unique q for hash test", query="SELECT 999")
        await any_backend.initialize("contract_test", 8)
        await any_backend.upsert("contract_test", [entry])

        result = await any_backend.search_by_query_hash("contract_test", entry.query_hash)

        assert result is not None
        assert result.generated_query == "SELECT 999"

    async def test_search_by_query_hash_not_found(self, any_backend):
        await any_backend.initialize("contract_test", 8)

        result = await any_backend.search_by_query_hash("contract_test", "deadbeef" * 8)

        assert result is None

    async def test_update_usage_count(self, any_backend):
        entry = _make_entry(question="usage count test", query="SELECT usage")
        await any_backend.initialize("contract_test", 8)
        await any_backend.upsert("contract_test", [entry])

        await any_backend.update_usage_count("contract_test", entry.id)

        results, _ = await any_backend.scroll("contract_test")
        matching = [r for r in results if r.id == entry.id]
        assert matching
        assert matching[0].usage_count == 2

    async def test_find_expired_returns_empty_when_none_expired(self, any_backend):
        from datetime import datetime, timedelta, timezone
        entry = _make_entry(question="future ttl", query="SELECT future")
        entry = CacheEntry(
            id=entry.id, vector=entry.vector,
            original_question=entry.original_question,
            normalized_question=entry.normalized_question,
            generated_query=entry.generated_query,
            query_hash=entry.query_hash,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        await any_backend.initialize("contract_test", 8)
        await any_backend.upsert("contract_test", [entry])

        expired_ids = await any_backend.find_expired("contract_test")

        assert entry.id not in expired_ids

    async def test_drop_collection_removes_data(self, any_backend):
        await any_backend.initialize("drop_coll_test", 8)
        await any_backend.upsert("drop_coll_test", [_make_entry()])

        await any_backend.drop_collection("drop_coll_test")

        # After drop, collection is gone; re-initialize should give count=0
        await any_backend.initialize("drop_coll_test", 8)
        assert await any_backend.count("drop_coll_test") == 0

    async def test_find_by_query_hash(self, any_backend):
        entry = _make_entry(question="find by qhash", query="SELECT hash_test")
        await any_backend.initialize("contract_test", 8)
        await any_backend.upsert("contract_test", [entry])

        ids = await any_backend.find_by_query_hash("contract_test", entry.query_hash)

        assert entry.id in ids

    async def test_search_by_normalized_question(self, any_backend):
        entry = _make_entry(question="normalized q test", query="SELECT norm")
        await any_backend.initialize("contract_test", 8)
        await any_backend.upsert("contract_test", [entry])

        result = await any_backend.search_by_normalized_question(
            "contract_test", entry.normalized_question
        )

        assert result is not None
        assert result.generated_query == "SELECT norm"

    async def test_update_feedback_correct_returns_one(self, any_backend, make_entry_fixture):
        entry = make_entry_fixture(question="fb correct contract", query="SELECT fb_c")
        await any_backend.initialize("contract_test", 8)
        await any_backend.upsert("contract_test", [entry])

        result = await any_backend.update_feedback("contract_test", entry.id, correct=True)

        assert result == 1

    async def test_update_feedback_incorrect_returns_one(self, any_backend, make_entry_fixture):
        entry = make_entry_fixture(question="fb incorrect contract", query="SELECT fb_i")
        await any_backend.initialize("contract_test", 8)
        await any_backend.upsert("contract_test", [entry])

        result = await any_backend.update_feedback("contract_test", entry.id, correct=False)

        assert result == 1

    async def test_update_feedback_accumulates(self, any_backend, make_entry_fixture):
        entry = make_entry_fixture(question="fb accum contract", query="SELECT fb_accum")
        await any_backend.initialize("contract_test", 8)
        await any_backend.upsert("contract_test", [entry])

        await any_backend.update_feedback("contract_test", entry.id, correct=True)
        await any_backend.update_feedback("contract_test", entry.id, correct=True)
        await any_backend.update_feedback("contract_test", entry.id, correct=False)

        results, _ = await any_backend.scroll("contract_test")
        matching = [r for r in results if r.id == entry.id]
        assert matching
        assert matching[0].feedback_correct == 2
        assert matching[0].feedback_incorrect == 1

    async def test_update_feedback_missing_id_returns_zero(self, any_backend, make_entry_fixture):
        await any_backend.initialize("contract_test", 8)

        result = await any_backend.update_feedback(
            "contract_test", "no-such-id-000", correct=True
        )

        assert result == 0
