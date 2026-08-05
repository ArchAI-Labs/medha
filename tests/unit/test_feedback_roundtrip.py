"""Every backend must read feedback counters back into CacheResult.

Regression guard for the 0.5.0 bug where only InMemoryBackend populated
``feedback_correct`` / ``feedback_incorrect`` on the way out.  The counters were
written by ``update_feedback()`` but never read back, so ``feedback_boost_factor``
silently did nothing on nine of the ten built-in backends.

The result mappers are pure functions, so they are tested directly: no server,
no driver round-trip, and one test per backend that cannot rot silently.
"""

from datetime import datetime, timezone

import pytest

FB_CORRECT = 7
FB_INCORRECT = 3


def _assert_counters(result) -> None:
    assert result.feedback_correct == FB_CORRECT, (
        f"feedback_correct dropped by the mapper (got {result.feedback_correct})"
    )
    assert result.feedback_incorrect == FB_INCORRECT, (
        f"feedback_incorrect dropped by the mapper (got {result.feedback_incorrect})"
    )


def _payload() -> dict:
    """A stored record as the schemaless backends keep it."""
    return {
        "original_question": "how many users",
        "normalized_question": "how many users",
        "generated_query": "SELECT count(*) FROM users",
        "query_hash": "abc123",
        "response_summary": None,
        "template_id": None,
        "usage_count": 1,
        "feedback_correct": FB_CORRECT,
        "feedback_incorrect": FB_INCORRECT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": None,
    }


# --- backends with no optional dependency -----------------------------------


def test_memory_mapper_reads_feedback():
    from medha.backends.memory import _point_to_cache_result

    point = {"id": "1", "vector": [0.1], "payload": _payload()}
    _assert_counters(_point_to_cache_result(point, 0.9))


def test_qdrant_mapper_reads_feedback():
    pytest.importorskip("qdrant_client")
    from medha.backends.qdrant import QdrantBackend

    class _Point:
        id = "1"
        score = 0.9
        payload = _payload()

    _assert_counters(QdrantBackend._point_to_cache_result(_Point()))


def test_qdrant_payload_writes_feedback():
    """The counters must survive the CacheEntry -> payload direction too."""
    pytest.importorskip("qdrant_client")
    import uuid

    from medha.backends.qdrant import QdrantBackend
    from medha.types import CacheEntry

    entry = CacheEntry(
        id=str(uuid.uuid4()),
        vector=[0.1, 0.2],
        original_question="q",
        normalized_question="q",
        generated_query="SELECT 1",
        query_hash="h",
        feedback_correct=FB_CORRECT,
        feedback_incorrect=FB_INCORRECT,
    )
    payload = QdrantBackend._entry_to_point(entry).payload
    assert payload["feedback_correct"] == FB_CORRECT
    assert payload["feedback_incorrect"] == FB_INCORRECT


def test_asyncpg_mapper_reads_feedback():
    """Covers both PgVectorBackend and VectorChordBackend (shared mixin)."""
    from medha.backends._asyncpg_mixin import _row_to_cache_result

    row = dict(_payload(), id="1", created_at=None, expires_at=None)
    _assert_counters(_row_to_cache_result(row, score=1.0))


def test_asyncpg_mapper_tolerates_null_counters():
    """Rows written before the ALTER TABLE migration can carry NULL."""
    from medha.backends._asyncpg_mixin import _row_to_cache_result

    row = dict(
        _payload(),
        id="1",
        created_at=None,
        expires_at=None,
        feedback_correct=None,
        feedback_incorrect=None,
    )
    result = _row_to_cache_result(row, score=1.0)
    assert result.feedback_correct == 0
    assert result.feedback_incorrect == 0


# --- backends behind an optional dependency ---------------------------------


def test_elasticsearch_mapper_reads_feedback():
    pytest.importorskip("elasticsearch")
    from medha.backends.elasticsearch import _hit_to_cache_result

    _assert_counters(_hit_to_cache_result("1", _payload(), 0.9))


def test_azure_search_mapper_reads_feedback():
    pytest.importorskip("azure.search.documents")
    from medha.backends.azure_search import _doc_to_result

    _assert_counters(_doc_to_result(dict(_payload(), id="1"), 0.9))


def test_weaviate_mapper_reads_feedback():
    pytest.importorskip("weaviate")
    from medha.backends.weaviate import _obj_to_result

    class _Obj:
        uuid = "1"
        properties = _payload()

    _assert_counters(_obj_to_result(_Obj(), 0.9))


def test_redis_mapper_reads_feedback():
    pytest.importorskip("redis")
    from medha.backends.redis_vector import _doc_to_result

    class _Doc:
        id = "prefix:1"

        def __init__(self, fields):
            for k, v in fields.items():
                setattr(self, k, "" if v is None else str(v))

    _assert_counters(_doc_to_result(_Doc(_payload()), 0.9))


def test_lancedb_mapper_reads_feedback():
    pytest.importorskip("lancedb")
    from medha.backends.lancedb import _row_to_result

    _assert_counters(_row_to_result(dict(_payload(), id="1"), 0.9))


def test_chroma_mapper_reads_feedback():
    pytest.importorskip("chromadb")
    from medha.backends.chroma import _meta_to_result

    _assert_counters(_meta_to_result("1", 0.9, _payload()))


# --- schema coverage --------------------------------------------------------


def _module_sql(module_name: str) -> str:
    """Source of a backend module with runs of whitespace collapsed."""
    import importlib
    import re
    from pathlib import Path

    mod = importlib.import_module(f"medha.backends.{module_name}")
    return re.sub(r"\s+", " ", Path(mod.__file__).read_text(encoding="utf-8"))


@pytest.mark.parametrize("module_name", ["pgvector", "vectorchord"])
def test_pg_ddl_declares_feedback_columns(module_name):
    """update_feedback() runs UPDATE ... SET feedback_correct — the column must exist.

    Both tables predate the counters, so a fresh CREATE TABLE and an ALTER for
    already-deployed tables are each required.
    """
    sql = _module_sql(module_name)
    assert "feedback_correct INTEGER NOT NULL DEFAULT 0" in sql
    assert "feedback_incorrect INTEGER NOT NULL DEFAULT 0" in sql
    assert "ADD COLUMN IF NOT EXISTS feedback_correct" in sql
    assert "ADD COLUMN IF NOT EXISTS feedback_incorrect" in sql


@pytest.mark.parametrize("module_name", ["pgvector", "vectorchord"])
def test_pg_select_projects_feedback_columns(module_name):
    """A column that exists but is never SELECTed still yields a 0 counter."""
    sql = _module_sql(module_name)
    assert "feedback_correct, feedback_incorrect," in sql


@pytest.mark.parametrize(
    "module_name",
    ["azure_search", "redis_vector", "lancedb"],
)
def test_projection_lists_request_feedback_fields(module_name):
    """Backends that project an explicit field list must ask for the counters.

    Azure (``select=_SCALAR_FIELDS``), Redis (``return_fields``) and LanceDB
    (``select(columns)``) return only what they are asked for, so a mapper that
    reads the counters is not enough on its own.
    """
    src = _module_sql(module_name)
    assert '"feedback_correct", "feedback_incorrect"' in src
