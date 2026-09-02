"""Every backend stores metadata and hands it back.

Ten backends encode an entry ten different ways, and a field one of them
forgets does not fail — it reads back as "this entry has no scope", which
turns a filtered search into a permanent NO_MATCH. These tests walk an entry
through each backend's own encode/decode pair, without needing the service
running, so a new field cannot be added to nine of them and missed on the
tenth.
"""

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from medha.types import CacheEntry

METADATA = {"resolved_date": "2026-08-12", "hour": 10, "ratio": 0.25, "draft": True}


def _entry(metadata=None) -> CacheEntry:
    return CacheEntry(
        id=str(uuid.uuid4()),
        vector=[0.1, 0.2, 0.3],
        original_question="total revenue for the period",
        normalized_question="total revenue for the period",
        generated_query="SELECT SUM(amount) FROM sales",
        query_hash="deadbeef",
        created_at=datetime.now(timezone.utc),
        metadata=METADATA if metadata is None else metadata,
    )


# ---------------------------------------------------------------------------
# Every backend declares support
# ---------------------------------------------------------------------------

BACKENDS = [
    ("medha.backends.memory", "InMemoryBackend"),
    ("medha.backends.qdrant", "QdrantBackend"),
    ("medha.backends.pgvector", "PgVectorBackend"),
    ("medha.backends.vectorchord", "VectorChordBackend"),
    ("medha.backends.elasticsearch", "ElasticsearchBackend"),
    ("medha.backends.chroma", "ChromaBackend"),
    ("medha.backends.weaviate", "WeaviateBackend"),
    ("medha.backends.redis_vector", "RedisVectorBackend"),
    ("medha.backends.azure_search", "AzureSearchBackend"),
    ("medha.backends.lancedb", "LanceDBBackend"),
]


@pytest.mark.parametrize("module_path,class_name", BACKENDS, ids=lambda v: v.split(".")[-1])
def test_backend_declares_metadata_support(module_path, class_name):
    """The flag Medha checks before accepting filters or metadata."""
    module = pytest.importorskip(module_path)
    backend_cls = getattr(module, class_name)

    assert backend_cls.supports_metadata is True


# ---------------------------------------------------------------------------
# Encode/decode round-trips, backend by backend
# ---------------------------------------------------------------------------

class TestRoundTrips:
    def test_memory(self):
        from medha.backends.memory import _point_to_cache_result

        entry = _entry()
        point = {
            "id": entry.id,
            "vector": entry.vector,
            "payload": {
                "original_question": entry.original_question,
                "normalized_question": entry.normalized_question,
                "generated_query": entry.generated_query,
                "query_hash": entry.query_hash,
                "created_at": entry.created_at.isoformat(),
                "metadata": dict(entry.metadata),
            },
        }

        assert _point_to_cache_result(point, 1.0).metadata == METADATA

    def test_qdrant(self):
        # The module imports without the driver, but _entry_to_point needs the
        # models it could not bind — so guard on the driver, not the module.
        pytest.importorskip("qdrant_client")
        from medha.backends.qdrant import QdrantBackend

        point = QdrantBackend._entry_to_point(_entry())
        scored = SimpleNamespace(id=point.id, payload=point.payload, score=0.9)

        assert QdrantBackend._point_to_cache_result(scored).metadata == METADATA

    def test_chroma(self):
        from medha.backends.chroma import _entry_to_metadata, _meta_to_result

        entry = _entry()
        stored = _entry_to_metadata(entry)

        # Chroma only accepts scalars, so the encoded form must stay flat.
        assert all(isinstance(v, (str, int, float, bool)) for v in stored.values())
        assert _meta_to_result(entry.id, 1.0, stored).metadata == METADATA

    def test_lancedb(self):
        from medha.backends.lancedb import _entry_to_row, _row_to_result

        row = _entry_to_row(_entry())

        assert isinstance(row["metadata_json"], str)
        assert _row_to_result(row, 1.0).metadata == METADATA

    def test_lancedb_schema_declares_the_column(self):
        pytest.importorskip("pyarrow")
        from medha.backends.lancedb import _build_schema

        assert "metadata_json" in _build_schema(3).names

    def test_lancedb_migrates_an_older_table(self):
        """An existing table gains the column, backfilled to the empty string."""
        pytest.importorskip("pyarrow")
        import pyarrow as pa

        from medha.backends.lancedb import (
            _backfill_expression,
            _build_schema,
            _missing_fields,
            _row_to_result,
        )

        expected = _build_schema(3)
        older = pa.schema([f for f in expected if f.name != "metadata_json"])
        missing = _missing_fields(older, expected)

        assert [f.name for f in missing] == ["metadata_json"]
        assert _backfill_expression(missing[0]) == "''"
        assert _row_to_result({"id": "x", "metadata_json": ""}, 1.0).metadata == {}

    def test_azure(self):
        from medha.backends.azure_search import _doc_to_result, _entry_to_doc

        doc = _entry_to_doc(_entry())

        assert _doc_to_result(doc, 1.0).metadata == METADATA

    def test_azure_projects_the_field_on_reads(self):
        """A field stored but left out of the projection is never returned."""
        from medha.backends.azure_search import _SCALAR_FIELDS, _entry_to_doc

        stored = set(_entry_to_doc(_entry())) - {"vector"}

        assert stored <= set(_SCALAR_FIELDS)

    def test_weaviate(self):
        from medha.backends.weaviate import _entry_to_properties, _obj_to_result

        entry = _entry()
        props = _entry_to_properties(entry)
        obj = SimpleNamespace(uuid=entry.id, properties=props)

        assert _obj_to_result(obj, 1.0).metadata == METADATA

    def test_elasticsearch(self):
        from medha.backends.elasticsearch import _hit_to_cache_result, _index_properties

        # The source document as upsert() builds it.
        src = {"generated_query": "SELECT 1", "metadata": dict(METADATA)}

        assert _hit_to_cache_result("id", src, 1.0).metadata == METADATA
        assert _index_properties(3)["metadata"] == {"type": "flattened"}

    def test_redis(self):
        from medha.backends.redis_vector import _doc_to_result
        from medha.utils.metadata import dumps_metadata

        doc = SimpleNamespace(
            id="prefix:abc",
            generated_query="SELECT 1",
            metadata_json=dumps_metadata(METADATA),
        )

        assert _doc_to_result(doc, 1.0).metadata == METADATA

    def test_asyncpg(self):
        from medha.backends._asyncpg_mixin import _RESULT_COLUMNS, _row_to_cache_result
        from medha.utils.metadata import canonical_json

        # asyncpg returns jsonb as text unless a codec says otherwise.
        row = {
            "id": "x",
            "original_question": "q",
            "normalized_question": "q",
            "generated_query": "SELECT 1",
            "query_hash": "h",
            "metadata": canonical_json(METADATA),
        }

        assert _row_to_cache_result(row, score=1.0).metadata == METADATA
        assert "metadata" in _RESULT_COLUMNS

    def test_asyncpg_accepts_a_decoded_column(self):
        """A pool with a json codec registered hands back a dict instead."""
        from medha.backends._asyncpg_mixin import _row_to_cache_result

        row = {
            "id": "x",
            "original_question": "q",
            "normalized_question": "q",
            "generated_query": "SELECT 1",
            "query_hash": "h",
            "metadata": dict(METADATA),
        }

        assert _row_to_cache_result(row, score=1.0).metadata == METADATA


class TestEmptyMetadataRoundTrips:
    """An entry without metadata must read back as {}, never as a failure."""

    def test_chroma(self):
        from medha.backends.chroma import _entry_to_metadata, _meta_to_result

        stored = _entry_to_metadata(_entry(metadata={}))
        assert _meta_to_result("x", 1.0, stored).metadata == {}

    def test_lancedb(self):
        from medha.backends.lancedb import _entry_to_row, _row_to_result

        assert _row_to_result(_entry_to_row(_entry(metadata={})), 1.0).metadata == {}

    def test_azure(self):
        from medha.backends.azure_search import _doc_to_result, _entry_to_doc

        assert _doc_to_result(_entry_to_doc(_entry(metadata={})), 1.0).metadata == {}

    def test_weaviate(self):
        from medha.backends.weaviate import _entry_to_properties, _obj_to_result

        props = _entry_to_properties(_entry(metadata={}))
        assert _obj_to_result(SimpleNamespace(uuid="x", properties=props), 1.0).metadata == {}

    def test_redis(self):
        from medha.backends.redis_vector import _doc_to_result

        doc = SimpleNamespace(id="p:x", generated_query="SELECT 1", metadata_json="")
        assert _doc_to_result(doc, 1.0).metadata == {}

    def test_redis_document_written_before_the_field_existed(self):
        from medha.backends.redis_vector import _doc_to_result

        doc = SimpleNamespace(id="p:x", generated_query="SELECT 1")
        assert _doc_to_result(doc, 1.0).metadata == {}
