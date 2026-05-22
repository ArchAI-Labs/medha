"""LanceDBBackend — LanceDB vector storage backend."""

import logging
from datetime import datetime, timezone
from typing import Any

from medha.exceptions import ConfigurationError, StorageError, StorageInitializationError
from medha.interfaces.storage import VectorStorageBackend
from medha.types import CacheEntry, CacheResult

logger = logging.getLogger(__name__)

try:
    import lancedb
    import pyarrow as pa
    HAS_LANCEDB = True
except ImportError:
    HAS_LANCEDB = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_schema(dimension: int) -> "pa.Schema":
    return pa.schema([
        pa.field("id", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), dimension)),
        pa.field("original_question", pa.string()),
        pa.field("normalized_question", pa.string()),
        pa.field("generated_query", pa.string()),
        pa.field("query_hash", pa.string()),
        pa.field("response_summary", pa.string()),
        pa.field("template_id", pa.string()),
        pa.field("usage_count", pa.int64()),
        pa.field("created_at", pa.string()),
        pa.field("expires_at", pa.string()),
    ])


def _entry_to_row(entry: CacheEntry) -> dict[str, Any]:
    return {
        "id": entry.id,
        "vector": entry.vector,
        "original_question": entry.original_question,
        "normalized_question": entry.normalized_question,
        "generated_query": entry.generated_query,
        "query_hash": entry.query_hash,
        "response_summary": entry.response_summary or "",
        "template_id": entry.template_id or "",
        "usage_count": entry.usage_count,
        "created_at": entry.created_at.isoformat() if entry.created_at else "",
        "expires_at": entry.expires_at.isoformat() if entry.expires_at else "",
    }


def _row_to_result(row: dict[str, Any], score: float) -> CacheResult:
    expires_at = None
    if row.get("expires_at"):
        try:
            expires_at = datetime.fromisoformat(row["expires_at"])
        except (ValueError, TypeError):
            pass
    created_at = None
    if row.get("created_at"):
        try:
            created_at = datetime.fromisoformat(row["created_at"])
        except (ValueError, TypeError):
            pass
    return CacheResult(
        id=row["id"],
        score=max(0.0, min(1.0, score)),
        original_question=row.get("original_question", ""),
        normalized_question=row.get("normalized_question", ""),
        generated_query=row.get("generated_query", ""),
        query_hash=row.get("query_hash", ""),
        response_summary=row.get("response_summary") or None,
        template_id=row.get("template_id") or None,
        usage_count=int(row.get("usage_count", 0)),
        created_at=created_at,
        expires_at=expires_at,
    )


def _distance_to_score(distance: float, metric: str) -> float:
    if metric == "cosine":
        # LanceDB cosine distance = 1 - cosine_similarity, range [0, 2]
        return 1.0 - distance
    if metric == "l2":
        # For unit-normalized vectors: L2² = 2*(1 - cosine_sim)
        # → cosine_sim = 1 - L2²/2, which keeps scores comparable across metrics
        return max(0.0, 1.0 - (distance ** 2) / 2.0)
    # dot: LanceDB stores negative dot product as distance; negate and clamp
    return max(0.0, -distance)


class LanceDBBackend(VectorStorageBackend):
    """LanceDB vector backend. Supports local, S3, GCS, and Azure storage.

    Uses the native async API (lancedb.connect_async) for non-blocking I/O.
    Local mode requires no external services; cloud URIs (s3://, gs://, az://)
    require the appropriate credentials to be set in the environment.
    """

    def __init__(self, settings: Any = None) -> None:
        if not HAS_LANCEDB:
            raise ConfigurationError(
                "lancedb backend requires 'lancedb>=0.6'. "
                "Install with: pip install medha-archai[lancedb]"
            )
        from medha.config import Settings
        self._settings = settings or Settings()
        self._db: Any = None
        self._tables: dict[str, Any] = {}
        self._dimensions: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        uri = self._settings.lancedb_uri
        try:
            self._db = await lancedb.connect_async(uri)
        except Exception as e:
            raise StorageInitializationError(
                f"Failed to connect to LanceDB at '{uri}': {e}"
            ) from e

    async def initialize(self, collection_name: str, dimension: int, **kwargs: Any) -> None:
        if self._db is None:
            raise StorageError("Not connected. Call connect() first.")
        if collection_name in self._tables:
            return
        table_name = self._table_name(collection_name)
        schema = _build_schema(dimension)
        self._dimensions[collection_name] = dimension
        try:
            table = await self._db.create_table(table_name, schema=schema, exist_ok=True)
            self._tables[collection_name] = table
        except Exception as e:
            raise StorageInitializationError(
                f"Failed to initialize LanceDB table '{table_name}': {e}"
            ) from e

    async def close(self) -> None:
        self._tables.clear()
        self._dimensions.clear()
        self._db = None

    async def update_feedback(self, collection_name: str, point_id: str, correct: bool) -> int:
        table = self._get_table(collection_name)
        safe_id = point_id.replace("'", "''")
        try:
            rows: list[dict[str, Any]] = await (
                table.query()
                .where(f"id = '{safe_id}'")
                .limit(1)
                .to_list()
            )
        except Exception as e:
            raise StorageError(
                f"LanceDB update_feedback failed on '{collection_name}': {e}"
            ) from e
        if not rows:
            return 0
        field = "feedback_correct" if correct else "feedback_incorrect"
        current = int(rows[0].get(field, 0)) if field in rows[0] else 0
        new_val = current + 1
        try:
            await table.update(where=f"id = '{safe_id}'", values={field: new_val})
        except Exception as e:
            raise StorageError(
                f"LanceDB update_feedback update failed on '{collection_name}': {e}"
            ) from e
        return new_val

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _table_name(self, collection_name: str) -> str:
        import re
        prefix = self._settings.lancedb_table_prefix
        safe = re.sub(r"[^a-zA-Z0-9_]", "_", collection_name)
        return f"{prefix}_{safe}" if prefix else safe

    def _get_table(self, collection_name: str) -> Any:
        tbl = self._tables.get(collection_name)
        if tbl is None:
            raise StorageError(
                f"Collection '{collection_name}' not initialized. Call initialize() first."
            )
        return tbl

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def search(
        self,
        collection_name: str,
        vector: list[float],
        limit: int = 5,
        score_threshold: float = 0.0,
    ) -> list[CacheResult]:
        table = self._get_table(collection_name)
        now_iso = _now_iso()
        where = f"expires_at = '' OR expires_at > '{now_iso}'"
        metric: str = self._settings.lancedb_metric
        try:
            rows: list[dict[str, Any]] = await (
                table.vector_search(vector)
                .distance_type(metric)
                .where(where)
                .limit(limit)
                .to_list()
            )
        except Exception as e:
            raise StorageError(f"LanceDB search failed on '{collection_name}': {e}") from e

        out: list[CacheResult] = []
        for row in rows:
            score = _distance_to_score(float(row.get("_distance", 0.0)), metric)
            score = max(0.0, min(1.0, score))
            if score >= score_threshold:
                out.append(_row_to_result(row, score))
        return out

    async def upsert(self, collection_name: str, entries: list[CacheEntry]) -> None:
        if not entries:
            return
        table = self._get_table(collection_name)
        rows = [_entry_to_row(e) for e in entries]
        try:
            await (
                table.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(rows)
            )
        except Exception as e:
            raise StorageError(f"LanceDB upsert failed on '{collection_name}': {e}") from e

    async def scroll(
        self,
        collection_name: str,
        limit: int = 100,
        offset: str | None = None,
        with_vectors: bool = False,
    ) -> tuple[list[CacheResult], str | None]:
        table = self._get_table(collection_name)
        int_offset = int(offset) if offset else 0
        columns = None if with_vectors else [
            "id", "original_question", "normalized_question", "generated_query",
            "query_hash", "response_summary", "template_id", "usage_count",
            "created_at", "expires_at",
        ]
        try:
            q = table.query().limit(limit).offset(int_offset)
            if columns is not None:
                q = q.select(columns)
            rows: list[dict[str, Any]] = await q.to_list()
        except Exception as e:
            raise StorageError(f"LanceDB scroll failed on '{collection_name}': {e}") from e

        next_offset = str(int_offset + len(rows)) if len(rows) == limit else None
        return [_row_to_result(row, 1.0) for row in rows], next_offset

    async def count(self, collection_name: str) -> int:
        table = self._get_table(collection_name)
        try:
            return await table.count_rows()
        except Exception as e:
            raise StorageError(f"LanceDB count failed on '{collection_name}': {e}") from e

    async def delete(self, collection_name: str, ids: list[str]) -> None:
        if not ids:
            return
        table = self._get_table(collection_name)
        safe_ids = ", ".join(f"'{id_.replace(chr(39), chr(39) * 2)}'" for id_ in ids)
        try:
            await table.delete(f"id IN ({safe_ids})")
        except Exception as e:
            raise StorageError(f"LanceDB delete failed on '{collection_name}': {e}") from e

    async def find_expired(self, collection_name: str) -> list[str]:
        table = self._get_table(collection_name)
        now_iso = _now_iso()
        try:
            rows: list[dict[str, Any]] = await (
                table.query()
                .where(f"expires_at != '' AND expires_at < '{now_iso}'")
                .select(["id"])
                .to_list()
            )
        except Exception as e:
            raise StorageError(f"LanceDB find_expired failed on '{collection_name}': {e}") from e
        return [row["id"] for row in rows]

    async def search_by_normalized_question(
        self, collection_name: str, normalized_question: str
    ) -> CacheResult | None:
        table = self._get_table(collection_name)
        safe_q = normalized_question.replace("'", "''")
        try:
            rows: list[dict[str, Any]] = await (
                table.query()
                .where(f"normalized_question = '{safe_q}'")
                .limit(1)
                .to_list()
            )
        except Exception as e:
            raise StorageError(
                f"LanceDB search_by_normalized_question failed on '{collection_name}': {e}"
            ) from e
        return _row_to_result(rows[0], 1.0) if rows else None

    async def find_by_query_hash(self, collection_name: str, query_hash: str) -> list[str]:
        table = self._get_table(collection_name)
        safe_hash = query_hash.replace("'", "''")
        try:
            rows: list[dict[str, Any]] = await (
                table.query()
                .where(f"query_hash = '{safe_hash}'")
                .select(["id"])
                .to_list()
            )
        except Exception as e:
            raise StorageError(
                f"LanceDB find_by_query_hash failed on '{collection_name}': {e}"
            ) from e
        return [row["id"] for row in rows]

    async def find_by_template_id(self, collection_name: str, template_id: str) -> list[str]:
        table = self._get_table(collection_name)
        safe_tid = template_id.replace("'", "''")
        try:
            rows: list[dict[str, Any]] = await (
                table.query()
                .where(f"template_id = '{safe_tid}'")
                .select(["id"])
                .to_list()
            )
        except Exception as e:
            raise StorageError(
                f"LanceDB find_by_template_id failed on '{collection_name}': {e}"
            ) from e
        return [row["id"] for row in rows]

    async def drop_collection(self, collection_name: str) -> None:
        if self._db is None:
            raise StorageError("Not connected. Call connect() first.")
        table_name = self._table_name(collection_name)
        try:
            await self._db.drop_table(table_name)
            self._tables.pop(collection_name, None)
            self._dimensions.pop(collection_name, None)
        except Exception as e:
            raise StorageError(
                f"LanceDB drop_collection failed on '{collection_name}': {e}"
            ) from e
