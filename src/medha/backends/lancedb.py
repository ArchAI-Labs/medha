"""LanceDBBackend — LanceDB vector storage backend."""

import contextlib
import inspect
import json
import logging
from datetime import datetime, timezone
from typing import Any

from medha.backends._escape import quote_sql_literal
from medha.exceptions import ConfigurationError, StorageError, StorageInitializationError
from medha.interfaces.storage import VectorStorageBackend
from medha.types import CacheEntry, CacheResult, MetadataDict, PersistedStats
from medha.utils.metadata import (
    dumps_metadata,
    filter_fetch_size,
    loads_metadata,
    split_filters,
    verify_filters,
)

logger = logging.getLogger(__name__)

# pyarrow is imported separately from lancedb, though the backend needs both.
# The schema helpers below are pure functions over pyarrow schemas, and keeping
# their dependency independent means they stay importable — and testable —
# wherever pyarrow is present, without a lancedb install or any storage.
try:
    import pyarrow as pa
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

try:
    import lancedb
    HAS_LANCEDB = True
except ImportError:
    HAS_LANCEDB = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_meta_schema() -> "pa.Schema":
    return pa.schema([
        pa.field("id", pa.string()),
        pa.field("stats_json", pa.string()),
    ])


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
        pa.field("feedback_correct", pa.int64()),
        pa.field("feedback_incorrect", pa.int64()),
        pa.field("created_at", pa.string()),
        pa.field("expires_at", pa.string()),
        # A JSON string, not a struct: LanceDB fixes the schema at creation and
        # metadata keys are only known per entry, so a struct would need a
        # schema migration on every new key.
        pa.field("metadata_json", pa.string()),
    ])


def _missing_fields(existing: "pa.Schema", expected: "pa.Schema") -> list["pa.Field"]:
    """Fields of *expected* that *existing* does not carry.

    ``create_table(..., exist_ok=True)`` opens a table that is already there and
    ignores the schema it was handed, so a table written by an older version
    keeps exactly the columns it was created with. Every column added to
    ``_build_schema`` since then is absent, and the first upsert supplying one
    fails on the whole batch.

    Matching on name alone is enough: columns are only ever appended to the
    schema, never retyped. A table carrying *extra* columns — written by a
    newer version — is left alone.
    """
    have = set(existing.names)
    return [field for field in expected if field.name not in have]


def _backfill_expression(field: "pa.Field") -> str:
    """SQL literal LanceDB stores in *field* for rows that predate it.

    ``add_columns`` takes an expression per column rather than a default, so
    each one needs the value its reader already assumes for a missing column:
    ``_row_to_result`` coerces counters through ``int(... or 0)`` and text
    through ``... or None``, so zero and the empty string keep old rows
    reading exactly as they do now.
    """
    if pa.types.is_integer(field.type):
        return "0"
    if pa.types.is_floating(field.type):
        return "0.0"
    return "''"


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
        "feedback_correct": entry.feedback_correct,
        "feedback_incorrect": entry.feedback_incorrect,
        "created_at": entry.created_at.isoformat() if entry.created_at else "",
        "expires_at": entry.expires_at.isoformat() if entry.expires_at else "",
        "metadata_json": dumps_metadata(entry.metadata),
    }


def _row_to_result(row: dict[str, Any], score: float) -> CacheResult:
    expires_at = None
    if row.get("expires_at"):
        with contextlib.suppress(ValueError, TypeError):
            expires_at = datetime.fromisoformat(row["expires_at"])
    created_at = None
    if row.get("created_at"):
        with contextlib.suppress(ValueError, TypeError):
            created_at = datetime.fromisoformat(row["created_at"])
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
        feedback_correct=int(row.get("feedback_correct") or 0),
        feedback_incorrect=int(row.get("feedback_incorrect") or 0),
        created_at=created_at,
        expires_at=expires_at,
        metadata=loads_metadata(row.get("metadata_json")),
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

    supports_metadata = True

    def __init__(self, settings: Any = None) -> None:
        if not (HAS_LANCEDB and HAS_PYARROW):
            missing = " and ".join(
                name for name, present in (("lancedb>=0.6", HAS_LANCEDB), ("pyarrow", HAS_PYARROW))
                if not present
            )
            raise ConfigurationError(
                f"lancedb backend requires {missing}. "
                "Install with: pip install medha-archai[lancedb]"
            )
        from medha.config import Settings
        self._settings = settings or Settings()
        self._db: Any = None
        self._tables: dict[str, Any] = {}
        self._dimensions: dict[str, int] = {}
        self._meta_table: Any = None

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
            await self._reconcile_schema(table, table_name, schema)
            self._tables[collection_name] = table
        except StorageInitializationError:
            raise
        except Exception as e:
            raise StorageInitializationError(
                f"Failed to initialize LanceDB table '{table_name}': {e}"
            ) from e

    async def _reconcile_schema(
        self, table: Any, table_name: str, expected: "pa.Schema"
    ) -> None:
        """Add the columns a table created by an older version is missing.

        ``initialize()`` is idempotent about a table's *existence* but was not
        about its *shape*: it handed ``create_table`` the current schema and
        ``exist_ok=True`` silently ignored it. The mismatch then surfaced at the
        first upsert, far from its cause and with the whole batch failing.

        Failing here instead, naming the columns, is the floor. Adding them is
        the repair.
        """
        existing = table.schema
        # The async table exposes schema() as a coroutine method; the sync one
        # exposes it as a property. Accept either rather than pinning a shape.
        if callable(existing):
            existing = existing()
            if inspect.isawaitable(existing):
                existing = await existing

        missing = _missing_fields(existing, expected)
        if not missing:
            return

        names = [field.name for field in missing]
        try:
            await table.add_columns(
                {field.name: _backfill_expression(field) for field in missing}
            )
        except Exception as e:
            raise StorageInitializationError(
                f"LanceDB table '{table_name}' was created by an older version of "
                f"medha and is missing {names}. Adding the columns automatically "
                f"failed ({e}). Add them by hand or recreate the table before using "
                f"this collection — otherwise every upsert that supplies one of them "
                f"fails."
            ) from e

        logger.info(
            "Added %s to LanceDB table '%s', created by an older version",
            names,
            table_name,
        )

    async def close(self) -> None:
        self._tables.clear()
        self._dimensions.clear()
        self._meta_table = None
        self._db = None

    # ------------------------------------------------------------------
    # Stats persistence
    # ------------------------------------------------------------------

    def _meta_table_name(self) -> str:
        """Name of the shared stats table, one row per collection.

        Routed through ``_table_name`` so it picks up the same prefix and
        sanitiser as the data tables. ``meta`` is therefore reserved: a
        collection literally named ``meta`` would map to the same table.
        """
        return self._table_name("meta")

    async def _get_meta_table(self) -> Any:
        if self._db is None:
            raise StorageError("Not connected. Call connect() first.")
        if self._meta_table is None:
            self._meta_table = await self._db.create_table(
                self._meta_table_name(), schema=_build_meta_schema(), exist_ok=True
            )
        return self._meta_table

    async def load_stats(self, collection_name: str) -> PersistedStats | None:
        try:
            table = await self._get_meta_table()
            rows: list[dict[str, Any]] = await (
                table.query()
                .where(f"id = '{quote_sql_literal(collection_name)}'")
                .limit(1)
                .to_list()
            )
        except StorageError:
            raise
        except Exception as e:
            raise StorageError(
                f"LanceDB load_stats failed on '{collection_name}': {e}"
            ) from e

        if not rows:
            return None
        raw = rows[0].get("stats_json")
        if not raw:
            return None
        try:
            return PersistedStats.model_validate_json(raw)
        except Exception as e:
            raise StorageError(
                f"LanceDB load_stats failed to parse stats for '{collection_name}': {e}"
            ) from e

    async def save_stats(self, collection_name: str, stats: PersistedStats) -> None:
        try:
            table = await self._get_meta_table()
            await (
                table.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute([{
                    "id": collection_name,
                    "stats_json": stats.model_dump_json(),
                }])
            )
        except StorageError:
            raise
        except Exception as e:
            raise StorageError(
                f"LanceDB save_stats failed on '{collection_name}': {e}"
            ) from e

    async def update_feedback(self, collection_name: str, point_id: str, correct: bool) -> int:
        table = self._get_table(collection_name)
        safe_id = quote_sql_literal(point_id)
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

    @staticmethod
    def _build_where(pushable: MetadataDict) -> str:
        """The ``where`` expression: not expired, and narrowed by *pushable*.

        The TTL test is parenthesised. It is an ``OR``, and appending
        ``AND <metadata>`` to a bare ``a OR b`` would bind as
        ``a OR (b AND metadata)`` — entries with no expiry would slip past the
        metadata condition entirely.
        """
        clause = f"(expires_at = '' OR expires_at > '{_now_iso()}')"
        for key, value in pushable.items():
            # metadata_json holds canonical JSON: sorted keys, no spaces. So a
            # matching row contains this exact pair as a substring, and the
            # encoder here has to be the one that wrote it — hence
            # ensure_ascii=False, matching canonical_json.
            pair = (
                f"{json.dumps(key, ensure_ascii=False)}:"
                f"{json.dumps(value, ensure_ascii=False)}"
            )
            clause += f" AND metadata_json LIKE '%{quote_sql_literal(pair)}%'"
        return clause

    async def search(
        self,
        collection_name: str,
        vector: list[float],
        limit: int = 5,
        score_threshold: float = 0.0,
    ) -> list[CacheResult]:
        return await self.search_filtered(
            collection_name, vector, limit, score_threshold, filters=None
        )

    async def search_filtered(
        self,
        collection_name: str,
        vector: list[float],
        limit: int = 5,
        score_threshold: float = 0.0,
        filters: MetadataDict | None = None,
        overfetch: int = 10,
    ) -> list[CacheResult]:
        """Search, narrowing the scan by a substring test on the metadata JSON.

        LanceDB has no JSON accessor to filter on, so the pushdown is a
        ``LIKE`` over the encoded blob. That makes it a *narrowing* step rather
        than a decision: it never hides a row that matches — a matching row
        contains the pair verbatim — but a value carrying a ``%`` or a ``_``
        matches more rows than it should, since those are LIKE wildcards.

        So the fetch is widened whenever any filter is present, not only when
        something is left over for Python: a false positive ranked above a true
        match must not be able to fill up ``limit``. The Python pass then has
        the last word, as everywhere else.

        Only strings are pushed down. A number's JSON spelling is not unique
        enough to match on — ``10`` and ``10.0`` are the same value to
        ``metadata_matches`` and different substrings here.

        Raises:
            StorageError: If the search fails.
        """
        table = self._get_table(collection_name)
        pushable, _ = split_filters(filters)
        fetch = filter_fetch_size(limit, filters or {}, overfetch)
        metric: str = self._settings.lancedb_metric
        try:
            rows: list[dict[str, Any]] = await (
                table.vector_search(vector)
                .distance_type(metric)
                .where(self._build_where(pushable))
                .limit(fetch)
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
        return verify_filters(out, filters, limit)

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
            "feedback_correct", "feedback_incorrect",
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
        safe_ids = ", ".join(f"'{quote_sql_literal(id_)}'" for id_ in ids)
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
        safe_q = quote_sql_literal(normalized_question)
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
        safe_hash = quote_sql_literal(query_hash)
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
        safe_tid = quote_sql_literal(template_id)
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
