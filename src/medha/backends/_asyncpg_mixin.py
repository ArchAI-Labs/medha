"""Shared asyncpg helpers for PostgreSQL-based backends."""

import re
import uuid
from typing import Any

from medha.exceptions import StorageError
from medha.types import CacheResult, MetadataDict, PersistedStats
from medha.utils.metadata import (
    canonical_json,
    filter_fetch_size,
    loads_metadata,
    split_filters,
    verify_filters,
)

# Columns every read path selects. Kept in one place so a column added to the
# schema reaches search, scroll and the lookups together — a SELECT that
# forgets one silently returns entries with that field at its default.
_RESULT_COLUMNS = (
    "id::text, original_question, normalized_question, generated_query, "
    "query_hash, response_summary, template_id, usage_count, "
    "feedback_correct, feedback_incorrect, created_at, metadata"
)

# Fixed literal, never derived from user input. The only identifier
# interpolated into the stats SQL is the schema, which Settings validates
# against _SAFE_IDENTIFIER_RE; collection_name and stats_json always travel as
# bind parameters ($1/$2).
_STATS_TABLE = "_medha_stats"

# PostgreSQL SQLSTATE 42P01 (undefined_table): the stats table has never been
# created, so no stats were ever saved -> load_stats returns None.
_UNDEFINED_TABLE = "42P01"


def _row_to_cache_result(row: Any, score: float | None = None) -> CacheResult:
    row_score = score if score is not None else max(0.0, min(1.0, float(row["score"])))
    return CacheResult(
        id=row["id"],
        score=max(0.0, min(1.0, row_score)),
        original_question=row["original_question"],
        normalized_question=row["normalized_question"],
        generated_query=row["generated_query"],
        query_hash=row["query_hash"],
        response_summary=row.get("response_summary"),
        template_id=row.get("template_id"),
        usage_count=row.get("usage_count", 0),
        feedback_correct=row.get("feedback_correct") or 0,
        feedback_incorrect=row.get("feedback_incorrect") or 0,
        created_at=row.get("created_at"),
        expires_at=row.get("expires_at"),
        # asyncpg hands back jsonb as a string unless a codec says otherwise.
        metadata=loads_metadata(row.get("metadata")),
    )


class _AsyncpgBackendMixin:
    """Mixin with common asyncpg operations shared between PgVector and VectorChord backends."""

    _pool: Any
    _initialized_tables: set[str]
    _settings: Any

    def _table_name(self, collection_name: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_]", "_", collection_name)
        return f"{self._settings.pg_table_prefix}_{safe}"

    def _vector_search_sql(
        self,
        collection_name: str,
        vector: list[float],
        score_threshold: float,
        limit: int,
        pushable: MetadataDict | None = None,
    ) -> tuple[str, list[Any]]:
        """The KNN query both PostgreSQL backends run, and its bind parameters.

        Identical between pgvector and VectorChord: they differ in the index
        that answers the ``ORDER BY``, not in the statement. Built in one place
        so a column or predicate cannot reach one of them and not the other.

        *pushable* adds a JSONB containment test. Containment is the right
        operator here: it asks whether the row's metadata includes these pairs
        and ignores the keys the row carries beyond them — exactly what
        ``metadata_matches`` means. It also answers from a GIN index, which
        :meth:`metadata_index_sql` hands to whoever wants one.
        """
        schema = self._settings.pg_schema
        table = self._table_name(collection_name)
        params: list[Any] = [vector, score_threshold]

        metadata_clause = ""
        if pushable:
            params.append(canonical_json(pushable))
            metadata_clause = f"AND metadata @> ${len(params)}::jsonb"

        params.append(limit)
        sql = f"""
            SELECT
                {_RESULT_COLUMNS},
                expires_at,
                (1 - (vector <=> $1::vector))::float AS score
            FROM {schema}.{table}
            WHERE (1 - (vector <=> $1::vector)) >= $2
              AND (expires_at IS NULL OR expires_at > NOW())
              {metadata_clause}
            ORDER BY vector <=> $1::vector
            LIMIT ${len(params)}
        """
        return sql, params

    async def search_filtered(
        self,
        collection_name: str,
        vector: list[float],
        limit: int = 5,
        score_threshold: float = 0.0,
        filters: MetadataDict | None = None,
        overfetch: int = 10,
    ) -> list[CacheResult]:
        """Search with the string-valued constraints pushed into the WHERE clause.

        Shared by pgvector and VectorChord, which run the same statement.

        Only the part of the filter :func:`split_filters` calls pushable goes
        into SQL; whatever is left is applied in Python, and the fetch widens
        to compensate. Note that neither of these backends is covered by the
        test suite — their integration tests skip unless a PostgreSQL instance
        is configured, and CI runs no service containers — so the pushdown is
        deliberately restricted to the case that cannot be misread.

        Raises:
            StorageError: If the search fails.
        """
        if self._pool is None:
            raise StorageError("Not connected. Call connect() first.")

        pushable, residual = split_filters(filters)
        sql, params = self._vector_search_sql(
            collection_name,
            vector,
            score_threshold,
            filter_fetch_size(limit, residual, overfetch),
            pushable,
        )
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql, *params)
        except Exception as e:
            raise StorageError(
                f"asyncpg filtered search failed on '{collection_name}': {e}"
            ) from e

        return verify_filters([_row_to_cache_result(row) for row in rows], filters, limit)

    def _metadata_ddl(self, table: str) -> str:
        """DDL adding the metadata column to an existing table.

        Executed on every ``initialize()``, not only at creation: a table
        written by an earlier version is missing the column, and the first
        upsert supplying it would fail the whole batch. ``IF NOT EXISTS`` makes
        it idempotent.

        JSONB rather than TEXT so the filter can go down as
        ``metadata @> $n::jsonb``. From PostgreSQL 11 a column added with a
        default is a catalog-only change, so this costs nothing however large
        the table is.

        No index is built here, deliberately — see
        :meth:`metadata_index_sql`.
        """
        return f"""
            ALTER TABLE {self._settings.pg_schema}.{table}
                ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
        """

    def metadata_index_sql(self, collection_name: str) -> str:
        """The GIN index statement for a collection, for an operator to run.

        ``metadata @> ...`` works without it — the planner filters the rows the
        vector index produced — and it only becomes worth having once a
        collection holds enough entries that actually carry metadata.

        It is not created at startup on purpose. A plain ``CREATE INDEX`` locks
        the table against writes for the whole build, which an upgrade would
        have imposed on every existing deployment to build an index over a
        column that is ``{}`` in every row written before the upgrade —
        a stall in exchange for nothing.

        So it is handed over instead. ``CONCURRENTLY`` keeps writes running
        while it builds; it cannot run inside a transaction block, and a failed
        build leaves an invalid index behind that must be dropped before
        retrying, which is exactly the kind of thing to do deliberately rather
        than during someone else's deploy::

            print(backend.metadata_index_sql("my_cache"))

        Args:
            collection_name: The collection whose table to index.

        Returns:
            A single SQL statement, ready to run.
        """
        table = self._table_name(collection_name)
        return (
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {table}_metadata_gin_idx "
            f"ON {self._settings.pg_schema}.{table} USING gin (metadata);"
        )

    def _stats_table_ddl(self) -> str:
        """DDL for the shared stats table, executed from ``initialize()``.

        One row per collection, keyed by ``collection_name``. Kept out of the
        vector tables so a ``DROP TABLE`` of a collection cannot take the
        metadata of the others with it.
        """
        return f"""
            CREATE TABLE IF NOT EXISTS {self._settings.pg_schema}.{_STATS_TABLE} (
                collection_name TEXT PRIMARY KEY,
                stats_json      TEXT NOT NULL,
                updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """

    async def load_stats(self, collection_name: str) -> PersistedStats | None:
        if self._pool is None:
            raise StorageError("Not connected. Call connect() first.")

        schema = self._settings.pg_schema
        sql = (
            f"SELECT stats_json FROM {schema}.{_STATS_TABLE}"
            f" WHERE collection_name = $1"
        )
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(sql, collection_name)
        except Exception as e:
            if getattr(e, "sqlstate", None) == _UNDEFINED_TABLE:
                return None
            raise StorageError(
                f"asyncpg load_stats failed on '{collection_name}': {e}"
            ) from e

        if row is None or row["stats_json"] is None:
            return None
        try:
            return PersistedStats.model_validate_json(row["stats_json"])
        except Exception as e:
            raise StorageError(
                f"asyncpg load_stats failed to parse stats for '{collection_name}': {e}"
            ) from e

    async def save_stats(self, collection_name: str, stats: PersistedStats) -> None:
        if self._pool is None:
            raise StorageError("Not connected. Call connect() first.")

        schema = self._settings.pg_schema
        sql = f"""
            INSERT INTO {schema}.{_STATS_TABLE} (collection_name, stats_json)
            VALUES ($1, $2)
            ON CONFLICT (collection_name) DO UPDATE
                SET stats_json = $2,
                    updated_at = now()
        """
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(sql, collection_name, stats.model_dump_json())
        except Exception as e:
            raise StorageError(
                f"asyncpg save_stats failed on '{collection_name}': {e}"
            ) from e

    async def scroll(
        self,
        collection_name: str,
        limit: int = 100,
        offset: str | None = None,
        with_vectors: bool = False,
    ) -> tuple[list[CacheResult], str | None]:
        if self._pool is None:
            raise StorageError("Not connected. Call connect() first.")

        schema = self._settings.pg_schema
        table = self._table_name(collection_name)
        int_offset = int(offset) if offset is not None else 0

        vector_col = ", vector" if with_vectors else ""
        sql = f"""
            SELECT {_RESULT_COLUMNS}{vector_col}
            FROM {schema}.{table}
            ORDER BY created_at ASC, id ASC
            LIMIT $1 OFFSET $2
        """
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql, limit, int_offset)
        except Exception as e:
            raise StorageError(f"asyncpg operation failed on '{collection_name}': {e}") from e

        results = [_row_to_cache_result(row, score=1.0) for row in rows]
        next_offset = str(int_offset + limit) if len(rows) == limit else None
        return results, next_offset

    async def count(self, collection_name: str) -> int:
        if self._pool is None:
            raise StorageError("Not connected. Call connect() first.")

        schema = self._settings.pg_schema
        table = self._table_name(collection_name)

        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(f"SELECT COUNT(*) FROM {schema}.{table}")
                return int(row[0])
        except Exception as e:
            raise StorageError(f"asyncpg operation failed on '{collection_name}': {e}") from e

    async def delete(self, collection_name: str, ids: list[str]) -> None:
        if self._pool is None:
            raise StorageError("Not connected. Call connect() first.")
        if not ids:
            return

        schema = self._settings.pg_schema
        table = self._table_name(collection_name)
        uuid_ids = [uuid.UUID(id_) for id_ in ids]

        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    f"DELETE FROM {schema}.{table} WHERE id = ANY($1::uuid[])",
                    uuid_ids,
                )
        except Exception as e:
            raise StorageError(f"asyncpg operation failed on '{collection_name}': {e}") from e

    async def search_by_query_hash(
        self, collection_name: str, query_hash: str
    ) -> CacheResult | None:
        if self._pool is None:
            raise StorageError("Not connected. Call connect() first.")

        schema = self._settings.pg_schema
        table = self._table_name(collection_name)

        sql = f"""
            SELECT {_RESULT_COLUMNS}
            FROM {schema}.{table}
            WHERE query_hash = $1
            LIMIT 1
        """
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(sql, query_hash)
        except Exception as e:
            raise StorageError(f"asyncpg operation failed on '{collection_name}': {e}") from e

        if row is None:
            return None
        return _row_to_cache_result(row, score=1.0)

    async def update_usage_count(self, collection_name: str, point_id: str) -> None:
        if self._pool is None:
            raise StorageError("Not connected. Call connect() first.")

        import logging
        logger = logging.getLogger(__name__)

        schema = self._settings.pg_schema
        table = self._table_name(collection_name)

        try:
            async with self._pool.acquire() as conn:
                result = await conn.execute(
                    f"UPDATE {schema}.{table} SET usage_count = usage_count + 1 WHERE id = $1::uuid",
                    uuid.UUID(point_id),
                )
        except Exception as e:
            raise StorageError(f"asyncpg operation failed on '{collection_name}': {e}") from e

        updated = int(result.split()[-1]) if result else 0
        if updated == 0:
            logger.warning(
                "update_usage_count: id '%s' not found in collection '%s'",
                point_id,
                collection_name,
            )

    async def update_feedback(self, collection_name: str, point_id: str, correct: bool) -> int:
        if self._pool is None:
            raise StorageError("Not connected. Call connect() first.")

        import logging
        logger = logging.getLogger(__name__)

        schema = self._settings.pg_schema
        table = self._table_name(collection_name)
        col = "feedback_correct" if correct else "feedback_incorrect"

        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    f"UPDATE {schema}.{table}"
                    f" SET {col} = COALESCE({col}, 0) + 1"
                    f" WHERE id = $1::uuid"
                    f" RETURNING {col}",
                    uuid.UUID(point_id),
                )
        except Exception as e:
            raise StorageError(f"asyncpg operation failed on '{collection_name}': {e}") from e

        if row is None:
            logger.warning(
                "update_feedback: id '%s' not found in collection '%s'",
                point_id,
                collection_name,
            )
            return 0
        return int(row[col])

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            self._initialized_tables.clear()

    async def find_by_query_hash(
        self, collection_name: str, query_hash: str
    ) -> list[str]:
        if self._pool is None:
            raise StorageError("Not connected. Call connect() first.")
        schema = self._settings.pg_schema
        table = self._table_name(collection_name)
        sql = f"SELECT id::text FROM {schema}.{table} WHERE query_hash = $1"
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql, query_hash)
            return [row["id"] for row in rows]
        except Exception as e:
            raise StorageError(f"asyncpg operation failed on '{collection_name}': {e}") from e

    async def find_by_template_id(
        self, collection_name: str, template_id: str
    ) -> list[str]:
        if self._pool is None:
            raise StorageError("Not connected. Call connect() first.")
        schema = self._settings.pg_schema
        table = self._table_name(collection_name)
        sql = f"SELECT id::text FROM {schema}.{table} WHERE template_id = $1"
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql, template_id)
            return [row["id"] for row in rows]
        except Exception as e:
            raise StorageError(f"asyncpg operation failed on '{collection_name}': {e}") from e

    async def search_by_normalized_question(
        self, collection_name: str, normalized_question: str
    ) -> CacheResult | None:
        if self._pool is None:
            raise StorageError("Not connected. Call connect() first.")
        schema = self._settings.pg_schema
        table = self._table_name(collection_name)
        sql = f"""
            SELECT {_RESULT_COLUMNS}, expires_at
            FROM {schema}.{table}
            WHERE normalized_question = $1
            LIMIT 1
        """
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(sql, normalized_question)
        except Exception as e:
            raise StorageError(f"asyncpg operation failed on '{collection_name}': {e}") from e
        if row is None:
            return None
        return _row_to_cache_result(row, score=1.0)

    async def drop_collection(self, collection_name: str) -> None:
        if self._pool is None:
            raise StorageError("Not connected. Call connect() first.")

        import logging
        logger = logging.getLogger(__name__)

        schema = self._settings.pg_schema
        table = self._table_name(collection_name)
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(f"DROP TABLE IF EXISTS {schema}.{table}")
            self._initialized_tables.discard(collection_name)
            logger.info("Dropped table '%s.%s'", schema, table)
        except Exception as e:
            raise StorageError(f"asyncpg drop_collection failed on '{collection_name}': {e}") from e
