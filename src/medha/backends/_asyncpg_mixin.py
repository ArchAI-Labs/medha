"""Shared asyncpg helpers for PostgreSQL-based backends."""

import re
import uuid
from typing import Any

from medha.exceptions import StorageError
from medha.types import CacheResult


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
        created_at=row.get("created_at"),
        expires_at=row.get("expires_at"),
    )


class _AsyncpgBackendMixin:
    """Mixin with common asyncpg operations shared between PgVector and VectorChord backends."""

    _pool: Any
    _initialized_tables: set[str]

    def _table_name(self, collection_name: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_]", "_", collection_name)
        return f"{self._settings.pg_table_prefix}_{safe}"

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
            SELECT id::text, original_question, normalized_question, generated_query,
                   query_hash, response_summary, template_id, usage_count, created_at{vector_col}
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
            SELECT id::text, original_question, normalized_question, generated_query,
                   query_hash, response_summary, template_id, usage_count, created_at
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
            SELECT id::text, original_question, normalized_question, generated_query,
                   query_hash, response_summary, template_id, usage_count, created_at, expires_at
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
