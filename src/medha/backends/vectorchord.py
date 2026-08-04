"""VectorChordBackend — PostgreSQL + VectorChord (vchordrq) storage backend."""

import json
import logging
import uuid
from typing import Any

from medha.backends._asyncpg_mixin import _AsyncpgBackendMixin, _row_to_cache_result
from medha.exceptions import ConfigurationError, StorageError, StorageInitializationError
from medha.interfaces.storage import VectorStorageBackend
from medha.types import CacheEntry, CacheResult

logger = logging.getLogger(__name__)

try:
    import asyncpg
    HAS_VECTORCHORD = True
except ImportError:
    HAS_VECTORCHORD = False


def _encode_vector(v: list[float]) -> str:
    return "[" + ",".join(str(x) for x in v) + "]"


def _decode_vector(s: str) -> list[float]:
    return [float(x) for x in s.strip("[]").split(",")]


def _validate_vc_lists(vc_lists: Any) -> list[int]:
    """Validate ``vc_lists`` before it is interpolated into the vchordrq DDL.

    ``vc_lists`` is embedded into ``CREATE INDEX ... WITH (lists = '...')`` as a
    JSON literal. From :class:`Settings` it is already typed ``list[int]``, but
    the ``initialize(..., vc_lists=...)`` kwargs path bypasses Pydantic, so we
    reject anything that is not a list/tuple of plain integers. ``bool`` is an
    ``int`` subclass and is excluded; an empty list is allowed (vchordrq
    accepts it).
    """
    if not isinstance(vc_lists, (list, tuple)) or any(
        not isinstance(n, int) or isinstance(n, bool) for n in vc_lists
    ):
        raise StorageInitializationError(
            f"vc_lists must be a list of integers, got: {vc_lists!r}"
        )
    return list(vc_lists)


class VectorChordBackend(_AsyncpgBackendMixin, VectorStorageBackend):
    """PostgreSQL + VectorChord backend using vchordrq index with RaBitQ quantization.

    Drop-in replacement for PgVectorBackend. Requires asyncpg (no pgvector Python package).
    The vectorchord PostgreSQL extension must be installed in the database.
    """

    def __init__(self, settings: Any = None) -> None:
        if not HAS_VECTORCHORD:
            raise ConfigurationError(
                "vectorchord backend requires 'asyncpg'. "
                "Install with: pip install medha-archai[vectorchord]"
            )
        from medha.config import Settings
        self._settings = settings or Settings()
        self._pool: asyncpg.Pool | None = None
        self._initialized_tables: set[str] = set()

    async def _register_codecs(self, conn: Any) -> None:
        await conn.set_type_codec(
            "vector",
            encoder=_encode_vector,
            decoder=_decode_vector,
            schema="public",
            format="text",
        )

    async def connect(self) -> None:
        try:
            kwargs = dict(
                min_size=self._settings.pg_pool_min_size,
                max_size=self._settings.pg_pool_max_size,
                init=self._register_codecs,
            )
            if self._settings.pg_dsn:
                self._pool = await asyncpg.create_pool(dsn=self._settings.pg_dsn, **kwargs)
            else:
                self._pool = await asyncpg.create_pool(
                    host=self._settings.pg_host,
                    port=self._settings.pg_port,
                    database=self._settings.pg_database,
                    user=self._settings.pg_user,
                    password=self._settings.pg_password.get_secret_value(),
                    **kwargs,
                )
        except Exception as e:
            raise StorageInitializationError(f"Failed to connect to PostgreSQL: {e}") from e

    async def initialize(self, collection_name: str, dimension: int, **kwargs: Any) -> None:
        if self._pool is None:
            raise StorageError("Not connected. Call connect() first.")
        if collection_name in self._initialized_tables:
            return

        schema = self._settings.pg_schema
        table = self._table_name(collection_name)

        vc_lists = _validate_vc_lists(kwargs.get("vc_lists", self._settings.vc_lists))
        vc_residual = kwargs.get("vc_residual_quantization", self._settings.vc_residual_quantization)

        lists_sql = json.dumps(vc_lists)

        try:
            async with self._pool.acquire() as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vectorchord")

                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {schema}.{table} (
                        id                   UUID        PRIMARY KEY,
                        vector               vector({dimension}) NOT NULL,
                        original_question    TEXT NOT NULL DEFAULT '',
                        normalized_question  TEXT NOT NULL DEFAULT '',
                        generated_query      TEXT NOT NULL DEFAULT '',
                        query_hash           TEXT NOT NULL DEFAULT '',
                        response_summary     TEXT,
                        template_id          TEXT,
                        usage_count          INTEGER NOT NULL DEFAULT 1,
                        created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        expires_at           TIMESTAMPTZ
                    )
                """)

                residual_str = "true" if vc_residual else "false"
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {table}_vector_vchordrq_idx
                        ON {schema}.{table}
                        USING vchordrq (vector vector_cosine_ops)
                        WITH (residual_quantization = {residual_str}, lists = '{lists_sql}')
                """)

                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {table}_query_hash_idx
                        ON {schema}.{table} (query_hash)
                """)

                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {table}_template_id_idx
                        ON {schema}.{table} (template_id)
                        WHERE template_id IS NOT NULL
                """)

                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {table}_expires_at_idx
                        ON {schema}.{table} (expires_at)
                        WHERE expires_at IS NOT NULL
                """)

                await conn.execute(self._stats_table_ddl())

        except asyncpg.PostgresError as e:
            raise StorageInitializationError(
                f"Failed to initialize collection '{collection_name}': {e}"
            ) from e

        self._initialized_tables.add(collection_name)

    async def search(
        self,
        collection_name: str,
        vector: list[float],
        limit: int = 5,
        score_threshold: float = 0.0,
    ) -> list[CacheResult]:
        if self._pool is None:
            raise StorageError("Not connected. Call connect() first.")

        schema = self._settings.pg_schema
        table = self._table_name(collection_name)

        sql = f"""
            SELECT
                id::text,
                original_question,
                normalized_question,
                generated_query,
                query_hash,
                response_summary,
                template_id,
                usage_count,
                created_at,
                expires_at,
                (1 - (vector <=> $1::vector))::float AS score
            FROM {schema}.{table}
            WHERE (1 - (vector <=> $1::vector)) >= $2
              AND (expires_at IS NULL OR expires_at > NOW())
            ORDER BY vector <=> $1::vector
            LIMIT $3
        """
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql, vector, score_threshold, limit)
        except asyncpg.PostgresError as e:
            raise StorageError(f"VectorChord operation failed on '{collection_name}': {e}") from e

        return [_row_to_cache_result(row) for row in rows]

    async def upsert(self, collection_name: str, entries: list[CacheEntry]) -> None:
        if self._pool is None:
            raise StorageError("Not connected. Call connect() first.")
        if not entries:
            return

        schema = self._settings.pg_schema
        table = self._table_name(collection_name)

        sql = f"""
            INSERT INTO {schema}.{table} (
                id, vector, original_question, normalized_question,
                generated_query, query_hash, response_summary,
                template_id, usage_count, created_at, expires_at
            )
            VALUES ($1, $2::vector, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (id) DO UPDATE SET
                vector              = EXCLUDED.vector,
                original_question   = EXCLUDED.original_question,
                normalized_question = EXCLUDED.normalized_question,
                generated_query     = EXCLUDED.generated_query,
                query_hash          = EXCLUDED.query_hash,
                response_summary    = EXCLUDED.response_summary,
                template_id         = EXCLUDED.template_id,
                usage_count         = EXCLUDED.usage_count,
                created_at          = EXCLUDED.created_at,
                expires_at          = EXCLUDED.expires_at
        """
        try:
            async with self._pool.acquire() as conn:
                await conn.executemany(sql, [
                    (
                        uuid.UUID(entry.id),
                        entry.vector,
                        entry.original_question,
                        entry.normalized_question,
                        entry.generated_query,
                        entry.query_hash,
                        entry.response_summary,
                        entry.template_id,
                        entry.usage_count,
                        entry.created_at,
                        entry.expires_at,
                    )
                    for entry in entries
                ])
        except asyncpg.PostgresError as e:
            raise StorageError(f"VectorChord operation failed on '{collection_name}': {e}") from e

    async def find_expired(self, collection_name: str) -> list[str]:
        if self._pool is None:
            raise StorageError("Not connected. Call connect() first.")

        schema = self._settings.pg_schema
        table = self._table_name(collection_name)

        sql = f"""
            SELECT id::text FROM {schema}.{table}
            WHERE expires_at IS NOT NULL AND expires_at < NOW()
            LIMIT 10000
        """
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql)
            return [row["id"] for row in rows]
        except asyncpg.PostgresError as e:
            raise StorageError(f"VectorChord find_expired failed on '{collection_name}': {e}") from e
