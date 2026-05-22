"""RedisVectorBackend — Redis Stack (RediSearch) vector storage backend."""

import logging
import re
import time
from typing import Any

from medha.exceptions import ConfigurationError, StorageError, StorageInitializationError
from medha.interfaces.storage import VectorStorageBackend
from medha.types import CacheEntry, CacheResult

logger = logging.getLogger(__name__)

try:
    import numpy as np
    import redis.asyncio as aioredis
    from redis.asyncio import Redis, Sentinel
    from redis.commands.search.field import (
        NumericField,
        TagField,
        TextField,
        VectorField,
    )
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
    from redis.commands.search.query import Query
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

_TAG_ESCAPE_RE = re.compile(r'([,.<>{}\[\]"\':;!@#$%^&*()\-+=~|/\\])')
_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_\-]")


def _escape_tag(value: str) -> str:
    return _TAG_ESCAPE_RE.sub(r"\\\1", value)


def _safe_name(name: str) -> str:
    return _SAFE_NAME_RE.sub("_", name).strip("_").lower()


def _index_name(key_prefix: str, collection_name: str) -> str:
    return f"{key_prefix}:idx:{_safe_name(collection_name)}"


def _key_prefix(key_prefix: str, collection_name: str) -> str:
    return f"{key_prefix}:{_safe_name(collection_name)}"


def _doc_to_result(doc: Any, score: float) -> CacheResult:
    def _get(field: str, default: Any = "") -> Any:
        return getattr(doc, field, default) or default

    expires_at_raw = _get("expires_at", "0")
    expires_at = None
    try:
        ts = float(expires_at_raw)
        if ts > 0:
            from datetime import datetime, timezone
            expires_at = datetime.fromtimestamp(ts, tz=timezone.utc)
    except (ValueError, TypeError):
        pass

    created_at_raw = _get("created_at", "0")
    created_at = None
    try:
        ts = float(created_at_raw)
        if ts > 0:
            from datetime import datetime, timezone
            created_at = datetime.fromtimestamp(ts, tz=timezone.utc)
    except (ValueError, TypeError):
        pass

    return CacheResult(
        id=doc.id.rsplit(":", 1)[-1],
        score=max(0.0, min(1.0, score)),
        original_question=_get("original_question"),
        normalized_question=_get("normalized_question"),
        generated_query=_get("generated_query"),
        query_hash=_get("query_hash"),
        response_summary=_get("response_summary") or None,
        template_id=_get("template_id") or None,
        usage_count=int(_get("usage_count", 0)),
        created_at=created_at,
        expires_at=expires_at,
    )


class RedisVectorBackend(VectorStorageBackend):
    """Redis Stack (RediSearch) vector backend. Supports standalone and sentinel modes."""

    def __init__(self, settings: Any = None) -> None:
        if not HAS_REDIS:
            raise ConfigurationError(
                "redis backend requires 'redis[hiredis]>=4.6'. "
                "Install with: pip install medha-archai[redis]"
            )
        from medha.config import Settings
        self._settings = settings or Settings()
        self._client: Any = None

    async def connect(self) -> None:
        mode = self._settings.redis_mode
        s = self._settings
        try:
            if mode == "sentinel":
                hosts = []
                for h in s.redis_sentinel_hosts:
                    if ":" in h:
                        host, port_str = h.rsplit(":", 1)
                        hosts.append((host, int(port_str)))
                    else:
                        hosts.append((h, 26379))
                sentinel = Sentinel(hosts)
                self._client = sentinel.master_for(s.redis_sentinel_master)
            else:
                ssl_params: dict[str, Any] = {}
                if s.redis_ssl:
                    ssl_params = {
                        "ssl": True,
                        "ssl_certfile": s.redis_ssl_certfile,
                        "ssl_keyfile": s.redis_ssl_keyfile,
                        "ssl_ca_certs": s.redis_ssl_ca_certs,
                    }
                common = {
                    "socket_timeout": s.redis_socket_timeout,
                    "socket_connect_timeout": s.redis_socket_connect_timeout,
                    **ssl_params,
                }
                if s.redis_url:
                    self._client = aioredis.from_url(s.redis_url, **common)
                else:
                    kwargs: dict[str, Any] = {
                        "host": s.redis_host,
                        "port": s.redis_port,
                        "db": s.redis_db,
                        **common,
                    }
                    if s.redis_username:
                        kwargs["username"] = s.redis_username
                    if s.redis_password:
                        kwargs["password"] = s.redis_password.get_secret_value()
                    self._client = Redis(**kwargs)
            await self._client.ping()
        except Exception as e:
            raise StorageInitializationError(f"Failed to connect to Redis ({mode}): {e}") from e

    def _build_schema(self, dimension: int) -> list[Any]:
        s = self._settings
        algo = s.redis_index_algorithm
        if algo == "HNSW":
            vec_attrs: dict[str, Any] = {
                "TYPE": "FLOAT32",
                "DIM": dimension,
                "DISTANCE_METRIC": "COSINE",
                "M": s.redis_hnsw_m,
                "EF_CONSTRUCTION": s.redis_hnsw_ef_construction,
                "EF_RUNTIME": s.redis_hnsw_ef_runtime,
            }
        else:
            vec_attrs = {
                "TYPE": "FLOAT32",
                "DIM": dimension,
                "DISTANCE_METRIC": "COSINE",
            }
        return [
            TextField("original_question"),
            TextField("generated_query"),
            TagField("normalized_question", separator="|"),
            TagField("query_hash", separator="|"),
            TagField("template_id", separator="|"),
            NumericField("usage_count"),
            NumericField("created_at"),
            NumericField("expires_at"),
            VectorField("vector", algo, vec_attrs),
        ]

    async def initialize(self, collection_name: str, dimension: int, **kwargs: Any) -> None:
        if self._client is None:
            raise StorageError("Not connected. Call connect() first.")
        idx = _index_name(self._settings.redis_key_prefix, collection_name)
        col_key = _key_prefix(self._settings.redis_key_prefix, collection_name)
        try:
            await self._client.ft(idx).info()
            return  # index already exists
        except Exception as e:
            if "unknown index name" not in str(e).lower() and "no such index" not in str(e).lower():
                # not an "index not found" error — re-raise only if it's truly unexpected
                # Some redis versions say "Unknown Index name" so we check case-insensitively
                err_lower = str(e).lower()
                if "unknown" not in err_lower and "index" not in err_lower:
                    raise StorageInitializationError(
                        f"Redis initialize failed on '{collection_name}': {e}"
                    ) from e
        try:
            schema = self._build_schema(dimension)
            definition = IndexDefinition(prefix=[f"{col_key}:"], index_type=IndexType.HASH)
            await self._client.ft(idx).create_index(schema, definition=definition)
            logger.debug("Created Redis index '%s' for collection '%s'", idx, collection_name)
        except Exception as e:
            err_lower = str(e).lower()
            if "index already exists" in err_lower:
                return
            raise StorageInitializationError(
                f"Redis initialize failed on '{collection_name}': {e}"
            ) from e

    async def search(
        self,
        collection_name: str,
        vector: list[float],
        limit: int = 5,
        score_threshold: float = 0.0,
    ) -> list[CacheResult]:
        if self._client is None:
            raise StorageError("Not connected. Call connect() first.")
        idx = _index_name(self._settings.redis_key_prefix, collection_name)
        try:
            import numpy as np
            vec_bytes = np.array(vector, dtype=np.float32).tobytes()
            now_ts = int(time.time())
            ttl_filter = f"(@expires_at:[0 0] | @expires_at:[({now_ts} +inf])"
            q = (
                Query(f"({ttl_filter})=>[KNN {limit} @vector $vec AS __score]")
                .sort_by("__score", asc=True)
                .return_fields(
                    "original_question", "normalized_question", "generated_query",
                    "query_hash", "response_summary", "template_id",
                    "usage_count", "created_at", "expires_at", "__score",
                )
                .paging(0, limit)
                .dialect(2)
            )
            result = await self._client.ft(idx).search(q, query_params={"vec": vec_bytes})
        except Exception as e:
            import redis as redis_lib
            if isinstance(e, redis_lib.exceptions.ResponseError):
                raise StorageError(
                    f"Redis search failed on '{collection_name}': {e}. "
                    "Did you call initialize()?"
                ) from e
            raise StorageError(f"Redis search failed on '{collection_name}': {e}") from e

        out = []
        for doc in result.docs:
            raw_score = getattr(doc, "__score", "1.0")
            try:
                dist = float(raw_score)
            except (ValueError, TypeError):
                dist = 1.0
            score = max(0.0, min(1.0, 1.0 - dist))
            if score >= score_threshold:
                out.append(_doc_to_result(doc, score))
        return out

    async def upsert(self, collection_name: str, entries: list[CacheEntry]) -> None:
        if not entries:
            return
        if self._client is None:
            raise StorageError("Not connected. Call connect() first.")
        col_key = _key_prefix(self._settings.redis_key_prefix, collection_name)
        try:
            import numpy as np
            pipe = self._client.pipeline(transaction=False)
            for entry in entries:
                vec_bytes = np.array(entry.vector, dtype=np.float32).tobytes()
                expires_at = (
                    str(int(entry.expires_at.timestamp())) if entry.expires_at else "0"
                )
                created_at = (
                    str(int(entry.created_at.timestamp())) if entry.created_at else "0"
                )
                mapping: dict[str, Any] = {
                    "original_question": entry.original_question,
                    "normalized_question": entry.normalized_question,
                    "generated_query": entry.generated_query,
                    "query_hash": entry.query_hash,
                    "response_summary": entry.response_summary or "",
                    "template_id": entry.template_id or "",
                    "usage_count": entry.usage_count,
                    "created_at": created_at,
                    "expires_at": expires_at,
                    "vector": vec_bytes,
                }
                pipe.hset(f"{col_key}:{entry.id}", mapping=mapping)
            await pipe.execute()
        except Exception as e:
            raise StorageError(f"Redis upsert failed on '{collection_name}': {e}") from e

    async def scroll(
        self,
        collection_name: str,
        limit: int = 100,
        offset: str | None = None,
        with_vectors: bool = False,
    ) -> tuple[list[CacheResult], str | None]:
        if self._client is None:
            raise StorageError("Not connected. Call connect() first.")
        idx = _index_name(self._settings.redis_key_prefix, collection_name)
        int_offset = int(offset) if offset else 0
        return_fields = [
            "original_question", "normalized_question", "generated_query",
            "query_hash", "response_summary", "template_id",
            "usage_count", "created_at", "expires_at",
        ]
        if with_vectors:
            return_fields.append("vector")
        try:
            q = (
                Query("*")
                .sort_by("created_at", asc=True)
                .return_fields(*return_fields)
                .paging(int_offset, limit)
            )
            result = await self._client.ft(idx).search(q)
        except Exception as e:
            raise StorageError(f"Redis scroll failed on '{collection_name}': {e}") from e

        items = result.docs
        cache_results = [_doc_to_result(doc, 1.0) for doc in items]
        next_offset = str(int_offset + len(items)) if len(items) == limit else None
        return cache_results, next_offset

    async def count(self, collection_name: str) -> int:
        if self._client is None:
            raise StorageError("Not connected. Call connect() first.")
        idx = _index_name(self._settings.redis_key_prefix, collection_name)
        try:
            info = await self._client.ft(idx).info()
            return int(info.get("num_docs", 0))
        except Exception as e:
            raise StorageError(f"Redis count failed on '{collection_name}': {e}") from e

    async def delete(self, collection_name: str, ids: list[str]) -> None:
        if not ids:
            return
        if self._client is None:
            raise StorageError("Not connected. Call connect() first.")
        col_key = _key_prefix(self._settings.redis_key_prefix, collection_name)
        try:
            pipe = self._client.pipeline(transaction=False)
            for id_ in ids:
                pipe.delete(f"{col_key}:{id_}")
            await pipe.execute()
        except Exception as e:
            raise StorageError(f"Redis delete failed on '{collection_name}': {e}") from e

    async def search_by_query_hash(
        self, collection_name: str, query_hash: str
    ) -> CacheResult | None:
        if self._client is None:
            raise StorageError("Not connected. Call connect() first.")
        idx = _index_name(self._settings.redis_key_prefix, collection_name)
        try:
            escaped = _escape_tag(query_hash)
            q = (
                Query(f"@query_hash:{{{escaped}}}")
                .return_fields(
                    "original_question", "normalized_question", "generated_query",
                    "query_hash", "response_summary", "template_id",
                    "usage_count", "created_at", "expires_at",
                )
                .paging(0, 1)
                .dialect(2)
            )
            result = await self._client.ft(idx).search(q)
        except Exception as e:
            raise StorageError(
                f"Redis search_by_query_hash failed on '{collection_name}': {e}"
            ) from e
        if not result.docs:
            return None
        return _doc_to_result(result.docs[0], 1.0)

    async def update_usage_count(self, collection_name: str, id_: str) -> None:
        if self._client is None:
            raise StorageError("Not connected. Call connect() first.")
        col_key = _key_prefix(self._settings.redis_key_prefix, collection_name)
        key = f"{col_key}:{id_}"
        try:
            exists = await self._client.hexists(key, "original_question")
            if not exists:
                return
            await self._client.hincrby(key, "usage_count", 1)
        except Exception as e:
            raise StorageError(
                f"Redis update_usage_count failed on '{collection_name}': {e}"
            ) from e

    async def update_feedback(self, collection_name: str, id_: str, correct: bool) -> int:
        if self._client is None:
            raise StorageError("Not connected. Call connect() first.")
        col_key = _key_prefix(self._settings.redis_key_prefix, collection_name)
        key = f"{col_key}:{id_}"
        field = "feedback_correct" if correct else "feedback_incorrect"
        try:
            exists = await self._client.hexists(key, "original_question")
            if not exists:
                return 0
            return int(await self._client.hincrby(key, field, 1))
        except Exception as e:
            raise StorageError(
                f"Redis update_feedback failed on '{collection_name}': {e}"
            ) from e

    async def find_expired(self, collection_name: str) -> list[str]:
        if self._client is None:
            raise StorageError("Not connected. Call connect() first.")
        idx = _index_name(self._settings.redis_key_prefix, collection_name)
        now_ts = int(time.time())
        try:
            q = (
                Query(f"@expires_at:[(0 ({now_ts}]")
                .return_fields("expires_at")
                .paging(0, 10000)
            )
            result = await self._client.ft(idx).search(q)
        except Exception as e:
            raise StorageError(f"Redis find_expired failed on '{collection_name}': {e}") from e
        return [doc.id.rsplit(":", 1)[-1] for doc in result.docs]

    async def search_by_normalized_question(
        self, collection_name: str, normalized_question: str
    ) -> CacheResult | None:
        if self._client is None:
            raise StorageError("Not connected. Call connect() first.")
        idx = _index_name(self._settings.redis_key_prefix, collection_name)
        try:
            escaped = _escape_tag(normalized_question)
            q = (
                Query(f"@normalized_question:{{{escaped}}}")
                .return_fields(
                    "original_question", "normalized_question", "generated_query",
                    "query_hash", "response_summary", "template_id",
                    "usage_count", "created_at", "expires_at",
                )
                .paging(0, 1)
                .dialect(2)
            )
            result = await self._client.ft(idx).search(q)
        except Exception as e:
            raise StorageError(
                f"Redis search_by_normalized_question failed on '{collection_name}': {e}"
            ) from e
        if not result.docs:
            return None
        return _doc_to_result(result.docs[0], 1.0)

    async def find_by_query_hash(self, collection_name: str, query_hash: str) -> list[str]:
        if self._client is None:
            raise StorageError("Not connected. Call connect() first.")
        idx = _index_name(self._settings.redis_key_prefix, collection_name)
        try:
            escaped = _escape_tag(query_hash)
            q = (
                Query(f"@query_hash:{{{escaped}}}")
                .return_fields("query_hash")
                .paging(0, 10000)
                .dialect(2)
            )
            result = await self._client.ft(idx).search(q)
        except Exception as e:
            raise StorageError(
                f"Redis find_by_query_hash failed on '{collection_name}': {e}"
            ) from e
        return [doc.id.rsplit(":", 1)[-1] for doc in result.docs]

    async def find_by_template_id(self, collection_name: str, template_id: str) -> list[str]:
        if self._client is None:
            raise StorageError("Not connected. Call connect() first.")
        idx = _index_name(self._settings.redis_key_prefix, collection_name)
        try:
            escaped = _escape_tag(template_id)
            q = (
                Query(f"@template_id:{{{escaped}}}")
                .return_fields("template_id")
                .paging(0, 10000)
                .dialect(2)
            )
            result = await self._client.ft(idx).search(q)
        except Exception as e:
            raise StorageError(
                f"Redis find_by_template_id failed on '{collection_name}': {e}"
            ) from e
        return [doc.id.rsplit(":", 1)[-1] for doc in result.docs]

    async def drop_collection(self, collection_name: str) -> None:
        if self._client is None:
            raise StorageError("Not connected. Call connect() first.")
        idx = _index_name(self._settings.redis_key_prefix, collection_name)
        col_key = _key_prefix(self._settings.redis_key_prefix, collection_name)
        try:
            await self._client.ft(idx).dropindex(delete_documents=True)
        except Exception as e:
            if "unknown index name" not in str(e).lower() and "no such index" not in str(e).lower():
                logger.warning("Redis dropindex warning on '%s': %s", collection_name, e)
            # fallback: scan and delete orphan hashes
            try:
                async for key in self._client.scan_iter(match=f"{col_key}:*", count=100):
                    await self._client.delete(key)
            except Exception as scan_e:
                raise StorageError(
                    f"Redis drop_collection fallback scan failed on '{collection_name}': {scan_e}"
                ) from scan_e

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
        self._client = None
