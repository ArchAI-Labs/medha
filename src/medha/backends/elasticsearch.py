"""ElasticsearchBackend — Elasticsearch 8.x storage backend."""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from medha.exceptions import ConfigurationError, StorageError, StorageInitializationError
from medha.interfaces.storage import VectorStorageBackend
from medha.types import CacheEntry, CacheResult

logger = logging.getLogger(__name__)

try:
    from elasticsearch import AsyncElasticsearch, NotFoundError, TransportError
    from elasticsearch.helpers import async_bulk
    HAS_ELASTICSEARCH = True
except ImportError:
    HAS_ELASTICSEARCH = False

_INDEX_UNSAFE_RE = re.compile(r"[^a-z0-9_-]")


class ElasticsearchBackend(VectorStorageBackend):
    """Elasticsearch 8.x backend. Requires elasticsearch[async]>=8.12."""

    def __init__(self, settings: Any = None) -> None:
        if not HAS_ELASTICSEARCH:
            raise ConfigurationError(
                "elasticsearch backend requires 'elasticsearch[async]>=8.12'. "
                "Install with: pip install medha-archai[elasticsearch]"
            )
        from medha.config import Settings
        self._settings = settings or Settings()
        self._client: Any = None

    def _index_name(self, collection_name: str) -> str:
        safe = _INDEX_UNSAFE_RE.sub("_", collection_name.lower())
        prefix = self._settings.es_index_prefix
        return f"{prefix}_{safe}"[:255]

    async def connect(self) -> None:
        kwargs: dict[str, Any] = {
            "hosts": self._settings.es_hosts,
            "request_timeout": self._settings.es_timeout,
        }
        if self._settings.es_api_key is not None:
            kwargs["api_key"] = self._settings.es_api_key.get_secret_value()
        elif self._settings.es_username is not None and self._settings.es_password is not None:
            kwargs["basic_auth"] = (
                self._settings.es_username,
                self._settings.es_password.get_secret_value(),
            )
        try:
            self._client = AsyncElasticsearch(**kwargs)
            await self._client.info()
        except Exception as e:
            raise StorageInitializationError(f"Failed to connect to Elasticsearch: {e}") from e

    async def initialize(self, collection_name: str, dimension: int, **kwargs: Any) -> None:
        if self._client is None:
            raise StorageError("Not connected. Call connect() first.")
        index = self._index_name(collection_name)
        try:
            exists = await self._client.indices.exists(index=index)
            if exists:
                return
            mapping = {
                "mappings": {
                    "properties": {
                        "vector": {
                            "type": "dense_vector",
                            "dims": dimension,
                            "index": True,
                            "similarity": "cosine",
                        },
                        "original_question": {"type": "text"},
                        "normalized_question": {"type": "keyword"},
                        "generated_query": {"type": "text"},
                        "query_hash": {"type": "keyword"},
                        "response_summary": {"type": "text"},
                        "template_id": {"type": "keyword"},
                        "usage_count": {"type": "integer"},
                        "created_at": {"type": "date"},
                        "expires_at": {"type": "date"},
                    }
                },
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                },
            }
            await self._client.indices.create(index=index, body=mapping)
            logger.info("Created Elasticsearch index '%s'", index)
        except Exception as e:
            raise StorageInitializationError(
                f"Failed to initialize Elasticsearch index '{index}': {e}"
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
        index = self._index_name(collection_name)
        ttl_filter = {
            "bool": {
                "should": [
                    {"bool": {"must_not": {"exists": {"field": "expires_at"}}}},
                    {"range": {"expires_at": {"gt": "now"}}},
                ]
            }
        }
        query = {
            "knn": {
                "field": "vector",
                "query_vector": vector,
                "k": limit,
                "num_candidates": self._settings.es_num_candidates,
                "filter": ttl_filter,
            },
            "size": limit,
            "_source": {
                "excludes": ["vector"]
            },
        }
        try:
            resp = await self._client.search(index=index, body=query)
        except NotFoundError as e:
            raise StorageError(f"Elasticsearch index '{index}' not found: {e}") from e
        except TransportError as e:
            raise StorageError(f"Elasticsearch transport error on '{collection_name}': {e}") from e

        results = []
        for hit in resp["hits"]["hits"]:
            score = (hit["_score"] * 2) - 1
            if score < score_threshold:
                continue
            src = hit["_source"]
            results.append(_hit_to_cache_result(hit["_id"], src, score))
        return results

    async def upsert(self, collection_name: str, entries: list[CacheEntry]) -> None:
        if self._client is None:
            raise StorageError("Not connected. Call connect() first.")
        if not entries:
            return
        index = self._index_name(collection_name)

        def _actions() -> Any:
            for entry in entries:
                doc: dict[str, Any] = {
                    "original_question": entry.original_question,
                    "normalized_question": entry.normalized_question,
                    "generated_query": entry.generated_query,
                    "query_hash": entry.query_hash,
                    "response_summary": entry.response_summary,
                    "template_id": entry.template_id,
                    "usage_count": entry.usage_count,
                    "created_at": _dt_to_str(entry.created_at),
                    "vector": entry.vector,
                }
                if entry.expires_at is not None:
                    doc["expires_at"] = _dt_to_str(entry.expires_at)
                yield {
                    "_op_type": "index",
                    "_index": index,
                    "_id": entry.id,
                    "_source": doc,
                }

        try:
            await async_bulk(self._client, _actions())
        except Exception as e:
            raise StorageError(f"Elasticsearch upsert failed on '{collection_name}': {e}") from e

    async def scroll(
        self,
        collection_name: str,
        limit: int = 100,
        offset: str | None = None,
        with_vectors: bool = False,
    ) -> tuple[list[CacheResult], str | None]:
        if self._client is None:
            raise StorageError("Not connected. Call connect() first.")
        index = self._index_name(collection_name)

        search_after = json.loads(offset) if offset is not None else None
        source_excludes = [] if with_vectors else ["vector"]

        query: dict[str, Any] = {
            "query": {"match_all": {}},
            "size": limit,
            "sort": [{"created_at": "asc"}, {"_id": "asc"}],
            "_source": {"excludes": source_excludes},
        }
        if search_after is not None:
            query["search_after"] = search_after

        try:
            resp = await self._client.search(index=index, body=query)
        except NotFoundError as e:
            raise StorageError(f"Elasticsearch index '{index}' not found: {e}") from e
        except TransportError as e:
            raise StorageError(f"Elasticsearch transport error on '{collection_name}': {e}") from e

        hits = resp["hits"]["hits"]
        results = [_hit_to_cache_result(h["_id"], h["_source"], 1.0) for h in hits]
        next_offset: str | None = None
        if len(hits) == limit:
            next_offset = json.dumps(hits[-1]["sort"])
        return results, next_offset

    async def count(self, collection_name: str) -> int:
        if self._client is None:
            raise StorageError("Not connected. Call connect() first.")
        index = self._index_name(collection_name)
        try:
            resp = await self._client.count(index=index)
            return int(resp["count"])
        except NotFoundError:
            return 0
        except TransportError as e:
            raise StorageError(f"Elasticsearch count failed on '{collection_name}': {e}") from e

    async def delete(self, collection_name: str, ids: list[str]) -> None:
        if self._client is None:
            raise StorageError("Not connected. Call connect() first.")
        if not ids:
            return
        index = self._index_name(collection_name)

        actions = [
            {"_op_type": "delete", "_index": index, "_id": id_}
            for id_ in ids
        ]
        try:
            await async_bulk(self._client, actions, ignore_status=[404])
        except Exception as e:
            raise StorageError(f"Elasticsearch delete failed on '{collection_name}': {e}") from e

    async def search_by_query_hash(
        self, collection_name: str, query_hash: str
    ) -> CacheResult | None:
        if self._client is None:
            raise StorageError("Not connected. Call connect() first.")
        index = self._index_name(collection_name)
        query = {
            "query": {"term": {"query_hash": query_hash}},
            "size": 1,
            "_source": {"excludes": ["vector"]},
        }
        try:
            resp = await self._client.search(index=index, body=query)
        except NotFoundError:
            return None
        except TransportError as e:
            raise StorageError(f"Elasticsearch search_by_query_hash failed on '{collection_name}': {e}") from e
        hits = resp["hits"]["hits"]
        if not hits:
            return None
        return _hit_to_cache_result(hits[0]["_id"], hits[0]["_source"], 1.0)

    async def update_usage_count(self, collection_name: str, point_id: str) -> None:
        if self._client is None:
            raise StorageError("Not connected. Call connect() first.")
        index = self._index_name(collection_name)
        try:
            await self._client.update(
                index=index,
                id=point_id,
                body={"script": {"source": "ctx._source.usage_count += 1", "lang": "painless"}},
            )
        except NotFoundError:
            logger.warning(
                "update_usage_count: id '%s' not found in collection '%s'",
                point_id,
                collection_name,
            )
        except TransportError as e:
            raise StorageError(f"Elasticsearch update_usage_count failed on '{collection_name}': {e}") from e

    async def update_feedback(self, collection_name: str, point_id: str, correct: bool) -> int:
        if self._client is None:
            raise StorageError("Not connected. Call connect() first.")
        index_name = self._index_name(collection_name)
        field = "feedback_correct" if correct else "feedback_incorrect"
        script = {
            "source": (
                "if (ctx._source.containsKey(params.field)) "
                "{ ctx._source[params.field]++; } "
                "else { ctx._source[params.field] = 1; }"
            ),
            "params": {"field": field},
        }
        try:
            resp = await self._client.update(
                index=index_name,
                id=point_id,
                body={"script": script},
                ignore=[404],
            )
        except TransportError as e:
            raise StorageError(f"Elasticsearch update_feedback failed on '{collection_name}': {e}") from e

        if resp.get("result") == "not_found" or resp.get("_shards", {}).get("successful", 1) == 0:
            return 0

        try:
            get_resp = await self._client.get(index=index_name, id=point_id, _source_includes=[field])
            return int(get_resp["_source"].get(field, 0))
        except NotFoundError:
            return 0

    async def find_expired(self, collection_name: str) -> list[str]:
        if self._client is None:
            raise StorageError("Not connected. Call connect() first.")
        index = self._index_name(collection_name)
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"exists": {"field": "expires_at"}},
                        {"range": {"expires_at": {"lt": "now"}}},
                    ]
                }
            },
            "size": 10000,
            "_source": False,
        }
        try:
            resp = await self._client.search(index=index, body=query)
            return [hit["_id"] for hit in resp["hits"]["hits"]]
        except NotFoundError:
            return []
        except TransportError as e:
            raise StorageError(f"Elasticsearch find_expired failed on '{collection_name}': {e}") from e

    async def search_by_normalized_question(
        self, collection_name: str, normalized_question: str
    ) -> CacheResult | None:
        if self._client is None:
            raise StorageError("Not connected. Call connect() first.")
        index = self._index_name(collection_name)
        query = {
            "query": {"term": {"normalized_question": normalized_question}},
            "size": 1,
            "_source": {"excludes": ["vector"]},
        }
        try:
            resp = await self._client.search(index=index, body=query)
        except NotFoundError:
            return None
        except TransportError as e:
            raise StorageError(
                f"Elasticsearch search_by_normalized_question failed on '{collection_name}': {e}"
            ) from e
        hits = resp["hits"]["hits"]
        if not hits:
            return None
        return _hit_to_cache_result(hits[0]["_id"], hits[0]["_source"], 1.0)

    async def find_by_query_hash(
        self, collection_name: str, query_hash: str
    ) -> list[str]:
        if self._client is None:
            raise StorageError("Not connected. Call connect() first.")
        index = self._index_name(collection_name)
        query = {
            "query": {"term": {"query_hash": query_hash}},
            "size": 10000,
            "_source": False,
        }
        try:
            resp = await self._client.search(index=index, body=query)
            return [hit["_id"] for hit in resp["hits"]["hits"]]
        except NotFoundError:
            return []
        except TransportError as e:
            raise StorageError(f"Elasticsearch find_by_query_hash failed on '{collection_name}': {e}") from e

    async def find_by_template_id(
        self, collection_name: str, template_id: str
    ) -> list[str]:
        if self._client is None:
            raise StorageError("Not connected. Call connect() first.")
        index = self._index_name(collection_name)
        query = {
            "query": {"term": {"template_id": template_id}},
            "size": 10000,
            "_source": False,
        }
        try:
            resp = await self._client.search(index=index, body=query)
            return [hit["_id"] for hit in resp["hits"]["hits"]]
        except NotFoundError:
            return []
        except TransportError as e:
            raise StorageError(f"Elasticsearch find_by_template_id failed on '{collection_name}': {e}") from e

    async def drop_collection(self, collection_name: str) -> None:
        if self._client is None:
            raise StorageError("Not connected. Call connect() first.")
        index = self._index_name(collection_name)
        try:
            await self._client.indices.delete(index=index, ignore_unavailable=True)
            logger.info("Dropped Elasticsearch index '%s'", index)
        except TransportError as e:
            raise StorageError(f"Elasticsearch drop_collection failed on '{collection_name}': {e}") from e

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


def _dt_to_str(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _hit_to_cache_result(doc_id: str, src: dict[str, Any], score: float) -> CacheResult:
    from medha.types import CacheResult

    def _parse_dt(val: Any) -> datetime | None:
        if val is None:
            return None
        if isinstance(val, datetime):
            return val
        try:
            return datetime.fromisoformat(str(val))
        except ValueError:
            return None

    return CacheResult(
        id=doc_id,
        score=max(0.0, min(1.0, score)),
        original_question=src.get("original_question", ""),
        normalized_question=src.get("normalized_question", ""),
        generated_query=src.get("generated_query", ""),
        query_hash=src.get("query_hash", ""),
        response_summary=src.get("response_summary"),
        template_id=src.get("template_id"),
        usage_count=src.get("usage_count", 1),
        created_at=_parse_dt(src.get("created_at")),
        expires_at=_parse_dt(src.get("expires_at")),
        vector=src.get("vector"),
    )
