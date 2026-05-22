"""WeaviateBackend — Weaviate v4 vector storage backend."""

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from medha.exceptions import ConfigurationError, StorageError, StorageInitializationError
from medha.interfaces.storage import VectorStorageBackend
from medha.types import CacheEntry, CacheResult

logger = logging.getLogger(__name__)

try:
    import weaviate
    import weaviate.classes as wvc
    from weaviate.classes.data import DataObject
    from weaviate.classes.query import Filter, MetadataQuery
    HAS_WEAVIATE = True
except ImportError:
    HAS_WEAVIATE = False


def _wv_collection_name(prefix: str, collection_name: str) -> str:
    pascal = "".join(word.capitalize() for word in collection_name.split("_"))
    return f"{prefix}{pascal}"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _ttl_filter() -> Any:
    now = _now_utc()
    return Filter.by_property("expires_at").is_none(True) | Filter.by_property("expires_at").greater_than(now)


def _entry_to_properties(entry: CacheEntry) -> dict[str, Any]:
    return {
        "original_question": entry.original_question,
        "normalized_question": entry.normalized_question,
        "generated_query": entry.generated_query,
        "query_hash": entry.query_hash,
        "response_summary": entry.response_summary or "",
        "template_id": entry.template_id or "",
        "usage_count": entry.usage_count,
        "created_at": entry.created_at,
        "expires_at": entry.expires_at,
    }


def _obj_to_result(obj: Any, score: float) -> CacheResult:
    props = obj.properties
    return CacheResult(
        id=str(obj.uuid),
        score=max(0.0, min(1.0, score)),
        original_question=props.get("original_question", ""),
        normalized_question=props.get("normalized_question", ""),
        generated_query=props.get("generated_query", ""),
        query_hash=props.get("query_hash", ""),
        response_summary=props.get("response_summary") or None,
        template_id=props.get("template_id") or None,
        usage_count=int(props.get("usage_count", 0)),
        created_at=props.get("created_at"),
        expires_at=props.get("expires_at"),
    )


class WeaviateBackend(VectorStorageBackend):
    """Weaviate v4 vector backend. Supports local and cloud modes."""

    def __init__(self, settings: Any = None) -> None:
        if not HAS_WEAVIATE:
            raise ConfigurationError(
                "weaviate backend requires 'weaviate-client>=4.6'. "
                "Install with: pip install medha-archai[weaviate]"
            )
        from medha.config import Settings
        self._settings = settings or Settings()
        self._client: Any = None
        self._collections: dict[str, Any] = {}

    async def connect(self) -> None:
        mode = self._settings.weaviate_mode
        try:
            auth = None
            if self._settings.weaviate_api_key:
                auth = wvc.init.Auth.api_key(self._settings.weaviate_api_key.get_secret_value())

            if mode == "cloud":
                self._client = weaviate.use_async_with_weaviate_cloud(
                    cluster_url=self._settings.weaviate_cloud_url,
                    auth_credentials=auth,
                )
            else:
                self._client = weaviate.use_async_with_local(
                    host=self._settings.weaviate_host,
                    port=self._settings.weaviate_http_port,
                    grpc_port=self._settings.weaviate_grpc_port,
                    auth_credentials=auth,
                )
            await self._client.connect()
        except Exception as e:
            raise StorageInitializationError(f"Failed to connect to Weaviate ({mode}): {e}") from e

    async def initialize(self, collection_name: str, dimension: int, **kwargs: Any) -> None:
        if self._client is None:
            raise StorageError("Not connected. Call connect() first.")
        if collection_name in self._collections:
            return
        wv_name = _wv_collection_name(self._settings.weaviate_collection_prefix, collection_name)
        try:
            if await self._client.collections.exists(wv_name):
                # index_null_state is immutable — drop and recreate if it is not set.
                # Data loss is acceptable: this is a cache.
                col = self._client.collections.get(wv_name)
                try:
                    cfg = await col.config.get()
                    needs_recreate = not getattr(cfg.inverted_index_config, "index_null_state", False)
                except Exception:
                    needs_recreate = False
                if needs_recreate:
                    logger.warning(
                        "Collection '%s' was created without indexNullState=True "
                        "(required for TTL filters). Dropping and recreating — cached entries will be lost.",
                        wv_name,
                    )
                    await self._client.collections.delete(wv_name)
                else:
                    self._collections[collection_name] = col
                    return

            await self._client.collections.create(
                name=wv_name,
                properties=[
                    wvc.config.Property(name="original_question", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="normalized_question", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="generated_query", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="query_hash", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="response_summary", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="template_id", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="usage_count", data_type=wvc.config.DataType.INT),
                    wvc.config.Property(name="created_at", data_type=wvc.config.DataType.DATE),
                    wvc.config.Property(name="expires_at", data_type=wvc.config.DataType.DATE),
                ],
                inverted_index_config=wvc.config.Configure.inverted_index(
                    index_null_state=True,
                ),
                vectorizer_config=wvc.config.Configure.Vectorizer.none(),
                vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                    distance_metric=wvc.config.VectorDistances.COSINE
                ),
            )
            self._collections[collection_name] = self._client.collections.get(wv_name)
        except Exception as e:
            raise StorageInitializationError(
                f"Failed to initialize Weaviate collection '{wv_name}': {e}"
            ) from e

    def _get_collection(self, collection_name: str) -> Any:
        col = self._collections.get(collection_name)
        if col is None:
            raise StorageError(
                f"Collection '{collection_name}' not initialized. Call initialize() first."
            )
        return col

    async def search(
        self,
        collection_name: str,
        vector: list[float],
        limit: int = 5,
        score_threshold: float = 0.0,
    ) -> list[CacheResult]:
        collection = self._get_collection(collection_name)
        try:
            result = await collection.query.near_vector(
                near_vector=vector,
                limit=limit,
                filters=_ttl_filter(),
                return_metadata=MetadataQuery(distance=True),
            )
        except Exception as e:
            raise StorageError(f"Weaviate search failed on '{collection_name}': {e}") from e

        out = []
        for obj in result.objects:
            score = 1.0 - (obj.metadata.distance or 0.0)
            if score >= score_threshold:
                out.append(_obj_to_result(obj, score))
        return out

    async def upsert(self, collection_name: str, entries: list[CacheEntry]) -> None:
        if not entries:
            return
        collection = self._get_collection(collection_name)
        objects = [
            DataObject(
                uuid=entry.id,
                properties=_entry_to_properties(entry),
                vector=entry.vector,
            )
            for entry in entries
        ]
        try:
            result = await collection.data.insert_many(objects)
            if result.has_errors:
                errors = [str(e) for e in result.errors.values()]
                raise StorageError(f"Weaviate upsert errors on '{collection_name}': {errors}")
        except StorageError:
            raise
        except Exception as e:
            raise StorageError(f"Weaviate upsert failed on '{collection_name}': {e}") from e

    async def scroll(
        self,
        collection_name: str,
        limit: int = 100,
        offset: str | None = None,
        with_vectors: bool = False,
    ) -> tuple[list[CacheResult], str | None]:
        collection = self._get_collection(collection_name)
        after = UUID(offset) if offset else None
        try:
            result = await collection.query.fetch_objects(
                limit=limit,
                after=after,
                include_vector=with_vectors,
            )
        except Exception as e:
            raise StorageError(f"Weaviate scroll failed on '{collection_name}': {e}") from e

        objects = result.objects
        cache_results = [_obj_to_result(obj, 1.0) for obj in objects]
        next_offset = str(objects[-1].uuid) if len(objects) == limit else None
        return cache_results, next_offset

    async def count(self, collection_name: str) -> int:
        collection = self._get_collection(collection_name)
        try:
            result = await collection.aggregate.over_all(total_count=True)
            return result.total_count or 0
        except Exception as e:
            raise StorageError(f"Weaviate count failed on '{collection_name}': {e}") from e

    async def delete(self, collection_name: str, ids: list[str]) -> None:
        if not ids:
            return
        collection = self._get_collection(collection_name)
        try:
            if len(ids) <= 10:
                for id_ in ids:
                    await collection.data.delete_by_id(id_)
            else:
                await collection.data.delete_many(
                    where=Filter.by_id().contains_any(ids)
                )
        except Exception as e:
            raise StorageError(f"Weaviate delete failed on '{collection_name}': {e}") from e

    async def search_by_query_hash(
        self, collection_name: str, query_hash: str
    ) -> CacheResult | None:
        collection = self._get_collection(collection_name)
        try:
            result = await collection.query.fetch_objects(
                filters=Filter.by_property("query_hash").equal(query_hash),
                limit=1,
            )
        except Exception as e:
            raise StorageError(
                f"Weaviate search_by_query_hash failed on '{collection_name}': {e}"
            ) from e
        if not result.objects:
            return None
        return _obj_to_result(result.objects[0], 1.0)

    async def update_usage_count(self, collection_name: str, id_: str) -> None:
        collection = self._get_collection(collection_name)
        try:
            obj = await collection.query.fetch_object_by_id(id_)
            if obj is None:
                return
            new_count = int(obj.properties.get("usage_count", 0)) + 1
            await collection.data.update(uuid=id_, properties={"usage_count": new_count})
        except Exception as e:
            raise StorageError(
                f"Weaviate update_usage_count failed on '{collection_name}': {e}"
            ) from e

    async def update_feedback(self, collection_name: str, id_: str, correct: bool) -> int:
        collection = self._get_collection(collection_name)
        try:
            obj = await collection.query.fetch_object_by_id(id_)
            if obj is None:
                return 0
            field = "feedback_correct" if correct else "feedback_incorrect"
            current = int(obj.properties.get(field, 0))
            await collection.data.update(uuid=id_, properties={field: current + 1})
            return current + 1
        except Exception as e:
            raise StorageError(
                f"Weaviate update_feedback failed on '{collection_name}': {e}"
            ) from e

    async def find_expired(self, collection_name: str) -> list[str]:
        collection = self._get_collection(collection_name)
        now = _now_utc()
        try:
            result = await collection.query.fetch_objects(
                filters=(
                    Filter.by_property("expires_at").is_none(False)
                    & Filter.by_property("expires_at").less_than(now)
                ),
            )
        except Exception as e:
            raise StorageError(f"Weaviate find_expired failed on '{collection_name}': {e}") from e
        return [str(obj.uuid) for obj in result.objects]

    async def search_by_normalized_question(
        self, collection_name: str, normalized_question: str
    ) -> CacheResult | None:
        collection = self._get_collection(collection_name)
        try:
            result = await collection.query.fetch_objects(
                filters=Filter.by_property("normalized_question").equal(normalized_question),
                limit=1,
            )
        except Exception as e:
            raise StorageError(
                f"Weaviate search_by_normalized_question failed on '{collection_name}': {e}"
            ) from e
        if not result.objects:
            return None
        return _obj_to_result(result.objects[0], 1.0)

    async def find_by_query_hash(self, collection_name: str, query_hash: str) -> list[str]:
        collection = self._get_collection(collection_name)
        try:
            result = await collection.query.fetch_objects(
                filters=Filter.by_property("query_hash").equal(query_hash),
            )
        except Exception as e:
            raise StorageError(
                f"Weaviate find_by_query_hash failed on '{collection_name}': {e}"
            ) from e
        return [str(obj.uuid) for obj in result.objects]

    async def find_by_template_id(self, collection_name: str, template_id: str) -> list[str]:
        collection = self._get_collection(collection_name)
        try:
            result = await collection.query.fetch_objects(
                filters=Filter.by_property("template_id").equal(template_id),
            )
        except Exception as e:
            raise StorageError(
                f"Weaviate find_by_template_id failed on '{collection_name}': {e}"
            ) from e
        return [str(obj.uuid) for obj in result.objects]

    async def drop_collection(self, collection_name: str) -> None:
        if self._client is None:
            raise StorageError("Not connected. Call connect() first.")
        wv_name = _wv_collection_name(self._settings.weaviate_collection_prefix, collection_name)
        try:
            await self._client.collections.delete(wv_name)
            self._collections.pop(collection_name, None)
        except Exception as e:
            raise StorageError(
                f"Weaviate drop_collection failed on '{collection_name}': {e}"
            ) from e

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass
        self._client = None
        self._collections.clear()
