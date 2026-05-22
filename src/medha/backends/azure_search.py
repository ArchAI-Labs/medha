"""AzureSearchBackend — Azure AI Search (azure-search-documents v11) vector storage backend."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from medha.exceptions import ConfigurationError, StorageError, StorageInitializationError
from medha.interfaces.storage import VectorStorageBackend
from medha.types import CacheEntry, CacheResult

logger = logging.getLogger(__name__)

try:
    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import HttpResponseError, ResourceNotFoundError, ServiceRequestError
    from azure.search.documents.aio import SearchClient as AsyncSearchClient
    from azure.search.documents.indexes.aio import SearchIndexClient as AsyncSearchIndexClient
    from azure.search.documents.indexes.models import (
        HnswAlgorithmConfiguration,
        SearchField,
        SearchFieldDataType,
        SearchIndex,
        SearchableField,
        SimpleField,
        VectorSearch,
        VectorSearchProfile,
    )
    from azure.search.documents.models import VectorizedQuery
    HAS_AZURE_SEARCH = True
except ImportError:
    HAS_AZURE_SEARCH = False

_SCALAR_FIELDS = [
    "id", "original_question", "normalized_question", "generated_query",
    "query_hash", "response_summary", "template_id", "usage_count",
    "created_at", "expires_at",
]


def _az_index_name(collection_name: str, prefix: str) -> str:
    return re.sub(r"[^a-z0-9-]", "-", f"{prefix}-{collection_name}".lower()).strip("-")[:128]


def _esc(value: str) -> str:
    return value.replace("'", "''")


def _dt_to_iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_dt(val: Any) -> datetime | None:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    try:
        return datetime.fromisoformat(str(val).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _doc_to_result(doc: dict[str, Any], score: float) -> CacheResult:
    return CacheResult(
        id=doc.get("id", ""),
        score=max(0.0, min(1.0, score)),
        original_question=doc.get("original_question", ""),
        normalized_question=doc.get("normalized_question", ""),
        generated_query=doc.get("generated_query", ""),
        query_hash=doc.get("query_hash", ""),
        response_summary=doc.get("response_summary"),
        template_id=doc.get("template_id"),
        usage_count=doc.get("usage_count", 1),
        created_at=_parse_dt(doc.get("created_at")),
        expires_at=_parse_dt(doc.get("expires_at")),
        vector=doc.get("vector"),
    )


def _entry_to_doc(entry: CacheEntry) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "id": entry.id,
        "original_question": entry.original_question,
        "normalized_question": entry.normalized_question,
        "generated_query": entry.generated_query,
        "query_hash": entry.query_hash,
        "response_summary": entry.response_summary,
        "template_id": entry.template_id,
        "usage_count": entry.usage_count,
        "created_at": _dt_to_iso(entry.created_at),
        "vector": entry.vector,
    }
    if entry.expires_at is not None:
        doc["expires_at"] = _dt_to_iso(entry.expires_at)
    return doc


class AzureSearchBackend(VectorStorageBackend):
    """Azure AI Search backend. Requires azure-search-documents>=11.4,<12."""

    def __init__(self, settings: Any = None) -> None:
        if not HAS_AZURE_SEARCH:
            raise ConfigurationError(
                "azure-search backend requires 'azure-search-documents>=11.4,<12'. "
                "Install with: pip install medha-archai[azure-search]"
            )
        from medha.config import Settings
        self._settings = settings or Settings()
        self._search_clients: dict[str, AsyncSearchClient] = {}
        self._index_client: AsyncSearchIndexClient | None = None
        self._credential: Any = None

    async def connect(self) -> None:
        endpoint = self._settings.azure_search_endpoint
        api_version = self._settings.azure_search_api_version

        if self._settings.azure_search_api_key is not None:
            self._credential = AzureKeyCredential(
                self._settings.azure_search_api_key.get_secret_value()
            )
        else:
            try:
                from azure.identity.aio import DefaultAzureCredential
                self._credential = DefaultAzureCredential()
            except ImportError as e:
                raise ConfigurationError(
                    "AAD authentication requires 'azure-identity'. "
                    "Install it separately: pip install azure-identity"
                ) from e

        try:
            self._index_client = AsyncSearchIndexClient(
                endpoint, self._credential, api_version=api_version
            )
            [idx async for idx in self._index_client.list_index_names()]
        except ServiceRequestError as e:
            raise StorageInitializationError(
                f"Failed to connect to Azure AI Search (network error): {e}"
            ) from e
        except HttpResponseError as e:
            if e.status_code in (401, 403):
                raise StorageInitializationError(
                    f"Azure AI Search authentication failed (HTTP {e.status_code}): {e}"
                ) from e
            raise StorageInitializationError(
                f"Failed to connect to Azure AI Search: {e}"
            ) from e

    async def initialize(self, collection_name: str, dimension: int, **kwargs: Any) -> None:
        if self._index_client is None:
            raise StorageError("Not connected. Call connect() first.")

        index_name = _az_index_name(collection_name, self._settings.azure_search_index_name)
        api_version = self._settings.azure_search_api_version
        endpoint = self._settings.azure_search_endpoint

        try:
            await self._index_client.get_index(index_name)
            # Index already exists — create client and return
            if collection_name not in self._search_clients:
                self._search_clients[collection_name] = AsyncSearchClient(
                    endpoint, index_name, self._credential, api_version=api_version
                )
            return
        except ResourceNotFoundError:
            pass
        except HttpResponseError as e:
            raise StorageInitializationError(
                f"Failed to check Azure Search index '{index_name}': {e}"
            ) from e

        try:
            vector_search = VectorSearch(
                algorithms=[HnswAlgorithmConfiguration(name="medha-hnsw")],
                profiles=[VectorSearchProfile(
                    name="medha-hnsw-profile",
                    algorithm_configuration_name="medha-hnsw",
                )],
            )
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
                SearchableField(name="original_question", analyzer_name="standard.lucene"),
                SimpleField(name="normalized_question", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="generated_query", type=SearchFieldDataType.String),
                SimpleField(name="query_hash", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="response_summary", type=SearchFieldDataType.String, nullable=True),
                SimpleField(name="template_id", type=SearchFieldDataType.String, filterable=True, nullable=True),
                SimpleField(name="usage_count", type=SearchFieldDataType.Int32, filterable=True),
                SimpleField(name="created_at", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
                SimpleField(name="expires_at", type=SearchFieldDataType.DateTimeOffset, filterable=True, nullable=True),
                SearchField(
                    name="vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=dimension,
                    vector_search_profile_name="medha-hnsw-profile",
                ),
            ]
            index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
            await self._index_client.create_index(index)
            logger.info("Created Azure Search index '%s'", index_name)
        except HttpResponseError as e:
            raise StorageInitializationError(
                f"Failed to create Azure Search index '{index_name}': {e}"
            ) from e

        self._search_clients[collection_name] = AsyncSearchClient(
            endpoint, index_name, self._credential, api_version=api_version
        )

    def _get_client(self, collection_name: str) -> AsyncSearchClient:
        if collection_name in self._search_clients:
            return self._search_clients[collection_name]
        if self._index_client is not None:
            index_name = _az_index_name(collection_name, self._settings.azure_search_index_name)
            client = AsyncSearchClient(
                self._settings.azure_search_endpoint,
                index_name,
                self._credential,
                api_version=self._settings.azure_search_api_version,
            )
            self._search_clients[collection_name] = client
            return client
        raise StorageError("Not connected.")

    async def search(
        self,
        collection_name: str,
        vector: list[float],
        limit: int = 5,
        score_threshold: float = 0.0,
    ) -> list[CacheResult]:
        client = self._get_client(collection_name)
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        ttl_filter = f"(expires_at eq null) or (expires_at gt {now_iso})"
        vq = VectorizedQuery(
            vector=vector,
            k_nearest_neighbors=limit + self._settings.azure_search_top_k_candidates,
            fields="vector",
        )
        try:
            items: list[CacheResult] = []
            async for result in await client.search(
                search_text=None,
                vector_queries=[vq],
                filter=ttl_filter,
                top=limit,
                select=_SCALAR_FIELDS,
            ):
                score = result.get("@search.score", 0.0)
                if score < score_threshold:
                    continue
                items.append(_doc_to_result(dict(result), score))
            return items
        except HttpResponseError as e:
            raise StorageError(f"Azure Search search failed on '{collection_name}': {e}") from e

    async def upsert(self, collection_name: str, entries: list[CacheEntry]) -> None:
        if not entries:
            return
        client = self._get_client(collection_name)
        docs = [_entry_to_doc(e) for e in entries]
        try:
            await client.merge_or_upload_documents(docs)
        except HttpResponseError as e:
            raise StorageError(f"Azure Search upsert failed on '{collection_name}': {e}") from e

    async def scroll(
        self,
        collection_name: str,
        limit: int = 100,
        offset: str | None = None,
        with_vectors: bool = False,
    ) -> tuple[list[CacheResult], str | None]:
        client = self._get_client(collection_name)
        skip = int(offset) if offset is not None else 0
        select = None if with_vectors else _SCALAR_FIELDS
        try:
            items: list[CacheResult] = []
            async for result in await client.search(
                search_text="*",
                skip=skip,
                top=limit,
                order_by=["created_at asc", "id asc"],
                select=select,
            ):
                items.append(_doc_to_result(dict(result), 1.0))
            next_offset = str(skip + len(items)) if len(items) == limit else None
            return items, next_offset
        except HttpResponseError as e:
            raise StorageError(f"Azure Search scroll failed on '{collection_name}': {e}") from e

    async def count(self, collection_name: str) -> int:
        client = self._get_client(collection_name)
        try:
            return await client.get_document_count()
        except HttpResponseError as e:
            raise StorageError(f"Azure Search count failed on '{collection_name}': {e}") from e

    async def delete(self, collection_name: str, ids: list[str]) -> None:
        if not ids:
            return
        client = self._get_client(collection_name)
        try:
            await client.delete_documents([{"id": id_} for id_ in ids])
        except HttpResponseError as e:
            raise StorageError(f"Azure Search delete failed on '{collection_name}': {e}") from e

    async def search_by_query_hash(
        self, collection_name: str, query_hash: str
    ) -> CacheResult | None:
        client = self._get_client(collection_name)
        filter_expr = f"query_hash eq '{_esc(query_hash)}'"
        try:
            async for result in await client.search(
                search_text=None,
                filter=filter_expr,
                top=1,
                select=_SCALAR_FIELDS,
            ):
                return _doc_to_result(dict(result), 1.0)
            return None
        except HttpResponseError as e:
            raise StorageError(
                f"Azure Search search_by_query_hash failed on '{collection_name}': {e}"
            ) from e

    async def update_usage_count(self, collection_name: str, point_id: str) -> None:
        """Increment usage_count for a document.

        Note: This is not atomic. A race condition exists if two callers update
        the same document concurrently — both may read the same value and write
        the same incremented result, causing one increment to be lost.
        """
        client = self._get_client(collection_name)
        try:
            doc = await client.get_document(key=point_id)
        except ResourceNotFoundError:
            logger.warning(
                "update_usage_count: id '%s' not found in collection '%s'",
                point_id,
                collection_name,
            )
            return
        except HttpResponseError as e:
            raise StorageError(
                f"Azure Search update_usage_count failed on '{collection_name}': {e}"
            ) from e
        try:
            await client.merge_documents([{"id": point_id, "usage_count": doc["usage_count"] + 1}])
        except HttpResponseError as e:
            raise StorageError(
                f"Azure Search update_usage_count merge failed on '{collection_name}': {e}"
            ) from e

    async def update_feedback(self, collection_name: str, point_id: str, correct: bool) -> int:
        client = self._get_client(collection_name)
        try:
            result = await client.get_document(key=point_id)
        except ResourceNotFoundError:
            return 0
        except HttpResponseError as e:
            raise StorageError(
                f"Azure Search update_feedback failed on '{collection_name}': {e}"
            ) from e
        field = "feedback_correct" if correct else "feedback_incorrect"
        new_val = int(result.get(field, 0)) + 1
        try:
            await client.merge_or_upload_documents([{"id": point_id, field: new_val}])
        except HttpResponseError as e:
            raise StorageError(
                f"Azure Search update_feedback merge failed on '{collection_name}': {e}"
            ) from e
        return new_val

    async def find_expired(self, collection_name: str) -> list[str]:
        client = self._get_client(collection_name)
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        filter_expr = f"expires_at ne null and expires_at lt {now_iso}"
        try:
            ids: list[str] = []
            async for result in await client.search(
                search_text=None,
                filter=filter_expr,
                select=["id"],
                top=10000,
            ):
                ids.append(result["id"])
            return ids
        except HttpResponseError as e:
            raise StorageError(
                f"Azure Search find_expired failed on '{collection_name}': {e}"
            ) from e

    async def search_by_normalized_question(
        self, collection_name: str, normalized_question: str
    ) -> CacheResult | None:
        client = self._get_client(collection_name)
        filter_expr = f"normalized_question eq '{_esc(normalized_question)}'"
        try:
            async for result in await client.search(
                search_text=None,
                filter=filter_expr,
                top=1,
                select=_SCALAR_FIELDS,
            ):
                return _doc_to_result(dict(result), 1.0)
            return None
        except HttpResponseError as e:
            raise StorageError(
                f"Azure Search search_by_normalized_question failed on '{collection_name}': {e}"
            ) from e

    async def find_by_query_hash(
        self, collection_name: str, query_hash: str
    ) -> list[str]:
        client = self._get_client(collection_name)
        filter_expr = f"query_hash eq '{_esc(query_hash)}'"
        try:
            ids: list[str] = []
            async for result in await client.search(
                search_text=None,
                filter=filter_expr,
                select=["id"],
                top=10000,
            ):
                ids.append(result["id"])
            return ids
        except HttpResponseError as e:
            raise StorageError(
                f"Azure Search find_by_query_hash failed on '{collection_name}': {e}"
            ) from e

    async def find_by_template_id(
        self, collection_name: str, template_id: str
    ) -> list[str]:
        client = self._get_client(collection_name)
        filter_expr = f"template_id eq '{_esc(template_id)}'"
        try:
            ids: list[str] = []
            async for result in await client.search(
                search_text=None,
                filter=filter_expr,
                select=["id"],
                top=10000,
            ):
                ids.append(result["id"])
            return ids
        except HttpResponseError as e:
            raise StorageError(
                f"Azure Search find_by_template_id failed on '{collection_name}': {e}"
            ) from e

    async def drop_collection(self, collection_name: str) -> None:
        if self._index_client is None:
            raise StorageError("Not connected. Call connect() first.")
        index_name = _az_index_name(collection_name, self._settings.azure_search_index_name)
        try:
            await self._index_client.delete_index(index_name)
            logger.info("Dropped Azure Search index '%s'", index_name)
        except ResourceNotFoundError:
            logger.warning("drop_collection: index '%s' not found", index_name)
        except HttpResponseError as e:
            raise StorageError(
                f"Azure Search drop_collection failed on '{collection_name}': {e}"
            ) from e
        self._search_clients.pop(collection_name, None)

    async def close(self) -> None:
        for client in self._search_clients.values():
            await client.close()
        self._search_clients.clear()
        if self._index_client is not None:
            await self._index_client.close()
            self._index_client = None
