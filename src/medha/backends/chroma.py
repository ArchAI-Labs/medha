"""ChromaBackend — Chroma vector storage backend."""

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any, Callable

from medha.exceptions import ConfigurationError, StorageError, StorageInitializationError
from medha.interfaces.storage import VectorStorageBackend
from medha.types import CacheEntry, CacheResult

logger = logging.getLogger(__name__)

try:
    import chromadb
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

_CHROMA_UNSAFE_RE = re.compile(r"[^a-z0-9_-]")


def _chroma_collection_name(name: str) -> str:
    safe = _CHROMA_UNSAFE_RE.sub("_", name.lower())
    return safe[:63]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _entry_to_metadata(entry: CacheEntry) -> dict[str, Any]:
    return {
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


def _meta_to_result(id_: str, score: float, meta: dict[str, Any]) -> CacheResult:
    expires_at = None
    if meta.get("expires_at"):
        try:
            expires_at = datetime.fromisoformat(meta["expires_at"])
        except (ValueError, TypeError):
            pass
    created_at = None
    if meta.get("created_at"):
        try:
            created_at = datetime.fromisoformat(meta["created_at"])
        except (ValueError, TypeError):
            pass
    return CacheResult(
        id=id_,
        score=max(0.0, min(1.0, score)),
        original_question=meta.get("original_question", ""),
        normalized_question=meta.get("normalized_question", ""),
        generated_query=meta.get("generated_query", ""),
        query_hash=meta.get("query_hash", ""),
        response_summary=meta.get("response_summary") or None,
        template_id=meta.get("template_id") or None,
        usage_count=int(meta.get("usage_count", 0)),
        created_at=created_at,
        expires_at=expires_at,
    )


class ChromaBackend(VectorStorageBackend):
    """Chroma vector backend. Supports ephemeral, persistent, and http modes.

    Only 'http' uses the native async client; the other two wrap sync calls with
    asyncio.to_thread.
    """

    def __init__(self, settings: Any = None) -> None:
        if not HAS_CHROMA:
            raise ConfigurationError(
                "chroma backend requires 'chromadb>=0.5'. "
                "Install with: pip install medha-archai[chroma]"
            )
        from medha.config import Settings
        self._settings = settings or Settings()
        self._client: Any = None
        self._is_async: bool = False
        self._collections: dict[str, Any] = {}

    async def _run(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if self._is_async:
            return await fn(*args, **kwargs)
        return await asyncio.to_thread(fn, *args, **kwargs)

    async def connect(self) -> None:
        mode = self._settings.chroma_mode
        try:
            if mode == "http":
                self._is_async = True
                extra: dict[str, Any] = {}
                if self._settings.chroma_auth_token:
                    extra["headers"] = {
                        "Authorization": f"Bearer {self._settings.chroma_auth_token.get_secret_value()}"
                    }
                self._client = await chromadb.AsyncHttpClient(
                    host=self._settings.chroma_host,
                    port=self._settings.chroma_port,
                    ssl=self._settings.chroma_ssl,
                    **extra,
                )
            elif mode == "persistent":
                self._is_async = False
                path = self._settings.chroma_persist_path or "./chroma_data"
                self._client = await asyncio.to_thread(chromadb.PersistentClient, path=path)
            else:
                self._is_async = False
                self._client = await asyncio.to_thread(chromadb.EphemeralClient)
        except Exception as e:
            raise StorageInitializationError(f"Failed to connect to Chroma ({mode}): {e}") from e

    async def initialize(self, collection_name: str, dimension: int, **kwargs: Any) -> None:
        if self._client is None:
            raise StorageError("Not connected. Call connect() first.")
        if collection_name in self._collections:
            return
        chroma_name = _chroma_collection_name(collection_name)
        try:
            collection = await self._run(
                self._client.get_or_create_collection,
                name=chroma_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._collections[collection_name] = collection
        except Exception as e:
            raise StorageInitializationError(
                f"Failed to initialize Chroma collection '{chroma_name}': {e}"
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
        cnt = await self._run(collection.count)
        if cnt == 0:
            return []
        now_iso = _now_iso()
        where = {"$or": [{"expires_at": {"$eq": ""}}, {"expires_at": {"$gt": now_iso}}]}
        try:
            result = await self._run(
                collection.query,
                query_embeddings=[vector],
                n_results=min(limit, cnt),
                where=where,
                include=["metadatas", "distances"],
            )
        except Exception as e:
            raise StorageError(f"Chroma search failed on '{collection_name}': {e}") from e

        ids = result["ids"][0]
        distances = result["distances"][0]
        metadatas = result["metadatas"][0]
        out = []
        for id_, dist, meta in zip(ids, distances, metadatas):
            score = 1.0 - dist
            if score >= score_threshold:
                out.append(_meta_to_result(id_, score, meta))
        return out

    async def upsert(self, collection_name: str, entries: list[CacheEntry]) -> None:
        if not entries:
            return
        collection = self._get_collection(collection_name)
        ids = [e.id for e in entries]
        embeddings = [e.vector for e in entries]
        metadatas = [_entry_to_metadata(e) for e in entries]
        try:
            await self._run(collection.upsert, ids=ids, embeddings=embeddings, metadatas=metadatas)
        except Exception as e:
            raise StorageError(f"Chroma upsert failed on '{collection_name}': {e}") from e

    async def scroll(
        self,
        collection_name: str,
        limit: int = 100,
        offset: str | None = None,
        with_vectors: bool = False,
    ) -> tuple[list[CacheResult], str | None]:
        collection = self._get_collection(collection_name)
        int_offset = int(offset) if offset else 0
        include = ["metadatas", "embeddings"] if with_vectors else ["metadatas"]
        try:
            result = await self._run(
                collection.get,
                limit=limit,
                offset=int_offset,
                include=include,
            )
        except Exception as e:
            raise StorageError(f"Chroma scroll failed on '{collection_name}': {e}") from e

        ids: list[str] = result.get("ids", [])
        metadatas: list[dict[str, Any]] = result.get("metadatas", [])
        cache_results = [_meta_to_result(id_, 1.0, meta) for id_, meta in zip(ids, metadatas)]
        next_offset = str(int_offset + len(ids)) if len(ids) == limit else None
        return cache_results, next_offset

    async def count(self, collection_name: str) -> int:
        collection = self._get_collection(collection_name)
        try:
            return await self._run(collection.count)
        except Exception as e:
            raise StorageError(f"Chroma count failed on '{collection_name}': {e}") from e

    async def delete(self, collection_name: str, ids: list[str]) -> None:
        if not ids:
            return
        collection = self._get_collection(collection_name)
        try:
            await self._run(collection.delete, ids=ids)
        except Exception as e:
            raise StorageError(f"Chroma delete failed on '{collection_name}': {e}") from e

    async def search_by_query_hash(
        self, collection_name: str, query_hash: str
    ) -> CacheResult | None:
        collection = self._get_collection(collection_name)
        try:
            result = await self._run(
                collection.get,
                where={"query_hash": {"$eq": query_hash}},
                limit=1,
                include=["metadatas"],
            )
        except Exception as e:
            raise StorageError(
                f"Chroma search_by_query_hash failed on '{collection_name}': {e}"
            ) from e
        ids: list[str] = result.get("ids", [])
        metadatas: list[dict[str, Any]] = result.get("metadatas", [])
        if not ids:
            return None
        return _meta_to_result(ids[0], 1.0, metadatas[0])

    async def update_usage_count(self, collection_name: str, id_: str) -> None:
        collection = self._get_collection(collection_name)
        try:
            result = await self._run(collection.get, ids=[id_], include=["metadatas"])
            ids: list[str] = result.get("ids", [])
            if not ids:
                return
            meta = dict(result["metadatas"][0])
            meta["usage_count"] = int(meta.get("usage_count", 0)) + 1
            await self._run(collection.upsert, ids=[id_], metadatas=[meta])
        except Exception as e:
            raise StorageError(
                f"Chroma update_usage_count failed on '{collection_name}': {e}"
            ) from e

    async def update_feedback(self, collection_name: str, id_: str, correct: bool) -> int:
        collection = self._get_collection(collection_name)
        try:
            result = await self._run(collection.get, ids=[id_], include=["metadatas"])
            if not result["ids"]:
                return 0
            metadata = result["metadatas"][0] or {}
            field = "feedback_correct" if correct else "feedback_incorrect"
            new_val = int(metadata.get(field, 0)) + 1
            await self._run(collection.update, ids=[id_], metadatas=[{**metadata, field: new_val}])
            return new_val
        except StorageError:
            raise
        except Exception as e:
            raise StorageError(
                f"Chroma update_feedback failed on '{collection_name}': {e}"
            ) from e

    async def find_expired(self, collection_name: str) -> list[str]:
        collection = self._get_collection(collection_name)
        now_iso = _now_iso()
        try:
            result = await self._run(
                collection.get,
                where={"$and": [{"expires_at": {"$ne": ""}}, {"expires_at": {"$lt": now_iso}}]},
                include=["metadatas"],
            )
        except Exception as e:
            raise StorageError(f"Chroma find_expired failed on '{collection_name}': {e}") from e
        return result.get("ids", [])

    async def search_by_normalized_question(
        self, collection_name: str, normalized_question: str
    ) -> CacheResult | None:
        collection = self._get_collection(collection_name)
        try:
            result = await self._run(
                collection.get,
                where={"normalized_question": {"$eq": normalized_question}},
                limit=1,
                include=["metadatas"],
            )
        except Exception as e:
            raise StorageError(
                f"Chroma search_by_normalized_question failed on '{collection_name}': {e}"
            ) from e
        ids: list[str] = result.get("ids", [])
        metadatas: list[dict[str, Any]] = result.get("metadatas", [])
        if not ids:
            return None
        return _meta_to_result(ids[0], 1.0, metadatas[0])

    async def find_by_query_hash(self, collection_name: str, query_hash: str) -> list[str]:
        collection = self._get_collection(collection_name)
        try:
            result = await self._run(
                collection.get,
                where={"query_hash": {"$eq": query_hash}},
                include=["metadatas"],
            )
        except Exception as e:
            raise StorageError(
                f"Chroma find_by_query_hash failed on '{collection_name}': {e}"
            ) from e
        return result.get("ids", [])

    async def find_by_template_id(self, collection_name: str, template_id: str) -> list[str]:
        collection = self._get_collection(collection_name)
        try:
            result = await self._run(
                collection.get,
                where={"template_id": {"$eq": template_id}},
                include=["metadatas"],
            )
        except Exception as e:
            raise StorageError(
                f"Chroma find_by_template_id failed on '{collection_name}': {e}"
            ) from e
        return result.get("ids", [])

    async def drop_collection(self, collection_name: str) -> None:
        if self._client is None:
            raise StorageError("Not connected. Call connect() first.")
        chroma_name = _chroma_collection_name(collection_name)
        try:
            if self._is_async:
                await self._client.delete_collection(name=chroma_name)
            else:
                await asyncio.to_thread(self._client.delete_collection, name=chroma_name)
            self._collections.pop(collection_name, None)
        except Exception as e:
            raise StorageError(
                f"Chroma drop_collection failed on '{collection_name}': {e}"
            ) from e

    async def close(self) -> None:
        if self._is_async and self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
        self._client = None
        self._collections.clear()
