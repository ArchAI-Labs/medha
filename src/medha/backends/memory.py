"""InMemoryBackend — zero-external-deps vector storage backend."""

import asyncio
import contextlib
import logging
from datetime import datetime, timezone
from typing import Any

from medha.exceptions import StorageError
from medha.interfaces.storage import VectorStorageBackend
from medha.types import CacheEntry, CacheResult

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity, using numpy if available."""
    try:
        import numpy as np

        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom == 0.0:
            return 0.0
        return float(np.dot(va, vb) / denom)
    except ImportError:
        dot: float = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a: float = sum(x * x for x in a) ** 0.5
        norm_b: float = sum(x * x for x in b) ** 0.5
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)


class InMemoryBackend(VectorStorageBackend):
    """Pure-Python in-process vector backend. No external dependencies required."""

    def __init__(self) -> None:
        # _store: collection_name -> {"dimension": int, "entries": {id: stored_point}}
        self._store: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        """No-op — backend is always connected."""

    async def initialize(self, collection_name: str, dimension: int, **kwargs: Any) -> None:
        if dimension <= 0:
            raise StorageError(f"dimension must be > 0, got {dimension}")
        async with self._lock:
            if collection_name not in self._store:
                self._store[collection_name] = {"dimension": dimension, "entries": {}}

    async def search(
        self,
        collection_name: str,
        vector: list[float],
        limit: int = 5,
        score_threshold: float = 0.0,
    ) -> list[CacheResult]:
        if collection_name not in self._store:
            raise StorageError(f"Collection '{collection_name}' does not exist.")

        entries = self._store[collection_name]["entries"]
        if not entries:
            return []

        now = datetime.now(timezone.utc)
        scored: list[tuple[float, dict[str, Any]]] = []
        for point in entries.values():
            ea_raw = point["payload"].get("expires_at")
            if ea_raw is not None:
                ea = _parse_dt(ea_raw)
                if ea is not None and ea <= now:
                    continue
            score = _cosine_similarity(vector, point["vector"])
            score = max(0.0, min(1.0, score))
            if score >= score_threshold:
                scored.append((score, point))

        scored.sort(key=lambda t: t[0], reverse=True)

        return [
            _point_to_cache_result(point, score)
            for score, point in scored[:limit]
        ]

    async def upsert(self, collection_name: str, entries: list[CacheEntry]) -> None:
        if collection_name not in self._store:
            raise StorageError(f"Collection '{collection_name}' does not exist.")
        if not entries:
            return
        async with self._lock:
            store_entries = self._store[collection_name]["entries"]
            for entry in entries:
                store_entries[entry.id] = {
                    "id": entry.id,
                    "vector": entry.vector,
                    "payload": {
                        "original_question": entry.original_question,
                        "normalized_question": entry.normalized_question,
                        "generated_query": entry.generated_query,
                        "query_hash": entry.query_hash,
                        "response_summary": entry.response_summary,
                        "template_id": entry.template_id,
                        "usage_count": entry.usage_count,
                        "created_at": entry.created_at.isoformat(),
                        "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
                        "feedback_correct": entry.feedback_correct,
                        "feedback_incorrect": entry.feedback_incorrect,
                    },
                }

    async def scroll(
        self,
        collection_name: str,
        limit: int = 100,
        offset: str | None = None,
        with_vectors: bool = False,
    ) -> tuple[list[CacheResult], str | None]:
        if collection_name not in self._store:
            raise StorageError(f"Collection '{collection_name}' does not exist.")

        entries = self._store[collection_name]["entries"]
        ids = list(entries.keys())
        start = int(offset) if offset is not None else 0
        page_ids = ids[start : start + limit]
        next_offset = str(start + limit) if start + limit < len(ids) else None

        results = [
            _point_to_cache_result(entries[id_], score=1.0)
            for id_ in page_ids
        ]
        return results, next_offset

    async def count(self, collection_name: str) -> int:
        return len(self._store.get(collection_name, {}).get("entries", {}))

    async def delete(self, collection_name: str, ids: list[str]) -> None:
        if not ids:
            return
        async with self._lock:
            entries = self._store.get(collection_name, {}).get("entries", {})
            for id_ in ids:
                entries.pop(id_, None)

    async def close(self) -> None:
        self._store.clear()

    async def find_expired(self, collection_name: str) -> list[str]:
        if collection_name not in self._store:
            return []
        now = datetime.now(timezone.utc)
        return [
            id_
            for id_, point in self._store[collection_name]["entries"].items()
            if (ea_raw := point["payload"].get("expires_at"))
            and (ea := _parse_dt(ea_raw)) is not None
            and ea < now
        ]

    async def search_by_normalized_question(
        self, collection_name: str, normalized_question: str
    ) -> CacheResult | None:
        if collection_name not in self._store:
            raise StorageError(f"Collection '{collection_name}' does not exist.")
        for point in self._store[collection_name]["entries"].values():
            if point["payload"]["normalized_question"] == normalized_question:
                return _point_to_cache_result(point, score=1.0)
        return None

    async def find_by_query_hash(
        self, collection_name: str, query_hash: str
    ) -> list[str]:
        if collection_name not in self._store:
            raise StorageError(f"Collection '{collection_name}' does not exist.")
        return [
            point["id"]
            for point in self._store[collection_name]["entries"].values()
            if point["payload"].get("query_hash") == query_hash
        ]

    async def find_by_template_id(
        self, collection_name: str, template_id: str
    ) -> list[str]:
        if collection_name not in self._store:
            raise StorageError(f"Collection '{collection_name}' does not exist.")
        return [
            point["id"]
            for point in self._store[collection_name]["entries"].values()
            if point["payload"].get("template_id") == template_id
        ]

    async def drop_collection(self, collection_name: str) -> None:
        async with self._lock:
            self._store.pop(collection_name, None)

    async def search_by_query_hash(
        self, collection_name: str, query_hash: str
    ) -> CacheResult | None:
        if collection_name not in self._store:
            raise StorageError(f"Collection '{collection_name}' does not exist.")

        for point in self._store[collection_name]["entries"].values():
            if point["payload"]["query_hash"] == query_hash:
                return _point_to_cache_result(point, score=1.0)
        return None

    async def update_usage_count(self, collection_name: str, point_id: str) -> None:
        async with self._lock:
            entries = self._store.get(collection_name, {}).get("entries", {})
            if point_id not in entries:
                logger.warning(
                    "update_usage_count: id '%s' not found in collection '%s'",
                    point_id,
                    collection_name,
                )
                return
            entries[point_id]["payload"]["usage_count"] += 1

    async def update_feedback(
        self, collection_name: str, point_id: str, correct: bool
    ) -> int:
        async with self._lock:
            entries = self._store.get(collection_name, {}).get("entries", {})
            if point_id not in entries:
                logger.warning(
                    "update_feedback: id '%s' not found in collection '%s'",
                    point_id,
                    collection_name,
                )
                return 0
            payload = entries[point_id]["payload"]
            if correct:
                new_value = payload.get("feedback_correct", 0) + 1
                payload["feedback_correct"] = new_value
            else:
                new_value = payload.get("feedback_incorrect", 0) + 1
                payload["feedback_incorrect"] = new_value
            return new_value


def _parse_dt(value: str) -> datetime | None:
    with contextlib.suppress(ValueError, TypeError):
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    return None


def _point_to_cache_result(point: dict[str, Any], score: float) -> CacheResult:
    payload = point["payload"]
    return CacheResult(
        id=point["id"],
        score=max(0.0, min(1.0, score)),
        original_question=payload["original_question"],
        normalized_question=payload["normalized_question"],
        generated_query=payload["generated_query"],
        query_hash=payload["query_hash"],
        response_summary=payload.get("response_summary"),
        template_id=payload.get("template_id"),
        usage_count=payload.get("usage_count", 0),
        created_at=_parse_dt(payload["created_at"]) if payload.get("created_at") else None,
        expires_at=_parse_dt(payload["expires_at"]) if payload.get("expires_at") else None,
        feedback_correct=payload.get("feedback_correct", 0),
        feedback_incorrect=payload.get("feedback_incorrect", 0),
    )
