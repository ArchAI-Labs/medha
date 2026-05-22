"""Qdrant backend supporting memory, Docker, and cloud deployment modes."""

from __future__ import annotations

import contextlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, cast

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    BinaryQuantization,
    BinaryQuantizationConfig,
    DatetimeRange,
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchValue,
    OptimizersConfigDiff,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    QuantizationSearchParams,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    SearchParams,
    TextIndexParams,
    TextIndexType,
    TokenizerType,
    VectorParams,
)

from medha.config import Settings
from medha.exceptions import StorageError, StorageInitializationError
from medha.interfaces.storage import VectorStorageBackend
from medha.types import CacheEntry, CacheResult

logger = logging.getLogger(__name__)


class QdrantBackend(VectorStorageBackend):
    """Qdrant-based vector storage backend.

    Supports three deployment modes:
        - "memory": In-process, no persistence. Best for testing.
        - "docker": Connect to a local/remote Qdrant Docker instance.
        - "cloud": Connect to Qdrant Cloud with API key.

    Args:
        settings: Medha Settings instance. If None, loads from environment.
    """

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or Settings()
        self._client: AsyncQdrantClient | None = None
        self._initialized_collections: set[str] = set()

    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            raise StorageError("Backend not connected. Call connect() first.")
        return self._client

    async def connect(self) -> None:
        """Establish connection to Qdrant based on settings.

        Must be called before any other operation.

        Raises:
            StorageInitializationError: If connection fails.
        """
        try:
            self._client = self._build_client()
            logger.info(
                "Connected to Qdrant in '%s' mode", self._settings.qdrant_mode
            )
        except Exception as e:
            raise StorageInitializationError(
                f"Failed to connect to Qdrant in '{self._settings.qdrant_mode}' mode: {e}"
            ) from e

    async def initialize(
        self, collection_name: str, dimension: int, **kwargs: Any
    ) -> None:
        """Create and configure a Qdrant collection.

        Idempotent: skips creation if collection already exists.

        Configuration includes:
            - Vector params (dimension, cosine distance)
            - Quantization (scalar INT8 by default, binary for dim >= 512)
            - HNSW parameters (m=16, ef_construct=100)
            - Optimizer config (indexing threshold, memmap threshold)
            - Payload indexes on frequently queried fields

        Args:
            collection_name: Name of the collection.
            dimension: Vector dimensionality.
            **kwargs: Override settings (hnsw_m, enable_quantization, etc.)
        """
        if collection_name in self._initialized_collections:
            return

        try:
            logger.debug("Initializing collection '%s' (dim=%d)", collection_name, dimension)
            collections = await self.client.get_collections()
            existing = {c.name for c in collections.collections}

            if collection_name not in existing:
                quantization = self._build_quantization_config(dimension, **kwargs)
                hnsw = self._build_hnsw_config(**kwargs)

                await self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=dimension,
                        distance=Distance.COSINE,
                        on_disk=self._settings.on_disk,
                    ),
                    quantization_config=quantization,
                    hnsw_config=hnsw,
                    optimizers_config=OptimizersConfigDiff(
                        indexing_threshold=20000,
                        memmap_threshold=50000,
                    ),
                )
                logger.info(
                    "Created collection '%s' (dim=%d)", collection_name, dimension
                )

                # Create payload indexes only on main collection (not template collections)
                # and only when not in memory mode (indexes have no effect in memory mode
                # and Qdrant emits a UserWarning when they are created there).
                if (
                    not collection_name.startswith("__medha_templates_")
                    and self._settings.qdrant_mode != "memory"
                ):
                    try:
                        await self._create_payload_indexes(collection_name)
                    except Exception as idx_err:
                        logger.warning(
                            "Payload indexes not created on '%s' (non-critical): %s",
                            collection_name,
                            idx_err,
                        )
            else:
                logger.info(
                    "Collection '%s' already exists, skipping creation",
                    collection_name,
                )

            self._initialized_collections.add(collection_name)
        except (StorageError, StorageInitializationError):
            raise
        except Exception as e:
            raise StorageInitializationError(
                f"Failed to initialize collection '{collection_name}': {e}"
            ) from e

    async def search(
        self,
        collection_name: str,
        vector: list[float],
        limit: int = 5,
        score_threshold: float = 0.0,
    ) -> list[CacheResult]:
        """Search for similar vectors using query_points.

        Args:
            collection_name: Collection to search.
            vector: Query vector.
            limit: Max number of results.
            score_threshold: Minimum similarity score (0.0 - 1.0).

        Returns:
            List of CacheResult, sorted by descending score.

        Raises:
            StorageError: If the search fails.
        """
        try:
            logger.debug(
                "Searching '%s': limit=%d, threshold=%.3f",
                collection_name,
                limit,
                score_threshold,
            )
            search_params = self._build_search_params()
            now_iso = datetime.now(timezone.utc).isoformat()
            ttl_filter = Filter(
                must_not=[
                    FieldCondition(
                        key="expires_at",
                        range=DatetimeRange(lte=now_iso),
                    )
                ]
            )
            response = await self.client.query_points(
                collection_name=collection_name,
                query=vector,
                limit=limit,
                score_threshold=score_threshold if score_threshold > 0.0 else None,
                search_params=search_params,
                query_filter=ttl_filter,
                with_payload=True,
            )
            results = [self._point_to_cache_result(point) for point in response.points]
            if results:
                logger.debug(
                    "Search '%s' returned %d results (top score=%.4f)",
                    collection_name,
                    len(results),
                    results[0].score,
                )
            else:
                logger.debug("Search '%s' returned 0 results", collection_name)
            return results
        except StorageError:
            raise
        except Exception as e:
            raise StorageError(
                f"Qdrant search failed on '{collection_name}': {e}"
            ) from e

    async def upsert(self, collection_name: str, entries: list[CacheEntry]) -> None:
        """Insert or update cache entries in batches.

        Args:
            collection_name: Target collection.
            entries: List of CacheEntry objects to upsert.

        Raises:
            StorageError: If the upsert fails.
        """
        try:
            points = [self._entry_to_point(e) for e in entries]
            batch_size = self._settings.batch_size

            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                await self.client.upsert(
                    collection_name=collection_name,
                    wait=True,
                    points=batch,
                )
                logger.info(
                    "Upserted batch %d: %d points", i // batch_size + 1, len(batch)
                )
        except StorageError:
            raise
        except Exception as e:
            raise StorageError(
                f"Qdrant upsert failed on '{collection_name}': {e}"
            ) from e

    async def scroll(
        self,
        collection_name: str,
        limit: int = 100,
        offset: str | None = None,
        with_vectors: bool = False,
    ) -> tuple[list[CacheResult], str | None]:
        """Iterate over all points in a collection.

        Args:
            collection_name: Collection to scroll.
            limit: Batch size per scroll.
            offset: Pagination token from a previous scroll.
            with_vectors: Whether to include vectors in results.

        Returns:
            Tuple of (results, next_offset). next_offset is None when done.

        Raises:
            StorageError: If the scroll fails.
        """
        try:
            records, next_offset = await self.client.scroll(
                collection_name=collection_name,
                limit=limit,
                offset=offset,
                with_vectors=with_vectors,
                with_payload=True,
            )

            results = []
            for record in records:
                payload = record.payload or {}
                results.append(
                    CacheResult(
                        id=str(record.id),
                        score=0.0,
                        original_question=payload.get("original_question", ""),
                        normalized_question=payload.get("normalized_question", ""),
                        generated_query=payload.get("generated_query", ""),
                        query_hash=payload.get("query_hash", ""),
                        response_summary=payload.get("response_summary"),
                        template_id=payload.get("template_id"),
                        usage_count=payload.get("usage_count", 0),
                        created_at=payload.get("created_at"),
                    )
                )

            next_offset_str = str(next_offset) if next_offset is not None else None
            logger.debug(
                "Scroll '%s': returned %d records, has_more=%s",
                collection_name,
                len(results),
                next_offset_str is not None,
            )
            return results, next_offset_str
        except StorageError:
            raise
        except Exception as e:
            raise StorageError(
                f"Qdrant scroll failed on '{collection_name}': {e}"
            ) from e

    async def count(self, collection_name: str) -> int:
        """Return the number of points in a collection.

        Raises:
            StorageError: If the count fails.
        """
        try:
            result = await self.client.count(collection_name=collection_name)
            return result.count
        except StorageError:
            raise
        except Exception as e:
            raise StorageError(
                f"Qdrant count failed on '{collection_name}': {e}"
            ) from e

    async def delete(self, collection_name: str, ids: list[str]) -> None:
        """Delete points by ID.

        Args:
            collection_name: Target collection.
            ids: List of point IDs to delete.

        Raises:
            StorageError: If the delete fails.
        """
        try:
            await self.client.delete(
                collection_name=collection_name,
                points_selector=PointIdsList(points=cast("list[int | str | uuid.UUID]", ids)),
                wait=True,
            )
            logger.info("Deleted %d points from '%s'", len(ids), collection_name)
        except StorageError:
            raise
        except Exception as e:
            raise StorageError(
                f"Qdrant delete failed on '{collection_name}': {e}"
            ) from e

    async def find_expired(self, collection_name: str) -> list[str]:
        try:
            now_iso = datetime.now(timezone.utc).isoformat()
            results, _ = await self.client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="expires_at",
                            range=DatetimeRange(lt=now_iso),
                        )
                    ]
                ),
                with_payload=False,
                with_vectors=False,
                limit=1000,
            )
            return [str(p.id) for p in results]
        except StorageError:
            raise
        except Exception as e:
            raise StorageError(
                f"Qdrant find_expired failed on '{collection_name}': {e}"
            ) from e

    async def close(self) -> None:
        """Close the Qdrant client."""
        if self._client is not None:
            await self._client.close()
            self._client = None
            self._initialized_collections.clear()
            logger.info("Qdrant client closed")

    # --- Collection-specific operations ---

    async def search_by_normalized_question(
        self, collection_name: str, normalized_question: str
    ) -> CacheResult | None:
        try:
            results, _ = await self.client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="normalized_question",
                            match=MatchValue(value=normalized_question),
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
            )
            if not results:
                return None
            record = results[0]
            payload = record.payload or {}
            return CacheResult(
                id=str(record.id),
                score=1.0,
                original_question=payload.get("original_question", ""),
                normalized_question=payload.get("normalized_question", ""),
                generated_query=payload.get("generated_query", ""),
                query_hash=payload.get("query_hash", ""),
                response_summary=payload.get("response_summary"),
                template_id=payload.get("template_id"),
                usage_count=payload.get("usage_count", 0),
                created_at=payload.get("created_at"),
            )
        except StorageError:
            raise
        except Exception as e:
            raise StorageError(
                f"Qdrant search_by_normalized_question failed on '{collection_name}': {e}"
            ) from e

    async def find_by_query_hash(
        self, collection_name: str, query_hash: str
    ) -> list[str]:
        try:
            ids: list[str] = []
            offset: str | None = None
            while True:
                results, next_offset = await self.client.scroll(
                    collection_name=collection_name,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="query_hash",
                                match=MatchValue(value=query_hash),
                            )
                        ]
                    ),
                    limit=1000,
                    offset=offset,
                    with_payload=False,
                )
                ids.extend(str(r.id) for r in results)
                if next_offset is None:
                    break
                offset = str(next_offset)
            return ids
        except StorageError:
            raise
        except Exception as e:
            raise StorageError(
                f"Qdrant find_by_query_hash failed on '{collection_name}': {e}"
            ) from e

    async def find_by_template_id(
        self, collection_name: str, template_id: str
    ) -> list[str]:
        try:
            ids: list[str] = []
            offset: str | None = None
            while True:
                results, next_offset = await self.client.scroll(
                    collection_name=collection_name,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="template_id",
                                match=MatchValue(value=template_id),
                            )
                        ]
                    ),
                    limit=1000,
                    offset=offset,
                    with_payload=False,
                )
                ids.extend(str(r.id) for r in results)
                if next_offset is None:
                    break
                offset = str(next_offset)
            return ids
        except StorageError:
            raise
        except Exception as e:
            raise StorageError(
                f"Qdrant find_by_template_id failed on '{collection_name}': {e}"
            ) from e

    async def drop_collection(self, collection_name: str) -> None:
        try:
            await self.client.delete_collection(collection_name)
            self._initialized_collections.discard(collection_name)
            logger.info("Dropped collection '%s'", collection_name)
        except StorageError:
            raise
        except Exception as e:
            raise StorageError(
                f"Qdrant drop_collection failed on '{collection_name}': {e}"
            ) from e

    async def search_by_query_hash(
        self,
        collection_name: str,
        query_hash: str,
    ) -> CacheResult | None:
        """Find a cache entry by its query hash (exact payload filter).

        Used to check if a template-generated query already has a cached response.

        Args:
            collection_name: Collection to search.
            query_hash: MD5 hash of the generated query.

        Returns:
            CacheResult if found, None otherwise.
        """
        try:
            results, _ = await self.client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="query_hash",
                            match=MatchValue(value=query_hash),
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
            )

            if not results:
                return None

            record = results[0]
            payload = record.payload or {}
            return CacheResult(
                id=str(record.id),
                score=1.0,
                original_question=payload.get("original_question", ""),
                normalized_question=payload.get("normalized_question", ""),
                generated_query=payload.get("generated_query", ""),
                query_hash=payload.get("query_hash", ""),
                response_summary=payload.get("response_summary"),
                template_id=payload.get("template_id"),
                usage_count=payload.get("usage_count", 0),
                created_at=payload.get("created_at"),
            )
        except StorageError:
            raise
        except Exception as e:
            raise StorageError(
                f"Qdrant search_by_query_hash failed on '{collection_name}': {e}"
            ) from e

    async def update_usage_count(
        self, collection_name: str, point_id: str
    ) -> None:
        """Increment the usage_count for a specific point.

        Used for cache analytics and potential eviction policies.
        """
        try:
            points = await self.client.retrieve(
                collection_name=collection_name,
                ids=[point_id],
                with_payload=True,
            )

            if not points:
                logger.warning(
                    "Point '%s' not found in '%s'", point_id, collection_name
                )
                return

            current_count = (points[0].payload or {}).get("usage_count", 0)

            await self.client.set_payload(
                collection_name=collection_name,
                payload={"usage_count": current_count + 1},
                points=[point_id],
                wait=True,
            )
            logger.debug(
                "Updated usage_count for '%s' to %d", point_id, current_count + 1
            )
        except StorageError:
            raise
        except Exception as e:
            raise StorageError(
                f"Qdrant update_usage_count failed on '{collection_name}': {e}"
            ) from e

    async def update_feedback(
        self, collection_name: str, point_id: str, correct: bool
    ) -> int:
        """Increment feedback_correct or feedback_incorrect for a stored entry."""
        try:
            points = await self.client.retrieve(
                collection_name=collection_name,
                ids=[point_id],
                with_payload=True,
            )

            if not points:
                logger.warning(
                    "Point '%s' not found in '%s'", point_id, collection_name
                )
                return 0

            payload = points[0].payload or {}
            field = "feedback_correct" if correct else "feedback_incorrect"
            new_val = payload.get(field, 0) + 1

            await self.client.set_payload(
                collection_name=collection_name,
                payload={field: new_val},
                points=[point_id],
                wait=True,
            )
            logger.debug(
                "Updated %s for '%s' to %d", field, point_id, new_val
            )
            return new_val
        except StorageError:
            raise
        except Exception as e:
            raise StorageError(
                f"Qdrant update_feedback failed on '{collection_name}': {e}"
            ) from e

    # --- Private methods ---

    def _build_client(self) -> AsyncQdrantClient:
        """Create the Qdrant client based on settings.mode."""
        mode = self._settings.qdrant_mode
        logger.debug("Building Qdrant client for mode='%s'", mode)

        if mode == "memory":
            return AsyncQdrantClient(":memory:")
        elif mode == "docker":
            url = self._settings.qdrant_url or (
                f"http://{self._settings.qdrant_host}:{self._settings.qdrant_port}"
            )
            logger.debug("Qdrant Docker URL: %s", url)
            return AsyncQdrantClient(url=url)
        elif mode == "cloud":
            logger.debug("Qdrant Cloud URL: %s", self._settings.qdrant_url)
            return AsyncQdrantClient(
                url=self._settings.qdrant_url,
                api_key=self._settings.qdrant_api_key.get_secret_value() if self._settings.qdrant_api_key else None,
            )
        else:
            raise StorageInitializationError(f"Unknown qdrant_mode: '{mode}'")

    def _build_quantization_config(self, dimension: int, **kwargs: Any) -> Any:
        """Choose quantization config based on dimension and settings.

        When ``on_disk=True`` and ``quantization_always_ram=True`` (the defaults
        for hybrid storage), original vectors live on disk while quantized
        vectors stay in RAM — giving a good balance of speed and memory.
        """
        enable = kwargs.get("enable_quantization", self._settings.enable_quantization)
        if not enable:
            return None

        q_type = kwargs.get("quantization_type", self._settings.quantization_type)
        always_ram = kwargs.get(
            "quantization_always_ram", self._settings.quantization_always_ram
        )

        if q_type == "binary" and dimension >= 512:
            return BinaryQuantization(
                binary=BinaryQuantizationConfig(always_ram=always_ram)
            )

        # Default: Scalar INT8
        return ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8,
                quantile=0.99,
                always_ram=always_ram,
            )
        )

    def _build_hnsw_config(self, **kwargs: Any) -> HnswConfigDiff:
        """Build HNSW config from settings."""
        return HnswConfigDiff(
            m=kwargs.get("hnsw_m", self._settings.hnsw_m),
            ef_construct=kwargs.get(
                "hnsw_ef_construct", self._settings.hnsw_ef_construct
            ),
            full_scan_threshold=10000,
        )

    def _build_search_params(self) -> SearchParams:
        """Build search params with quantization-aware settings.

        Controls how quantized vectors are used at query time:
        - **ignore**: bypass quantization entirely (use original vectors).
        - **rescore**: re-evaluate top candidates with original vectors for
          better accuracy.  Disable when originals are on slow storage.
        - **oversampling**: fetch ``limit * oversampling`` candidates from the
          quantized index before re-scoring.  Higher values improve recall at
          the cost of latency.
        """
        return SearchParams(
            quantization=QuantizationSearchParams(
                ignore=self._settings.quantization_ignore,
                rescore=self._settings.quantization_rescore,
                oversampling=self._settings.quantization_oversampling,
            )
        )

    async def _create_payload_indexes(self, collection_name: str) -> None:
        """Create payload indexes for fast filtering."""
        # KEYWORD indexes
        for field in ("template_id", "query_hash"):
            await self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )

        # INTEGER index
        await self.client.create_payload_index(
            collection_name=collection_name,
            field_name="usage_count",
            field_schema=PayloadSchemaType.INTEGER,
        )

        # TEXT index with word tokenizer
        await self.client.create_payload_index(
            collection_name=collection_name,
            field_name="normalized_question",
            field_schema=TextIndexParams(
                type=TextIndexType.TEXT,
                tokenizer=TokenizerType.WORD,
                min_token_len=2,
                max_token_len=20,
                lowercase=True,
            ),
        )

        logger.info("Created payload indexes on '%s'", collection_name)

    @staticmethod
    def _point_to_cache_result(point: Any) -> CacheResult:
        """Convert a Qdrant ScoredPoint to a CacheResult."""
        payload = point.payload or {}
        score = point.score if point.score is not None else 0.0
        score = max(0.0, min(1.0, score))

        expires_at: datetime | None = None
        if raw_ea := payload.get("expires_at"):
            with contextlib.suppress(ValueError, TypeError):
                expires_at = datetime.fromisoformat(str(raw_ea))
                if expires_at.tzinfo is None:
                    expires_at = expires_at.replace(tzinfo=timezone.utc)

        return CacheResult(
            id=str(point.id),
            score=score,
            original_question=payload.get("original_question", ""),
            normalized_question=payload.get("normalized_question", ""),
            generated_query=payload.get("generated_query", ""),
            query_hash=payload.get("query_hash", ""),
            response_summary=payload.get("response_summary"),
            template_id=payload.get("template_id"),
            usage_count=payload.get("usage_count", 0),
            created_at=payload.get("created_at"),
            expires_at=expires_at,
        )

    @staticmethod
    def _entry_to_point(entry: CacheEntry) -> PointStruct:
        """Convert a CacheEntry to a Qdrant PointStruct."""
        return PointStruct(
            id=entry.id,
            vector=entry.vector,
            payload={
                "original_question": entry.original_question,
                "normalized_question": entry.normalized_question,
                "generated_query": entry.generated_query,
                "query_hash": entry.query_hash,
                "response_summary": entry.response_summary,
                "template_id": entry.template_id,
                "usage_count": entry.usage_count,
                "created_at": entry.created_at.isoformat(),
                "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
            },
        )
