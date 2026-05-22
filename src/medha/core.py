"""Core Medha class implementing the waterfall search strategy."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from collections import OrderedDict, deque
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_UNSET = object()

from medha.config import Settings
from medha.exceptions import ConfigurationError, EmbeddingError, MedhaError, StorageError, TemplateError
from medha.interfaces.embedder import BaseEmbedder
from medha.interfaces.l1_cache import L1CacheBackend
from medha.interfaces.storage import VectorStorageBackend
from medha.types import CacheEntry, CacheHit, CacheResult, CacheStats, QueryTemplate, SearchStrategy, StrategyStats
from medha.utils.nlp import ParameterExtractor, keyword_overlap_score
from medha.utils.normalization import normalize_question, query_hash, question_hash

logger = logging.getLogger(__name__)

_HIT_STRATEGIES = frozenset({
    SearchStrategy.L1_CACHE,
    SearchStrategy.TEMPLATE_MATCH,
    SearchStrategy.EXACT_MATCH,
    SearchStrategy.SEMANTIC_MATCH,
    SearchStrategy.FUZZY_MATCH,
})


class _StatsCollector:
    """Thread-safe async collector for cache performance metrics."""

    def __init__(self, enabled: bool = True, max_latency_samples: int = 10_000) -> None:
        self._enabled = enabled
        self._max_latency_samples = max_latency_samples
        self._lock = asyncio.Lock()
        self._reset_state()

    def _reset_state(self) -> None:
        self._total_requests = 0
        self._total_hits = 0
        self._total_misses = 0
        self._total_errors = 0
        self._total_latency_ms = 0.0
        self._by_strategy: dict[str, dict[str, float | int]] = {}
        self._latencies: deque[float] = deque(maxlen=self._max_latency_samples)
        self._since = datetime.now(timezone.utc)

    async def record(self, strategy: SearchStrategy, latency_ms: float) -> None:
        if not self._enabled:
            return
        async with self._lock:
            self._total_requests += 1
            self._total_latency_ms += latency_ms
            self._latencies.append(latency_ms)

            key = strategy.value
            if key not in self._by_strategy:
                self._by_strategy[key] = {"count": 0, "total_latency_ms": 0.0}
            self._by_strategy[key]["count"] = int(self._by_strategy[key]["count"]) + 1
            self._by_strategy[key]["total_latency_ms"] = float(self._by_strategy[key]["total_latency_ms"]) + latency_ms

            if strategy in _HIT_STRATEGIES:
                self._total_hits += 1
            elif strategy == SearchStrategy.ERROR:
                self._total_errors += 1
            else:
                self._total_misses += 1

    async def snapshot(self, backend_count: int) -> CacheStats:
        async with self._lock:
            until = datetime.now(timezone.utc)
            sorted_latencies = sorted(self._latencies)
            n = len(sorted_latencies)

            def _pct(p: float) -> float:
                if n == 0:
                    return 0.0
                return sorted_latencies[min(int(n * p), n - 1)]

            by_strategy = {
                k: StrategyStats(
                    count=int(v["count"]),
                    total_latency_ms=float(v["total_latency_ms"]),
                )
                for k, v in self._by_strategy.items()
            }

            return CacheStats(
                by_strategy=by_strategy,
                total_requests=self._total_requests,
                total_hits=self._total_hits,
                total_misses=self._total_misses,
                total_errors=self._total_errors,
                total_latency_ms=self._total_latency_ms,
                p50_latency_ms=_pct(0.50),
                p95_latency_ms=_pct(0.95),
                p99_latency_ms=_pct(0.99),
                backend_count=backend_count,
                since=self._since,
                until=until,
            )

    async def reset(self) -> None:
        async with self._lock:
            self._reset_state()


class Medha:
    """Semantic Memory for AI Text-to-Query systems.

    Provides a multi-tier "waterfall" cache search strategy to maximize
    cache hits and minimize LLM costs.

    Args:
        collection_name: Name of the cache collection.
        embedder: An instance of BaseEmbedder (e.g., FastEmbedAdapter).
        backend: An instance of VectorStorageBackend (e.g., QdrantBackend).
            If None, creates a QdrantBackend from settings.
        settings: Configuration. If None, loads from environment.
        templates: Pre-loaded query templates. If None, loads from
            settings.template_file (if configured).

    Example:
        >>> from medha import Medha, Settings
        >>> from medha.embeddings.fastembed_adapter import FastEmbedAdapter
        >>>
        >>> embedder = FastEmbedAdapter()
        >>> medha = Medha(collection_name="my_cache", embedder=embedder)
        >>> await medha.start()
        >>>
        >>> # Store a question-query pair
        >>> await medha.store("How many users?", "SELECT COUNT(*) FROM users")
        >>>
        >>> # Search
        >>> hit = await medha.search("Show me user count")
        >>> print(hit.generated_query)  # "SELECT COUNT(*) FROM users"
        >>> print(hit.strategy)         # SearchStrategy.SEMANTIC_MATCH
        >>>
        >>> await medha.close()
    """

    def __init__(
        self,
        collection_name: str,
        embedder: BaseEmbedder,
        backend: VectorStorageBackend | None = None,
        settings: Settings | None = None,
        templates: list[QueryTemplate] | None = None,
        l1_backend: L1CacheBackend | None = None,
    ):
        self._collection_name = collection_name
        self._template_collection = f"__medha_templates_{collection_name}"
        self._embedder = embedder
        self._settings = settings or Settings()
        self._templates = templates or []

        # Backend: use provided instance or build from settings.backend_type
        self._backend = backend if backend is not None else self._build_backend()

        # L1 cache (Tier 0) — pluggable: in-memory (default) or Redis
        if l1_backend is not None:
            self._l1_backend = l1_backend
        else:
            from medha.l1_cache.memory import InMemoryL1Cache
            self._l1_backend = InMemoryL1Cache(max_size=self._settings.l1_cache_max_size)

        # Embedding cache (avoids redundant embedding calls)
        self._embedding_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._embedding_cache_max = 10000
        self._embedding_cache_lock = asyncio.Lock()

        # Deduplication: tracks in-flight embedding computations
        self._pending_embeddings: dict[str, asyncio.Future[list[float]]] = {}

        # NLP parameter extractor
        self._param_extractor = ParameterExtractor()

        # Stats
        self._stats = _StatsCollector(
            enabled=self._settings.collect_stats,
            max_latency_samples=self._settings.stats_max_latency_samples,
        )
        self._total_stored = 0
        self._warm_loaded = 0
        self._cleanup_task: asyncio.Task[None] | None = None
        self._known_collections = [self._collection_name]

    # --- Private helpers ---

    def _build_backend(self) -> VectorStorageBackend:
        """Instantiate the correct vector backend from settings.backend_type.

        Called only when no backend is passed to __init__. Allows users to
        configure the backend via Settings without importing backend classes.

        Returns:
            A VectorStorageBackend instance (not yet connected/initialized).

        Raises:
            ConfigurationError: If backend_type is unknown or required deps are missing.
        """
        bt = self._settings.backend_type
        if bt == "qdrant":
            from medha.backends.qdrant import QdrantBackend
            return QdrantBackend(self._settings)
        elif bt == "memory":
            from medha.backends.memory import InMemoryBackend
            return InMemoryBackend()
        elif bt == "pgvector":
            from medha.backends.pgvector import PgVectorBackend
            return PgVectorBackend(self._settings)
        elif bt == "elasticsearch":
            from medha.backends.elasticsearch import ElasticsearchBackend
            return ElasticsearchBackend(self._settings)
        elif bt == "vectorchord":
            from medha.backends.vectorchord import VectorChordBackend
            return VectorChordBackend(self._settings)
        elif bt == "chroma":
            from medha.backends.chroma import ChromaBackend
            return ChromaBackend(self._settings)
        elif bt == "weaviate":
            from medha.backends.weaviate import WeaviateBackend
            return WeaviateBackend(self._settings)
        elif bt == "redis":
            from medha.backends.redis_vector import RedisVectorBackend
            return RedisVectorBackend(self._settings)
        elif bt == "azure-search":
            from medha.backends.azure_search import AzureSearchBackend
            return AzureSearchBackend(self._settings)
        elif bt == "lancedb":
            from medha.backends.lancedb import LanceDBBackend
            return LanceDBBackend(self._settings)
        else:
            raise ConfigurationError(f"Unknown backend_type: '{bt}'")

    # --- Lifecycle ---

    async def start(self) -> None:
        """Initialize the backend and sync templates.

        Must be called before search/store operations.

        Steps:
            1. Connect to the vector backend.
            2. Initialize the main collection.
            3. Initialize the template collection.
            4. Load templates from file (if configured).
            5. Sync templates to the template collection.
        """
        logger.debug(
            "Starting Medha: collection='%s', embedder=%s, backend=%s",
            self._collection_name,
            type(self._embedder).__name__,
            type(self._backend).__name__,
        )
        if hasattr(self._backend, "connect"):
            await self._backend.connect()

        dimension = self._embedder.dimension
        logger.debug("Embedder dimension: %d", dimension)

        await self._backend.initialize(self._collection_name, dimension)
        await self._backend.initialize(self._template_collection, dimension)

        # Warn once if a legacy-named template collection still exists
        legacy_collection = f"{self._collection_name}_templates"
        try:
            legacy_count = await self._backend.count(legacy_collection)
            if legacy_count > 0:
                logger.warning(
                    "Legacy template collection '%s' found with %d entries. "
                    "Templates will be re-synced to '%s'. "
                    "Delete the old collection manually when ready.",
                    legacy_collection,
                    legacy_count,
                    self._template_collection,
                )
        except StorageError:
            pass  # Old collection does not exist — fresh deployment

        if self._settings.template_file and not self._templates:
            await self.load_templates_from_file(self._settings.template_file)

        if self._templates:
            await self._sync_templates_to_backend()

        # Load persistent embedding cache from disk (if configured)
        if self._settings.embedding_cache_path:
            self._load_embedding_cache_from_disk()

        if self._settings.cleanup_interval_seconds:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info(
            "Medha started: collection='%s', templates=%d",
            self._collection_name,
            len(self._templates),
        )

    async def close(self) -> None:
        """Shut down the backend and release resources."""
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._cleanup_task
            self._cleanup_task = None
        if self._settings.embedding_cache_path:
            self._save_embedding_cache_to_disk()
        await self._l1_backend.close()
        await self._backend.close()
        logger.info("Medha closed")

    async def __aenter__(self) -> Medha:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> bool:
        await self.close()
        return False

    # --- Waterfall Search ---

    async def search(self, question: str) -> CacheHit:
        """Search the cache using the waterfall strategy.

        Tiers (checked in order, first hit wins):
            0. L1 In-Memory Cache (exact hash match)
            1. Template Matching (intent recognition + parameter extraction)
            2. Exact Vector Match (score >= score_threshold_exact)
            3. Semantic Similarity (score >= score_threshold_semantic)
            4. Fuzzy Matching (Levenshtein distance, optional)

        Args:
            question: Natural language question from the user.

        Returns:
            CacheHit with the matched query, confidence, and strategy.
            Returns CacheHit(strategy=NO_MATCH) if no tier matches.
        """
        t0 = time.monotonic()
        result = await self._search_impl(question)
        latency_ms = (time.monotonic() - t0) * 1000
        await self._stats.record(result.strategy, latency_ms)
        return result

    async def _search_impl(self, question: str) -> CacheHit:
        try:
            if not question or not question.strip():
                logger.warning("Search called with empty question")
                return CacheHit(strategy=SearchStrategy.ERROR)

            if len(question) > self._settings.max_question_length:
                logger.warning(
                    "Search rejected: question length %d exceeds max %d",
                    len(question),
                    self._settings.max_question_length,
                )
                return CacheHit(strategy=SearchStrategy.ERROR)

            logger.debug("Search started for: '%s'", question[:80])

            # --- Tier 0: L1 Cache ---
            l1_hit = await self._check_l1_cache(question)
            if l1_hit:
                logger.debug("Tier 0 L1 cache HIT for: '%s'", question[:50])
                return l1_hit
            logger.debug("Tier 0 L1 cache MISS")

            # --- Tier 1: Template Matching ---
            template_hit = await self._search_templates(question)
            if template_hit:
                await self._store_in_l1(question, template_hit)
                logger.debug(
                    "Tier 1 template HIT: template='%s', confidence=%.3f",
                    template_hit.template_used,
                    template_hit.confidence,
                )
                return template_hit
            logger.debug("Tier 1 template MISS")

            # --- Get embedding (shared by Tier 2, 3) ---
            embedding = await self._get_embedding(question)
            if embedding is None:
                logger.error("Embedding failed, aborting search for: '%s'", question[:50])
                return CacheHit(strategy=SearchStrategy.ERROR)

            # --- Tier 2 + 3: Exact and Semantic in parallel ---
            # Running them concurrently reduces wall-clock latency from
            # ~(t_exact + t_semantic) to ~max(t_exact, t_semantic).
            exact_hit, semantic_hit = await asyncio.gather(
                self._search_exact(embedding),
                self._search_semantic(embedding),
            )

            if exact_hit:
                await self._store_in_l1(question, exact_hit)
                logger.debug(
                    "Tier 2 exact HIT: confidence=%.4f, query='%s'",
                    exact_hit.confidence,
                    exact_hit.generated_query[:50] if exact_hit.generated_query else "",
                )
                return exact_hit
            logger.debug("Tier 2 exact MISS (threshold=%.2f)", self._settings.score_threshold_exact)

            if semantic_hit:
                await self._store_in_l1(question, semantic_hit)
                logger.debug(
                    "Tier 3 semantic HIT: confidence=%.4f, query='%s'",
                    semantic_hit.confidence,
                    semantic_hit.generated_query[:50] if semantic_hit.generated_query else "",
                )
                return semantic_hit
            logger.debug("Tier 3 semantic MISS (threshold=%.2f)", self._settings.score_threshold_semantic)

            # --- Tier 4: Fuzzy Matching ---
            fuzzy_hit = await self._search_fuzzy(question, embedding)
            if fuzzy_hit:
                await self._store_in_l1(question, fuzzy_hit)
                logger.debug("Tier 4 fuzzy HIT: confidence=%.4f", fuzzy_hit.confidence)
                return fuzzy_hit
            logger.debug("Tier 4 fuzzy MISS (threshold=%.1f)", self._settings.score_threshold_fuzzy)

            logger.debug("All tiers exhausted, NO_MATCH for: '%s'", question[:50])
            return CacheHit(strategy=SearchStrategy.NO_MATCH)

        except Exception as e:
            logger.error("Search failed for '%s': %s", question[:50], e, exc_info=True)
            return CacheHit(strategy=SearchStrategy.ERROR)

    # --- Tier Implementations ---

    async def _check_l1_cache(self, question: str) -> CacheHit | None:
        """Check the L1 cache (pluggable backend).

        Key: MD5 hash of normalized question.
        Returns: CacheHit with strategy=L1_CACHE if found and not expired, None otherwise.
        """
        key = question_hash(question)
        hit = await self._l1_backend.get(key)
        if hit is not None:
            if hit.expires_at is not None and hit.expires_at <= datetime.now(timezone.utc):
                await self._l1_backend.invalidate(key)
                return None
            return hit.model_copy(update={"strategy": SearchStrategy.L1_CACHE})
        return None

    async def _store_in_l1(self, question: str, hit: CacheHit) -> None:
        """Store a result in the L1 cache."""
        key = question_hash(question)
        await self._l1_backend.set(key, hit)
        logger.debug(
            "L1 cache store: key=%s, strategy=%s",
            key[:8],
            hit.strategy.value if hit.strategy else "?",
        )

    async def _search_templates(self, question: str) -> CacheHit | None:
        """Search for template matches using parameter extraction + keyword scoring.

        Iterates over all loaded templates and attempts parameter extraction.
        Templates where all required parameters are successfully extracted are
        scored using keyword overlap and parameter completeness.

        Steps:
            1. For each template, try to extract parameters from the question.
            2. Skip templates where extraction fails or is incomplete.
            3. Score candidates: keyword_overlap * 0.5 + param_completeness * 0.3
               + priority_bonus.
            4. Return the best match above the configured threshold.
        """
        if not self._templates:
            logger.debug("Template search skipped: no templates loaded")
            return None

        best_hit: CacheHit | None = None
        best_score = 0.0
        normalized = normalize_question(question)

        for template in self._templates:
            # Try to extract parameters
            try:
                params = self._param_extractor.extract(question, template)
            except Exception:
                logger.debug(
                    "Parameter extraction failed for template '%s'", template.intent
                )
                continue

            # All parameters must be extracted for a valid match
            if template.parameters and len(params) != len(template.parameters):
                logger.debug(
                    "Incomplete params for template '%s': got %d/%d",
                    template.intent,
                    len(params),
                    len(template.parameters),
                )
                continue

            # Compute score from keyword overlap + param completeness
            keyword_bonus = keyword_overlap_score(normalized, template.template_text)
            param_completeness = 1.0 if not template.parameters else (
                len(params) / len(template.parameters)
            )
            final_score = (
                (keyword_bonus * 0.5)
                + (param_completeness * 0.3)
            )
            # Priority bonus: priority 1 (highest) gets most bonus
            final_score += (5 - template.priority) * 0.02

            logger.debug(
                "Template '%s': keyword=%.2f, params=%d/%d, score=%.3f",
                template.intent,
                keyword_bonus,
                len(params),
                len(template.parameters) if template.parameters else 0,
                final_score,
            )

            if final_score > best_score:
                best_score = final_score
                try:
                    rendered_query = self._param_extractor.render_query(
                        template, params
                    )
                except Exception:
                    continue

                best_hit = CacheHit(
                    generated_query=rendered_query,
                    confidence=min(final_score, 1.0),
                    strategy=SearchStrategy.TEMPLATE_MATCH,
                    template_used=template.intent,
                )

        if best_hit is None:
            logger.debug("Template search: no template could extract all required parameters")
            return None
        if best_score < self._settings.score_threshold_template:
            logger.debug(
                "Template search: best_score=%.3f below threshold=%.3f",
                best_score,
                self._settings.score_threshold_template,
            )
            return None
        return best_hit

    async def _search_exact(self, embedding: list[float]) -> CacheHit | None:
        """Search for exact vector match (score >= score_threshold_exact).

        Uses settings.score_threshold_exact (default 0.99).
        Returns the top-1 result if above threshold.
        """
        results = await self._backend.search(
            collection_name=self._collection_name,
            vector=embedding,
            limit=1,
            score_threshold=self._settings.score_threshold_exact,
        )
        if results:
            r = results[0]
            return CacheHit(
                generated_query=r.generated_query,
                response_summary=r.response_summary,
                confidence=r.score,
                strategy=SearchStrategy.EXACT_MATCH,
                template_used=r.template_id,
                expires_at=r.expires_at,
            )
        return None

    async def _search_semantic(self, embedding: list[float]) -> CacheHit | None:
        """Search for semantic similarity (score >= score_threshold_semantic).

        Uses settings.score_threshold_semantic (default 0.90).
        Returns the top-1 result with a slight confidence penalty (0.9x).
        """
        results = await self._backend.search(
            collection_name=self._collection_name,
            vector=embedding,
            limit=3,
            score_threshold=self._settings.score_threshold_semantic,
        )
        if results:
            r = results[0]
            return CacheHit(
                generated_query=r.generated_query,
                response_summary=r.response_summary,
                confidence=r.score * 0.9,  # Penalize slightly
                strategy=SearchStrategy.SEMANTIC_MATCH,
                template_used=r.template_id,
                expires_at=r.expires_at,
            )
        return None

    async def _search_fuzzy(
        self, question: str, embedding: list[float] | None = None
    ) -> CacheHit | None:
        """Search using Levenshtein distance (optional, requires rapidfuzz).

        When an embedding is provided, a vector pre-filter is applied first:
        only the top-K most similar candidates (by cosine similarity) are
        considered for fuzzy scoring.  This reduces complexity from O(n) to
        O(top_k) for large collections while preserving recall.

        Falls back to a full collection scroll when no embedding is available.

        Only activated if rapidfuzz is installed.
        """
        try:
            from rapidfuzz import fuzz
        except ImportError:
            logger.debug("Fuzzy search skipped: rapidfuzz not installed")
            return None

        normalized = normalize_question(question)
        threshold = self._settings.score_threshold_fuzzy

        best_match: CacheResult | None = None
        best_score = 0.0
        early_exit_score = 99.0

        if embedding is not None:
            # Fast path: vector pre-filter → fuzzy only on top-K candidates
            candidates = await self._backend.search(
                collection_name=self._collection_name,
                vector=embedding,
                limit=self._settings.fuzzy_prefilter_top_k,
                score_threshold=self._settings.score_threshold_fuzzy_prefilter,
            )
            logger.debug(
                "Fuzzy pre-filter: %d candidates (vector threshold=%.2f, top_k=%d)",
                len(candidates),
                self._settings.score_threshold_fuzzy_prefilter,
                self._settings.fuzzy_prefilter_top_k,
            )
            for r in candidates:
                score = fuzz.ratio(normalized, r.normalized_question)
                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = r
                    if best_score >= early_exit_score:
                        break
        else:
            # Slow path: full collection scroll (no embedding available)
            logger.debug("Fuzzy search: no embedding, falling back to full scroll")
            offset = None
            while True:
                results, offset = await self._backend.scroll(
                    collection_name=self._collection_name,
                    limit=500,
                    offset=offset,
                )
                for r in results:
                    score = fuzz.ratio(normalized, r.normalized_question)
                    if score > best_score and score >= threshold:
                        best_score = score
                        best_match = r
                        if best_score >= early_exit_score:
                            break

                if offset is None or best_score >= early_exit_score:
                    break

        if best_match:
            return CacheHit(
                generated_query=best_match.generated_query,
                response_summary=best_match.response_summary,
                confidence=best_score / 100.0,
                strategy=SearchStrategy.FUZZY_MATCH,
                template_used=best_match.template_id,
                expires_at=best_match.expires_at,
            )
        return None

    # --- Store Operations ---

    async def store(
        self,
        question: str,
        generated_query: str,
        response_summary: str | None = None,
        template_id: str | None = None,
        ttl: int | None = _UNSET,  # type: ignore[assignment]
    ) -> bool:
        """Store a question-query pair in the cache.

        Also stores in L1 cache for immediate subsequent hits.

        Args:
            question: The natural language question.
            generated_query: The SQL/Cypher/GraphQL query.
            response_summary: Optional response summary.
            template_id: Optional template intent identifier.

        Returns:
            True if stored successfully, False otherwise.
        """
        if not question or not question.strip():
            logger.warning("Store skipped: question is empty or whitespace-only")
            return False
        if len(question) > self._settings.max_question_length:
            raise ValueError(
                f"Question length {len(question)} exceeds max_question_length "
                f"({self._settings.max_question_length})"
            )
        if not generated_query or not generated_query.strip():
            logger.warning("Store skipped: generated_query is empty or whitespace-only")
            return False

        try:
            logger.debug("Storing: '%s' -> '%s'", question[:50], generated_query[:50])
            embedding = await self._get_embedding(question)
            if embedding is None:
                logger.error("Store aborted: embedding failed for '%s'", question[:50])
                return False

            resolved_ttl = ttl if ttl is not _UNSET else self._settings.default_ttl_seconds
            expires_at = (
                datetime.now(timezone.utc) + timedelta(seconds=resolved_ttl)
                if resolved_ttl is not None
                else None
            )

            normalized = normalize_question(question)
            entry = CacheEntry(
                id=str(uuid.uuid4()),
                vector=embedding,
                original_question=question,
                normalized_question=normalized,
                generated_query=generated_query,
                query_hash=query_hash(generated_query),
                response_summary=response_summary,
                template_id=template_id,
                expires_at=expires_at,
            )

            await self._backend.upsert(self._collection_name, [entry])

            # Also store in L1
            await self._store_in_l1(
                question,
                CacheHit(
                    generated_query=generated_query,
                    response_summary=response_summary,
                    confidence=1.0,
                    strategy=SearchStrategy.EXACT_MATCH,
                    template_used=template_id,
                    expires_at=expires_at,
                ),
            )

            self._total_stored += 1
            logger.info("Stored: '%s' -> '%s'", question[:50], generated_query[:50])
            return True

        except Exception as e:
            logger.error("Store failed for '%s': %s", question[:50], e, exc_info=True)
            return False

    async def store_batch(self, entries: list[dict[str, Any]]) -> bool:
        """Store multiple question-query pairs efficiently.

        Uses aembed_batch() for a single round-trip to the embedder, then
        upserts all entries and populates the L1 cache.

        Args:
            entries: List of dicts with keys: question, generated_query,
                response_summary (optional), template_id (optional).

        Returns:
            True if all stored successfully, False if embedding or upsert fails.
        """
        if not entries:
            return True

        valid_entries = []
        for i, item in enumerate(entries):
            if not item.get("question", "").strip():
                logger.warning("store_batch: entry %d skipped — empty question", i)
                continue
            if not item.get("generated_query", "").strip():
                logger.warning("store_batch: entry %d skipped — empty generated_query", i)
                continue
            valid_entries.append(item)

        if not valid_entries:
            logger.warning("store_batch: no valid entries to store")
            return False
        entries = valid_entries

        try:
            logger.debug("Batch store started: %d entries", len(entries))

            questions = [item["question"] for item in entries]
            normalized_questions = [normalize_question(q) for q in questions]

            # Single batch embedding call — much faster than N sequential calls
            try:
                coro = self._embedder.aembed_batch(normalized_questions, is_document=True)
                if self._settings.embedding_timeout is not None:
                    coro = asyncio.wait_for(coro, timeout=self._settings.embedding_timeout)
                embeddings = await coro
            except asyncio.TimeoutError:
                logger.error(
                    "Batch store: embedding timed out after %.1fs for %d entries",
                    self._settings.embedding_timeout,
                    len(entries),
                )
                return False
            except EmbeddingError as e:
                logger.error("Batch store: embedding failed: %s", e)
                return False

            # Populate embedding cache with all computed vectors
            async with self._embedding_cache_lock:
                for question, vec in zip(questions, embeddings, strict=False):
                    cache_key = question_hash(question)
                    if len(self._embedding_cache) >= self._embedding_cache_max:
                        self._embedding_cache.popitem(last=False)
                    self._embedding_cache[cache_key] = vec

            # Build CacheEntry objects
            cache_entries = []
            for item, embedding in zip(entries, embeddings, strict=False):
                question = item["question"]
                normalized = normalize_question(question)
                entry = CacheEntry(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    original_question=question,
                    normalized_question=normalized,
                    generated_query=item["generated_query"],
                    query_hash=query_hash(item["generated_query"]),
                    response_summary=item.get("response_summary"),
                    template_id=item.get("template_id"),
                )
                cache_entries.append(entry)

            await self._backend.upsert(self._collection_name, cache_entries)

            # Populate L1 cache — consistent with store()
            for item in entries:
                await self._store_in_l1(
                    item["question"],
                    CacheHit(
                        generated_query=item["generated_query"],
                        response_summary=item.get("response_summary"),
                        confidence=1.0,
                        strategy=SearchStrategy.EXACT_MATCH,
                        template_used=item.get("template_id"),
                    ),
                )

            self._total_stored += len(cache_entries)
            logger.info("Batch stored %d entries", len(cache_entries))
            return True

        except Exception as e:
            logger.error("Batch store failed: %s", e, exc_info=True)
            return False

    # --- TTL / Expiry ---

    async def expire(self, collection_name: str | None = None) -> int:
        """Delete expired entries from the collection.

        Args:
            collection_name: Target collection. None = all known collections.

        Returns:
            Total number of entries deleted.
        """
        collections = [collection_name] if collection_name else self._known_collections
        total = 0
        for coll in collections:
            try:
                expired_ids = await self._backend.find_expired(coll)
                if expired_ids:
                    await self._backend.delete(coll, expired_ids)
                    total += len(expired_ids)
            except Exception:
                logger.exception("expire() failed for collection '%s'", coll)
        return total

    # --- Invalidation API ---

    async def invalidate(self, question: str) -> bool:
        """Invalidate a single cache entry by its original question.

        Finds the entry in the vector backend via normalized_question match,
        deletes it, and removes the corresponding L1 key.

        Args:
            question: Natural language question whose cached entry to remove.

        Returns:
            True if an entry was found and deleted, False if not found.
        """
        normalized = normalize_question(question)
        try:
            result = await self._backend.search_by_normalized_question(
                self._collection_name, normalized
            )
        except Exception:
            logger.exception("invalidate: backend lookup failed for '%s'", question[:50])
            return False

        if result is None:
            logger.debug("invalidate: no entry found for '%s'", question[:50])
            return False

        try:
            await self._backend.delete(self._collection_name, [result.id])
        except Exception:
            logger.exception("invalidate: backend delete failed for id='%s'", result.id)
            return False

        key = question_hash(question)
        await self._l1_backend.invalidate(key)
        logger.info("Invalidated entry for '%s' (id=%s)", question[:50], result.id)
        return True

    async def invalidate_by_query_hash(self, query_hash: str) -> int:
        """Invalidate all entries whose generated query matches *query_hash*.

        Args:
            query_hash: MD5 hash of the generated query (as stored in the backend).

        Returns:
            Number of entries deleted.
        """
        try:
            ids = await self._backend.find_by_query_hash(self._collection_name, query_hash)
        except Exception:
            logger.exception("invalidate_by_query_hash: lookup failed for hash='%s'", query_hash)
            return 0

        if not ids:
            return 0

        try:
            await self._backend.delete(self._collection_name, ids)
        except Exception:
            logger.exception("invalidate_by_query_hash: delete failed")
            return 0

        await self._l1_backend.invalidate_all()
        logger.info("Invalidated %d entries for query_hash='%s'", len(ids), query_hash)
        return len(ids)

    async def invalidate_by_template(self, template_id: str) -> int:
        """Invalidate all entries belonging to a template.

        Args:
            template_id: Template intent identifier.

        Returns:
            Number of entries deleted.
        """
        try:
            ids = await self._backend.find_by_template_id(self._collection_name, template_id)
        except Exception:
            logger.exception("invalidate_by_template: lookup failed for template_id='%s'", template_id)
            return 0

        if not ids:
            return 0

        try:
            await self._backend.delete(self._collection_name, ids)
        except Exception:
            logger.exception("invalidate_by_template: delete failed")
            return 0

        await self._l1_backend.invalidate_all()
        logger.info("Invalidated %d entries for template_id='%s'", len(ids), template_id)
        return len(ids)

    async def invalidate_collection(self, collection_name: str | None = None) -> int:
        """Drop and re-initialize a collection, clearing all its entries.

        Args:
            collection_name: Target collection. None = main collection.

        Returns:
            Number of entries that were in the collection before deletion.
        """
        coll = collection_name or self._collection_name
        try:
            count = await self._backend.count(coll)
        except Exception:
            count = 0

        try:
            await self._backend.drop_collection(coll)
        except Exception:
            logger.exception("invalidate_collection: drop failed for '%s'", coll)
            return 0

        try:
            await self._backend.initialize(coll, self._embedder.dimension)
        except Exception:
            logger.exception("invalidate_collection: re-initialize failed for '%s'", coll)

        await self._l1_backend.invalidate_all()
        logger.info("Invalidated collection '%s' (%d entries dropped)", coll, count)
        return count

    async def feedback(self, question: str, correct: bool) -> bool:
        """Record feedback for a previously cached question.

        Locates the entry by exact normalized-question match, increments
        feedback_correct or feedback_incorrect, and — if
        Settings.feedback_incorrect_threshold is set and the incorrect count
        has reached it — automatically invalidates the entry.

        Args:
            question: The original natural-language question.
            correct:  True → the cached query was correct; False → it was wrong.

        Returns:
            True  if the entry was found and updated.
            False if no entry exists for the question (expired, invalidated, or
                  never stored).
        """
        normalized = normalize_question(question)
        result = await self._backend.search_by_normalized_question(
            self._collection_name, normalized
        )
        if result is None:
            logger.warning("feedback: no entry found for '%s'", question[:50])
            return False
        new_count = await self._backend.update_feedback(
            self._collection_name, result.id, correct
        )
        if (
            not correct
            and self._settings.feedback_incorrect_threshold is not None
            and new_count >= self._settings.feedback_incorrect_threshold
        ):
            await self.invalidate(question)
            logger.info(
                "Auto-invalidated '%s' after %d incorrect feedbacks",
                question[:50],
                new_count,
            )
        return True

    async def _cleanup_loop(self) -> None:
        interval = self._settings.cleanup_interval_seconds
        while True:
            await asyncio.sleep(interval)
            try:
                n = await self.expire()
                if n > 0:
                    logger.info("TTL cleanup: removed %d expired entries", n)
            except Exception:
                logger.exception("TTL cleanup failed")

    # --- Template Management ---

    def _resolve_and_check_path(self, raw_path: str, label: str) -> Path:
        """Resolve path and optionally enforce allowed_file_dir restriction."""
        resolved = Path(raw_path).resolve()
        if self._settings.allowed_file_dir is not None:
            allowed = Path(self._settings.allowed_file_dir).resolve()
            try:
                resolved.relative_to(allowed)
            except ValueError as err:
                raise ValueError(
                    f"{label}: path '{resolved}' is outside allowed_file_dir '{allowed}'"
                ) from err
        return resolved

    async def load_templates(self, templates: list[QueryTemplate]) -> None:
        """Load templates into memory and sync to the template collection.

        Args:
            templates: List of QueryTemplate objects.
        """
        self._templates = templates
        await self._sync_templates_to_backend()
        logger.info("Loaded %d templates", len(templates))

    async def load_templates_from_file(self, file_path: str) -> None:
        """Load templates from a JSON file.

        Expected format: List of objects matching QueryTemplate schema.

        Args:
            file_path: Path to the JSON template file.

        Raises:
            TemplateError: If the file cannot be read or parsed.
        """
        try:
            resolved = self._resolve_and_check_path(file_path, "load_templates_from_file")
            _max_bytes = self._settings.max_file_size_mb * 1024 * 1024
            file_size = os.path.getsize(resolved)
            if file_size > _max_bytes:
                raise TemplateError(
                    f"Template file '{resolved}' is {file_size / 1_048_576:.1f} MB, "
                    f"exceeds max_file_size_mb={self._settings.max_file_size_mb}"
                )
            with open(resolved, encoding="utf-8") as f:
                data = json.load(f)
            templates = [QueryTemplate(**item) for item in data]
            self._templates = templates
            logger.info("Loaded %d templates from '%s'", len(templates), resolved)
        except TemplateError:
            raise
        except Exception as e:
            raise TemplateError(
                f"Failed to load templates from '{file_path}': {e}"
            ) from e

    async def warm_from_file(
        self,
        path: str,
        batch_size: int | None = None,
        on_progress: Any = None,
    ) -> int:
        """Warm the cache from a JSON or JSONL file.

        Supports two formats:
          - JSON array: ``[{"question": ..., "generated_query": ...}, ...]``
          - JSONL: one JSON object per line (same keys)

        Optional per-entry keys: ``response_summary``, ``template_id``.

        Args:
            path: Path to the file.
            batch_size: Override the default batch size for chunked upserts.
            on_progress: Optional callback ``(done, total)`` called after each chunk.

        Returns:
            Number of entries successfully stored.

        Raises:
            MedhaError: If the file cannot be read or parsed.
        """
        try:
            resolved = self._resolve_and_check_path(path, "warm_from_file")
            _max_bytes = self._settings.max_file_size_mb * 1024 * 1024
            file_size = os.path.getsize(resolved)
            if file_size > _max_bytes:
                raise MedhaError(
                    f"warm_from_file: '{resolved}' is {file_size / 1_048_576:.1f} MB, "
                    f"exceeds max_file_size_mb={self._settings.max_file_size_mb}"
                )
            with open(resolved, encoding="utf-8") as f:
                content = f.read().strip()

            if content.startswith("["):
                data = json.loads(content)
            else:
                data = [
                    json.loads(line)
                    for line in content.splitlines()
                    if line.strip()
                ]
        except MedhaError:
            raise
        except Exception as e:
            raise MedhaError(f"warm_from_file: cannot read '{path}': {e}") from e

        if not data:
            logger.warning("warm_from_file: no entries found in '%s'", path)
            return 0

        count = await self.store_many(data, batch_size=batch_size, on_progress=on_progress)
        if count:
            self._warm_loaded += count
            logger.info("Cache warmed: %d entries from '%s'", count, path)
        return count

    def _build_cache_entries(
        self,
        chunk: list[dict[str, Any]],
        embeddings: list[list[float]],
        resolved_ttl: int | None,
    ) -> list[CacheEntry]:
        entries = []
        for item, embedding in zip(chunk, embeddings, strict=False):
            question = item["question"]
            normalized = normalize_question(question)
            expires_at = (
                datetime.now(timezone.utc) + timedelta(seconds=resolved_ttl)
                if resolved_ttl is not None
                else None
            )
            entries.append(CacheEntry(
                id=str(uuid.uuid4()),
                vector=embedding,
                original_question=question,
                normalized_question=normalized,
                generated_query=item["generated_query"],
                query_hash=query_hash(item["generated_query"]),
                response_summary=item.get("response_summary"),
                template_id=item.get("template_id"),
                expires_at=expires_at,
            ))
        return entries

    async def store_many(
        self,
        entries: list[dict[str, Any]],
        *,
        batch_size: int | None = None,
        on_progress: Any = None,
        ttl: int | None = _UNSET,  # type: ignore[assignment]
    ) -> int:
        """Store multiple entries using chunked, optionally concurrent embedding.

        Fail-fast: raises ValueError if any entry is missing 'question' or
        'generated_query' before any embedding or upsert is performed.

        Args:
            entries: List of dicts, each with at minimum 'question' and
                'generated_query'. Optional keys: 'response_summary', 'template_id'.
            batch_size: Chunk size. Defaults to settings.batch_size.
            on_progress: Optional callback ``(done: int, total: int)`` called
                after each chunk is upserted.
            ttl: TTL in seconds for the new entries. _UNSET = use settings default.

        Returns:
            Number of entries stored.

        Raises:
            ValueError: If any entry is missing required keys.
        """
        for i, item in enumerate(entries):
            if not item.get("question", "").strip():
                raise ValueError(f"store_many: entry {i} missing or empty 'question'")
            if not item.get("generated_query", "").strip():
                raise ValueError(f"store_many: entry {i} missing or empty 'generated_query'")

        if not entries:
            return 0

        chunk_size = batch_size or self._settings.batch_size
        total = len(entries)
        done = 0
        resolved_ttl = ttl if ttl is not _UNSET else self._settings.default_ttl_seconds
        concurrency = self._settings.batch_embed_concurrency
        chunks = [entries[i: i + chunk_size] for i in range(0, total, chunk_size)]

        async def _embed_chunk(chunk: list[dict[str, Any]]) -> list[list[float]]:
            normalized = [normalize_question(item["question"]) for item in chunk]
            coro = self._embedder.aembed_batch(normalized, is_document=True)
            if self._settings.embedding_timeout is not None:
                coro = asyncio.wait_for(coro, timeout=self._settings.embedding_timeout)
            return await coro

        async def _upsert_chunk(chunk: list[dict[str, Any]], embeddings: list[list[float]]) -> None:
            cache_entries = self._build_cache_entries(chunk, embeddings, resolved_ttl)
            await self._backend.upsert(self._collection_name, cache_entries)
            for item in chunk:
                await self._store_in_l1(
                    item["question"],
                    CacheHit(
                        generated_query=item["generated_query"],
                        response_summary=item.get("response_summary"),
                        confidence=1.0,
                        strategy=SearchStrategy.EXACT_MATCH,
                        template_used=item.get("template_id"),
                    ),
                )

        if concurrency > 1:
            for group_start in range(0, len(chunks), concurrency):
                group = chunks[group_start: group_start + concurrency]
                embeddings_list: list[list[list[float]]] = await asyncio.gather(
                    *[_embed_chunk(chunk) for chunk in group]
                )
                for chunk, embeddings in zip(group, embeddings_list, strict=False):
                    await _upsert_chunk(chunk, embeddings)
                    done += len(chunk)
                    if on_progress is not None:
                        on_progress(done, total)
        else:
            for chunk in chunks:
                embeddings = await _embed_chunk(chunk)
                await _upsert_chunk(chunk, embeddings)
                done += len(chunk)
                if on_progress is not None:
                    on_progress(done, total)

        self._total_stored += total
        logger.info("store_many: stored %d entries in %d chunks", total, len(chunks))
        return total

    async def warm_from_dataframe(
        self,
        df: Any,
        *,
        question_col: str = "question",
        query_col: str = "generated_query",
        response_col: str = "response_summary",
        template_col: str | None = None,
        batch_size: int | None = None,
        on_progress: Any = None,
        ttl: int | None = _UNSET,  # type: ignore[assignment]
    ) -> int:
        """Warm the cache from a pandas DataFrame.

        Args:
            df: A ``pandas.DataFrame`` with at least the question and query columns.
            question_col: Column name for questions.
            query_col: Column name for generated queries.
            response_col: Column name for optional response summaries.
            template_col: Column name for optional template IDs.
            batch_size: Override chunk size.
            on_progress: Optional ``(done, total)`` callback.
            ttl: TTL for new entries.

        Returns:
            Number of entries stored.

        Raises:
            ConfigurationError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ConfigurationError(
                "warm_from_dataframe requires pandas: pip install pandas"
            ) from exc

        rows: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            item: dict[str, Any] = {
                "question": row[question_col],
                "generated_query": row[query_col],
            }
            if response_col in df.columns:
                val = row.get(response_col)
                if val is not None and pd.notna(val):
                    item["response_summary"] = val
            if template_col is not None and template_col in df.columns:
                val = row.get(template_col)
                if val is not None and pd.notna(val):
                    item["template_id"] = val
            rows.append(item)

        return await self.store_many(rows, batch_size=batch_size, on_progress=on_progress, ttl=ttl)

    async def export_to_dataframe(
        self,
        collection_name: str | None = None,
        *,
        include_vectors: bool = False,
    ) -> Any:
        """Export all cache entries to a pandas DataFrame.

        Scrolls the entire collection in pages of 500 and returns the result
        as a DataFrame where each row is the ``model_dump()`` of a CacheResult.

        Args:
            collection_name: Target collection. None = main collection.
            include_vectors: Whether to request vectors from the backend.

        Returns:
            ``pandas.DataFrame`` with one row per cache entry.

        Raises:
            ConfigurationError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ConfigurationError(
                "export_to_dataframe requires pandas: pip install pandas"
            ) from exc

        coll = collection_name or self._collection_name
        rows: list[dict[str, Any]] = []
        offset: str | None = None
        while True:
            results, offset = await self._backend.scroll(
                collection_name=coll,
                limit=500,
                offset=offset,
                with_vectors=include_vectors,
            )
            for r in results:
                rows.append(r.model_dump())
            if offset is None:
                break

        return pd.DataFrame(rows)

    async def dedup_collection(
        self,
        collection_name: str | None = None,
        *,
        strategy: str = "keep_latest",
    ) -> int:
        """Remove duplicate entries that share the same generated query hash.

        Args:
            collection_name: Target collection. None = main collection.
            strategy: ``"keep_latest"`` retains the most recently created entry;
                ``"keep_first"`` retains the oldest.

        Returns:
            Number of duplicate entries deleted.
        """
        coll = collection_name or self._collection_name
        seen: dict[str, Any] = {}
        to_delete: list[str] = []
        offset: str | None = None

        while True:
            results, offset = await self._backend.scroll(
                collection_name=coll,
                limit=500,
                offset=offset,
            )
            for r in results:
                qhash = r.query_hash
                if qhash not in seen:
                    seen[qhash] = r
                else:
                    existing = seen[qhash]
                    if strategy == "keep_latest":
                        r_ts = r.created_at
                        ex_ts = existing.created_at
                        if r_ts is not None and (ex_ts is None or r_ts > ex_ts):
                            to_delete.append(existing.id)
                            seen[qhash] = r
                        else:
                            to_delete.append(r.id)
                    else:
                        to_delete.append(r.id)
            if offset is None:
                break

        if to_delete:
            await self._backend.delete(coll, to_delete)
            logger.info("dedup_collection: deleted %d duplicates from '%s'", len(to_delete), coll)

        return len(to_delete)

    async def _sync_templates_to_backend(self) -> None:
        """Sync in-memory templates to the template collection in the backend.

        Idempotent: skips if templates already exist in the collection.
        """
        if not self._templates:
            return

        try:
            count = await self._backend.count(self._template_collection)
            if count > 0:
                logger.info(
                    "Template collection already has %d entries, skipping sync",
                    count,
                )
                return
        except StorageError:
            pass

        # Flatten all texts across templates into a single list for batch embedding.
        # Each item: (template, original_text, normalized_embed_text)
        # Strip {placeholder} braces so the embedder sees natural words.
        items: list[tuple[Any, ...]] = []
        for template in self._templates:
            for text in [template.template_text] + template.aliases:
                embed_text = re.sub(r"\{(\w+)\}", r"\1", text)
                items.append((template, text, normalize_question(embed_text)))

        if not items:
            return

        # Single batch embedding call instead of N sequential aembed() calls
        try:
            coro = self._embedder.aembed_batch(
                [embed_text for _, _, embed_text in items],
                is_document=True,
            )
            if self._settings.embedding_timeout is not None:
                coro = asyncio.wait_for(coro, timeout=self._settings.embedding_timeout)
            vectors = await coro
        except asyncio.TimeoutError:
            logger.error(
                "Template sync: batch embedding timed out after %.1fs for %d texts",
                self._settings.embedding_timeout,
                len(items),
            )
            return
        except EmbeddingError as e:
            logger.error("Template sync: batch embedding failed: %s", e)
            return

        entries = [
            CacheEntry(
                id=str(uuid.uuid4()),
                vector=vec,
                original_question=original_text,
                normalized_question=normalize_question(original_text),
                generated_query=template.query_template,
                query_hash=query_hash(template.query_template),
                template_id=template.intent,
            )
            for (template, original_text, _), vec in zip(items, vectors, strict=False)
        ]

        await self._backend.upsert(self._template_collection, entries)
        logger.info("Synced %d template entries to backend", len(entries))

    # --- Embedding Cache ---

    async def _get_embedding(self, question: str) -> list[float] | None:
        """Get or compute the embedding for a question.

        Checks the internal LRU cache first. If another coroutine is already
        computing the same embedding, waits for its result instead of
        duplicating the work (deduplication via asyncio.Future).

        Returns:
            Embedding vector, or None on failure.
        """
        normalized = normalize_question(question)
        cache_key = question_hash(question)

        our_future: asyncio.Future[list[float]] | None = None
        wait_future: asyncio.Future[list[float]] | None = None

        async with self._embedding_cache_lock:
            # Cache hit → LRU bump and return
            if cache_key in self._embedding_cache:
                vec = self._embedding_cache.pop(cache_key)
                self._embedding_cache[cache_key] = vec
                logger.debug("Embedding cache HIT for key=%s", cache_key[:8])
                return vec

            # Another coroutine is already computing this key → join it
            if cache_key in self._pending_embeddings:
                wait_future = self._pending_embeddings[cache_key]
                logger.debug("Embedding deduplication: joining in-flight key=%s", cache_key[:8])
            else:
                # We are the first → register a Future so others can join
                our_future = asyncio.get_running_loop().create_future()
                # Suppress "Future exception was never retrieved" if no waiter joins
                our_future.add_done_callback(
                    lambda f: f.exception() if not f.cancelled() and f.done() and f.exception() else None
                )
                self._pending_embeddings[cache_key] = our_future
                logger.debug("Embedding cache MISS for key=%s, computing...", cache_key[:8])

        if wait_future is not None:
            try:
                return await asyncio.shield(wait_future)
            except (EmbeddingError, asyncio.CancelledError, asyncio.TimeoutError):
                logger.warning("In-flight embedding unavailable for key=%s", cache_key[:8])
                return None

        # We own this computation
        if our_future is None:
            logger.error("Embedding deduplication: invariant violated for key=%s", cache_key[:8])
            return None

        try:
            coro = self._embedder.aembed(normalized)
            if self._settings.embedding_timeout is not None:
                coro = asyncio.wait_for(coro, timeout=self._settings.embedding_timeout)
            vec = await coro
        except asyncio.TimeoutError:
            err = EmbeddingError(
                f"Embedding timed out after {self._settings.embedding_timeout}s"
            )
            logger.error("Embedding timed out for '%s'", question[:50])
            async with self._embedding_cache_lock:
                self._pending_embeddings.pop(cache_key, None)
            if not our_future.done():
                our_future.set_exception(err)
            return None
        except EmbeddingError as e:
            logger.error("Embedding failed for '%s': %s", question[:50], e)
            async with self._embedding_cache_lock:
                self._pending_embeddings.pop(cache_key, None)
            if not our_future.done():
                our_future.set_exception(e)
            return None
        except BaseException:
            # CancelledError, KeyboardInterrupt, etc. — always unblock waiters
            async with self._embedding_cache_lock:
                self._pending_embeddings.pop(cache_key, None)
            if not our_future.done():
                our_future.cancel()
            raise

        # Store in cache and notify waiters
        async with self._embedding_cache_lock:
            if len(self._embedding_cache) >= self._embedding_cache_max:
                self._embedding_cache.popitem(last=False)
            self._embedding_cache[cache_key] = vec
            self._pending_embeddings.pop(cache_key, None)

        if not our_future.done():
            our_future.set_result(vec)

        logger.debug(
            "Embedding computed: dim=%d, cache_size=%d",
            len(vec),
            len(self._embedding_cache),
        )
        return vec

    # --- Statistics & Monitoring ---

    async def stats(self, collection_name: str | None = None) -> CacheStats:
        """Return a frozen snapshot of cache performance metrics.

        Args:
            collection_name: Collection to count entries for. None = main collection.

        Returns:
            CacheStats with hit/miss rates, latency percentiles, and backend count.
        """
        backend_count = await self._backend.count(collection_name or self._collection_name)
        return await self._stats.snapshot(backend_count)

    async def reset_stats(self) -> None:
        """Reset all collected statistics to zero."""
        await self._stats.reset()

    async def clear_caches(self) -> None:
        """Clear all caches (L1, embedding). Backend data is preserved."""
        await self._l1_backend.clear()
        self._embedding_cache.clear()
        await self._stats.reset()
        self._total_stored = 0
        self._warm_loaded = 0
        logger.info("In-memory caches cleared")

    # --- Persistent Embedding Cache ---

    def _load_embedding_cache_from_disk(self) -> None:
        """Load persisted embeddings from disk into the in-memory cache.

        Silently skips if the file does not exist yet (first run).
        """
        path = self._settings.embedding_cache_path
        if not path:
            return
        try:
            import os
            if not os.path.exists(path):
                logger.debug("Embedding cache file not found at '%s', starting empty", path)
                return
            with open(path, encoding="utf-8") as f:
                data: dict[str, list[float]] = json.load(f)
            loaded = 0
            for key, vec in data.items():
                if len(self._embedding_cache) >= self._embedding_cache_max:
                    break
                self._embedding_cache[key] = vec
                loaded += 1
            logger.info("Loaded %d embeddings from disk cache '%s'", loaded, path)
        except Exception as exc:
            logger.warning("Failed to load embedding cache from '%s': %s", path, exc)

    def _save_embedding_cache_to_disk(self) -> None:
        """Persist the current in-memory embedding cache to disk."""
        path = self._settings.embedding_cache_path
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(dict(self._embedding_cache), f)
            logger.info(
                "Saved %d embeddings to disk cache '%s'",
                len(self._embedding_cache),
                path,
            )
        except Exception as exc:
            logger.warning("Failed to save embedding cache to '%s': %s", path, exc)

    # --- Sync Wrappers ---

    def search_sync(self, question: str) -> CacheHit:
        """Synchronous wrapper for search()."""
        return BaseEmbedder._run_sync(self.search(question))

    def store_sync(self, question: str, generated_query: str, **kwargs: Any) -> bool:
        """Synchronous wrapper for store()."""
        return BaseEmbedder._run_sync(self.store(question, generated_query, **kwargs))

    def warm_from_file_sync(self, path: str) -> int:
        """Synchronous wrapper for warm_from_file()."""
        return BaseEmbedder._run_sync(self.warm_from_file(path))

    def clear_caches_sync(self) -> None:
        """Synchronous wrapper for clear_caches()."""
        BaseEmbedder._run_sync(self.clear_caches())

    def feedback_sync(self, question: str, correct: bool) -> bool:
        """Synchronous wrapper for feedback()."""
        return BaseEmbedder._run_sync(self.feedback(question, correct))
