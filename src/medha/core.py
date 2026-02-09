"""Core Medha class implementing the waterfall search strategy."""

import json
import logging
import re
import uuid
from collections import OrderedDict
from typing import Dict, List, Optional

from medha.config import Settings
from medha.types import CacheHit, CacheEntry, CacheResult, QueryTemplate, SearchStrategy
from medha.interfaces.embedder import BaseEmbedder
from medha.interfaces.storage import VectorStorageBackend
from medha.utils.normalization import normalize_question, question_hash, query_hash
from medha.utils.nlp import ParameterExtractor, keyword_overlap_score
from medha.exceptions import MedhaError, EmbeddingError, StorageError, TemplateError

logger = logging.getLogger(__name__)


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
        templates: List[QueryTemplate] | None = None,
    ):
        self._collection_name = collection_name
        self._template_collection = f"{collection_name}_templates"
        self._embedder = embedder
        self._settings = settings or Settings()
        self._templates = templates or []

        # Backend: default to Qdrant if not provided
        if backend is None:
            from medha.backends.qdrant import QdrantBackend
            self._backend = QdrantBackend(self._settings)
        else:
            self._backend = backend

        # L1 in-memory cache (Tier 0)
        self._l1_cache: OrderedDict[str, CacheHit] = OrderedDict()
        self._l1_max_size = self._settings.l1_cache_max_size

        # Embedding cache (avoids redundant embedding calls)
        self._embedding_cache: OrderedDict[str, List[float]] = OrderedDict()
        self._embedding_cache_max = 10000

        # NLP parameter extractor
        self._param_extractor = ParameterExtractor()

        # Stats
        self._stats = {
            "l1_hits": 0,
            "template_hits": 0,
            "exact_hits": 0,
            "semantic_hits": 0,
            "fuzzy_hits": 0,
            "misses": 0,
            "errors": 0,
        }

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

        if self._settings.template_file and not self._templates:
            await self.load_templates_from_file(self._settings.template_file)

        if self._templates:
            await self._sync_templates_to_backend()

        logger.info(
            "Medha started: collection='%s', templates=%d",
            self._collection_name,
            len(self._templates),
        )

    async def close(self) -> None:
        """Shut down the backend and release resources."""
        await self._backend.close()
        logger.info("Medha closed")

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
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
        try:
            if not question or not question.strip():
                logger.warning("Search called with empty question")
                return CacheHit(strategy=SearchStrategy.ERROR)

            logger.debug("Search started for: '%s'", question[:80])

            # --- Tier 0: L1 Cache ---
            l1_hit = self._check_l1_cache(question)
            if l1_hit:
                self._stats["l1_hits"] += 1
                logger.debug("Tier 0 L1 cache HIT for: '%s'", question[:50])
                return l1_hit
            logger.debug("Tier 0 L1 cache MISS")

            # --- Tier 1: Template Matching ---
            template_hit = await self._search_templates(question)
            if template_hit:
                self._stats["template_hits"] += 1
                self._store_in_l1(question, template_hit)
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
                self._stats["errors"] += 1
                logger.error("Embedding failed, aborting search for: '%s'", question[:50])
                return CacheHit(strategy=SearchStrategy.ERROR)

            # --- Tier 2: Exact Vector Match ---
            exact_hit = await self._search_exact(embedding)
            if exact_hit:
                self._stats["exact_hits"] += 1
                self._store_in_l1(question, exact_hit)
                logger.debug(
                    "Tier 2 exact HIT: confidence=%.4f, query='%s'",
                    exact_hit.confidence,
                    exact_hit.generated_query[:50] if exact_hit.generated_query else "",
                )
                return exact_hit
            logger.debug("Tier 2 exact MISS (threshold=%.2f)", self._settings.score_threshold_exact)

            # --- Tier 3: Semantic Similarity ---
            semantic_hit = await self._search_semantic(embedding)
            if semantic_hit:
                self._stats["semantic_hits"] += 1
                self._store_in_l1(question, semantic_hit)
                logger.debug(
                    "Tier 3 semantic HIT: confidence=%.4f, query='%s'",
                    semantic_hit.confidence,
                    semantic_hit.generated_query[:50] if semantic_hit.generated_query else "",
                )
                return semantic_hit
            logger.debug("Tier 3 semantic MISS (threshold=%.2f)", self._settings.score_threshold_semantic)

            # --- Tier 4: Fuzzy Matching ---
            fuzzy_hit = await self._search_fuzzy(question)
            if fuzzy_hit:
                self._stats["fuzzy_hits"] += 1
                self._store_in_l1(question, fuzzy_hit)
                logger.debug(
                    "Tier 4 fuzzy HIT: confidence=%.4f", fuzzy_hit.confidence
                )
                return fuzzy_hit
            logger.debug("Tier 4 fuzzy MISS (threshold=%.1f)", self._settings.score_threshold_fuzzy)

            # --- No match ---
            self._stats["misses"] += 1
            logger.debug("All tiers exhausted, NO_MATCH for: '%s'", question[:50])
            return CacheHit(strategy=SearchStrategy.NO_MATCH)

        except Exception as e:
            logger.error("Search failed for '%s': %s", question[:50], e, exc_info=True)
            self._stats["errors"] += 1
            return CacheHit(strategy=SearchStrategy.ERROR)

    # --- Tier Implementations ---

    def _check_l1_cache(self, question: str) -> Optional[CacheHit]:
        """Check the L1 in-memory LRU cache.

        Key: MD5 hash of normalized question.
        Returns: CacheHit if found, None otherwise.
        """
        key = question_hash(question)
        if key in self._l1_cache:
            # Move to end (most recently used)
            hit = self._l1_cache.pop(key)
            self._l1_cache[key] = hit
            return hit
        return None

    def _store_in_l1(self, question: str, hit: CacheHit) -> None:
        """Store a result in the L1 cache with LRU eviction."""
        if self._l1_max_size <= 0:
            return
        key = question_hash(question)
        evicted = False
        if len(self._l1_cache) >= self._l1_max_size:
            self._l1_cache.popitem(last=False)  # Evict oldest
            evicted = True
        self._l1_cache[key] = hit
        logger.debug(
            "L1 cache store: key=%s, strategy=%s, size=%d%s",
            key[:8],
            hit.strategy.value if hit.strategy else "?",
            len(self._l1_cache),
            " (evicted oldest)" if evicted else "",
        )

    async def _search_templates(self, question: str) -> Optional[CacheHit]:
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

        best_hit: Optional[CacheHit] = None
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

        return best_hit

    async def _search_exact(self, embedding: List[float]) -> Optional[CacheHit]:
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
            )
        return None

    async def _search_semantic(self, embedding: List[float]) -> Optional[CacheHit]:
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
            )
        return None

    async def _search_fuzzy(self, question: str) -> Optional[CacheHit]:
        """Search using Levenshtein distance (optional, requires rapidfuzz).

        Scrolls through all entries in the main collection and compares
        normalized questions. Computationally expensive for large collections.

        Only activated if rapidfuzz is installed.
        """
        try:
            from rapidfuzz import fuzz
        except ImportError:
            logger.debug("Fuzzy search skipped: rapidfuzz not installed")
            return None

        normalized = normalize_question(question)
        threshold = self._settings.score_threshold_fuzzy

        best_match: Optional[CacheResult] = None
        best_score = 0.0
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

            if offset is None:
                break

        if best_match:
            return CacheHit(
                generated_query=best_match.generated_query,
                response_summary=best_match.response_summary,
                confidence=best_score / 100.0,
                strategy=SearchStrategy.FUZZY_MATCH,
                template_used=best_match.template_id,
            )
        return None

    # --- Store Operations ---

    async def store(
        self,
        question: str,
        generated_query: str,
        response_summary: Optional[str] = None,
        template_id: Optional[str] = None,
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
        try:
            logger.debug("Storing: '%s' -> '%s'", question[:50], generated_query[:50])
            embedding = await self._get_embedding(question)
            if embedding is None:
                logger.error("Store aborted: embedding failed for '%s'", question[:50])
                return False

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
            )

            await self._backend.upsert(self._collection_name, [entry])

            # Also store in L1
            self._store_in_l1(
                question,
                CacheHit(
                    generated_query=generated_query,
                    response_summary=response_summary,
                    confidence=1.0,
                    strategy=SearchStrategy.EXACT_MATCH,
                    template_used=template_id,
                ),
            )

            logger.info("Stored: '%s' -> '%s'", question[:50], generated_query[:50])
            return True

        except Exception as e:
            logger.error("Store failed for '%s': %s", question[:50], e, exc_info=True)
            return False

    async def store_batch(self, entries: List[Dict]) -> bool:
        """Store multiple question-query pairs efficiently.

        Args:
            entries: List of dicts with keys: question, generated_query,
                response_summary (optional), template_id (optional).

        Returns:
            True if all stored successfully, False otherwise.
        """
        try:
            logger.debug("Batch store started: %d entries", len(entries))
            cache_entries = []
            for item in entries:
                question = item["question"]
                gen_query = item["generated_query"]

                embedding = await self._get_embedding(question)
                if embedding is None:
                    logger.warning("Batch store: skipping entry, embedding failed for '%s'", question[:50])
                    continue

                normalized = normalize_question(question)
                entry = CacheEntry(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    original_question=question,
                    normalized_question=normalized,
                    generated_query=gen_query,
                    query_hash=query_hash(gen_query),
                    response_summary=item.get("response_summary"),
                    template_id=item.get("template_id"),
                )
                cache_entries.append(entry)

            if cache_entries:
                await self._backend.upsert(self._collection_name, cache_entries)
                logger.info("Batch stored %d entries", len(cache_entries))
            return True

        except Exception as e:
            logger.error("Batch store failed: %s", e, exc_info=True)
            return False

    # --- Template Management ---

    async def load_templates(self, templates: List[QueryTemplate]) -> None:
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
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            templates = [QueryTemplate(**item) for item in data]
            self._templates = templates
            logger.info("Loaded %d templates from '%s'", len(templates), file_path)
        except Exception as e:
            raise TemplateError(
                f"Failed to load templates from '{file_path}': {e}"
            ) from e

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

        entries = []
        for template in self._templates:
            # Embed the main template_text and each alias
            texts_to_embed = [template.template_text] + template.aliases
            for text in texts_to_embed:
                try:
                    # Strip {placeholder} braces so the embedder sees
                    # natural words (e.g. "department") instead of noise.
                    embed_text = re.sub(r"\{(\w+)\}", r"\1", text)
                    vec = await self._embedder.aembed(normalize_question(embed_text))
                except EmbeddingError:
                    logger.warning(
                        "Failed to embed template text: '%s'", text[:50]
                    )
                    continue

                entry = CacheEntry(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    original_question=text,
                    normalized_question=normalize_question(text),
                    generated_query=template.query_template,
                    query_hash=query_hash(template.query_template),
                    template_id=template.intent,
                )
                entries.append(entry)

        if entries:
            await self._backend.upsert(self._template_collection, entries)
            logger.info("Synced %d template entries to backend", len(entries))

    # --- Embedding Cache ---

    async def _get_embedding(self, question: str) -> Optional[List[float]]:
        """Get or compute the embedding for a question.

        Checks the internal LRU cache first. On miss, calls the embedder
        and caches the result.

        Returns:
            Embedding vector, or None on failure.
        """
        normalized = normalize_question(question)
        cache_key = question_hash(question)

        # Check cache
        if cache_key in self._embedding_cache:
            vec = self._embedding_cache.pop(cache_key)
            self._embedding_cache[cache_key] = vec  # Move to end
            logger.debug("Embedding cache HIT for key=%s", cache_key[:8])
            return vec

        # Compute
        logger.debug("Embedding cache MISS for key=%s, computing...", cache_key[:8])
        try:
            vec = await self._embedder.aembed(normalized)
        except EmbeddingError as e:
            logger.error("Embedding failed for '%s': %s", question[:50], e)
            return None

        # Store in cache with LRU eviction
        if len(self._embedding_cache) >= self._embedding_cache_max:
            self._embedding_cache.popitem(last=False)
        self._embedding_cache[cache_key] = vec
        logger.debug(
            "Embedding computed: dim=%d, cache_size=%d",
            len(vec),
            len(self._embedding_cache),
        )
        return vec

    # --- Statistics & Monitoring ---

    @property
    def stats(self) -> Dict:
        """Return cache performance statistics."""
        total = sum(self._stats.values())
        hit_count = total - self._stats["misses"] - self._stats["errors"]
        return {
            "total_requests": total,
            "hit_rate": (hit_count / total * 100) if total > 0 else 0.0,
            "by_strategy": dict(self._stats),
            "l1_cache_size": len(self._l1_cache),
            "embedding_cache_size": len(self._embedding_cache),
            "templates_loaded": len(self._templates),
        }

    def clear_caches(self) -> None:
        """Clear all in-memory caches (L1, embedding). Backend data is preserved."""
        self._l1_cache.clear()
        self._embedding_cache.clear()
        self._stats = {k: 0 for k in self._stats}
        logger.info("In-memory caches cleared")

    # --- Sync Wrappers ---

    def search_sync(self, question: str) -> CacheHit:
        """Synchronous wrapper for search()."""
        return BaseEmbedder._run_sync(self.search(question))

    def store_sync(self, question: str, generated_query: str, **kwargs) -> bool:
        """Synchronous wrapper for store()."""
        return BaseEmbedder._run_sync(self.store(question, generated_query, **kwargs))
