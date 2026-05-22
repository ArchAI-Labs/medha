# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] — 2026-05-19

### Breaking Changes

- **Custom `VectorStorageBackend` subclasses** must implement
  `update_feedback(collection_name, point_id, correct) -> int` or raise
  `TypeError` at instantiation.

### Added

- `Medha.feedback(question, correct)` — record a correct/incorrect signal for a
  cached question. Returns `True` if found and updated, `False` if not found.
- `Settings.feedback_incorrect_threshold` (`int | None`, default `None`,
  env `MEDHA_FEEDBACK_INCORRECT_THRESHOLD`) — when set, entries are
  automatically invalidated once `feedback_incorrect` reaches the threshold.
- `CacheEntry.feedback_correct` / `CacheEntry.feedback_incorrect` — new fields,
  default 0, backward compatible with 0.3.x entries.
- `CacheResult.feedback_correct` / `CacheResult.feedback_incorrect` — counters
  visible in backend search results.
- `VectorStorageBackend.update_feedback(collection, point_id, correct) -> int`
  — new abstract method returning the new counter value; implemented by all ten
  built-in backends.
- `medha` CLI — install with `pip install "medha-archai[cli]"`.
  Commands: `stats`, `warm`, `invalidate`, `invalidate-collection`, `expire`,
  `dedup`, `export`, `feedback`.
- `Settings.embedder_type` — declarative embedder selection via
  `MEDHA_EMBEDDER_TYPE` env var. Accepted values: `fastembed`, `openai`,
  `cohere`, `gemini`, `_noop` (default). Used by the CLI factory; no change to
  the `Medha(embedder=...)` call signature.
- `Settings.collection` — default collection name for CLI commands via
  `MEDHA_COLLECTION` env var (default: `"default"`).
- `Settings.fastembed_model` — FastEmbed model name via `MEDHA_FASTEMBED_MODEL`
  env var (default: `"BAAI/bge-small-en-v1.5"`).
- `_NoOpEmbedder` (private, `medha.cli._noop_embedder`) — placeholder embedder
  used by CLI admin commands that do not perform vector search or storage.
  `medha feedback` works with `_NoOpEmbedder` because it uses a plain text
  lookup, not vector search.

### Notes

- `medha stats` reports structural information only (entry count). In-process
  performance metrics (hit rate, latency percentiles) are not available from the
  CLI because `CacheStats` is a non-persistent in-memory accumulator.

---

## [0.3.0] — 2026-04-22

### Breaking Changes

- **`qdrant-client` is no longer a core dependency.** Install it explicitly with
  `pip install medha-archai[qdrant]`. Projects that do not use Qdrant no longer
  pull in the client package.

- **`Settings.backend_type` default changed from `"qdrant"` to `"memory"`.** New
  installations default to the zero-dependency in-process backend. Users relying on
  Qdrant must set `MEDHA_BACKEND_TYPE=qdrant` or pass `Settings(backend_type="qdrant")`.

### Added

#### New Vector Backends

- **`ElasticsearchBackend`**: Elasticsearch 8.x vector backend using the native
  `knn` dense-vector search. Install with `pip install medha-archai[elasticsearch]`.
  Select via `Settings(backend_type="elasticsearch")`.
  Configuration fields: `es_hosts`, `es_api_key`, `es_username`, `es_password`,
  `es_index_prefix`, `es_num_candidates`, `es_timeout`.

- **`VectorChordBackend`**: PostgreSQL + VectorChord (`vchordrq` index) backend for
  high-performance approximate search. Install with `pip install medha-archai[vectorchord]`.
  Select via `Settings(backend_type="vectorchord")`.
  Configuration fields: `vc_lists`, `vc_residual_quantization` (inherited from pgvector
  DSN/host/pool settings).
  Shares the new `_AsyncpgMixin` with `PgVectorBackend`, eliminating duplicated
  asyncpg connection-pool code.

- **`ChromaBackend`**: ChromaDB vector backend supporting ephemeral (in-memory),
  persistent (local disk), and HTTP (remote server) modes.
  Install with `pip install medha-archai[chroma]`.
  Select via `Settings(backend_type="chroma")`.
  Configuration fields: `chroma_mode`, `chroma_host`, `chroma_port`,
  `chroma_persist_path`, `chroma_ssl`, `chroma_auth_token`.

- **`WeaviateBackend`**: Weaviate vector backend supporting local (self-hosted) and
  cloud (Weaviate Cloud) modes. Install with `pip install medha-archai[weaviate]`.
  Select via `Settings(backend_type="weaviate")`.
  Configuration fields: `weaviate_mode`, `weaviate_host`, `weaviate_http_port`,
  `weaviate_grpc_port`, `weaviate_http_secure`, `weaviate_grpc_secure`,
  `weaviate_cloud_url`, `weaviate_api_key`, `weaviate_collection_prefix`.

- **`RedisVectorBackend`**: Redis Stack vector backend using RediSearch with HNSW or
  FLAT index algorithms. Supports standalone and Sentinel high-availability modes.
  Install with `pip install medha-archai[redis]`.
  Select via `Settings(backend_type="redis")`.
  Configuration fields: `redis_mode`, `redis_url`, `redis_host`, `redis_port`,
  `redis_db`, `redis_username`, `redis_password`, `redis_ssl` (+ TLS cert paths),
  `redis_sentinel_hosts`, `redis_sentinel_master`, `redis_key_prefix`,
  `redis_index_algorithm`, `redis_hnsw_m`, `redis_hnsw_ef_construction`,
  `redis_hnsw_ef_runtime`, `redis_socket_timeout`, `redis_socket_connect_timeout`.

- **`AzureSearchBackend`**: Azure AI Search vector backend using HNSW
  `VectorizedQuery` for approximate nearest-neighbour search. Supports both API-key
  and `DefaultAzureCredential` (managed identity) authentication.
  Install with `pip install medha-archai[azure-search]`.
  Select via `Settings(backend_type="azure-search")`.
  Configuration fields: `azure_search_endpoint`, `azure_search_api_key`,
  `azure_search_api_version`, `azure_search_index_name`,
  `azure_search_top_k_candidates`.

- **`LanceDBBackend`**: LanceDB embedded/cloud vector backend using the native async
  API (`lancedb.connect_async`). Local mode (default) requires no external services;
  cloud storage is supported via `s3://`, `gs://`, and `az://` URIs.
  Install with `pip install medha-archai[lancedb]`.
  Select via `Settings(backend_type="lancedb")`.
  Configuration fields: `lancedb_uri`, `lancedb_table_prefix`, `lancedb_metric`
  (`cosine` / `l2` / `dot`).

- **`_AsyncpgMixin`** (`medha.backends._asyncpg_mixin`): shared asyncpg connection-pool
  and SQL helpers extracted from `PgVectorBackend` and reused by `VectorChordBackend`.

#### New Embedders

- **`CohereAdapter`**: Cohere Embed v3 embedder (`cohere.AsyncClientV2`).
  Install with `pip install medha-archai[cohere]`. Supports `embed-multilingual-v3.0` and
  other Cohere embedding models; selects `search_document` / `search_query` input types
  automatically.

- **`GeminiAdapter`**: Google Gemini embedder (`google-generativeai`).
  Install with `pip install medha-archai[gemini]`. Supports `models/text-embedding-004` and
  other Gemini embedding models; batches requests in chunks of 100 to stay within
  API limits.

#### Cache Lifecycle (TTL & Invalidation)

- **TTL support on `store()` and `store_many()`**: new optional `ttl: int | None`
  parameter sets per-entry expiry. Entries with a past `expires_at` are excluded from
  all search results without being deleted.

- **`Settings.default_ttl_seconds`** (default `None`): global TTL applied to every new
  entry when `ttl` is not passed to `store()` / `store_many()`. `None` means immortal
  entries (unchanged behaviour).

- **`Settings.cleanup_interval_seconds`** (default `None`): when set, `Medha.start()`
  launches an asyncio background task that calls `expire()` periodically to hard-delete
  expired entries from the backend.

- **`Medha.expire(collection_name)`**: manually trigger deletion of all entries whose
  `expires_at` is in the past. Returns the number of deleted entries.

- **`Medha.invalidate(question)`**: remove a single entry by exact question text.
  Returns `True` if an entry was deleted.

- **`Medha.invalidate_by_query_hash(query_hash)`**: remove all entries that share the
  same generated-query MD5 hash. Returns the count of deleted entries.

- **`Medha.invalidate_by_template(template_id)`**: remove all entries associated with
  a given template intent. Returns the count of deleted entries.

- **`Medha.invalidate_collection(collection_name)`**: drop and recreate an entire
  collection. Returns the count of deleted entries.

- **`VectorStorageBackend.find_expired()`**, **`find_by_query_hash()`**,
  **`find_by_template_id()`**, **`drop_collection()`**: new abstract methods added to
  the storage interface, implemented by all backends.

#### Batch Operations

- **`Medha.store_batch(entries)`**: store a list of `{question, generated_query, ...}`
  dicts using a single `aembed_batch()` round-trip. Significantly faster than N
  sequential `store()` calls for bulk ingestion.

- **`Medha.store_many(data, batch_size, on_progress, ttl)`**: chunked bulk upsert with
  configurable concurrency (`Settings.batch_embed_concurrency`) and optional progress
  callback. Designed for large datasets that exceed memory or API-rate limits.

- **`Medha.warm_from_file()`** and **`Medha.warm_from_dataframe()`**: now delegate to
  `store_many()` internally, gaining chunked processing, concurrency control, and TTL
  support.

- **`Medha.export_to_dataframe()`**: export all entries in a collection to a
  `pandas.DataFrame` for inspection or migration.

- **`Medha.dedup_collection()`**: remove duplicate entries (same `query_hash`) from a
  collection, keeping the most-used entry per group.

- **`Settings.batch_size`** (default `100`): chunk size for bulk upsert operations.

- **`Settings.batch_embed_concurrency`** (default `1`): number of embedding chunks
  processed concurrently in `store_many()`.

#### Observability

- **`CacheStats`** model (`medha.types`): immutable Pydantic snapshot of cache
  performance metrics including `total_requests`, `total_hits`, `total_misses`,
  `total_errors`, `hit_rate`, `miss_rate`, `avg_latency_ms`, `p50_latency_ms`,
  `p95_latency_ms`, `p99_latency_ms`, `backend_count`, `since`, `until`, and
  per-strategy breakdown via `by_strategy: dict[str, StrategyStats]`.

- **`StrategyStats`** model (`medha.types`): per-strategy `count` and
  `total_latency_ms` with a computed `avg_latency_ms` property.

- **`Medha.stats(collection_name)`**: returns a `CacheStats` snapshot for the current
  or named collection.

- **`Medha.reset_stats()`**: resets all in-process statistics counters.

- **`Settings.collect_stats`** (default `True`): toggle statistics collection globally.

- **`Settings.stats_max_latency_samples`** (default `10 000`): FIFO buffer size for
  per-request latency samples used in percentile calculations.

#### Embedder Timeout

- **`Settings.embedding_timeout`** (default `None`): when set, wraps every `aembed()`
  and `aembed_batch()` call in `asyncio.wait_for()`, raising `EmbeddingError` if the
  embedder does not respond within the configured number of seconds. Prevents
  indefinite hangs on slow or unreachable embedding services.

#### Sync Convenience Wrappers

- **`Medha.search_sync()`**, **`Medha.store_sync()`**, **`Medha.warm_from_file_sync()`**,
  **`Medha.clear_caches_sync()`**: thin synchronous wrappers around their async
  counterparts for use in non-async contexts.

### Changed

- `QdrantBackend` imports in `medha.backends` and `medha` are now guarded with
  `try/except ImportError` so the package loads cleanly without `qdrant-client`.

- **`Settings.backend_type`** now accepts ten values:
  `"qdrant"`, `"memory"`, `"pgvector"`, `"elasticsearch"`, `"vectorchord"`,
  `"chroma"`, `"weaviate"`, `"redis"`, `"azure-search"`, `"lancedb"`.

- **`PgVectorBackend`** refactored to use the shared `_AsyncpgMixin`; its connection-pool
  logic is no longer duplicated in `VectorChordBackend`.

- **`CacheEntry`** and **`CacheResult`** gained an `expires_at: datetime | None` field
  for TTL support. Existing serialised entries without this field deserialise with
  `expires_at=None` (no expiry).

- **`VectorStorageBackend`** (abstract interface) gained four new abstract methods:
  `find_expired()`, `find_by_query_hash()`, `find_by_template_id()`,
  `drop_collection()`. All existing and new backends implement them.

- **`L1CacheBackend`** (abstract interface) gained `invalidate(key)` and
  `invalidate_all()` abstract methods.

- **Experiments** (`experiments/`): benchmark scripts updated to cover the new backends
  and the stats API.

---

## [0.2.0] — 2026-03-30

### Added

- **`InMemoryBackend`**: pure-Python in-process vector backend with zero external
  dependencies. Useful for testing, development, CI, and lightweight deployments.
  Select via `Settings(backend_type="memory")` or env var `MEDHA_BACKEND_TYPE=memory`.

- **`PgVectorBackend`**: PostgreSQL + pgvector vector backend for production deployments
  that already run PostgreSQL. Install with `pip install medha-archai[pgvector]`.
  Select via `Settings(backend_type="pgvector")` or env var `MEDHA_BACKEND_TYPE=pgvector`.

- **`Settings.backend_type`**: new `Literal["qdrant", "memory", "pgvector"]` field to
  select the vector backend declaratively without passing an instance to `Medha`.

- **PostgreSQL configuration fields** in `Settings`: `pg_dsn`, `pg_host`, `pg_port`,
  `pg_database`, `pg_user`, `pg_password` (SecretStr), `pg_schema`, `pg_table_prefix`,
  `pg_pool_min_size`, `pg_pool_max_size`.

- **Backend factory** in `Medha.__init__`: `backend=None` now resolves to the correct
  backend class based on `settings.backend_type` instead of always defaulting to Qdrant.

- **`connect()` method** added to `VectorStorageBackend` base class as a concrete no-op,
  ensuring all backends share the same lifecycle interface.

- **`Settings.max_question_length`** (default `8192`): rejects oversized question strings
  in `search()` (returns `SearchStrategy.ERROR`) and `store()` (raises `ValueError`),
  preventing DoS via oversized inputs.

- **`Settings.max_file_size_mb`** (default `100`): `warm_from_file()` and
  `load_templates_from_file()` reject files exceeding this limit before reading them.

- **`Settings.allowed_file_dir`** (default `None`): when set, restricts
  `warm_from_file()` and `load_templates_from_file()` to paths inside the specified
  directory, preventing path traversal attacks.

- **`Medha._resolve_and_check_path()`**: private method implementing path resolution
  and `allowed_file_dir` enforcement, used by all file-loading methods.

- **`pg_schema` and `pg_table_prefix` validators**: identifier regex
  `^[a-zA-Z_][a-zA-Z0-9_]{0,62}$` enforced at `Settings` construction time,
  preventing SQL injection via misconfigured table names.

- New demo notebooks:
  - `demo/11_inmemory_backend.ipynb` — zero-dependency setup, testing patterns,
    performance characteristics of the linear scan backend
  - `demo/12_pgvector_backend.ipynb` — PostgreSQL + pgvector setup, table naming,
    pool tuning, multi-tenant isolation, persistence verification

- Updated existing demo notebooks (`01`–`10`) to use explicit `backend_type` in
  all `Settings(...)` calls and to demonstrate new security settings where relevant.

### Changed

- **`Settings.qdrant_api_key`**: type changed from `str | None` to `SecretStr | None`.
  The secret value is never exposed in logs or `repr()`. Callers passing a plain
  string continue to work — Pydantic coerces it automatically.

- **`QdrantBackend._build_client()`**: updated to call `.get_secret_value()` when
  constructing the Qdrant Cloud client from `qdrant_api_key`.

- **`Medha.__init__`**: when `backend=None`, the backend is now selected via
  `settings.backend_type` (default `"qdrant"` — **fully backward compatible**).

- **`pyproject.toml`**: development status promoted from Alpha (3) to Beta (4).
  Added `[pgvector]` optional dependency group. `[all]` updated to include it.

- **Experiments** (`experiments/`): all benchmark scripts updated to use
  `backend_type="memory"` (zero-infra) where the backend is not the subject of
  measurement. `latency_benchmark.py` gains a `--backend qdrant|memory` flag for
  direct comparison.

### Fixed

- `qdrant_api_key` was previously logged as a plain string in debug output.
  Now stored as `SecretStr` and masked automatically.

- `warm_from_file()` and `load_templates_from_file()` previously allowed arbitrary
  file paths and sizes. Now validated against `allowed_file_dir` and `max_file_size_mb`.

- `pg_schema` and `pg_table_prefix` were previously passed directly to SQL queries
  without validation. Now validated at config construction time.

### Known Issues

- **`qdrant-client` is still a core dependency** even when `backend_type="memory"` or
  `backend_type="pgvector"` is selected. Users who only need `InMemoryBackend` or
  `PgVectorBackend` will still have `qdrant-client` installed unnecessarily.
  Moving it to an optional `[qdrant]` extra would be a breaking change (it would
  require existing users to add `[qdrant]` to their install command) and is deferred
  to 0.3.1, where it can be versioned as a breaking change alongside a potential
  change to the `backend_type` default.

## [0.1.0] — 2025-02

Initial release.

- Waterfall search strategy (L1 Cache → Template → Exact → Semantic → Fuzzy)
- Qdrant backend (memory / Docker / Cloud modes)
- FastEmbed and OpenAI embedding adapters
- In-memory and Redis L1 cache
- Template matching with parameter extraction (regex, spaCy, GLiNER, heuristics)
- Pydantic-based configuration with `MEDHA_` env var prefix
- Persistent embedding cache (JSON file)
- Comprehensive unit and integration test suite
- 10 demo notebooks
