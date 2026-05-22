# Configuration

Medha is configured through the `Settings` class, which is backed by Pydantic Settings. All fields can be set programmatically or via environment variables with the `MEDHA_` prefix.

---

!!! warning "Threshold Ordering"

    `MEDHA_SCORE_THRESHOLD_SEMANTIC` must be **strictly less than** `MEDHA_SCORE_THRESHOLD_EXACT`. Medha validates this at startup and raises `ConfigurationError` if the invariant is violated. Typical values: semantic = `0.85`, exact = `0.99`.

---

## Construction

=== "Programmatic"

    ```python
    from medha import Settings

    settings = Settings(
        backend_type="qdrant",
        score_threshold_semantic=0.85,
        score_threshold_exact=0.99,
        default_ttl_seconds=3600,
        collect_stats=True,
    )
    ```

=== "Environment Variables"

    ```bash
    export MEDHA_BACKEND_TYPE=qdrant
    export MEDHA_SCORE_THRESHOLD_SEMANTIC=0.85
    export MEDHA_SCORE_THRESHOLD_EXACT=0.99
    export MEDHA_DEFAULT_TTL_SECONDS=3600
    export MEDHA_COLLECT_STATS=true
    ```

    ```python
    from medha import Settings

    # Reads from environment automatically
    settings = Settings()
    ```

=== ".env File"

    ```ini
    # .env
    MEDHA_BACKEND_TYPE=qdrant
    MEDHA_SCORE_THRESHOLD_SEMANTIC=0.85
    MEDHA_SCORE_THRESHOLD_EXACT=0.99
    MEDHA_COLLECT_STATS=true
    ```

    ```python
    from medha import Settings

    settings = Settings(_env_file=".env")
    ```

---

## Field Reference

??? info "Search Thresholds"

    | Field | Type | Default | Description |
    |---|---|---|---|
    | `score_threshold_semantic` | `float` | `0.85` | Minimum cosine score for Tier 3 (Semantic Match) |
    | `score_threshold_exact` | `float` | `0.99` | Minimum cosine score for Tier 2 (Exact Vector Match) |
    | `score_threshold_fuzzy` | `float` | `0.85` | Minimum Levenshtein ratio for Tier 4 (Fuzzy Match) |
    | `enable_fuzzy` | `bool` | `False` | Whether to activate the Fuzzy tier at all |
    | `top_k` | `int` | `1` | Number of candidates to retrieve from the vector backend |

    **Environment variable prefix:** `MEDHA_SCORE_THRESHOLD_*`, `MEDHA_ENABLE_FUZZY`, `MEDHA_TOP_K`

??? info "L1 Cache"

    | Field | Type | Default | Description |
    |---|---|---|---|
    | `l1_cache_type` | `str` | `"memory"` | L1 implementation: `"memory"` or `"redis"` |
    | `l1_cache_max_size` | `int` | `1000` | Max items in the in-memory L1 cache (LRU eviction) |
    | `l1_redis_url` | `str` | `"redis://localhost:6379"` | Redis URL when `l1_cache_type="redis"` |
    | `l1_redis_ttl_seconds` | `int \| None` | `None` | TTL for L1 Redis entries |

    **Environment variable prefix:** `MEDHA_L1_*`

??? info "Backend Selection"

    | Field | Type | Default | Description |
    |---|---|---|---|
    | `backend_type` | `str` | `"memory"` | Vector backend: `memory`, `qdrant`, `pgvector`, `elasticsearch`, `vectorchord`, `chroma`, `weaviate`, `redis`, `azure_search`, `lancedb` |
    | `qdrant_host` | `str` | `"localhost"` | Qdrant server host |
    | `qdrant_port` | `int` | `6333` | Qdrant gRPC port |
    | `qdrant_api_key` | `str \| None` | `None` | Qdrant Cloud API key |
    | `qdrant_url` | `str \| None` | `None` | Full Qdrant URL (overrides host/port) |
    | `pg_dsn` | `str \| None` | `None` | PostgreSQL DSN for pgvector / VectorChord |
    | `es_url` | `str` | `"http://localhost:9200"` | Elasticsearch URL |
    | `chroma_host` | `str` | `"localhost"` | Chroma host (HTTP mode) |
    | `weaviate_url` | `str` | `"http://localhost:8080"` | Weaviate instance URL |
    | `redis_url` | `str` | `"redis://localhost:6379"` | Redis Stack URL for vector backend |
    | `azure_search_endpoint` | `str \| None` | `None` | Azure AI Search endpoint |
    | `azure_search_api_key` | `str \| None` | `None` | Azure AI Search admin key |
    | `lancedb_uri` | `str` | `"./lancedb"` | LanceDB database path or URI |

    **Environment variable prefix:** `MEDHA_BACKEND_TYPE`, `MEDHA_QDRANT_*`, `MEDHA_PG_DSN`, `MEDHA_ES_*`, etc.

??? info "TTL & Lifecycle"

    | Field | Type | Default | Description |
    |---|---|---|---|
    | `default_ttl_seconds` | `int \| None` | `None` | Global TTL for new entries; `None` means no expiry |
    | `cleanup_interval_seconds` | `int` | `300` | How often (seconds) the background cleanup task runs |
    | `enable_background_cleanup` | `bool` | `True` | Enable periodic sweep of expired entries |

    **Environment variable prefix:** `MEDHA_DEFAULT_TTL_SECONDS`, `MEDHA_CLEANUP_*`, `MEDHA_ENABLE_BACKGROUND_CLEANUP`

??? info "Batch Operations"

    | Field | Type | Default | Description |
    |---|---|---|---|
    | `batch_embed_concurrency` | `int` | `8` | Max concurrent embedding calls during batch ingestion |
    | `batch_upsert_size` | `int` | `100` | Number of entries per upsert call to the vector backend |

    **Environment variable prefix:** `MEDHA_BATCH_*`

??? info "Observability"

    | Field | Type | Default | Description |
    |---|---|---|---|
    | `collect_stats` | `bool` | `True` | Enable hit/miss counters and per-strategy tracking |
    | `log_level` | `str` | `"WARNING"` | Python logging level for the `medha` logger |
    | `log_format` | `str` | `"text"` | Log format: `"text"` or `"json"` |

    **Environment variable prefix:** `MEDHA_COLLECT_STATS`, `MEDHA_LOG_*`

??? info "Feedback Loop"

    | Field | Type | Default | Description |
    |---|---|---|---|
    | `feedback_incorrect_threshold` | `int \| None` | `None` | Auto-invalidate an entry after this many incorrect feedback signals. `None` disables auto-invalidation. Must be ≥ 1. |

    **Environment variable:** `MEDHA_FEEDBACK_INCORRECT_THRESHOLD`

??? info "CLI"

    | Field | Type | Default | Description |
    |---|---|---|---|
    | `embedder_type` | `str` | `"_noop"` | Embedder used by CLI commands. Accepted values: `fastembed`, `openai`, `cohere`, `gemini`, `_noop`. |
    | `collection` | `str` | `"default"` | Default collection name for CLI commands. Overridable per-command with `--collection`. |
    | `fastembed_model` | `str` | `"BAAI/bge-small-en-v1.5"` | FastEmbed model identifier used by `medha warm`. |

    **Environment variables:** `MEDHA_EMBEDDER_TYPE`, `MEDHA_COLLECTION`, `MEDHA_FASTEMBED_MODEL`

??? info "Security & File I/O"

    | Field | Type | Default | Description |
    |---|---|---|---|
    | `allowed_file_dir` | `str \| None` | `None` | Restrict file reads (`warm_from_file`) to this directory |
    | `max_file_size_mb` | `int` | `100` | Maximum allowed file size for bulk ingestion |
    | `max_question_length` | `int` | `2000` | Maximum character length of an input question |

    **Environment variable prefix:** `MEDHA_ALLOWED_FILE_DIR`, `MEDHA_MAX_*`

---

## See Also

- [Backends](backends.md) — backend-specific configuration fields
- [TTL & Lifecycle](ttl_and_lifecycle.md) — how TTL interacts with expiry
- [API: Settings](../api/config.md) — auto-generated full field reference
