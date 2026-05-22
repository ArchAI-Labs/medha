"""Pydantic-based configuration with environment variable support (MEDHA_ prefix)."""

import re
from typing import Literal

from pydantic import Field, SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_SAFE_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,62}$")


class Settings(BaseSettings):
    """Central configuration for a Medha instance."""

    model_config = SettingsConfigDict(
        env_prefix="MEDHA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Backend selection ---
    backend_type: Literal[
        "qdrant", "memory", "pgvector", "elasticsearch",
        "vectorchord", "chroma", "weaviate", "redis", "azure-search", "lancedb"
    ] = Field(
        default="memory",
        description=(
            "Vector storage backend to use. "
            "'memory' uses pure Python in-process storage, zero external deps (default). "
            "'qdrant' requires qdrant-client (pip install medha-archai[qdrant]). "
            "'pgvector' requires asyncpg and pgvector (pip install medha-archai[pgvector]). "
            "'elasticsearch' requires elasticsearch[async]>=8.12 (pip install medha-archai[elasticsearch]). "
            "'vectorchord' requires asyncpg (pip install medha-archai[vectorchord]). "
            "'chroma' requires chromadb>=0.5 (pip install medha-archai[chroma]). "
            "'weaviate' requires weaviate-client>=4.6 (pip install medha-archai[weaviate]). "
            "'redis' requires redis[hiredis]>=4.6 (pip install medha-archai[redis]). "
            "'azure-search' requires azure-search-documents>=11.4 (pip install medha-archai[azure-search]). "
            "'lancedb' requires lancedb>=0.6 (pip install medha-archai[lancedb])."
        ),
    )

    # --- Backend ---
    qdrant_mode: Literal["memory", "docker", "cloud"] = Field(
        default="memory",
        description="Qdrant connection mode",
    )
    qdrant_host: str = Field(default="localhost", description="Qdrant host for docker/cloud mode")
    qdrant_port: int = Field(default=6333, ge=1, le=65535, description="Qdrant gRPC port")
    qdrant_url: str | None = Field(default=None, description="Full Qdrant URL (overrides host:port)")
    qdrant_api_key: SecretStr | None = Field(default=None, description="Qdrant Cloud API key")

    # --- Query language ---
    query_language: Literal["sql", "cypher", "graphql", "generic"] = Field(
        default="generic",
        description="Target query language (informational, no behavioral change)",
    )

    # --- Search thresholds ---
    score_threshold_exact: float = Field(default=0.99, ge=0.0, le=1.0)
    score_threshold_semantic: float = Field(default=0.85, ge=0.0, le=1.0)
    score_threshold_template: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum score for a template match to be returned. "
            "The maximum achievable score is ~0.88 "
            "(keyword_bonus=1.0 × 0.5 + param_completeness=1.0 × 0.3 + priority_1 × 0.08). "
            "Values above 0.88 will never match any template."
        ),
    )
    score_threshold_fuzzy: float = Field(default=85.0, ge=0.0, le=100.0, description="Fuzzy match threshold (0-100)")
    score_threshold_fuzzy_prefilter: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum cosine similarity for the vector pre-filter used in fuzzy search. "
            "Candidates below this threshold are excluded before Levenshtein scoring, "
            "reducing fuzzy search from O(n) to O(top_k). Lower values increase recall "
            "at the cost of more fuzzy comparisons."
        ),
    )
    fuzzy_prefilter_top_k: int = Field(
        default=50,
        ge=1,
        le=1000,
        description=(
            "Maximum number of vector-similar candidates to retrieve for fuzzy pre-filtering. "
            "Fuzzy scoring is applied only to these candidates instead of the full collection."
        ),
    )

    # --- L1 Cache ---
    l1_cache_max_size: int = Field(default=1000, ge=0, description="Max entries in L1 in-memory cache (0=disabled)")

    # --- Qdrant tuning ---
    hnsw_m: int = Field(default=16, ge=4, le=64, description="HNSW edges per node")
    hnsw_ef_construct: int = Field(default=100, ge=50, le=500, description="HNSW construction search depth")
    enable_quantization: bool = Field(default=True, description="Enable vector quantization")
    quantization_type: Literal["scalar", "binary"] = Field(
        default="scalar",
        description="Quantization method. Binary only for dim >= 512",
    )
    on_disk: bool = Field(default=False, description="Store vectors on disk (large datasets)")

    # --- PostgreSQL / pgvector ---
    pg_dsn: str | None = Field(
        default=None,
        description=(
            "Full asyncpg DSN for PostgreSQL connection "
            "(e.g. 'postgresql://user:pass@localhost:5432/dbname'). "
            "When set, overrides pg_host, pg_port, pg_database, pg_user, pg_password."
        ),
    )
    pg_host: str = Field(default="localhost", description="PostgreSQL host")
    pg_port: int = Field(default=5432, ge=1, le=65535, description="PostgreSQL port")
    pg_database: str = Field(default="medha", description="PostgreSQL database name")
    pg_user: str = Field(default="postgres", description="PostgreSQL user")
    pg_password: SecretStr = Field(default=SecretStr(""), description="PostgreSQL password")
    pg_schema: str = Field(default="public", description="PostgreSQL schema for Medha tables")
    pg_table_prefix: str = Field(
        default="medha",
        description="Prefix for Medha table names (e.g. 'medha' → table 'medha_my_cache')",
    )
    pg_pool_min_size: int = Field(default=2, ge=1, description="Min connections in asyncpg pool")
    pg_pool_max_size: int = Field(default=10, ge=1, description="Max connections in asyncpg pool")

    # --- VectorChord ---
    vc_lists: list[int] = Field(
        default_factory=lambda: [1000],
        description="Number of centroids per level for the vchordrq index.",
    )
    vc_residual_quantization: bool = Field(
        default=True,
        description="Enable residual quantization in the vchordrq index.",
    )

    # --- Weaviate ---
    weaviate_mode: Literal["local", "cloud"] = Field(
        default="local",
        description="Weaviate connection mode: 'local' (self-hosted) or 'cloud' (Weaviate Cloud).",
    )
    weaviate_host: str = Field(default="localhost", description="Weaviate host (local mode)")
    weaviate_http_port: int = Field(default=8080, ge=1, le=65535, description="Weaviate HTTP port (local mode)")
    weaviate_grpc_port: int = Field(default=50051, ge=1, le=65535, description="Weaviate gRPC port (local mode)")
    weaviate_http_secure: bool = Field(default=False, description="Use HTTPS for Weaviate HTTP connection")
    weaviate_grpc_secure: bool = Field(default=False, description="Use TLS for Weaviate gRPC connection")
    weaviate_cloud_url: str | None = Field(default=None, description="Weaviate Cloud cluster URL (cloud mode)")
    weaviate_api_key: SecretStr | None = Field(default=None, description="Weaviate API key")
    weaviate_collection_prefix: str = Field(
        default="Medha",
        description="Prefix for Weaviate collection names in PascalCase (e.g. 'Medha' → 'MedhaMyCache')",
    )

    # --- Redis Stack ---
    redis_mode: Literal["standalone", "sentinel"] = Field(
        default="standalone",
        description="Redis connection mode: 'standalone' or 'sentinel'.",
    )
    redis_url: str | None = Field(default=None, description="Full Redis URL (overrides host/port/db)")
    redis_host: str = Field(default="localhost", description="Redis host (standalone mode)")
    redis_port: int = Field(default=6379, ge=1, le=65535, description="Redis port")
    redis_db: int = Field(default=0, ge=0, description="Redis database index")
    redis_username: str | None = Field(default=None, description="Redis ACL username")
    redis_password: SecretStr | None = Field(default=None, description="Redis password")
    redis_ssl: bool = Field(default=False, description="Enable TLS for Redis connection")
    redis_ssl_certfile: str | None = Field(default=None, description="Path to client TLS certificate")
    redis_ssl_keyfile: str | None = Field(default=None, description="Path to client TLS key")
    redis_ssl_ca_certs: str | None = Field(default=None, description="Path to CA certificate bundle")
    redis_sentinel_hosts: list[str] = Field(
        default_factory=lambda: ["localhost:26379"],
        description="Sentinel host:port list (sentinel mode)",
    )
    redis_sentinel_master: str = Field(default="mymaster", description="Sentinel master name")
    redis_key_prefix: str = Field(default="medha", description="Prefix for all Redis keys and index names")
    redis_index_algorithm: Literal["HNSW", "FLAT"] = Field(
        default="HNSW",
        description="RediSearch vector index algorithm: 'HNSW' (approx) or 'FLAT' (exact brute-force)",
    )
    redis_hnsw_m: int = Field(default=16, ge=4, le=64, description="HNSW: edges per node")
    redis_hnsw_ef_construction: int = Field(default=200, ge=50, le=500, description="HNSW: build search depth")
    redis_hnsw_ef_runtime: int = Field(default=10, ge=10, le=500, description="HNSW: query search depth")
    redis_socket_timeout: float = Field(default=5.0, gt=0.0, description="Redis socket read timeout (s)")
    redis_socket_connect_timeout: float = Field(default=5.0, gt=0.0, description="Redis socket connect timeout (s)")

    # --- Chroma ---
    chroma_mode: Literal["ephemeral", "persistent", "http"] = Field(
        default="ephemeral",
        description="Chroma connection mode: 'ephemeral' (in-memory), 'persistent' (local disk), 'http' (remote server).",
    )
    chroma_host: str = Field(default="localhost", description="Chroma server host (http mode)")
    chroma_port: int = Field(default=8000, ge=1, le=65535, description="Chroma server port (http mode)")
    chroma_persist_path: str | None = Field(default=None, description="Directory for Chroma persistent storage")
    chroma_ssl: bool = Field(default=False, description="Use SSL for Chroma http connection")
    chroma_auth_token: SecretStr | None = Field(default=None, description="Bearer token for Chroma http authentication")

    # --- LanceDB ---
    lancedb_uri: str = Field(
        default="./lancedb_data",
        description=(
            "LanceDB storage URI. Use a local path (e.g. './lancedb_data') for embedded mode, "
            "or a cloud URI (s3://, gs://, az://) for cloud storage."
        ),
    )
    lancedb_table_prefix: str = Field(
        default="medha",
        description="Prefix for LanceDB table names (e.g. 'medha' → 'medha_my_cache').",
    )
    lancedb_metric: Literal["cosine", "l2", "dot"] = Field(
        default="cosine",
        description="Distance metric for LanceDB vector search: 'cosine' (default), 'l2', or 'dot'.",
    )

    # --- Azure AI Search ---
    azure_search_endpoint: str = Field(
        default="",
        description="Azure AI Search service endpoint (e.g. https://my-service.search.windows.net).",
    )
    azure_search_api_key: SecretStr | None = Field(
        default=None,
        description="Azure AI Search API key. If None, uses DefaultAzureCredential (requires azure-identity).",
    )
    azure_search_api_version: str = Field(
        default="2024-05-01-preview",
        description="Azure AI Search REST API version.",
    )
    azure_search_index_name: str = Field(
        default="medha",
        description=(
            "Prefix for Azure Search index names. "
            "Final index = '{azure_search_index_name}-{collection_name}' "
            "(e.g. 'medha' + 'my_cache' → 'medha-my-cache'). "
            "Corresponds to the env var MEDHA_AZURE_SEARCH_INDEX_NAME."
        ),
    )
    azure_search_top_k_candidates: int = Field(
        default=50,
        ge=1,
        le=10000,
        description=(
            "Extra candidates retrieved by HNSW before score filtering. "
            "Added to limit in VectorizedQuery to improve recall without increasing returned results."
        ),
    )

    # --- Elasticsearch ---
    es_hosts: list[str] = Field(
        default_factory=lambda: ["http://localhost:9200"],
        description="Elasticsearch node URLs",
    )
    es_api_key: SecretStr | None = Field(default=None, description="Elasticsearch API key")
    es_username: str | None = Field(default=None, description="Elasticsearch basic-auth username")
    es_password: SecretStr | None = Field(default=None, description="Elasticsearch basic-auth password")
    es_index_prefix: str = Field(default="medha", description="Prefix for Elasticsearch index names")
    es_num_candidates: int = Field(
        default=100, ge=1, le=10000, description="num_candidates for kNN search"
    )
    es_timeout: float = Field(default=30.0, gt=0.0, description="Request timeout in seconds")

    # --- Quantization search ---
    quantization_rescore: bool = Field(
        default=True,
        description="Re-score top results using original vectors after quantized search",
    )
    quantization_oversampling: float | None = Field(
        default=None,
        ge=1.0,
        description=(
            "Oversampling factor for quantized search "
            "(e.g. 2.0 fetches 2x candidates before re-scoring). None = Qdrant default"
        ),
    )
    quantization_ignore: bool = Field(
        default=False,
        description="Bypass quantized vectors and search only original vectors",
    )
    quantization_always_ram: bool = Field(
        default=True,
        description=(
            "Keep quantized vectors in RAM. Combined with on_disk=True enables "
            "hybrid storage (original on disk, quantized in RAM)"
        ),
    )

    # --- Template loading ---
    template_file: str | None = Field(default=None, description="Path to JSON template file")

    # --- Persistent embedding cache ---
    embedding_cache_path: str | None = Field(
        default=None,
        description=(
            "Path to a JSON file used to persist the embedding cache across restarts. "
            "When set, embeddings are loaded from disk on start() and saved to disk on close(). "
            "Speeds up warm-start scenarios where the same questions recur across sessions."
        ),
    )

    # --- File operations ---
    allowed_file_dir: str | None = Field(
        default=None,
        description=(
            "If set, warm_from_file() and load_templates_from_file() will reject paths "
            "outside this directory. Useful when the caller is not trusted. "
            "Example: '/app/data'. Default: None (no restriction)."
        ),
    )
    max_file_size_mb: int = Field(
        default=100,
        ge=1,
        le=10_000,
        description=(
            "Maximum file size in MB for warm_from_file() and load_templates_from_file(). "
            "Files exceeding this limit are rejected before reading."
        ),
    )

    # --- Input validation ---
    max_question_length: int = Field(
        default=8192,
        ge=64,
        le=1_000_000,
        description=(
            "Maximum allowed length (characters) for a question string. "
            "Questions exceeding this limit are rejected with SearchStrategy.ERROR "
            "to prevent DoS via oversized inputs. Default: 8192 chars (~8KB)."
        ),
    )

    # --- Cache lifecycle ---
    default_ttl_seconds: int | None = Field(
        default=None,
        ge=1,
        description=(
            "TTL di default in secondi per le nuove entry. "
            "None = entry immortali (comportamento attuale). "
            "Può essere sovrascritto entry per entry tramite il parametro ttl di store()."
        ),
    )
    cleanup_interval_seconds: int | None = Field(
        default=None,
        ge=60,
        description=(
            "Intervallo in secondi per il cleanup automatico delle entry scadute. "
            "None = nessun cleanup automatico. "
            "Se impostato, Medha.start() avvia un task asyncio che chiama expire() periodicamente."
        ),
    )
    feedback_incorrect_threshold: int | None = Field(
        default=None,
        ge=1,
        description=(
            "When set, feedback(correct=False) automatically invalidates the entry "
            "once feedback_incorrect reaches this value. "
            "None disables auto-invalidation. Env var: MEDHA_FEEDBACK_INCORRECT_THRESHOLD."
        ),
    )

    # --- Batch operations ---
    batch_size: int = Field(default=100, ge=1, le=10000, description="Batch size for bulk upsert")
    batch_embed_concurrency: int = Field(default=1, ge=1, le=10, description="Chunk di embedding processati concorrentemente in store_many().")

    # --- Observability ---
    collect_stats: bool = Field(
        default=True,
        description="Enable collection of cache performance statistics.",
    )
    stats_max_latency_samples: int = Field(
        default=10_000,
        ge=100,
        le=1_000_000,
        description=(
            "Maximum number of per-request latency samples retained for percentile calculation. "
            "Older samples are evicted when the buffer is full (FIFO)."
        ),
    )

    # --- Timeouts ---
    embedding_timeout: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Timeout in seconds for embedding calls (aembed and aembed_batch). "
            "None disables the timeout. Increase for large batches or slow networks."
        ),
    )

    # --- CLI ---
    embedder_type: Literal["fastembed", "openai", "cohere", "gemini", "_noop"] = Field(
        default="_noop",
        description=(
            "Embedder to use. '_noop' is the default (no embedding). "
            "Real embedders require the matching extra. "
            "Env var: MEDHA_EMBEDDER_TYPE."
        ),
    )

    collection: str = Field(
        default="default",
        description="Default collection name for CLI commands. Env var: MEDHA_COLLECTION.",
    )

    fastembed_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="FastEmbed model identifier used by the CLI. Env var: MEDHA_FASTEMBED_MODEL.",
    )

    # --- Validators ---
    @field_validator("pg_schema", "pg_table_prefix")
    @classmethod
    def validate_pg_identifier(cls, v: str) -> str:
        if not _SAFE_IDENTIFIER_RE.match(v):
            raise ValueError(
                f"Invalid PostgreSQL identifier '{v}': must match ^[a-zA-Z_][a-zA-Z0-9_]{{0,62}}$"
            )
        return v

    @field_validator("pg_pool_max_size")
    @classmethod
    def pool_max_gte_min(cls, v: int, info: ValidationInfo) -> int:
        min_size = info.data.get("pg_pool_min_size", 2)
        if v < min_size:
            raise ValueError(
                f"pg_pool_max_size ({v}) must be >= pg_pool_min_size ({min_size})"
            )
        return v

    @field_validator("score_threshold_exact")
    @classmethod
    def exact_must_be_high(cls, v: float) -> float:
        if v < 0.90:
            raise ValueError("Exact threshold should be >= 0.90 to avoid false positives")
        return v

    @field_validator("score_threshold_semantic")
    @classmethod
    def semantic_below_exact(cls, v: float, info: ValidationInfo) -> float:
        exact = info.data.get("score_threshold_exact", 0.99)
        if v >= exact:
            raise ValueError("Semantic threshold must be lower than exact threshold")
        return v
