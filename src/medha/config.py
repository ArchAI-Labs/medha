"""Pydantic-based configuration with environment variable support (MEDHA_ prefix)."""

from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for a Medha instance."""

    model_config = SettingsConfigDict(
        env_prefix="MEDHA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Backend ---
    qdrant_mode: Literal["memory", "docker", "cloud"] = Field(
        default="memory",
        description="Qdrant connection mode",
    )
    qdrant_host: str = Field(default="localhost", description="Qdrant host for docker/cloud mode")
    qdrant_port: int = Field(default=6333, ge=1, le=65535, description="Qdrant gRPC port")
    qdrant_url: Optional[str] = Field(default=None, description="Full Qdrant URL (overrides host:port)")
    qdrant_api_key: Optional[str] = Field(default=None, description="Qdrant Cloud API key")

    # --- Query language ---
    query_language: Literal["sql", "cypher", "graphql", "generic"] = Field(
        default="generic",
        description="Target query language (informational, no behavioral change)",
    )

    # --- Search thresholds ---
    score_threshold_exact: float = Field(default=0.99, ge=0.0, le=1.0)
    score_threshold_semantic: float = Field(default=0.90, ge=0.0, le=1.0)
    score_threshold_template: float = Field(default=0.70, ge=0.0, le=1.0)
    score_threshold_fuzzy: float = Field(default=85.0, ge=0.0, le=100.0, description="Fuzzy match threshold (0-100)")

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

    # --- Quantization search ---
    quantization_rescore: bool = Field(
        default=True,
        description="Re-score top results using original vectors after quantized search",
    )
    quantization_oversampling: Optional[float] = Field(
        default=None,
        ge=1.0,
        description="Oversampling factor for quantized search (e.g. 2.0 fetches 2x candidates before re-scoring). None = Qdrant default",
    )
    quantization_ignore: bool = Field(
        default=False,
        description="Bypass quantized vectors and search only original vectors",
    )
    quantization_always_ram: bool = Field(
        default=True,
        description="Keep quantized vectors in RAM. Combined with on_disk=True enables hybrid storage (original on disk, quantized in RAM)",
    )

    # --- Template loading ---
    template_file: Optional[str] = Field(default=None, description="Path to JSON template file")

    # --- Batch operations ---
    batch_size: int = Field(default=100, ge=1, le=10000, description="Batch size for bulk upsert")

    # --- Validators ---
    @field_validator("score_threshold_exact")
    @classmethod
    def exact_must_be_high(cls, v: float) -> float:
        if v < 0.90:
            raise ValueError("Exact threshold should be >= 0.90 to avoid false positives")
        return v

    @field_validator("score_threshold_semantic")
    @classmethod
    def semantic_below_exact(cls, v: float, info) -> float:  # type: ignore[type-arg]
        exact = info.data.get("score_threshold_exact", 0.99)
        if v >= exact:
            raise ValueError("Semantic threshold must be lower than exact threshold")
        return v
