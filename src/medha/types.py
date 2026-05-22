"""Data models: CacheHit, QueryTemplate, CacheEntry, CacheResult, and enums."""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class StrategyStats(BaseModel):
    """Per-strategy hit count and total latency."""

    model_config = ConfigDict(frozen=True)

    count: int = Field(default=0, ge=0)
    total_latency_ms: float = Field(default=0.0, ge=0.0)

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.count if self.count > 0 else 0.0


class CacheStats(BaseModel):
    """Snapshot of cache performance metrics."""

    model_config = ConfigDict(frozen=True)

    by_strategy: dict[str, StrategyStats] = Field(default_factory=dict)
    total_requests: int = Field(default=0, ge=0)
    total_hits: int = Field(default=0, ge=0)
    total_misses: int = Field(default=0, ge=0)
    total_errors: int = Field(default=0, ge=0)
    total_latency_ms: float = Field(default=0.0, ge=0.0)
    p50_latency_ms: float = Field(default=0.0, ge=0.0)
    p95_latency_ms: float = Field(default=0.0, ge=0.0)
    p99_latency_ms: float = Field(default=0.0, ge=0.0)
    backend_count: int = Field(default=0, ge=0)
    since: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    until: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def hit_rate(self) -> float:
        return self.total_hits / self.total_requests if self.total_requests > 0 else 0.0

    @property
    def miss_rate(self) -> float:
        return self.total_misses / self.total_requests if self.total_requests > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total_requests if self.total_requests > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"CacheStats(requests={self.total_requests}, "
            f"hit_rate={self.hit_rate:.1%}, "
            f"miss_rate={self.miss_rate:.1%}, "
            f"avg_latency={self.avg_latency_ms:.1f}ms, "
            f"p50={self.p50_latency_ms:.1f}ms, "
            f"p95={self.p95_latency_ms:.1f}ms, "
            f"p99={self.p99_latency_ms:.1f}ms, "
            f"backend_count={self.backend_count})"
        )


class SearchStrategy(str, Enum):
    """Identifies which tier of the waterfall produced the hit."""

    L1_CACHE = "l1_cache"
    TEMPLATE_MATCH = "template_match"
    EXACT_MATCH = "exact_match"
    SEMANTIC_MATCH = "semantic_match"
    FUZZY_MATCH = "fuzzy_match"
    NO_MATCH = "no_match"
    ERROR = "error"


class CacheHit(BaseModel):
    """Result of a cache lookup."""

    model_config = ConfigDict(frozen=True)

    generated_query: str = Field(
        default="", description="The SQL/Cypher/GraphQL query string"
    )
    response_summary: str | None = Field(
        default=None, description="Optional cached response/summary"
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Match confidence score"
    )
    strategy: SearchStrategy = Field(default=SearchStrategy.NO_MATCH)
    template_used: str | None = Field(
        default=None, description="Template intent if matched via template"
    )
    expires_at: datetime | None = Field(
        default=None, description="UTC expiry timestamp; None means immortal"
    )


class QueryTemplate(BaseModel):
    """A parameterized template for pattern-based cache matching."""

    intent: str = Field(description="Human-readable description of the template intent")
    template_text: str = Field(
        description="Parameterized question, e.g. 'Show top {count} {entity}'"
    )
    query_template: str = Field(
        description="Query with placeholders, e.g. 'SELECT * FROM {entity} LIMIT {count}'"
    )
    parameters: list[str] = Field(
        default_factory=list, description="List of parameter names"
    )
    priority: int = Field(default=1, ge=1, le=5, description="Priority (1=highest)")
    aliases: list[str] = Field(
        default_factory=list, description="Alternative phrasings"
    )
    parameter_patterns: dict[str, str] = Field(
        default_factory=dict,
        description="Regex patterns per parameter for extraction",
    )


class CacheEntry(BaseModel):
    """A single cache entry to be stored in the vector backend."""

    id: str = Field(description="Unique identifier (UUID)")
    vector: list[float] = Field(description="Embedding vector")
    original_question: str
    normalized_question: str
    generated_query: str = Field(
        description="The SQL/Cypher/GraphQL query string"
    )
    query_hash: str = Field(
        description="MD5 hash of generated_query for deduplication"
    )
    response_summary: str | None = None
    template_id: str | None = Field(
        default=None, description="Template intent if generated via template"
    )
    usage_count: int = Field(default=1, ge=0)
    feedback_correct: int = Field(default=0, ge=0)
    feedback_incorrect: int = Field(default=0, ge=0)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    expires_at: datetime | None = Field(
        default=None,
        description=(
            "Scadenza opzionale dell'entry. "
            "Se None, l'entry non scade mai. "
            "Se nel passato, l'entry viene esclusa dalle ricerche."
        ),
    )


class CacheResult(BaseModel):
    """A single result returned from a vector search."""

    id: str
    score: float = Field(ge=0.0, le=1.0)
    original_question: str
    normalized_question: str
    generated_query: str
    query_hash: str
    response_summary: str | None = None
    template_id: str | None = None
    usage_count: int = Field(default=0)
    feedback_correct: int = Field(default=0)
    feedback_incorrect: int = Field(default=0)
    created_at: datetime | None = None
    expires_at: datetime | None = Field(default=None, description="Scadenza dell'entry, se impostata.")
