"""Data models: CacheHit, QueryTemplate, CacheEntry, CacheResult, and enums."""

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


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
    response_summary: Optional[str] = Field(
        default=None, description="Optional cached response/summary"
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Match confidence score"
    )
    strategy: SearchStrategy = Field(default=SearchStrategy.NO_MATCH)
    template_used: Optional[str] = Field(
        default=None, description="Template intent if matched via template"
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
    parameters: List[str] = Field(
        default_factory=list, description="List of parameter names"
    )
    priority: int = Field(default=1, ge=1, le=5, description="Priority (1=highest)")
    aliases: List[str] = Field(
        default_factory=list, description="Alternative phrasings"
    )
    parameter_patterns: Dict[str, str] = Field(
        default_factory=dict,
        description="Regex patterns per parameter for extraction",
    )


class CacheEntry(BaseModel):
    """A single cache entry to be stored in the vector backend."""

    id: str = Field(description="Unique identifier (UUID)")
    vector: List[float] = Field(description="Embedding vector")
    original_question: str
    normalized_question: str
    generated_query: str = Field(
        description="The SQL/Cypher/GraphQL query string"
    )
    query_hash: str = Field(
        description="MD5 hash of generated_query for deduplication"
    )
    response_summary: Optional[str] = None
    template_id: Optional[str] = Field(
        default=None, description="Template intent if generated via template"
    )
    usage_count: int = Field(default=1, ge=0)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class CacheResult(BaseModel):
    """A single result returned from a vector search."""

    id: str
    score: float = Field(ge=0.0, le=1.0)
    original_question: str
    normalized_question: str
    generated_query: str
    query_hash: str
    response_summary: Optional[str] = None
    template_id: Optional[str] = None
    usage_count: int = Field(default=0)
    created_at: Optional[datetime] = None
