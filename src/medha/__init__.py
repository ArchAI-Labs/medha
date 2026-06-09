"""Medha - Semantic Memory for AI Text-to-Query systems."""

__version__ = "0.4.2"

from medha.backends.memory import InMemoryBackend
from medha.config import Settings
from medha.core import Medha
from medha.interfaces.embedder import BaseEmbedder
from medha.interfaces.l1_cache import L1CacheBackend
from medha.interfaces.storage import VectorStorageBackend
from medha.l1_cache.memory import InMemoryL1Cache
from medha.l1_cache.redis_adapter import RedisL1Cache
from medha.logging import setup_logging
from medha.types import CacheEntry, CacheHit, CacheResult, CacheStats, QueryTemplate, SearchStrategy, StrategyStats

_optional: list[str] = []

try:
    from medha.backends.qdrant import QdrantBackend
    _optional.append("QdrantBackend")
except ImportError:
    pass

try:
    from medha.backends.pgvector import PgVectorBackend
    _optional.append("PgVectorBackend")
except ImportError:
    pass

try:
    from medha.backends.elasticsearch import ElasticsearchBackend
    _optional.append("ElasticsearchBackend")
except ImportError:
    pass

try:
    from medha.backends.vectorchord import VectorChordBackend
    _optional.append("VectorChordBackend")
except ImportError:
    pass

try:
    from medha.backends.chroma import ChromaBackend
    _optional.append("ChromaBackend")
except ImportError:
    pass

try:
    from medha.backends.weaviate import WeaviateBackend
    _optional.append("WeaviateBackend")
except ImportError:
    pass

try:
    from medha.backends.redis_vector import RedisVectorBackend
    _optional.append("RedisVectorBackend")
except ImportError:
    pass

try:
    from medha.backends.azure_search import AzureSearchBackend
    _optional.append("AzureSearchBackend")
except ImportError:
    pass

try:
    from medha.backends.lancedb import LanceDBBackend
    _optional.append("LanceDBBackend")
except ImportError:
    pass

try:
    from medha.embeddings.cohere_adapter import CohereAdapter
    _optional.append("CohereAdapter")
except ImportError:
    pass

try:
    from medha.embeddings.gemini_adapter import GeminiAdapter
    _optional.append("GeminiAdapter")
except ImportError:
    pass

try:
    from medha.embeddings.openai_compatible_adapter import OpenAICompatibleAdapter
    _optional.append("OpenAICompatibleAdapter")
except ImportError:
    pass

try:
    from medha.embeddings.mistral_adapter import MistralAdapter
    _optional.append("MistralAdapter")
except ImportError:
    pass

__all__ = [
    "Medha", "Settings", "CacheHit", "QueryTemplate", "CacheEntry",
    "CacheResult", "CacheStats", "StrategyStats", "SearchStrategy",
    "BaseEmbedder", "L1CacheBackend", "VectorStorageBackend",
    "InMemoryL1Cache", "RedisL1Cache", "setup_logging", "InMemoryBackend",
] + _optional
