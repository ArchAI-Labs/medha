"""Vector storage backend implementations."""

from typing import Any

from medha.backends.memory import InMemoryBackend

try:
    from medha.backends.qdrant import QdrantBackend
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False

_all_extra: list[str] = []

try:
    from medha.backends.pgvector import PgVectorBackend
    _all_extra.append("PgVectorBackend")
except ImportError:
    pass

try:
    from medha.backends.elasticsearch import ElasticsearchBackend
    _all_extra.append("ElasticsearchBackend")
except ImportError:
    pass

try:
    from medha.backends.vectorchord import VectorChordBackend
    _all_extra.append("VectorChordBackend")
except ImportError:
    pass

try:
    from medha.backends.chroma import ChromaBackend
    _all_extra.append("ChromaBackend")
except ImportError:
    pass

try:
    from medha.backends.weaviate import WeaviateBackend
    _all_extra.append("WeaviateBackend")
except ImportError:
    pass

try:
    from medha.backends.redis_vector import RedisVectorBackend
    _all_extra.append("RedisVectorBackend")
except ImportError:
    pass

try:
    from medha.backends.azure_search import AzureSearchBackend
    _all_extra.append("AzureSearchBackend")
except ImportError:
    pass

try:
    from medha.backends.lancedb import LanceDBBackend
    _all_extra.append("LanceDBBackend")
except ImportError:
    pass

__all__ = ["InMemoryBackend"] + (["QdrantBackend"] if HAS_QDRANT else []) + _all_extra

# Backend name -> extra that provides its driver. Used to turn the default
# "cannot import name 'QdrantBackend'" into something actionable.
_BACKEND_EXTRAS = {
    "QdrantBackend": "qdrant",
    "PgVectorBackend": "pgvector",
    "ElasticsearchBackend": "elasticsearch",
    "VectorChordBackend": "vectorchord",
    "ChromaBackend": "chroma",
    "WeaviateBackend": "weaviate",
    "RedisVectorBackend": "redis",
    "AzureSearchBackend": "azure-search",
    "LanceDBBackend": "lancedb",
}


def __getattr__(name: str) -> Any:
    """Explain which extra to install for a backend whose driver is missing."""
    extra = _BACKEND_EXTRAS.get(name)
    if extra is not None:
        raise ImportError(
            f"{name} requires the [{extra}] extra, which is not installed.\n"
            f'Install it with:  pip install "medha-archai[{extra}]"'
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
