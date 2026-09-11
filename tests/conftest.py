"""Shared pytest fixtures for Medha tests."""

import hashlib
import uuid

import pytest

from medha.config import Settings
from medha.interfaces.embedder import BaseEmbedder
from medha.types import CacheEntry, QueryTemplate


class MockEmbedder(BaseEmbedder):
    """Deterministic embedder for unit tests.

    Generates vectors by hashing the input text and spreading the hash
    across the requested dimension. Identical inputs always produce
    identical vectors; similar inputs produce similar (but not identical)
    vectors.
    """

    def __init__(self, dimension: int = 384):
        self._dimension = dimension
        self._model_name = "mock-embedder"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    async def aembed(self, text: str) -> list[float]:
        """Generate a deterministic embedding from text hash."""
        h = hashlib.sha256(text.lower().encode()).hexdigest()
        # Expand hash to fill dimension
        values = []
        for i in range(self._dimension):
            byte_val = int(h[(i * 2) % len(h) : (i * 2 + 2) % len(h) or len(h)], 16)
            values.append((byte_val / 255.0) * 2 - 1)  # Normalize to [-1, 1]
        # Normalize to unit vector
        magnitude = sum(v**2 for v in values) ** 0.5
        return [v / magnitude for v in values] if magnitude > 0 else values

    async def aembed_batch(self, texts: list[str], **kwargs: object) -> list[list[float]]:
        return [await self.aembed(t) for t in texts]


@pytest.fixture
def mock_embedder():
    """Provide a mock embedder with dimension=384."""
    return MockEmbedder(dimension=384)


@pytest.fixture
def test_settings():
    """Provide test-friendly settings (in-memory Qdrant, relaxed thresholds)."""
    return Settings(
        qdrant_mode="memory",
        score_threshold_exact=0.99,
        score_threshold_semantic=0.85,
        score_threshold_template=0.80,
        score_threshold_fuzzy=80.0,
        l1_cache_max_size=100,
    )


@pytest.fixture
async def medha_memory(mock_embedder):
    """Medha with InMemoryBackend and MockEmbedder (deterministic, no external deps)."""
    from medha.backends.memory import InMemoryBackend
    from medha.core import Medha

    backend = InMemoryBackend()
    await backend.connect()
    settings = Settings(
        backend_type="memory",
        score_threshold_exact=0.99,
        score_threshold_semantic=0.85,
        l1_cache_max_size=100,
    )
    m = Medha(
        collection_name="inmemory_e2e",
        embedder=mock_embedder,
        backend=backend,
        settings=settings,
    )
    await m.start()
    yield m
    await m.close()


@pytest.fixture
def test_settings_memory():
    """Settings con backend_type=memory."""
    return Settings(
        backend_type="memory",
        score_threshold_exact=0.99,
        score_threshold_semantic=0.85,
        score_threshold_template=0.80,
        score_threshold_fuzzy=80.0,
        l1_cache_max_size=100,
    )


@pytest.fixture
async def inmemory_backend():
    from medha.backends.memory import InMemoryBackend
    b = InMemoryBackend()
    await b.connect()
    yield b
    await b.close()


@pytest.fixture
def test_settings_pgvector():
    """Settings con backend_type=pgvector. Richiede PG reale."""
    import os
    return Settings(
        backend_type="pgvector",
        pg_dsn=os.environ.get(
            "MEDHA_TEST_PG_DSN",
            "postgresql://postgres:postgres@localhost:5432/medha_test",
        ),
        score_threshold_exact=0.99,
        score_threshold_semantic=0.85,
        score_threshold_template=0.80,
        score_threshold_fuzzy=80.0,
        l1_cache_max_size=100,
    )


def _chroma_available() -> bool:
    try:
        import chromadb  # noqa: F401
        return True
    except ImportError:
        return False


def _lancedb_available() -> bool:
    try:
        import lancedb  # noqa: F401
        return True
    except ImportError:
        return False


_any_backend_params = (
    ["memory"]
    + (["chroma"] if _chroma_available() else [])
    + (["lancedb"] if _lancedb_available() else [])
)


@pytest.fixture(params=_any_backend_params)
async def any_backend(request, tmp_path):
    """Parametrised over all backends testable without external services."""
    if request.param == "memory":
        from medha.backends.memory import InMemoryBackend
        b = InMemoryBackend()
        await b.connect()
        yield b
        await b.close()
    elif request.param == "chroma":
        from medha.backends.chroma import ChromaBackend
        from medha.config import Settings
        b = ChromaBackend(Settings(chroma_mode="ephemeral"))
        await b.connect()
        yield b
        await drop_all_chroma_collections(b)
        await b.close()
    elif request.param == "lancedb":
        from medha.backends.lancedb import LanceDBBackend
        from medha.config import Settings
        b = LanceDBBackend(Settings(lancedb_uri=str(tmp_path / "lancedb_any")))
        await b.connect()
        yield b
        await b.close()



async def drop_all_chroma_collections(backend) -> None:
    """Empty the process-wide Chroma store a backend was talking to.

    chromadb serves every ephemeral client in a process from one cached store,
    so a collection outlives the backend that created it and the next test
    inherits its rows. The other backends tested here need no equivalent: the
    in-memory one dies with its instance, and LanceDB gets a fresh ``tmp_path``.
    Closing the backend is not enough — it only drops medha's own references.
    """
    import asyncio

    client = backend._client
    if client is None:
        return
    for entry in await asyncio.to_thread(client.list_collections):
        # 0.6 returns names, earlier releases return collection objects.
        name = entry if isinstance(entry, str) else entry.name
        await asyncio.to_thread(client.delete_collection, name=name)

@pytest.fixture
async def pgvector_backend(test_settings_pgvector):
    pytest.importorskip("asyncpg")  # skip if not installed
    from medha.backends.pgvector import PgVectorBackend
    b = PgVectorBackend(test_settings_pgvector)
    await b.connect()
    yield b
    await b.close()


def make_entry(
    id: str | None = None,
    vector: list[float] | None = None,
    question: str = "test question",
    query: str = "SELECT 1",
    dim: int = 8,
) -> CacheEntry:
    """Factory per CacheEntry usata nei test."""
    vec = vector or [0.1] * dim
    return CacheEntry(
        id=id or str(uuid.uuid4()),
        vector=vec,
        original_question=question,
        normalized_question=question.lower(),
        generated_query=query,
        query_hash=hashlib.md5(query.encode()).hexdigest(),
    )


@pytest.fixture
def make_entry_fixture():
    """Fixture wrapper per make_entry."""
    return make_entry


@pytest.fixture
def sample_templates():
    """Provide sample QueryTemplate objects for testing."""
    return [
        QueryTemplate(
            intent="count_entities",
            template_text="How many {entity} are there",
            query_template="SELECT COUNT(*) FROM {entity}",
            parameters=["entity"],
            priority=1,
            parameter_patterns={"entity": r"\b(users|products|orders|employees)\b"},
        ),
        QueryTemplate(
            intent="top_n",
            template_text="Show top {count} {entity}",
            query_template="SELECT * FROM {entity} LIMIT {count}",
            parameters=["count", "entity"],
            priority=1,
            parameter_patterns={
                "count": r"\b(\d+)\b",
                "entity": r"\b(users|products|orders|employees)\b",
            },
        ),
        QueryTemplate(
            intent="filter_by_department",
            template_text="List employees in {department}",
            query_template="SELECT * FROM employees WHERE dept = '{department}'",
            parameters=["department"],
            priority=2,
        ),
    ]
