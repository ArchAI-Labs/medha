"""Unit tests for OpenAICompatibleAdapter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("openai")

import medha.embeddings.openai_compatible_adapter as oai_compat_mod  # noqa: E402
from medha.config import Settings  # noqa: E402
from medha.embeddings.openai_compatible_adapter import OpenAICompatibleAdapter  # noqa: E402
from medha.exceptions import EmbeddingError  # noqa: E402


def _make_embed_response(vectors: list[list[float]]) -> MagicMock:
    response = MagicMock()
    items = []
    for i, vec in enumerate(vectors):
        item = MagicMock()
        item.embedding = vec
        item.index = i
        items.append(item)
    response.data = items
    return response


@pytest.fixture
def settings():
    return Settings(
        oai_compat_base_url="http://localhost:11434/v1",
        oai_compat_model="nomic-embed-text",
    )


@pytest.fixture
def mock_client(settings):
    with patch.object(oai_compat_mod, "AsyncOpenAI") as mock_cls:
        instance = MagicMock()
        instance.embeddings.create = AsyncMock(
            return_value=_make_embed_response([[0.1, 0.2, 0.3]])
        )
        mock_cls.return_value = instance
        adapter = OpenAICompatibleAdapter(settings)
        yield adapter, instance


class TestOpenAICompatibleAdapterModelName:
    def test_model_name(self, settings):
        with patch.object(oai_compat_mod, "AsyncOpenAI"):
            adapter = OpenAICompatibleAdapter(settings)
        assert adapter.model_name == "nomic-embed-text"


class TestOpenAICompatibleAdapterUsesBaseUrl:
    def test_uses_base_url(self, settings):
        with patch.object(oai_compat_mod, "AsyncOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            OpenAICompatibleAdapter(settings)
        mock_cls.assert_called_once_with(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )


class TestOpenAICompatibleAdapterAembed:
    async def test_aembed_returns_vector(self, mock_client):
        adapter, _ = mock_client
        result = await adapter.aembed("test question")
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)

    async def test_aembed_sets_dimension(self, mock_client):
        adapter, _ = mock_client
        await adapter.aembed("hello")
        assert adapter.dimension == 3

    async def test_embedding_error_wrapped(self, settings):
        with patch.object(oai_compat_mod, "AsyncOpenAI") as mock_cls:
            instance = MagicMock()
            instance.embeddings.create = AsyncMock(side_effect=RuntimeError("network down"))
            mock_cls.return_value = instance
            adapter = OpenAICompatibleAdapter(settings)
        with pytest.raises(EmbeddingError):
            await adapter.aembed("test")


class TestOpenAICompatibleAdapterAembedBatch:
    async def test_aembed_batch_chunked(self, settings):
        with patch.object(oai_compat_mod, "AsyncOpenAI") as mock_cls:
            instance = MagicMock()
            instance.embeddings.create = AsyncMock(
                side_effect=[
                    _make_embed_response([[float(i)] * 3 for i in range(20)]),
                    _make_embed_response([[float(i)] * 3 for i in range(5)]),
                ]
            )
            mock_cls.return_value = instance
            adapter = OpenAICompatibleAdapter(settings)
            result = await adapter.aembed_batch([f"text {i}" for i in range(25)])

        assert instance.embeddings.create.call_count == 2
        assert len(result) == 25

    async def test_aembed_batch_error_wrapped(self, settings):
        with patch.object(oai_compat_mod, "AsyncOpenAI") as mock_cls:
            instance = MagicMock()
            instance.embeddings.create = AsyncMock(side_effect=RuntimeError("boom"))
            mock_cls.return_value = instance
            adapter = OpenAICompatibleAdapter(settings)
        with pytest.raises(EmbeddingError):
            await adapter.aembed_batch(["a", "b"])
