"""Unit tests for MistralAdapter."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("mistralai")

from medha.config import Settings  # noqa: E402
from medha.embeddings.mistral_adapter import MistralAdapter  # noqa: E402
from medha.exceptions import EmbeddingError  # noqa: E402


def _make_mistral_response(vectors: list[list[float]]) -> MagicMock:
    response = MagicMock()
    items = []
    for vec in vectors:
        item = MagicMock()
        item.embedding = vec
        items.append(item)
    response.data = items
    return response


@pytest.fixture
def settings():
    return Settings(mistral_model="mistral-embed", mistral_batch_size=5)


@pytest.fixture
def mock_client(settings):
    with patch("mistralai.Mistral") as mock_cls:
        instance = MagicMock()
        instance.embeddings.create_async = AsyncMock(
            return_value=_make_mistral_response([[0.1, 0.2, 0.3]])
        )
        mock_cls.return_value = instance
        adapter = MistralAdapter(settings)
        yield adapter, instance


class TestMistralAdapterModelName:
    def test_model_name(self, mock_client):
        adapter, _ = mock_client
        assert adapter.model_name == "mistral-embed"


class TestMistralAdapterDimension:
    def test_dimension_raises_before_embed(self, mock_client):
        adapter, _ = mock_client
        with pytest.raises(RuntimeError, match="not embedded yet"):
            _ = adapter.dimension

    async def test_dimension_set_after_aembed(self, mock_client):
        adapter, _ = mock_client
        await adapter.aembed("hello")
        assert adapter.dimension == 3


class TestMistralAdapterAembed:
    async def test_aembed_returns_vector(self, mock_client):
        adapter, _ = mock_client
        result = await adapter.aembed("test")
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)

    async def test_embedding_error_wrapped(self, settings):
        with patch("mistralai.Mistral") as mock_cls:
            instance = MagicMock()
            instance.embeddings.create_async = AsyncMock(side_effect=ValueError("api error"))
            mock_cls.return_value = instance
            adapter = MistralAdapter(settings)
        with pytest.raises(EmbeddingError):
            await adapter.aembed("test")


class TestMistralAdapterAembedBatch:
    async def test_aembed_batch_chunked(self, settings):
        with patch("mistralai.Mistral") as mock_cls:
            instance = MagicMock()
            instance.embeddings.create_async = AsyncMock(
                side_effect=[
                    _make_mistral_response([[float(i)] * 3 for i in range(5)]),
                    _make_mistral_response([[float(i)] * 3 for i in range(5)]),
                    _make_mistral_response([[float(i)] * 3 for i in range(2)]),
                ]
            )
            mock_cls.return_value = instance
            adapter = MistralAdapter(settings)
            result = await adapter.aembed_batch([f"text {i}" for i in range(12)])

        assert instance.embeddings.create_async.call_count == 3
        assert len(result) == 12

    async def test_aembed_batch_error_wrapped(self, settings):
        with patch("mistralai.Mistral") as mock_cls:
            instance = MagicMock()
            instance.embeddings.create_async = AsyncMock(side_effect=ValueError("boom"))
            mock_cls.return_value = instance
            adapter = MistralAdapter(settings)
        with pytest.raises(EmbeddingError):
            await adapter.aembed_batch(["a", "b"])


class TestMistralAdapterImportGuard:
    def test_import_error_when_mistralai_missing(self, settings, monkeypatch):
        monkeypatch.setitem(sys.modules, "mistralai", None)
        with pytest.raises(ImportError, match="pip install"):
            MistralAdapter(settings)
