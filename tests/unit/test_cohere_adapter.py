"""Unit tests for CohereAdapter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import medha.embeddings.cohere_adapter as cohere_mod
from medha.embeddings.cohere_adapter import CohereAdapter
from medha.exceptions import ConfigurationError


def _make_embed_response(vectors: list[list[float]]) -> MagicMock:
    response = MagicMock()
    response.embeddings.float_ = vectors
    return response


def _make_mock_cohere_client(vectors: list[list[float]]) -> MagicMock:
    mock_client = MagicMock()
    mock_client.embed = AsyncMock(return_value=_make_embed_response(vectors))
    return mock_client


class TestCohereAdapterImportGuard:
    def test_raises_configuration_error_when_cohere_missing(self):
        original = cohere_mod.HAS_COHERE
        try:
            cohere_mod.HAS_COHERE = False
            with pytest.raises(ConfigurationError, match="pip install medha"):
                CohereAdapter(api_key="test")
        finally:
            cohere_mod.HAS_COHERE = original


class TestCohereAdapterDimension:
    def test_dimension_raises_before_any_embed_call(self):
        mock_cohere = MagicMock()
        mock_cohere.AsyncClientV2.return_value = MagicMock()
        with patch.object(cohere_mod, "HAS_COHERE", True), patch.object(cohere_mod, "cohere", mock_cohere):
            adapter = CohereAdapter(api_key="test")
            with pytest.raises(RuntimeError, match="Dimension not available"):
                _ = adapter.dimension

    async def test_dimension_returns_value_after_aembed(self):
        mock_client = _make_mock_cohere_client([[0.1, 0.2, 0.3]])
        mock_cohere = MagicMock()
        mock_cohere.AsyncClientV2.return_value = mock_client
        with patch.object(cohere_mod, "HAS_COHERE", True), patch.object(cohere_mod, "cohere", mock_cohere):
            adapter = CohereAdapter(api_key="test")
            await adapter.aembed("hello")
            assert adapter.dimension == 3


class TestCohereAdapterAembed:
    async def test_aembed_uses_query_input_type(self):
        mock_client = _make_mock_cohere_client([[0.1, 0.2]])
        mock_cohere = MagicMock()
        mock_cohere.AsyncClientV2.return_value = mock_client
        with patch.object(cohere_mod, "HAS_COHERE", True), patch.object(cohere_mod, "cohere", mock_cohere):
            adapter = CohereAdapter(api_key="test", input_type_query="search_query")
            result = await adapter.aembed("hello world")

        mock_client.embed.assert_called_once()
        call_kwargs = mock_client.embed.call_args.kwargs
        assert call_kwargs["input_type"] == "search_query"
        assert result == [0.1, 0.2]

    async def test_aembed_updates_dimension(self):
        mock_client = _make_mock_cohere_client([[0.1, 0.2, 0.3, 0.4]])
        mock_cohere = MagicMock()
        mock_cohere.AsyncClientV2.return_value = mock_client
        with patch.object(cohere_mod, "HAS_COHERE", True), patch.object(cohere_mod, "cohere", mock_cohere):
            adapter = CohereAdapter(api_key="test")
            await adapter.aembed("test")
            assert adapter.dimension == 4


class TestCohereAdapterAembedBatch:
    async def test_aembed_batch_uses_query_input_type_by_default(self):
        mock_client = _make_mock_cohere_client([[0.1, 0.2], [0.3, 0.4]])
        mock_cohere = MagicMock()
        mock_cohere.AsyncClientV2.return_value = mock_client
        with patch.object(cohere_mod, "HAS_COHERE", True), patch.object(cohere_mod, "cohere", mock_cohere):
            adapter = CohereAdapter(api_key="test", input_type_query="search_query")
            await adapter.aembed_batch(["a", "b"])

        call_kwargs = mock_client.embed.call_args.kwargs
        assert call_kwargs["input_type"] == "search_query"

    async def test_aembed_batch_uses_document_input_type_when_is_document_true(self):
        mock_client = _make_mock_cohere_client([[0.1, 0.2], [0.3, 0.4]])
        mock_cohere = MagicMock()
        mock_cohere.AsyncClientV2.return_value = mock_client
        with patch.object(cohere_mod, "HAS_COHERE", True), patch.object(cohere_mod, "cohere", mock_cohere):
            adapter = CohereAdapter(
                api_key="test",
                input_type_query="search_query",
                input_type_document="search_document",
            )
            await adapter.aembed_batch(["a", "b"], is_document=True)

        call_kwargs = mock_client.embed.call_args.kwargs
        assert call_kwargs["input_type"] == "search_document"

    async def test_aembed_batch_returns_correct_vectors(self):
        expected = [[0.1, 0.2], [0.3, 0.4]]
        mock_client = _make_mock_cohere_client(expected)
        mock_cohere = MagicMock()
        mock_cohere.AsyncClientV2.return_value = mock_client
        with patch.object(cohere_mod, "HAS_COHERE", True), patch.object(cohere_mod, "cohere", mock_cohere):
            adapter = CohereAdapter(api_key="test")
            result = await adapter.aembed_batch(["a", "b"])

        assert result == expected
