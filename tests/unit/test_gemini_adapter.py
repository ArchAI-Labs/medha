"""Unit tests for GeminiAdapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import medha.embeddings.gemini_adapter as gemini_mod
from medha.embeddings.gemini_adapter import GeminiAdapter, _GEMINI_CHUNK_SIZE
from medha.exceptions import ConfigurationError


def _make_embed_result(vectors: list[list[float]]) -> MagicMock:
    """Build a mock result where result.embeddings[i].values == vectors[i]."""
    mock_result = MagicMock()
    embeddings = []
    for v in vectors:
        emb = MagicMock()
        emb.values = v
        embeddings.append(emb)
    mock_result.embeddings = embeddings
    return mock_result


def _single_embed_result(vector: list[float]) -> MagicMock:
    return _make_embed_result([vector])


def _mock_genai(embed_side_effect=None, embed_return=None) -> MagicMock:
    """Return a mock genai module with client.models.embed_content pre-configured."""
    mock = MagicMock()
    client_mock = mock.Client.return_value
    if embed_side_effect is not None:
        client_mock.models.embed_content.side_effect = embed_side_effect
    elif embed_return is not None:
        client_mock.models.embed_content.return_value = embed_return
    return mock


class TestGeminiAdapterImportGuard:
    def test_raises_configuration_error_when_gemini_missing(self):
        original = gemini_mod.HAS_GEMINI
        try:
            gemini_mod.HAS_GEMINI = False
            with pytest.raises(ConfigurationError, match="pip install medha"):
                GeminiAdapter(api_key="test")
        finally:
            gemini_mod.HAS_GEMINI = original


class TestGeminiAdapterDimension:
    def test_dimension_raises_before_any_embed_call(self):
        mock_g = _mock_genai()
        with patch.object(gemini_mod, "HAS_GEMINI", True):
            with patch.object(gemini_mod, "genai", mock_g):
                adapter = GeminiAdapter(api_key="test")
                with pytest.raises(RuntimeError, match="Dimension not available"):
                    _ = adapter.dimension

    async def test_dimension_set_after_aembed(self):
        mock_g = _mock_genai(embed_return=_single_embed_result([0.1, 0.2, 0.3]))
        mock_types = MagicMock()
        with patch.object(gemini_mod, "HAS_GEMINI", True):
            with patch.object(gemini_mod, "genai", mock_g):
                with patch.object(gemini_mod, "genai_types", mock_types):
                    adapter = GeminiAdapter(api_key="test")
                    await adapter.aembed("hello")
                    assert adapter.dimension == 3


class TestGeminiAdapterAembed:
    async def test_aembed_uses_query_task_type(self):
        mock_g = _mock_genai(embed_return=_single_embed_result([0.1, 0.2]))
        mock_types = MagicMock()
        with patch.object(gemini_mod, "HAS_GEMINI", True):
            with patch.object(gemini_mod, "genai", mock_g):
                with patch.object(gemini_mod, "genai_types", mock_types):
                    adapter = GeminiAdapter(api_key="test", task_type_query="RETRIEVAL_QUERY")
                    result = await adapter.aembed("test query")

        mock_types.EmbedContentConfig.assert_called_once()
        call_kwargs = mock_types.EmbedContentConfig.call_args.kwargs
        assert call_kwargs["task_type"] == "RETRIEVAL_QUERY"
        assert result == [0.1, 0.2]


class TestGeminiAdapterAembedBatch:
    async def test_aembed_batch_uses_query_task_type_by_default(self):
        mock_g = _mock_genai(embed_return=_make_embed_result([[0.1], [0.2]]))
        mock_types = MagicMock()
        with patch.object(gemini_mod, "HAS_GEMINI", True):
            with patch.object(gemini_mod, "genai", mock_g):
                with patch.object(gemini_mod, "genai_types", mock_types):
                    adapter = GeminiAdapter(api_key="test", task_type_query="RETRIEVAL_QUERY")
                    await adapter.aembed_batch(["a", "b"])

        call_kwargs = mock_types.EmbedContentConfig.call_args.kwargs
        assert call_kwargs["task_type"] == "RETRIEVAL_QUERY"

    async def test_aembed_batch_uses_document_task_type_when_is_document_true(self):
        mock_g = _mock_genai(embed_return=_make_embed_result([[0.1], [0.2]]))
        mock_types = MagicMock()
        with patch.object(gemini_mod, "HAS_GEMINI", True):
            with patch.object(gemini_mod, "genai", mock_g):
                with patch.object(gemini_mod, "genai_types", mock_types):
                    adapter = GeminiAdapter(
                        api_key="test",
                        task_type_query="RETRIEVAL_QUERY",
                        task_type_document="RETRIEVAL_DOCUMENT",
                    )
                    await adapter.aembed_batch(["a", "b"], is_document=True)

        call_kwargs = mock_types.EmbedContentConfig.call_args.kwargs
        assert call_kwargs["task_type"] == "RETRIEVAL_DOCUMENT"

    async def test_aembed_batch_chunks_large_input(self):
        """Verify input >100 texts is split into multiple API calls."""
        total = _GEMINI_CHUNK_SIZE + 50  # 150 texts
        texts = [f"text {i}" for i in range(total)]
        call_count = 0
        expected_chunks = [_GEMINI_CHUNK_SIZE, 50]

        def fake_embed(**kwargs):
            nonlocal call_count
            n = len(kwargs["contents"])
            assert n == expected_chunks[call_count], (
                f"chunk {call_count}: expected {expected_chunks[call_count]} texts, got {n}"
            )
            call_count += 1
            return _make_embed_result([[0.1] * 4 for _ in range(n)])

        mock_g = _mock_genai(embed_side_effect=fake_embed)
        mock_types = MagicMock()
        with patch.object(gemini_mod, "HAS_GEMINI", True):
            with patch.object(gemini_mod, "genai", mock_g):
                with patch.object(gemini_mod, "genai_types", mock_types):
                    adapter = GeminiAdapter(api_key="test")
                    result = await adapter.aembed_batch(texts)

        assert call_count == 2
        assert len(result) == total

    async def test_aembed_batch_returns_flattened_results(self):
        mock_g = _mock_genai(embed_return=_make_embed_result([[0.1, 0.2], [0.3, 0.4]]))
        mock_types = MagicMock()
        with patch.object(gemini_mod, "HAS_GEMINI", True):
            with patch.object(gemini_mod, "genai", mock_g):
                with patch.object(gemini_mod, "genai_types", mock_types):
                    adapter = GeminiAdapter(api_key="test")
                    result = await adapter.aembed_batch(["a", "b"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]
