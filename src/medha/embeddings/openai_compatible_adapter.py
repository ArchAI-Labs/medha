"""OpenAI-compatible embeddings adapter (Ollama, vLLM, LocalAI, LM Studio, …)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

try:
    from openai import APIConnectionError, AsyncOpenAI, AuthenticationError, RateLimitError
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    AsyncOpenAI = None  # type: ignore[assignment,misc]
    APIConnectionError = Exception  # type: ignore[assignment,misc]
    AuthenticationError = Exception  # type: ignore[assignment,misc]
    RateLimitError = Exception  # type: ignore[assignment,misc]

from medha.config import Settings
from medha.exceptions import EmbeddingError
from medha.interfaces.embedder import BaseEmbedder

logger = logging.getLogger(__name__)

_BATCH_SIZE = 20


class OpenAICompatibleAdapter(BaseEmbedder):
    """Embedding adapter for any OpenAI-compatible endpoint (Ollama, vLLM, LocalAI, LM Studio).

    Dimension is inferred lazily from the first embedding response.
    """

    def __init__(self, settings: Settings) -> None:
        if not _OPENAI_AVAILABLE:
            raise ImportError(
                "openai is required for OpenAICompatibleAdapter. "
                "Install it with: pip install medha[openai]"
            )
        self._model: str = settings.oai_compat_model
        self._dimension: int | None = None

        raw_key = settings.oai_compat_api_key
        api_key: str = raw_key.get_secret_value() if raw_key is not None else "ollama"

        try:
            self._aclient: AsyncOpenAI = AsyncOpenAI(
                base_url=settings.oai_compat_base_url,
                api_key=api_key,
            )
            logger.info(
                "OpenAICompatibleAdapter initialised: base_url=%s model=%s",
                settings.oai_compat_base_url,
                self._model,
            )
        except Exception as e:
            raise EmbeddingError(f"Failed to initialise OpenAI-compatible client: {e}") from e

    # ------------------------------------------------------------------
    # BaseEmbedder interface
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise EmbeddingError(
                "Dimension not yet known — call aembed() or aembed_batch() first."
            )
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model

    async def aembed(self, text: str) -> list[float]:
        try:
            logger.debug("OAICompat aembed: text_len=%d model='%s'", len(text), self._model)
            response = await self._aclient.embeddings.create(
                input=[text],
                model=self._model,
            )
            embedding = list(response.data[0].embedding)
            if self._dimension is None:
                self._dimension = len(embedding)
            return embedding
        except AuthenticationError as e:
            raise EmbeddingError(f"OpenAI-compatible auth failed: {e}") from e
        except RateLimitError as e:
            raise EmbeddingError(f"OpenAI-compatible rate limit exceeded: {e}") from e
        except APIConnectionError as e:
            raise EmbeddingError(f"OpenAI-compatible connection error: {e}") from e
        except Exception as e:
            raise EmbeddingError(
                f"Failed to embed text with model '{self._model}': {e}"
            ) from e

    async def aembed_batch(self, texts: list[str], **_kwargs: Any) -> list[list[float]]:
        chunks = [texts[i : i + _BATCH_SIZE] for i in range(0, len(texts), _BATCH_SIZE)]
        logger.debug(
            "OAICompat aembed_batch: %d texts → %d chunks, model='%s'",
            len(texts),
            len(chunks),
            self._model,
        )
        try:
            responses = await asyncio.gather(
                *[
                    self._aclient.embeddings.create(input=chunk, model=self._model)
                    for chunk in chunks
                ]
            )
        except AuthenticationError as e:
            raise EmbeddingError(f"OpenAI-compatible auth failed: {e}") from e
        except RateLimitError as e:
            raise EmbeddingError(f"OpenAI-compatible rate limit exceeded: {e}") from e
        except APIConnectionError as e:
            raise EmbeddingError(f"OpenAI-compatible connection error: {e}") from e
        except Exception as e:
            raise EmbeddingError(
                f"Failed to embed batch with model '{self._model}': {e}"
            ) from e

        result: list[list[float]] = []
        for resp in responses:
            for item in sorted(resp.data, key=lambda x: x.index):
                result.append(list(item.embedding))

        if self._dimension is None and result:
            self._dimension = len(result[0])

        return result
