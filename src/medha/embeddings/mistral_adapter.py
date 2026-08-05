"""Mistral embeddings adapter implementing the BaseEmbedder interface."""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Any

from medha.config import Settings
from medha.exceptions import EmbeddingError
from medha.interfaces.embedder import BaseEmbedder

logger = logging.getLogger(__name__)


def _chunks(lst: list[Any], n: int) -> Generator[list[Any], None, None]:
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class MistralAdapter(BaseEmbedder):
    """Embedding adapter using the Mistral Embeddings API."""

    def __init__(self, settings: Settings) -> None:
        try:
            from mistralai import Mistral
        except ImportError as exc:
            raise ImportError(
                "MistralAdapter requires the [mistral] extra, which is not installed.\n"
                'Install it with:  pip install "medha-archai[mistral]"'
            ) from exc

        self._model: str = settings.mistral_model
        self._batch_size: int = settings.mistral_batch_size
        self._dimension: int | None = None
        self._client = Mistral(
            api_key=settings.mistral_api_key.get_secret_value()
            if settings.mistral_api_key
            else None
        )
        logger.info("MistralAdapter initialised with model '%s'", self._model)

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise RuntimeError("not embedded yet")
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model

    async def aembed(self, text: str) -> list[float]:
        try:
            logger.debug("MistralAdapter aembed: text_len=%d model='%s'", len(text), self._model)
            result = await self._client.embeddings.create_async(
                model=self._model, inputs=[text]
            )
            vec = result.data[0].embedding
            if vec is None:
                raise EmbeddingError("Mistral returned an empty embedding")
            self._dimension = len(vec)
            return list(vec)
        except (RuntimeError, EmbeddingError):
            raise
        except Exception as e:
            logger.error("MistralAdapter aembed failed: %s", e)
            raise EmbeddingError(f"Mistral aembed failed: {e}") from e

    async def aembed_batch(self, texts: list[str], **_kwargs: Any) -> list[list[float]]:
        try:
            logger.debug(
                "MistralAdapter aembed_batch: %d texts model='%s'", len(texts), self._model
            )
            results: list[list[float]] = []
            for chunk in _chunks(texts, self._batch_size):
                resp = await self._client.embeddings.create_async(
                    model=self._model, inputs=chunk
                )
                for item in resp.data:
                    if item.embedding is None:
                        raise EmbeddingError("Mistral returned an empty embedding")
                    results.append(list(item.embedding))
            if results and self._dimension is None:
                self._dimension = len(results[0])
            return results
        except (RuntimeError, EmbeddingError):
            raise
        except Exception as e:
            logger.error("MistralAdapter aembed_batch failed: %s", e)
            raise EmbeddingError(f"Mistral aembed_batch failed: {e}") from e
