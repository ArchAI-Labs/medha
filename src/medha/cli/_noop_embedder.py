from typing import Any

from medha.interfaces.embedder import BaseEmbedder


class _NoOpEmbedder(BaseEmbedder):
    """Placeholder embedder for CLI commands that do not embed text."""

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return "_noop"

    async def aembed(self, text: str) -> list[float]:
        raise RuntimeError(
            "This command requires a real embedder. "
            "Set MEDHA_EMBEDDER_TYPE (e.g. fastembed) and install the "
            "corresponding extra: pip install 'medha-archai[fastembed]'."
        )

    async def aembed_batch(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        raise RuntimeError(
            "This command requires a real embedder. "
            "Set MEDHA_EMBEDDER_TYPE (e.g. fastembed) and install the "
            "corresponding extra: pip install 'medha-archai[fastembed]'."
        )
