"""Embedder adapter implementations (FastEmbed, OpenAI, Cohere, Gemini).

Lazy imports to avoid pulling in optional dependencies at package level.
"""

from typing import Any

from medha.interfaces.embedder import BaseEmbedder


def get_fastembed_adapter() -> type[BaseEmbedder]:
    """Import and return the FastEmbedAdapter class."""
    from medha.embeddings.fastembed_adapter import FastEmbedAdapter

    return FastEmbedAdapter


def get_openai_adapter() -> type[BaseEmbedder]:
    """Import and return the OpenAIAdapter class."""
    from medha.embeddings.openai_adapter import OpenAIAdapter

    return OpenAIAdapter


def get_cohere_adapter() -> type[BaseEmbedder]:
    """Import and return the CohereAdapter class."""
    from medha.embeddings.cohere_adapter import CohereAdapter

    return CohereAdapter


def get_gemini_adapter() -> type[BaseEmbedder]:
    """Import and return the GeminiAdapter class."""
    from medha.embeddings.gemini_adapter import GeminiAdapter

    return GeminiAdapter


__all__ = [
    "get_fastembed_adapter",
    "get_openai_adapter",
    "get_cohere_adapter",
    "get_gemini_adapter",
    "FastEmbedAdapter",
    "OpenAIAdapter",
    "OpenAICompatibleAdapter",
    "CohereAdapter",
    "GeminiAdapter",
    "MistralAdapter",
]

# Adapter name -> (module, extra that provides its SDK).
_ADAPTERS = {
    "FastEmbedAdapter": ("fastembed_adapter", "fastembed"),
    "OpenAIAdapter": ("openai_adapter", "openai"),
    "OpenAICompatibleAdapter": ("openai_compatible_adapter", "openai"),
    "CohereAdapter": ("cohere_adapter", "cohere"),
    "GeminiAdapter": ("gemini_adapter", "gemini"),
    "MistralAdapter": ("mistral_adapter", "mistral"),
}


def __getattr__(name: str) -> Any:
    """Expose the adapters lazily, naming the extra when the SDK is missing."""
    entry = _ADAPTERS.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, extra = entry
    import importlib

    try:
        module = importlib.import_module(f"medha.embeddings.{module_name}")
    except ImportError as exc:
        raise ImportError(
            f"{name} requires the [{extra}] extra, which is not installed.\n"
            f'Install it with:  pip install "medha-archai[{extra}]"\n\n'
            f"Original error: {exc}"
        ) from exc
    return getattr(module, name)
