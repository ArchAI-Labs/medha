# Getting Started

This guide walks you from zero to a working Medha semantic cache in under five minutes.

---

## Prerequisites

- Python **3.10** or higher
- `pip` or [`uv`](https://github.com/astral-sh/uv) (recommended)

---

## Installation

Choose the extras that match your use case:

=== "Minimal"

    ```bash
    pip install medha-archai
    ```

    Installs core Medha with the in-memory backend. You must supply your own embedder (e.g. OpenAI).

=== "Local Embedder"

    ```bash
    pip install "medha-archai[fastembed]"
    ```

    Adds FastEmbed — runs ONNX models locally with no API key or internet required after first model download.

=== "Qdrant Backend"

    ```bash
    pip install "medha-archai[qdrant,fastembed]"
    ```

    Production-grade setup: Qdrant vector store + local ONNX embeddings.

=== "CLI"

    ```bash
    pip install "medha-archai[cli,fastembed]"
    ```

    Installs the `medha` command-line tool plus FastEmbed. Run `medha --help` to see all commands. See the [CLI guide](user_guide/cli.md) for details.

=== "Everything"

    ```bash
    pip install "medha-archai[all]"
    ```

    All backends and all embedders. Useful for evaluation or when you haven't settled on a stack yet.

---

## Environment Variables

Medha is configured through `Settings` (Pydantic) with the `MEDHA_` prefix. All variables have sensible defaults.

| Variable | Default | Description |
|---|---|---|
| `MEDHA_BACKEND_TYPE` | `memory` | Which vector backend to use |
| `MEDHA_SCORE_THRESHOLD_SEMANTIC` | `0.85` | Minimum cosine score for semantic tier |
| `MEDHA_SCORE_THRESHOLD_EXACT` | `0.99` | Minimum cosine score for exact tier |
| `MEDHA_DEFAULT_TTL_SECONDS` | `None` | Entry TTL in seconds (`None` = no expiry) |
| `MEDHA_COLLECT_STATS` | `true` | Enable hit/miss statistics |
| `MEDHA_QDRANT_HOST` | `localhost` | Qdrant server hostname |
| `MEDHA_PG_DSN` | — | PostgreSQL DSN for pgvector backend |

---

## Hello World

The following script stores a question-query pair and searches for it using a semantically equivalent question:

```python
import asyncio
from medha import Medha, Settings
from medha.embeddings.fastembed_adapter import FastEmbedAdapter


async def main():
    # 1. Set up an embedder (local, no API key)
    embedder = FastEmbedAdapter()

    # 2. Configure Medha to use the in-memory backend
    settings = Settings(backend_type="memory", collect_stats=True)

    # 3. Use Medha as an async context manager
    async with Medha("demo", embedder=embedder, settings=settings) as cache:

        # Store a question → SQL query pair
        await cache.store(
            question="How many active users do we have?",
            generated_query="SELECT COUNT(*) FROM users WHERE active = true",
        )

        # Search with a semantically equivalent question
        hit = await cache.search("Count of active users")

        if hit:
            print(f"Query    : {hit.generated_query}")
            print(f"Strategy : {hit.strategy}")
            print(f"Confidence: {hit.confidence:.0%}")
        else:
            print("Cache miss — send to LLM")

        # Print overall statistics
        stats = await cache.get_stats()
        print(f"\nHit rate : {stats.hit_rate:.0%}")
        print(f"Total hits: {stats.total_hits}")
        print(f"Total misses: {stats.total_misses}")


asyncio.run(main())
```

Expected output:

```
Query    : SELECT COUNT(*) FROM users WHERE active = true
Strategy : SearchStrategy.SEMANTIC_MATCH
Confidence: 94%

Hit rate : 100%
Total hits: 1
Total misses: 0
```

---

## What Just Happened?

1. **Embedding** — FastEmbedAdapter converted "How many active users do we have?" into a 384-dimensional ONNX vector and stored it in the in-memory backend.
2. **Search** — When you asked "Count of active users", Medha ran the waterfall:
    - L1 Cache: miss (first search after store)
    - Template Match: miss (no templates defined)
    - Exact Vector (≥ 0.99): miss (similar but not identical phrasing)
    - Semantic Match (≥ 0.85): **HIT** — cosine similarity was above the 0.85 threshold
3. **Result** — Medha returned the cached SQL without touching the LLM.

The `strategy` field on `CacheHit` tells you exactly which tier fired, so you can tune thresholds based on your quality requirements.

---

## Next Steps

- [Core Concepts](user_guide/concepts.md) — understand the waterfall tiers and scoring model
- [Backends](user_guide/backends.md) — choose and configure a production vector store
- [Configuration](user_guide/configuration.md) — full `Settings` reference
- [Embedders](user_guide/embedders.md) — compare embedding providers
