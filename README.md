# Medha

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/medha-archai?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BLUE&left_text=downloads)](https://pepy.tech/projects/medha-archai)
[![PyPI](https://img.shields.io/pypi/v/medha-archai)](https://pypi.org/project/medha-archai/)
[![Python](https://img.shields.io/pypi/pyversions/medha-archai)](https://pypi.org/project/medha-archai/)
[![License](https://img.shields.io/github/license/ArchAI-Labs/medha)](https://github.com/ArchAI-Labs/medha/blob/main/LICENSE)
[![Stars](https://img.shields.io/github/stars/ArchAI-Labs/medha)](https://github.com/ArchAI-Labs/medha/stargazers)
[![Documentation](https://img.shields.io/badge/docs-archai--labs.github.io-blue)](https://archai-labs.github.io/medha/)

<br>

![medha_logo](https://raw.githubusercontent.com/ArchAI-Labs/medha/refs/heads/main/img/medha_hex_600.png)

<br>

## Semantic Memory for AI Data Agents

> ***Reduce LLM latency and costs by caching Text-to-Query generations (SQL, Cypher, GraphQL) with semantic understanding.***

---

## What is Medha?

**Medha** is an asynchronous, high-performance semantic cache library designed specifically for **Text-to-Query** systems.

Unlike traditional key-value caches that require exact string matches, Medha understands that *"Show me the top 5 users"* and *"List the first five users"* are the same question. It intercepts these queries and returns pre-calculated database queries (SQL, Cypher, etc.), bypassing the expensive and slow LLM generation step.

### Why Medha?
* **100x Faster:** Return cached queries in milliseconds vs. seconds for LLM generation.
* **Cost Efficient:** Reduce API calls to OpenAI/Anthropic by 40-60%.
* **Agnostic:** Works with **SQL**, **Cypher** (Neo4j), **GraphQL**, or any text-based query language.
* **Async Native:** Built on `asyncio` for high-concurrency API backends.
* **Pluggable:** Swap embedders (FastEmbed, OpenAI) and vector backends independently.

---

## The "Waterfall" Architecture

Medha uses a sophisticated multi-tier search strategy to maximize cache hits. If a tier fails, it cascades to the next:

1.  **Tier 0: L1 Memory (LRU)**
    * *Speed:* < 1ms
    * Exact hash match for identical, repeated questions.
2.  **Tier 1: Template Matching (Intent)**
    * *Speed:* ~10ms
    * Recognizes patterns like *"Show employees in {department}"*. Extracts parameters and injects them into a cached query template.
3.  **Tier 2 + 3: Exact Vector Match & Semantic Similarity** *(run in parallel)*
    * *Speed:* ~25ms (concurrent, not sequential)
    * Exact match uses a high threshold (≥ 0.99); Semantic uses a lower one (≥ 0.90). Both vector queries are fired simultaneously via `asyncio.gather` and the best result is chosen.
5.  **Tier 4: Fuzzy Fallback**
    * *Speed:* Variable
    * Handles typos and minor string variations using Levenshtein distance.

---

## Installation

### Core (minimal)

```bash
pip install medha-archai
```

Core dependencies: `pydantic`, `pydantic-settings`.  
> **Breaking change in 0.3.1:** `qdrant-client` is no longer a core dependency. Install it explicitly with `pip install "medha-archai[qdrant]"`. The default `backend_type` is now `"memory"`.

### With an embedding provider

```bash
# Local embeddings with FastEmbed (recommended for getting started)
pip install "medha-archai[fastembed]"

# OpenAI embeddings
pip install "medha-archai[openai]"

# Cohere Embed v3
pip install "medha-archai[cohere]"

# Google Gemini embeddings
pip install "medha-archai[gemini]"
```

### With a vector backend

```bash
# Qdrant (Docker / Cloud)
pip install "medha-archai[qdrant]"

# PostgreSQL + pgvector
pip install "medha-archai[pgvector]"

# Elasticsearch 8.x
pip install "medha-archai[elasticsearch]"

# PostgreSQL + VectorChord
pip install "medha-archai[vectorchord]"

# ChromaDB
pip install "medha-archai[chroma]"

# Weaviate
pip install "medha-archai[weaviate]"

# Redis Stack (vector backend + L1 cache)
pip install "medha-archai[redis]"

# Azure AI Search
pip install "medha-archai[azure-search]"

# LanceDB (embedded / S3 / GCS / Azure Blob)
pip install "medha-archai[lancedb]"

# CLI management tool
pip install "medha-archai[cli]"
```

### With optional extras

```bash
# Fuzzy matching (Tier 4 - Levenshtein distance)
pip install "medha-archai[fuzzy]"

# spaCy NLP for parameter extraction (pre-trained, fixed entity types, ~15 MB model)
pip install "medha-archai[nlp]"
python -m spacy download en_core_web_sm

# GLiNER NLP for zero-shot parameter extraction (uses param names as labels, ~500 MB model)
pip install "medha-archai[gliner]"

# All optional dependencies (excluding ChromaDB for env compatibility)
pip install "medha-archai[all-no-chroma]"

# Everything
pip install "medha-archai[all]"
```

### Install from source

```bash
# From GitHub
!pip install "medha-archai[all] @ git+https://github.com/ArchAI-Labs/medha.git"

# Development install
git clone https://github.com/ArchAI-Labs/medha.git
cd medha
pip install -e ".[dev,all]"
```

---

## Quick Start

```python
import asyncio
from medha import Medha
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

async def main():
    embedder = FastEmbedAdapter()
    cache = Medha(collection_name="text2sql_cache", embedder=embedder)

    async with cache:
        question = "How many users are active?"

        # 1. Search the cache
        hit = await cache.search(question)

        if hit.strategy.value != "no_match":
            print(f"Cache Hit! Strategy: {hit.strategy.value}")
            print(f"Query: {hit.generated_query}")
            print(f"Confidence: {hit.confidence:.2f}")
        else:
            print("Cache Miss. Calling LLM...")
            generated_sql = "SELECT count(*) FROM users WHERE status = 'active';"

            # 2. Store the result for next time
            await cache.store(
                question=question,
                generated_query=generated_sql,
            )
            print("Stored in cache.")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Demo Notebooks

Hands-on Jupyter notebooks covering every major feature. Run them locally or open them directly on GitHub.

### Query Languages

| Notebook | Description |
|---|---|
| [01 — Text-to-SQL (SQLite)](demo/01_text2sql_sqlite.ipynb) | End-to-end Text-to-SQL with a local SQLite database |
| [02 — Text-to-Cypher (Neo4j)](demo/02_text2cypher_neo4j.ipynb) | Graph query caching for Neo4j Cypher queries |
| [03 — Text-to-Query (MongoDB MQL)](demo/03_text2query_mongodb.ipynb) | MongoDB query caching with MQL |
| [10 — Text-to-SQL (DuckDB)](demo/10_text2sql_duckdb.ipynb) | Analytical SQL caching with DuckDB |

### Embedders

| Notebook | Description |
|---|---|
| [04 — Custom Embedder](demo/04_custom_embedder.ipynb) | Implement `BaseEmbedder` for any embedding provider |
| [07 — Cloud Embedders](demo/07_openai_embedder.ipynb) | OpenAI, Cohere, and Gemini embedding adapters |
| [05 — NER: spaCy vs GLiNER](demo/05_ner_spacy_vs_gliner.ipynb) | Parameter extraction for template matching |

### Vector Backends

| Notebook | Description |
|---|---|
| [11 — InMemory Backend](demo/11_inmemory_backend.ipynb) | Zero-dependency in-process caching |
| [24 — Qdrant Backend](demo/24_qdrant_backend.ipynb) | Production-grade HNSW with Qdrant (memory / Docker / Cloud) |
| [12 — pgvector Backend](demo/12_pgvector_backend.ipynb) | PostgreSQL-native vector search |
| [15 — VectorChord Backend](demo/15_vectorchord_backend.ipynb) | High-throughput PostgreSQL + VectorChord |
| [14 — Elasticsearch Backend](demo/14_elasticsearch_backend.ipynb) | Full-text + vector search with Elasticsearch 8.x |
| [16 — Chroma Backend](demo/16_chroma_backend.ipynb) | Lightweight local vector store with ChromaDB |
| [17 — Weaviate Backend](demo/17_weaviate_backend.ipynb) | Knowledge graph + vector search with Weaviate |
| [18 — Redis Vector Backend](demo/18_redis_vector_backend.ipynb) | Sub-millisecond semantic caching with Redis Stack |
| [19 — Azure AI Search Backend](demo/19_azure_search_backend.ipynb) | Managed cloud vector search on Azure |
| [20 — LanceDB Backend](demo/20_lancedb_backend.ipynb) | Embedded zero-infrastructure vector search |

### Features & Patterns

| Notebook | Description |
|---|---|
| [06 — Production Patterns](demo/06_production_patterns.ipynb) | Logging, retries, health checks, and deployment tips |
| [08 — Fuzzy Matching](demo/08_fuzzy_matching.ipynb) | Tier 4 Levenshtein fallback for typos and variants |
| [09 — Multi-Tenant Caching](demo/09_multi_tenant.ipynb) | Separate collections per tenant |
| [13 — Framework Integrations](demo/13_framework_integrations.ipynb) | LangChain, LlamaIndex, and Haystack integration |
| [21 — Cache Lifecycle](demo/21_cache_lifecycle.ipynb) | TTL, expiry, and invalidation strategies |
| [22 — Observability](demo/22_observability.ipynb) | Stats, latency percentiles, and logging |
| [23 — Batch Operations](demo/23_batch_ops.ipynb) | Bulk ingestion, export to DataFrame, and dedup |

---

## Choosing a Backend

| Backend | Extra | Persistence | Best For |
|---------|-------|-------------|----------|
| `memory` **(default)** | _(none)_ | No | Testing, development, CI |
| `qdrant` | `[qdrant]` | Yes (Docker/Cloud) | Production, large datasets |
| `pgvector` | `[pgvector]` | Yes (PostgreSQL) | Teams already using PostgreSQL |
| `vectorchord` | `[vectorchord]` | Yes (PostgreSQL + VectorChord) | High-performance approximate search |
| `elasticsearch` | `[elasticsearch]` | Yes (Elasticsearch 8.x) | Teams running the Elastic stack |
| `chroma` | `[chroma]` | Optional (ephemeral / disk / HTTP) | Quick experiments, local dev |
| `weaviate` | `[weaviate]` | Yes (local / Weaviate Cloud) | Weaviate-native deployments |
| `redis` | `[redis]` | Yes (Redis Stack / Sentinel) | Low-latency, Redis-native stacks |
| `azure-search` | `[azure-search]` | Yes (Azure AI Search) | Azure-hosted deployments |
| `lancedb` | `[lancedb]` | Yes (embedded / S3 / GCS / az) | Serverless, edge, embedded apps |

### InMemory Backend (zero dependencies)

```python
from medha import Medha, Settings
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

embedder = FastEmbedAdapter()
settings = Settings(backend_type="memory")

async with Medha(collection_name="my_cache", embedder=embedder, settings=settings) as m:
    await m.store("How many users?", "SELECT COUNT(*) FROM users")
    hit = await m.search("Count of users")
    print(hit.generated_query)
```

### PostgreSQL + pgvector Backend

```python
from medha import Medha, Settings
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

settings = Settings(
    backend_type="pgvector",
    pg_dsn="postgresql://user:password@localhost:5432/mydb",
)

async with Medha(collection_name="my_cache", embedder=FastEmbedAdapter(), settings=settings) as m:
    await m.store("How many users?", "SELECT COUNT(*) FROM users")
    hit = await m.search("Count of users")
    print(hit.generated_query)
```

### LanceDB Backend (embedded / cloud)

No external server needed for local mode. Supports S3, GCS, and Azure Blob Storage URIs for cloud storage.

```python
from medha import Medha, Settings
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

settings = Settings(
    backend_type="lancedb",
    lancedb_uri="/tmp/my_lancedb",   # local path; use s3://... for cloud
    lancedb_metric="cosine",         # cosine | l2 | dot
)

async with Medha(collection_name="my_cache", embedder=FastEmbedAdapter(), settings=settings) as m:
    await m.store("How many users?", "SELECT COUNT(*) FROM users")
    hit = await m.search("Count of users")
    print(hit.generated_query)
```

### Elasticsearch Backend

```python
from medha import Medha, Settings
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

settings = Settings(
    backend_type="elasticsearch",
    es_hosts=["http://localhost:9200"],
    es_api_key="your-api-key",   # or es_username / es_password
)

async with Medha(collection_name="my_cache", embedder=FastEmbedAdapter(), settings=settings) as m:
    await m.store("How many users?", "SELECT COUNT(*) FROM users")
    hit = await m.search("Count of users")
    print(hit.generated_query)
```

### Redis Stack Backend

```python
from medha import Medha, Settings
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

settings = Settings(
    backend_type="redis",
    redis_url="redis://localhost:6379/0",
    redis_index_algorithm="HNSW",  # HNSW | FLAT
)

async with Medha(collection_name="my_cache", embedder=FastEmbedAdapter(), settings=settings) as m:
    await m.store("How many users?", "SELECT COUNT(*) FROM users")
    hit = await m.search("Count of users")
    print(hit.generated_query)
```

Or via environment variables:

```bash
export MEDHA_BACKEND_TYPE=pgvector
export MEDHA_PG_DSN=postgresql://user:password@localhost:5432/mydb
```

---

## Configuration Examples

Medha is highly configurable. Below are examples covering every major use case.

### Basic: Zero-Dependency In-Memory Setup

The simplest setup, perfect for development, testing, and CI. No external services needed.

```python
import asyncio
from medha import Medha, Settings
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

async def main():
    # backend_type="memory" — pure-Python backend, zero external dependencies
    settings = Settings(backend_type="memory")
    embedder = FastEmbedAdapter()

    async with Medha(
        collection_name="dev_cache",
        embedder=embedder,
        settings=settings,
    ) as cache:
        await cache.store("List all users", "SELECT * FROM users;")
        hit = await cache.search("Show me all the users")
        print(hit.generated_query)  # SELECT * FROM users;

asyncio.run(main())
```

### Qdrant Docker (Local Persistence)

For persistent caching across restarts using a local Qdrant instance.

```bash
# Start Qdrant first
docker run -p 6333:6333 qdrant/qdrant
```

```python
import asyncio
from medha import Medha, Settings
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

async def main():
    settings = Settings(
        backend_type="qdrant",
        qdrant_mode="docker",
        qdrant_host="localhost",
        qdrant_port=6333,
    )
    embedder = FastEmbedAdapter()

    async with Medha(
        collection_name="persistent_cache",
        embedder=embedder,
        settings=settings,
    ) as cache:
        await cache.store(
            "Total revenue last quarter",
            "SELECT SUM(amount) FROM orders WHERE date >= '2024-10-01';",
        )
        hit = await cache.search("What was last quarter's revenue?")
        print(f"{hit.strategy.value}: {hit.generated_query}")

asyncio.run(main())
```

### Qdrant Cloud (Production)

For production deployments using Qdrant Cloud with API key authentication.

```python
import asyncio
from medha import Medha, Settings
from medha.embeddings.openai_adapter import OpenAIAdapter

async def main():
    settings = Settings(
        backend_type="qdrant",
        qdrant_mode="cloud",
        qdrant_url="https://your-cluster.cloud.qdrant.io",
        qdrant_api_key="your-qdrant-api-key",  # stored as SecretStr, never logged
    )
    embedder = OpenAIAdapter(
        model_name="text-embedding-3-small",
        api_key="sk-your-openai-key",
    )

    async with Medha(
        collection_name="production_cache",
        embedder=embedder,
        settings=settings,
    ) as cache:
        await cache.store(
            "Get all pending orders",
            "SELECT * FROM orders WHERE status = 'pending';",
        )
        hit = await cache.search("Show pending orders")
        print(f"Confidence: {hit.confidence:.2f}")

asyncio.run(main())
```

### Environment Variable Configuration

All settings can be configured via environment variables with the `MEDHA_` prefix. No code changes needed.

```bash
# .env or shell exports
export MEDHA_QDRANT_MODE=docker
export MEDHA_QDRANT_HOST=qdrant.internal.company.com
export MEDHA_QDRANT_PORT=6333
export MEDHA_SCORE_THRESHOLD_SEMANTIC=0.85
export MEDHA_SCORE_THRESHOLD_EXACT=0.98
export MEDHA_L1_CACHE_MAX_SIZE=5000
export MEDHA_QUERY_LANGUAGE=sql
export MEDHA_ENABLE_QUANTIZATION=true
export MEDHA_ON_DISK=false
export MEDHA_TEMPLATE_FILE=/etc/medha/templates.json
```

```python
import asyncio
from medha import Medha, Settings
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

async def main():
    # Settings automatically loads from MEDHA_* environment variables
    settings = Settings()
    embedder = FastEmbedAdapter()

    async with Medha(
        collection_name="my_cache",
        embedder=embedder,
        settings=settings,
    ) as cache:
        hit = await cache.search("Show me all employees")
        print(hit.strategy.value)

asyncio.run(main())
```

---

## Embedding Providers

### FastEmbed (Local, No API Key)

Runs entirely locally using ONNX Runtime. No API key, no network calls, no costs.

```python
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

# Default model (384 dimensions, fast and lightweight)
embedder = FastEmbedAdapter()

# Higher quality model
embedder = FastEmbedAdapter(
    model_name="BAAI/bge-base-en-v1.5",
    max_length=512,
)

# Custom cache directory for model files
embedder = FastEmbedAdapter(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_dir="/opt/models/fastembed",
)
```

### OpenAI Embeddings

Uses OpenAI's embedding API. Requires an API key (via parameter or `OPENAI_API_KEY` env var).

```python
from medha.embeddings.openai_adapter import OpenAIAdapter

# Default: text-embedding-3-small (1536 dimensions)
embedder = OpenAIAdapter(api_key="sk-your-key")

# High-quality large model (3072 dimensions)
embedder = OpenAIAdapter(
    model_name="text-embedding-3-large",
    api_key="sk-your-key",
)

# With custom dimensions (only supported by text-embedding-3-* models)
embedder = OpenAIAdapter(
    model_name="text-embedding-3-small",
    dimensions=512,
    api_key="sk-your-key",
)

# API key from environment variable (OPENAI_API_KEY)
embedder = OpenAIAdapter()
```

### Cohere Embeddings

Uses Cohere Embed v3 (`cohere.AsyncClientV2`). Requires an API key.

```python
from medha.embeddings.cohere_adapter import CohereAdapter

# Default: embed-multilingual-v3.0
embedder = CohereAdapter(api_key="your-cohere-key")

# Explicit model
embedder = CohereAdapter(
    api_key="your-cohere-key",
    model="embed-english-v3.0",
)
```

Input types `search_query` / `search_document` are selected automatically at embed time.

### Gemini Embeddings

Uses Google Gemini embeddings (`google-generativeai`). Requires an API key.

```python
from medha.embeddings.gemini_adapter import GeminiAdapter

# Default: models/text-embedding-004
embedder = GeminiAdapter(api_key="your-gemini-key")

# With reduced output dimensions (MRL — models/text-embedding-004 only)
embedder = GeminiAdapter(
    api_key="your-gemini-key",
    model="models/text-embedding-004",
    output_dimensionality=512,
)
```

Task types `RETRIEVAL_QUERY` / `RETRIEVAL_DOCUMENT` are selected automatically. Requests are batched in chunks of 100.

### Custom Embedder

Implement the `BaseEmbedder` interface to use any embedding provider.

```python
from medha.interfaces import BaseEmbedder
from typing import List

class MyCustomEmbedder(BaseEmbedder):
    @property
    def dimension(self) -> int:
        return 768

    @property
    def model_name(self) -> str:
        return "my-custom-model"

    async def aembed(self, text: str) -> List[float]:
        # Your embedding logic here
        ...

    async def aembed_batch(self, texts: List[str]) -> List[List[float]]:
        # Your batch embedding logic here
        ...

embedder = MyCustomEmbedder()
```

---

## Search Threshold Tuning

Fine-tune how aggressively Medha matches questions at each tier.

### Strict Matching (High Precision)

Only return cache hits when very confident. Minimizes false positives.

```python
from medha import Settings

settings = Settings(
    score_threshold_exact=0.995,     # Near-identical vectors only
    score_threshold_semantic=0.95,   # Very close meaning only
    score_threshold_template=0.90,   # Template must be a strong match
    score_threshold_fuzzy=95.0,      # Almost no typos allowed
)
```

### Relaxed Matching (High Recall)

Return more cache hits, accepting slightly lower confidence. Reduces LLM calls.

```python
from medha import Settings

settings = Settings(
    score_threshold_exact=0.97,
    score_threshold_semantic=0.82,
    score_threshold_template=0.75,
    score_threshold_fuzzy=75.0,
)
```

### Disable Specific Tiers

```python
from medha import Settings

# Disable L1 in-memory cache (always hit the vector store)
settings = Settings(l1_cache_max_size=0)

# Fuzzy matching is automatically disabled if rapidfuzz is not installed
# To install: pip install "medha-archai[fuzzy]"
```

---

## Cache Warming

Pre-populate the cache from a file before serving traffic. Supports both **JSON array** and **JSONL** formats.

```jsonc
// warm_queries.jsonl  — one entry per line
{"question": "How many users are active?", "generated_query": "SELECT COUNT(*) FROM users WHERE status = 'active';"}
{"question": "Total revenue this month", "generated_query": "SELECT SUM(amount) FROM orders WHERE date >= DATE_TRUNC('month', NOW());", "response_summary": "Monthly revenue total"}
```

```python
import asyncio
from medha import Medha
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

async def main():
    async with Medha(
        collection_name="my_cache",
        embedder=FastEmbedAdapter(),
    ) as cache:
        # Load from JSONL (also accepts JSON array files)
        loaded = await cache.warm_from_file("warm_queries.jsonl")
        print(f"Warmed {loaded} entries")

        # Sync variant
        # loaded = cache.warm_from_file_sync("warm_queries.json")

        print(cache.stats["warm_loaded"])  # 2

asyncio.run(main())
```

**Required keys per entry:** `question`, `generated_query`
**Optional keys:** `response_summary`, `template_id`

Internally calls `store_batch()` — a single embedding round-trip for all entries.

---

## Security Settings

Medha 0.2.0 adds three settings to defend against common attack vectors when
Medha is exposed to untrusted input.

### Input Length Guard — `max_question_length`

Prevent DoS via oversized question strings. `search()` returns
`SearchStrategy.ERROR`; `store()` raises `ValueError`.

```python
settings = Settings(max_question_length=2048)  # default: 8192
```

### File Size Limit — `max_file_size_mb`

`warm_from_file()` and `load_templates_from_file()` reject files larger than
this limit *before* reading them.

```python
settings = Settings(max_file_size_mb=50)  # default: 100 MB
```

### Path Traversal Protection — `allowed_file_dir`

When set, `warm_from_file()` and `load_templates_from_file()` reject any path
that resolves outside the specified directory.

```python
settings = Settings(allowed_file_dir="/app/data")
# warm_from_file("/app/data/../etc/passwd") → ValueError
```

---

## Distributed L1 Cache (Redis)

By default Medha's L1 cache is in-process. With multiple service instances (horizontal scaling) each process has its own isolated cache. Use `RedisL1Cache` to share the L1 cache across instances.

```bash
pip install "medha-archai[redis]"
```

```python
from medha import Medha
from medha.l1_cache.redis_adapter import RedisL1Cache
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

# Shared L1 cache — all instances see the same hits
redis_l1 = RedisL1Cache(
    url="redis://redis.internal:6379/0",
    prefix="myapp:medha:l1",   # namespace to avoid key collisions
    ttl=3600,                   # 1-hour TTL per entry (optional)
)

async with Medha(
    collection_name="prod_cache",
    embedder=FastEmbedAdapter(),
    l1_backend=redis_l1,
) as cache:
    await cache.store("How many users?", "SELECT COUNT(*) FROM users;")
    hit = await cache.search("How many users?")
    print(hit.strategy.value)  # l1_cache (served from Redis)
```

> **Redis eviction:** Configure `maxmemory-policy allkeys-lru` on the Redis server for automatic LRU eviction when memory is full.

### Custom L1 Backend

Implement `L1CacheBackend` to use any fast store (Memcached, DynamoDB DAX, etc.):

```python
from medha.interfaces.l1_cache import L1CacheBackend
from medha.types import CacheHit
from typing import Optional

class MyL1Cache(L1CacheBackend):
    async def get(self, key: str) -> Optional[CacheHit]: ...
    async def set(self, key: str, value: CacheHit) -> None: ...
    async def clear(self) -> None: ...

    @property
    def size(self) -> int: ...

cache = Medha(..., l1_backend=MyL1Cache())
```

---

## Persistent Embedding Cache

By default the embedding cache is in-memory and lost on restart. Set `embedding_cache_path` to persist it across sessions — useful when the same questions recur between deployments.

```bash
export MEDHA_EMBEDDING_CACHE_PATH=/var/cache/medha/embeddings.json
```

```python
from medha import Medha, Settings
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

settings = Settings(
    backend_type="qdrant",
    qdrant_mode="docker",
    embedding_cache_path="/var/cache/medha/embeddings.json",
)

async with Medha(
    collection_name="my_cache",
    embedder=FastEmbedAdapter(),
    settings=settings,
) as cache:
    # On start(): embeddings loaded from disk (if file exists)
    await cache.store("show active users", "SELECT * FROM users WHERE active = true;")
    # On close(): embeddings saved to disk automatically
```

No extra dependencies — uses stdlib `json`.

---

## Template Matching

Templates allow Medha to recognize parameterized patterns and generate queries dynamically without an LLM call.

### Define Templates in Code

```python
import asyncio
from medha import Medha, QueryTemplate
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

templates = [
    QueryTemplate(
        intent="top_n_entities",
        template_text="Show top {count} {entity}",
        query_template="SELECT * FROM {entity} ORDER BY id LIMIT {count}",
        parameters=["count", "entity"],
        priority=1,
        aliases=["List first {count} {entity}", "Get {count} {entity}"],
        parameter_patterns={
            "count": r"\b(\d+)\b",
            "entity": r"\b(users|orders|products|employees)\b",
        },
    ),
    QueryTemplate(
        intent="filter_by_status",
        template_text="Show {entity} with status {status}",
        query_template="SELECT * FROM {entity} WHERE status = '{status}'",
        parameters=["entity", "status"],
        priority=1,
        parameter_patterns={
            "entity": r"\b(users|orders|products)\b",
            "status": r"\b(active|inactive|pending|completed)\b",
        },
    ),
    QueryTemplate(
        intent="count_by_group",
        template_text="Count {entity} by {group}",
        query_template="SELECT {group}, COUNT(*) FROM {entity} GROUP BY {group}",
        parameters=["entity", "group"],
        priority=2,
        parameter_patterns={
            "entity": r"\b(users|orders|products|employees)\b",
            "group": r"\b(department|status|category|region)\b",
        },
    ),
]

async def main():
    embedder = FastEmbedAdapter()

    async with Medha(
        collection_name="template_demo",
        embedder=embedder,
        templates=templates,
    ) as cache:
        # Template matching with parameter extraction
        hit = await cache.search("Show top 10 users")
        print(f"Strategy: {hit.strategy.value}")
        # template_match
        print(f"Query: {hit.generated_query}")
        # SELECT * FROM users ORDER BY id LIMIT 10

        hit = await cache.search("Show orders with status pending")
        print(f"Query: {hit.generated_query}")
        # SELECT * FROM orders WHERE status = 'pending'

asyncio.run(main())
```

### Load Templates from a JSON File

```json
[
    {
        "intent": "top_n_entities",
        "template_text": "Show top {count} {entity}",
        "query_template": "SELECT * FROM {entity} ORDER BY id LIMIT {count}",
        "parameters": ["count", "entity"],
        "priority": 1,
        "aliases": ["List first {count} {entity}"],
        "parameter_patterns": {
            "count": "\\b(\\d+)\\b",
            "entity": "\\b(users|orders|products)\\b"
        }
    }
]
```

```python
from medha import Medha, Settings
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

settings = Settings(template_file="templates.json")

cache = Medha(
    collection_name="my_cache",
    embedder=FastEmbedAdapter(),
    settings=settings,
)
# Templates are loaded automatically during cache.start()
```

### Load Templates at Runtime

```python
async with Medha(
    collection_name="my_cache",
    embedder=FastEmbedAdapter(),
) as cache:
    await cache.load_templates_from_file("templates.json")
    # or
    await cache.load_templates([QueryTemplate(...), QueryTemplate(...)])
```

---

## Parameter Extraction (NER)

Template matching requires extracting parameter values (e.g. `{department}`, `{person}`) from the user's question. `ParameterExtractor` applies a cascading strategy:

1. **Regex** — patterns defined in `template.parameter_patterns` (fastest, most precise)
2. **GLiNER** — zero-shot NER, uses `template.parameters` directly as entity labels
3. **spaCy** — pre-trained NER with a fixed label set mapped to parameter names
4. **Heuristics** — numbers and capitalized words as last resort

### spaCy (pre-trained, fixed labels)

spaCy recognizes standard entity types (`PERSON`, `ORG`, `CARDINAL`) and maps them to parameter names.

```python
from medha.utils.nlp import ParameterExtractor

ext = ParameterExtractor(use_spacy=True)
print(ext.spacy_available)  # True if en_core_web_sm is installed
```

### GLiNER (zero-shot, arbitrary labels)

GLiNER receives `template.parameters` directly as entity labels — no mapping table needed.
It excels with domain-specific entities that spaCy cannot recognize without custom training.

```python
from medha.utils.nlp import ParameterExtractor

# Default model: urchade/gliner_medium-v2.1
ext = ParameterExtractor(use_gliner=True)

# Lighter variant (~250 MB)
ext = ParameterExtractor(use_gliner=True, gliner_model="urchade/gliner_small-v2.1")

print(ext.gliner_available)  # True if gliner package is installed
```

### Both enabled (recommended for mixed template sets)

```python
from medha.utils.nlp import ParameterExtractor
from medha.types import QueryTemplate

ext = ParameterExtractor(use_spacy=True, use_gliner=True)

template = QueryTemplate(
    intent="org_project_issues",
    template_text="Show open issues for {org} on project {project}",
    query_template="SELECT * FROM issues WHERE org='{org}' AND project='{project}' AND status='open'",
    parameters=["org", "project"],
    # No regex needed — GLiNER resolves both from the param names directly
)

params = ext.extract("Show open issues for Acme Corp on project Apollo", template)
# {"org": "Acme Corp", "project": "Apollo"}

query = ext.render_query(template, params)
# SELECT * FROM issues WHERE org='Acme Corp' AND project='Apollo' AND status='open'
```

| Scenario | Recommended backend |
|---|---|
| Numeric or enum parameters | Regex only (`use_spacy=False, use_gliner=False`) |
| Standard entities (person, org, number) | spaCy (`use_spacy=True`) |
| Domain-specific or unpredictable param names | GLiNER (`use_gliner=True`) |
| Mixed templates in the same app | Both enabled — cascade handles it |
| Edge / resource-constrained deployment | Regex + heuristics only |

Both backends fall back gracefully if the package is not installed.

---

## Batch Operations

### `store_batch` — single embedding round-trip

```python
import asyncio
from medha import Medha
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

entries = [
    {"question": "How many users are there?", "generated_query": "SELECT COUNT(*) FROM users;"},
    {"question": "List all active orders",    "generated_query": "SELECT * FROM orders WHERE status = 'active';"},
    {"question": "Average order value",       "generated_query": "SELECT AVG(amount) FROM orders;",
     "response_summary": "Returns the mean order amount."},
]

async def main():
    async with Medha(collection_name="batch_demo", embedder=FastEmbedAdapter()) as cache:
        success = await cache.store_batch(entries)
        print(f"Batch stored: {success}")

        hit = await cache.search("How many users exist?")
        print(f"{hit.strategy.value}: {hit.generated_query}")

asyncio.run(main())
```

### `store_many` — chunked bulk upsert with progress

For large datasets that exceed memory or API-rate limits. Chunking and concurrency are controlled by `Settings.batch_size` and `Settings.batch_embed_concurrency`.

```python
from medha import Medha, Settings
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

settings = Settings(
    batch_size=200,              # entries per embedding chunk
    batch_embed_concurrency=4,   # concurrent embedding requests
)

async def main():
    async with Medha(
        collection_name="large_cache",
        embedder=FastEmbedAdapter(),
        settings=settings,
    ) as cache:
        stored = await cache.store_many(
            entries,             # list of {question, generated_query, ...} dicts
            ttl=86400,           # optional per-entry TTL (seconds)
            on_progress=lambda done, total: print(f"{done}/{total}"),
        )
        print(f"Stored {stored} entries")
```

`warm_from_file()` and `warm_from_dataframe()` both delegate to `store_many()` internally.

### Export & Dedup

```python
async with Medha(collection_name="my_cache", embedder=FastEmbedAdapter()) as cache:
    # Export all entries to a pandas DataFrame
    df = await cache.export_to_dataframe()
    print(df.head())

    # Remove duplicate entries (same query_hash), keep most-used per group
    removed = await cache.dedup_collection()
    print(f"Removed {removed} duplicates")
```

---

## Cache Lifecycle (TTL & Invalidation)

### Per-entry TTL

Pass `ttl` (seconds) to `store()` or `store_many()`. Expired entries are excluded from all search results automatically.

```python
import asyncio
from medha import Medha, Settings
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

async def main():
    settings = Settings(
        backend_type="memory",
        default_ttl_seconds=3600,          # global default: 1 hour
        cleanup_interval_seconds=300,      # auto-delete expired entries every 5 min
    )

    async with Medha(
        collection_name="my_cache",
        embedder=FastEmbedAdapter(),
        settings=settings,
    ) as cache:
        # Per-entry TTL overrides the global default
        await cache.store(
            "Show live orders",
            "SELECT * FROM orders WHERE status = 'live';",
            ttl=60,    # expires in 60 seconds
        )

        # Entry with no TTL (immortal regardless of default)
        await cache.store(
            "Count all users",
            "SELECT COUNT(*) FROM users;",
            ttl=None,
        )

        # Manually expire all stale entries in the collection
        deleted = await cache.expire()
        print(f"Deleted {deleted} expired entries")

asyncio.run(main())
```

### Cache Invalidation

```python
async with Medha(collection_name="my_cache", embedder=FastEmbedAdapter()) as cache:
    # Remove a specific entry by exact question text
    removed = await cache.invalidate("Show live orders")
    print(removed)   # True if found and deleted

    # Remove all entries sharing the same query hash
    count = await cache.invalidate_by_query_hash("abc123...")

    # Remove all entries associated with a template intent
    count = await cache.invalidate_by_template("employee_lookup")

    # Drop and recreate the entire collection
    count = await cache.invalidate_collection()
```

---

## Observability

### CacheStats

`Medha.stats()` returns an immutable `CacheStats` snapshot with hit/miss rates, percentile latencies, and per-strategy breakdowns.

```python
import asyncio
from medha import Medha
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

async def main():
    async with Medha(
        collection_name="my_cache",
        embedder=FastEmbedAdapter(),
    ) as cache:
        await cache.store("Count all users", "SELECT COUNT(*) FROM users;")
        await cache.search("How many users are there?")
        await cache.search("Something unrelated")

        stats = await cache.stats()

        print(f"Hit rate:         {stats.hit_rate:.1f}%")
        print(f"Total requests:   {stats.total_requests}")
        print(f"Total hits:       {stats.total_hits}")
        print(f"Avg latency:      {stats.avg_latency_ms:.2f} ms")
        print(f"p50 / p95 / p99:  {stats.p50_latency_ms:.2f} / "
              f"{stats.p95_latency_ms:.2f} / {stats.p99_latency_ms:.2f} ms")

        for strategy, s in stats.by_strategy.items():
            print(f"  {strategy:16s}  count={s.count}  avg={s.avg_latency_ms:.2f} ms")

        # Reset counters
        await cache.reset_stats()

asyncio.run(main())
```

Relevant `Settings` fields:

```python
settings = Settings(
    collect_stats=True,               # default: True — disable to save overhead
    stats_max_latency_samples=10_000, # FIFO buffer size for percentile calculations
)
```

---

## Synchronous Usage

Medha provides sync wrappers for environments where `asyncio` is not available (scripts, notebooks, legacy code).

```python
from medha import Medha
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

# Initialize
embedder = FastEmbedAdapter()
cache = Medha(collection_name="sync_demo", embedder=embedder)

# Must call start manually (no async context manager)
import asyncio
asyncio.run(cache.start())

# Sync search and store
cache.store_sync("List all products", "SELECT * FROM products;")
hit = cache.search_sync("Show me all products")
print(f"{hit.strategy.value}: {hit.generated_query}")

# Warm from file synchronously
loaded = cache.warm_from_file_sync("warm_queries.jsonl")

# Clear caches synchronously
cache.clear_caches_sync()

# Clean up
asyncio.run(cache.close())
```

---

## Query Language Examples

Medha is query-language agnostic. Here are examples for different query languages.

### SQL (Text-to-SQL)

```python
from medha import Medha, Settings
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

settings = Settings(query_language="sql")

async with Medha(
    collection_name="text2sql",
    embedder=FastEmbedAdapter(),
    settings=settings,
) as cache:
    await cache.store(
        "What are the top 10 products by revenue?",
        "SELECT p.name, SUM(o.amount) as revenue FROM products p JOIN orders o ON p.id = o.product_id GROUP BY p.name ORDER BY revenue DESC LIMIT 10;",
    )
```

### Cypher (Text-to-Cypher for Neo4j)

```python
settings = Settings(query_language="cypher")

async with Medha(
    collection_name="text2cypher",
    embedder=FastEmbedAdapter(),
    settings=settings,
) as cache:
    await cache.store(
        "Find friends of Alice",
        "MATCH (a:Person {name: 'Alice'})-[:FRIEND]->(f:Person) RETURN f.name",
    )
    await cache.store(
        "Shortest path between Alice and Bob",
        "MATCH p = shortestPath((a:Person {name: 'Alice'})-[*]-(b:Person {name: 'Bob'})) RETURN p",
    )
```

### GraphQL

```python
settings = Settings(query_language="graphql")

async with Medha(
    collection_name="text2graphql",
    embedder=FastEmbedAdapter(),
    settings=settings,
) as cache:
    await cache.store(
        "Get user profile with posts",
        '{ user(id: "123") { name email posts { title createdAt } } }',
    )
```

---

## Qdrant Performance Tuning

### HNSW Index Tuning

Adjust the HNSW index parameters for your workload.

```python
from medha import Settings

# High-throughput production (more memory, faster search)
settings = Settings(
    hnsw_m=32,                # More edges per node (default: 16)
    hnsw_ef_construct=200,    # Deeper construction search (default: 100)
)

# Low-memory / edge deployment
settings = Settings(
    hnsw_m=8,
    hnsw_ef_construct=50,
)
```

### Quantization

Reduce memory usage while maintaining search quality.

```python
from medha import Settings

# Scalar quantization (default, ~4x memory reduction)
settings = Settings(
    enable_quantization=True,
    quantization_type="scalar",
    quantization_rescore=True,        # Re-score with original vectors
    quantization_always_ram=True,     # Keep quantized vectors in RAM
)

# Binary quantization (best for high-dimensional embeddings >= 512d)
settings = Settings(
    enable_quantization=True,
    quantization_type="binary",
    quantization_oversampling=2.0,    # Fetch 2x candidates before re-scoring
)

# No quantization (maximum accuracy, more memory)
settings = Settings(enable_quantization=False)
```

### On-Disk Storage

Store original vectors on disk to save RAM. Useful for large caches.

```python
settings = Settings(
    qdrant_mode="docker",
    on_disk=True,                     # Vectors stored on disk
    enable_quantization=True,         # Quantized copies in RAM for speed
    quantization_always_ram=True,
)
```

### Batch Size Tuning

Control how many entries are upserted per Qdrant API call.

```python
# Large batch inserts (reduce API overhead)
settings = Settings(batch_size=500)

# Small batches (lower memory per call)
settings = Settings(batch_size=50)
```

---

## Cache Monitoring

Track cache performance and hit rates at runtime. See the [Observability](#observability) section for `CacheStats` details.

```python
import asyncio
from medha import Medha
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

async def main():
    async with Medha(
        collection_name="monitored_cache",
        embedder=FastEmbedAdapter(),
    ) as cache:
        await cache.store("Count all users", "SELECT COUNT(*) FROM users;")
        await cache.store("List departments", "SELECT DISTINCT department FROM employees;")

        await cache.search("How many users are there?")
        await cache.search("Show all departments")
        await cache.search("Something completely unrelated")

        # stats() is an async method returning a CacheStats object
        stats = await cache.stats()
        print(f"Total requests:  {stats.total_requests}")
        print(f"Hit rate:        {stats.hit_rate:.1f}%")
        print(f"Avg latency:     {stats.avg_latency_ms:.2f} ms")
        print(f"p95 latency:     {stats.p95_latency_ms:.2f} ms")

        for strategy, s in stats.by_strategy.items():
            print(f"  {strategy:16s}  count={s.count}  avg={s.avg_latency_ms:.2f} ms")

asyncio.run(main())
```

---

## Logging

Configure Medha's logging for debugging and monitoring.

```python
from medha import setup_logging

# Basic: INFO level to console
setup_logging(level="INFO")

# Debug mode: see every tier of the waterfall search
setup_logging(level="DEBUG")

# Log to file + console with different levels
setup_logging(
    level="DEBUG",
    log_file="/var/log/medha/cache.log",
    console_level="WARNING",
)

# Custom format
setup_logging(
    level="INFO",
    fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    date_fmt="%Y-%m-%d %H:%M:%S",
)
```

---

## Full Production Example

A complete configuration combining all features for a production Text-to-SQL system.

```python
import asyncio
from medha import Medha, Settings, QueryTemplate, setup_logging
from medha.embeddings.openai_adapter import OpenAIAdapter

# Configure logging
setup_logging(level="INFO", log_file="medha.log")

# Production settings
settings = Settings(
    # Qdrant Cloud
    backend_type="qdrant",
    qdrant_mode="cloud",
    qdrant_url="https://your-cluster.cloud.qdrant.io",
    qdrant_api_key="your-api-key",  # stored as SecretStr, never logged

    # Query language
    query_language="sql",

    # Tuned thresholds
    score_threshold_exact=0.99,
    score_threshold_semantic=0.88,
    score_threshold_template=0.82,
    score_threshold_fuzzy=80.0,

    # L1 cache
    l1_cache_max_size=5000,

    # HNSW tuning
    hnsw_m=32,
    hnsw_ef_construct=200,

    # Quantization
    enable_quantization=True,
    quantization_type="scalar",
    quantization_rescore=True,
    quantization_always_ram=True,

    # Batch operations
    batch_size=200,

    # Templates from file
    template_file="production_templates.json",

    # Persist embedding cache across restarts
    embedding_cache_path="/var/cache/medha/embeddings.json",

    # Security
    max_question_length=8192,          # reject oversized questions (DoS guard)
    allowed_file_dir="/app/data",      # restrict warm_from_file() to this dir
    max_file_size_mb=100,              # reject files larger than 100 MB
)

# OpenAI embeddings
embedder = OpenAIAdapter(
    model_name="text-embedding-3-small",
    api_key="sk-your-key",
)

# Pre-defined templates
templates = [
    QueryTemplate(
        intent="employee_lookup",
        template_text="Find employees in {department}",
        query_template="SELECT * FROM employees WHERE department = '{department}'",
        parameters=["department"],
        priority=1,
        aliases=[
            "Show {department} employees",
            "Who works in {department}",
            "List {department} team",
        ],
        parameter_patterns={
            "department": r"\b(engineering|sales|marketing|hr|finance|ops)\b",
        },
    ),
]

async def main():
    from medha.l1_cache.redis_adapter import RedisL1Cache

    async with Medha(
        collection_name="production_text2sql",
        embedder=embedder,
        settings=settings,
        templates=templates,
        # Shared L1 cache across all service instances
        l1_backend=RedisL1Cache(url="redis://redis.internal:6379/0", ttl=3600),
    ) as cache:
        # Pre-warm cache from a curated file of known queries
        await cache.warm_from_file("common_queries.jsonl")

        # Or inline with store_batch for dynamic queries
        await cache.store_batch([
            {
                "question": "How many active users?",
                "generated_query": "SELECT COUNT(*) FROM users WHERE status = 'active';",
                "response_summary": "Count of active users",
            },
            {
                "question": "Total revenue this month",
                "generated_query": "SELECT SUM(amount) FROM orders WHERE date >= DATE_TRUNC('month', NOW());",
            },
            {
                "question": "Top customers by order count",
                "generated_query": "SELECT customer_id, COUNT(*) as n FROM orders GROUP BY customer_id ORDER BY n DESC LIMIT 10;",
            },
        ])

        # Search with full waterfall
        hit = await cache.search("Find employees in engineering")
        print(f"Strategy: {hit.strategy.value}")
        print(f"Query: {hit.generated_query}")
        print(f"Confidence: {hit.confidence:.3f}")

        # Monitor performance
        print(cache.stats)

asyncio.run(main())
```

---

## API Reference Summary

### Core

| Class / Method | Description |
|---|---|
| `Medha` | Core cache class with waterfall search |
| `Medha.search(question)` | Waterfall search → `CacheHit` |
| `Medha.store(question, query, *, ttl)` | Store a question-query pair with optional TTL |
| `Medha.store_batch(entries)` | Bulk store — single embedding round-trip |
| `Medha.store_many(entries, *, batch_size, on_progress, ttl)` | Chunked bulk upsert with concurrency control |
| `Medha.warm_from_file(path, *, ttl)` | Pre-populate cache from JSON / JSONL file |
| `Medha.warm_from_dataframe(df, *, ttl)` | Pre-populate cache from a pandas DataFrame |
| `Medha.export_to_dataframe(collection_name)` | Export collection to a pandas DataFrame |
| `Medha.dedup_collection(collection_name)` | Remove duplicate entries (same `query_hash`) |
| `Medha.expire(collection_name)` | Delete all expired entries; returns count |
| `Medha.invalidate(question)` | Remove entry by exact question text; returns `bool` |
| `Medha.invalidate_by_query_hash(hash)` | Remove all entries with a given query hash |
| `Medha.invalidate_by_template(template_id)` | Remove all entries for a template intent |
| `Medha.invalidate_collection(collection_name)` | Drop and recreate an entire collection |
| `Medha.stats(collection_name)` | Returns a `CacheStats` snapshot (async method) |
| `Medha.reset_stats()` | Reset all in-process statistics counters |
| `Medha.load_templates(templates)` | Load `QueryTemplate` list at runtime |
| `Medha.load_templates_from_file(path)` | Load templates from JSON file |
| `Medha.clear_caches()` | Clear L1 + embedding caches (async) |
| `Medha.feedback(question, correct)` | Record correct/incorrect signal for a cached entry |
| `Medha.feedback_sync` | Sync wrapper for `feedback()` |
| `Medha.search_sync` / `store_sync` / `warm_from_file_sync` / `clear_caches_sync` | Sync wrappers |

### Configuration & Types

| Class | Description |
|---|---|
| `Settings` | Pydantic configuration with env var support (`MEDHA_` prefix) |
| `Settings.feedback_incorrect_threshold` | Auto-invalidate a cache entry when its incorrect-feedback count reaches N |
| `CacheHit` | Search result: `generated_query`, `confidence`, `strategy`, `expires_at` |
| `CacheStats` | Immutable stats snapshot: hit/miss rates, latency percentiles, per-strategy breakdown |
| `StrategyStats` | Per-strategy `count`, `total_latency_ms`, `avg_latency_ms` |
| `QueryTemplate` | Parameterized question-to-query template |
| `CacheEntry` | Stored cache entry with vector and metadata |
| `CacheResult` | Backend search result with score |
| `SearchStrategy` | Enum: `l1_cache`, `template_match`, `exact_match`, `semantic_match`, `fuzzy_match`, `no_match`, `error` |

> **Breaking change in 0.4.0:** Custom `VectorStorageBackend` subclasses must implement `update_feedback(entry_id, correct)`. Add a no-op implementation to avoid `TypeError` on instantiation.

### Interfaces & Backends

| Class | Description |
|---|---|
| `BaseEmbedder` | Abstract interface for embedding providers |
| `L1CacheBackend` | Abstract interface for L1 cache backends |
| `VectorStorageBackend` | Abstract interface for vector storage backends |
| `FastEmbedAdapter` | Local embeddings via FastEmbed (ONNX) |
| `OpenAIAdapter` | OpenAI embedding API adapter |
| `CohereAdapter` | Cohere Embed v3 adapter (`pip install medha-archai[cohere]`) |
| `GeminiAdapter` | Google Gemini adapter (`pip install medha-archai[gemini]`) |
| `InMemoryBackend` | Pure-Python in-process backend, zero deps (`backend_type="memory"`) |
| `QdrantBackend` | Qdrant vector storage (memory / docker / cloud) |
| `PgVectorBackend` | PostgreSQL + pgvector (`pip install medha-archai[pgvector]`) |
| `VectorChordBackend` | PostgreSQL + VectorChord (`pip install medha-archai[vectorchord]`) |
| `ElasticsearchBackend` | Elasticsearch 8.x (`pip install medha-archai[elasticsearch]`) |
| `ChromaBackend` | ChromaDB ephemeral / disk / HTTP (`pip install medha-archai[chroma]`) |
| `WeaviateBackend` | Weaviate local / cloud (`pip install medha-archai[weaviate]`) |
| `RedisVectorBackend` | Redis Stack HNSW/FLAT (`pip install medha-archai[redis]`) |
| `AzureSearchBackend` | Azure AI Search HNSW (`pip install medha-archai[azure-search]`) |
| `LanceDBBackend` | LanceDB embedded / S3 / GCS / az (`pip install medha-archai[lancedb]`) |
| `InMemoryL1Cache` | Default in-process LRU L1 cache |
| `RedisL1Cache` | Redis-backed L1 cache (`pip install medha-archai[redis]`) |

### Utilities

| Function | Description |
|---|---|
| `setup_logging()` | Configure the `medha` logger |
| `ParameterExtractor` | NER-based parameter extractor (regex → GLiNER → spaCy → heuristics) |

---

## Roadmap

* [x] Redis L1 Cache backend (`RedisL1Cache`, `pip install medha[redis]`).
* [x] Cache warming from JSON / JSONL file (`warm_from_file`).
* [x] Per-tier latency stats (`tier_latencies_ms` in `cache.stats`).
* [x] Persistent embedding cache (`MEDHA_EMBEDDING_CACHE_PATH`).
* [x] Parallel execution of Tier 2 (exact) and Tier 3 (semantic).
* [x] `InMemoryBackend` — pure-Python vector backend, zero external deps.
* [x] `PgVectorBackend` — PostgreSQL + pgvector backend.
* [x] `backend_type` setting for declarative backend selection.
* [x] Security hardening: `max_question_length`, `max_file_size_mb`, `allowed_file_dir`, `qdrant_api_key` as `SecretStr`, PostgreSQL identifier validation.
* [x] `ElasticsearchBackend`, `VectorChordBackend`, `ChromaBackend`, `WeaviateBackend`, `RedisVectorBackend`, `AzureSearchBackend`, `LanceDBBackend` — seven new vector backends.
* [x] `CohereAdapter` and `GeminiAdapter` — two new embedding providers.
* [x] TTL support on `store()` / `store_many()` with per-entry and global defaults.
* [x] `expire()` and `invalidate*()` cache lifecycle methods.
* [x] `CacheStats` observability model with hit rate, latency percentiles, and per-strategy breakdown.
* [x] `store_many()`, `export_to_dataframe()`, `dedup_collection()` — batch and management operations.
* [x] `qdrant-client` moved to optional `[qdrant]` extra; default `backend_type` changed to `"memory"`.
* [x] Feedback loop — `Medha.feedback()` with optional auto-invalidation
      (`feedback_incorrect_threshold`).
* [x] `medha` CLI — `pip install "medha-archai[cli]"`. Commands: stats,
      warm, invalidate, invalidate-collection, expire, dedup, export, feedback.

---

## Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for details on how to set up the dev environment and run tests.

## License

This project is licensed under the Apache-2.0 License - see the [LICENSE](https://github.com/ArchAI-Labs/medha/blob/main/LICENSE) file for details.

---

*Built with ❤️ by **[ArchAI Labs](https://github.com/ArchAI-Labs)***
