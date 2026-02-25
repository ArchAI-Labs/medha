# Medha

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/medha-archai?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=BLUE&left_text=downloads)](https://pepy.tech/projects/medha-archai)

<br>

![medha_logo](https://raw.githubusercontent.com/ArchAI-Labs/medha/refs/heads/main/img/medha_logo.png)

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
3.  **Tier 2: Exact Vector Match**
    * *Speed:* ~20ms
    * Uses high-threshold vector search (Qdrant) to find semantically identical questions.
4.  **Tier 3: Semantic Similarity**
    * *Speed:* ~25ms
    * Finds questions with the same meaning but different phrasing (e.g., *"Who works here?"* vs *"List employees"*).
5.  **Tier 4: Fuzzy Fallback**
    * *Speed:* Variable
    * Handles typos and minor string variations using Levenshtein distance.

---

## Installation

### Core (minimal)

```bash
pip install medha-archai
```

Core dependencies: `pydantic`, `pydantic-settings`, `qdrant-client`.

### With an embedding provider

```bash
# Local embeddings with FastEmbed (recommended for getting started)
pip install "medha-archai[fastembed]"

# OpenAI embeddings
pip install "medha-archai[openai]"
```

### With optional extras

```bash
# Fuzzy matching (Tier 4 - Levenshtein distance)
pip install "medha-archai[fuzzy]"

# spaCy NLP for advanced parameter extraction
pip install "medha-archai[nlp]"

# Everything
pip install "medha-archai[all]"
```

### Install from source

```bash
# From GitHub
pip install git+https://github.com/ArchAI-Labs/medha.git

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

## Configuration Examples

Medha is highly configurable. Below are examples covering every major use case.

### Basic: In-Memory with FastEmbed (Default)

The simplest setup, perfect for development, testing, and single-process applications. No external services needed.

```python
import asyncio
from medha import Medha, Settings
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

async def main():
    settings = Settings(qdrant_mode="memory")  # default
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
        qdrant_mode="cloud",
        qdrant_url="https://your-cluster.cloud.qdrant.io",
        qdrant_api_key="your-qdrant-api-key",
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

## Batch Operations

Efficiently store many question-query pairs at once.

```python
import asyncio
from medha import Medha
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

entries = [
    {
        "question": "How many users are there?",
        "generated_query": "SELECT COUNT(*) FROM users;",
    },
    {
        "question": "List all active orders",
        "generated_query": "SELECT * FROM orders WHERE status = 'active';",
    },
    {
        "question": "Average order value",
        "generated_query": "SELECT AVG(amount) FROM orders;",
        "response_summary": "Returns the mean order amount.",
    },
    {
        "question": "Top 5 customers by spend",
        "generated_query": "SELECT customer_id, SUM(amount) AS total FROM orders GROUP BY customer_id ORDER BY total DESC LIMIT 5;",
    },
]

async def main():
    embedder = FastEmbedAdapter()

    async with Medha(
        collection_name="batch_demo",
        embedder=embedder,
    ) as cache:
        success = await cache.store_batch(entries)
        print(f"Batch stored: {success}")

        # Verify
        hit = await cache.search("How many users exist?")
        print(f"{hit.strategy.value}: {hit.generated_query}")
        # semantic_match: SELECT COUNT(*) FROM users;

asyncio.run(main())
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

Track cache performance and hit rates at runtime.

```python
import asyncio
from medha import Medha
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

async def main():
    embedder = FastEmbedAdapter()

    async with Medha(
        collection_name="monitored_cache",
        embedder=embedder,
    ) as cache:
        # Populate cache
        await cache.store("Count all users", "SELECT COUNT(*) FROM users;")
        await cache.store("List departments", "SELECT DISTINCT department FROM employees;")

        # Run some searches
        await cache.search("How many users are there?")
        await cache.search("Show all departments")
        await cache.search("Something completely unrelated")

        # Check stats
        stats = cache.stats
        print(f"Total requests: {stats['total_requests']}")
        print(f"Hit rate: {stats['hit_rate']:.1f}%")
        print(f"L1 hits: {stats['by_strategy']['l1_hits']}")
        print(f"Semantic hits: {stats['by_strategy']['semantic_hits']}")
        print(f"Misses: {stats['by_strategy']['misses']}")
        print(f"Templates loaded: {stats['templates_loaded']}")

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
    qdrant_mode="cloud",
    qdrant_url="https://your-cluster.cloud.qdrant.io",
    qdrant_api_key="your-api-key",

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
    async with Medha(
        collection_name="production_text2sql",
        embedder=embedder,
        settings=settings,
        templates=templates,
    ) as cache:
        # Pre-warm cache with common queries
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

| Class / Function | Description |
|---|---|
| `Medha` | Core cache class with waterfall search |
| `Settings` | Pydantic configuration with env var support (`MEDHA_` prefix) |
| `CacheHit` | Search result with `generated_query`, `confidence`, `strategy` |
| `QueryTemplate` | Parameterized question-to-query template |
| `CacheEntry` | Stored cache entry with vector and metadata |
| `CacheResult` | Backend search result with score |
| `SearchStrategy` | Enum: `l1_cache`, `template_match`, `exact_match`, `semantic_match`, `fuzzy_match`, `no_match`, `error` |
| `BaseEmbedder` | Abstract interface for embedding providers |
| `VectorStorageBackend` | Abstract interface for vector storage backends |
| `FastEmbedAdapter` | Local embeddings via FastEmbed (ONNX) |
| `OpenAIAdapter` | OpenAI embedding API adapter |
| `QdrantBackend` | Qdrant vector storage (memory / docker / cloud) |
| `setup_logging()` | Configure the `medha` logger |

---

## Roadmap

* [ ] Support for Redis as L1 Cache backend.
* [ ] Auto-eviction policies based on query execution feedback (RLHF).
* [ ] "Golden Query" tagging for verified SQL/Cypher.
* [ ] Dashboard for cache hit/miss analytics.

---

## Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for details on how to set up the dev environment and run tests.

## License

This project is licensed under the Apache-2.0 License - see the [LICENSE](https://github.com/ArchAI-Labs/medha/blob/main/LICENSE) file for details.

---

*Built with ❤️ by **[ArchAI Labs](https://github.com/ArchAI-Labs)***
