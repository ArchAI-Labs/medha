# Medha

![medha_logo](https://github.com/nickprock/medha/blob/develop/img/medha_logo.png)
  
## Semantic Memory for AI Data Agents

> ***Reduce LLM latency and costs by caching Text-to-Query generations (SQL, Cypher, GraphQL) with semantic understanding.***

---

## 🧠 What is Medha?

**Medha** is an asynchronous, high-performance semantic cache library designed specifically for **Text-to-Query** systems.

Unlike traditional key-value caches that require exact string matches, Medha understands that *"Show me the top 5 users"* and *"List the first five users"* are the same question. It intercepts these queries and returns pre-calculated database queries (SQL, Cypher, etc.), bypassing the expensive and slow LLM generation step.

### Why Medha?
* **🚀 100x Faster:** Return cached queries in milliseconds vs. seconds for LLM generation.
* **💰 Cost Efficient:** Reduce API calls to OpenAI/Anthropic by 40-60%.
* **🔌 Agnostic:** Works with **SQL**, **Cypher** (Neo4j), **GraphQL**, or any text-based query language.
* **⚡ Async Native:** Built on `asyncio` for high-concurrency API backends.

---

## 🌊 The "Waterfall" Architecture

Medha uses a sophisticated multi-tier search strategy to maximize cache hits. If a tier fails, it cascades to the next:

1.  **⚡ Tier 0: L1 Memory (LRU)**
    * *Speed:* < 1ms
    * Exact hash match for identical, repeated questions.
2.  **🧩 Tier 1: Template Matching (Intent)**
    * *Speed:* ~10ms
    * Recognizes patterns like *"Show employees in {department}"*. Extracts parameters and injects them into a cached query template.
3.  **🎯 Tier 2: Exact Vector Match**
    * *Speed:* ~20ms
    * Uses high-threshold vector search (Qdrant) to find semantically identical questions.
4.  **🔍 Tier 3: Semantic Similarity**
    * *Speed:* ~25ms
    * Finds questions with the same meaning but different phrasing (e.g., *"Who works here?"* vs *"List employees"*).
5.  **〰️ Tier 4: Fuzzy Fallback**
    * *Speed:* Variable
    * Handles typos and minor string variations using Levenshtein distance.

---

## 📦 Installation

Medha is currently in active development and is not yet available on PyPI. You can install the latest version directly from GitHub:

```bash
pip install git+[https://github.com/ArchAI-Labs/medha.git](https://github.com/ArchAI-Labs/medha.git)
cd medha
pip install -e .
```

---

## 🚀 Quick Start

```python
import asyncio
from medha import Medha

async def main():
    # Initialize Medha (uses local Qdrant in memory by default)
    cache = Medha(collection_name="text2sql_production")

    question = "How many users are active?"
    
    # 1. Search the cache
    hit = await cache.search(question)

    if hit:
        print(f"✅ Cache Hit! Strategy: {hit.strategy}")
        print(f"Query: {hit.generated_query}")
    else:
        print("❌ Cache Miss. Calling LLM...")
        # ... Call your LLM here to generate SQL ...
        generated_sql = "SELECT count(*) FROM users WHERE status = 'active';"
        
        # 2. Store the result for next time
        await cache.store(
            question=question,
            generated_query=generated_sql
        )
        print("💾 Stored in cache.")

if __name__ == "__main__":
    asyncio.run(main())

```

---

## 🧩 Modular & Pluggable

Medha is designed to be completely backend-agnostic. You can swap out the Embedder or the Vector Database.

### Changing the Embedder (e.g., OpenAI)

```python
from medha import Medha
from medha.embeddings import OpenAIAdapter

# Use OpenAI's text-embedding-3-small instead of local FastEmbed
embedder = OpenAIAdapter(
    model="text-embedding-3-small", 
    api_key="sk-..."
)

cache = Medha(
    collection_name="my_cache", 
    embedder=embedder
)

```

### Using Qdrant Cloud / Docker

```python
from medha.config import Settings

# Configure via code or Environment Variables
settings = Settings(
    qdrant_mode="cloud",
    qdrant_url="[https://xyz.qdrant.tech](https://xyz.qdrant.tech)",
    qdrant_api_key="th3-s3cr3t-k3y"
)

```

---

## 🛠️ Roadmap

* [ ] Support for Redis as L1 Cache backend.
* [ ] Auto-eviction policies based on query execution feedback (RLHF).
* [ ] "Golden Query" tagging for verified SQL/Cypher.
* [ ] Dashboard for cache hit/miss analytics.

---

## 🤝 Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for details on how to set up the dev environment and run tests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/ArchAI-Labs/medha/blob/main/LICENSE) file for details.

---

<div align="center">
<p>Built with ❤️ by <strong>ArchAI Labs</strong></p>
</div>

```