# Examples

Hands-on Jupyter notebooks covering every major feature of Medha.
Each notebook is self-contained and can be run locally or opened directly on GitHub.

```bash
git clone https://github.com/ArchAI-Labs/medha.git
cd medha
pip install "medha-archai[all]"
jupyter notebook demo/
```

---

## Query Languages

| Notebook | Description |
|---|---|
| [01 — Text-to-SQL (SQLite)](https://github.com/ArchAI-Labs/medha/blob/main/demo/01_text2sql_sqlite.ipynb) | End-to-end Text-to-SQL with a local SQLite database |
| [02 — Text-to-Cypher (Neo4j)](https://github.com/ArchAI-Labs/medha/blob/main/demo/02_text2cypher_neo4j.ipynb) | Graph query caching for Neo4j Cypher queries |
| [03 — Text-to-Query (MongoDB MQL)](https://github.com/ArchAI-Labs/medha/blob/main/demo/03_text2query_mongodb.ipynb) | MongoDB query caching with MQL |
| [10 — Text-to-SQL (DuckDB)](https://github.com/ArchAI-Labs/medha/blob/main/demo/10_text2sql_duckdb.ipynb) | Analytical SQL caching with DuckDB |

---

## Embedders

| Notebook | Description |
|---|---|
| [04 — Custom Embedder](https://github.com/ArchAI-Labs/medha/blob/main/demo/04_custom_embedder.ipynb) | Implement `BaseEmbedder` for any embedding provider |
| [07 — Cloud Embedders](https://github.com/ArchAI-Labs/medha/blob/main/demo/07_openai_embedder.ipynb) | OpenAI, Cohere, and Gemini embedding adapters |
| [05 — NER: spaCy vs GLiNER](https://github.com/ArchAI-Labs/medha/blob/main/demo/05_ner_spacy_vs_gliner.ipynb) | Parameter extraction strategies for template matching |

---

## Vector Backends

| Notebook | Description |
|---|---|
| [11 — InMemory Backend](https://github.com/ArchAI-Labs/medha/blob/main/demo/11_inmemory_backend.ipynb) | Zero-dependency in-process caching |
| [24 — Qdrant Backend](https://github.com/ArchAI-Labs/medha/blob/main/demo/24_qdrant_backend.ipynb) | Production-grade HNSW with Qdrant (memory / Docker / Cloud) |
| [12 — pgvector Backend](https://github.com/ArchAI-Labs/medha/blob/main/demo/12_pgvector_backend.ipynb) | PostgreSQL-native vector search |
| [15 — VectorChord Backend](https://github.com/ArchAI-Labs/medha/blob/main/demo/15_vectorchord_backend.ipynb) | High-throughput PostgreSQL + VectorChord |
| [14 — Elasticsearch Backend](https://github.com/ArchAI-Labs/medha/blob/main/demo/14_elasticsearch_backend.ipynb) | Full-text + vector search with Elasticsearch 8.x |
| [16 — Chroma Backend](https://github.com/ArchAI-Labs/medha/blob/main/demo/16_chroma_backend.ipynb) | Lightweight local vector store with ChromaDB |
| [17 — Weaviate Backend](https://github.com/ArchAI-Labs/medha/blob/main/demo/17_weaviate_backend.ipynb) | Knowledge graph + vector search with Weaviate |
| [18 — Redis Vector Backend](https://github.com/ArchAI-Labs/medha/blob/main/demo/18_redis_vector_backend.ipynb) | Sub-millisecond semantic caching with Redis Stack |
| [19 — Azure AI Search Backend](https://github.com/ArchAI-Labs/medha/blob/main/demo/19_azure_search_backend.ipynb) | Managed cloud vector search on Azure |
| [20 — LanceDB Backend](https://github.com/ArchAI-Labs/medha/blob/main/demo/20_lancedb_backend.ipynb) | Embedded zero-infrastructure vector search |

---

## Features & Patterns

| Notebook | Description |
|---|---|
| [06 — Production Patterns](https://github.com/ArchAI-Labs/medha/blob/main/demo/06_production_patterns.ipynb) | Logging, retries, health checks, and deployment tips |
| [08 — Fuzzy Matching](https://github.com/ArchAI-Labs/medha/blob/main/demo/08_fuzzy_matching.ipynb) | Tier 4 Levenshtein fallback for typos and variants |
| [09 — Multi-Tenant Caching](https://github.com/ArchAI-Labs/medha/blob/main/demo/09_multi_tenant.ipynb) | Separate collections per tenant |
| [13 — Framework Integrations](https://github.com/ArchAI-Labs/medha/blob/main/demo/13_framework_integrations.ipynb) | LangChain, LlamaIndex, and Haystack integration |
| [21 — Cache Lifecycle](https://github.com/ArchAI-Labs/medha/blob/main/demo/21_cache_lifecycle.ipynb) | TTL, expiry, and invalidation strategies |
| [22 — Observability](https://github.com/ArchAI-Labs/medha/blob/main/demo/22_observability.ipynb) | Stats, latency percentiles, and logging |
| [23 — Batch Operations](https://github.com/ArchAI-Labs/medha/blob/main/demo/23_batch_ops.ipynb) | Bulk ingestion, export to DataFrame, and dedup |
| [25 — Feedback Loop](https://github.com/ArchAI-Labs/medha/blob/main/demo/25_feedback_loop.ipynb) | Recording correct/incorrect signals and auto-invalidation |
| [26 — CLI](https://github.com/ArchAI-Labs/medha/blob/main/demo/26_cli.ipynb) | All `medha` CLI commands against an in-memory backend |
