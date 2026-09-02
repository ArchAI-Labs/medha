# Backends

Medha supports nine vector backends. Choose based on your infrastructure, persistence requirements, and scale.

---

## Comparison

| Backend | Use Case | Persistence | Setup | Cloud-native | Extra |
|---|---|---|---|---|---|
| `memory` | Dev/testing | No | None | No | built-in |
| `qdrant` | General production | Yes | Docker/Cloud | Yes | `[qdrant]` |
| `pgvector` | Existing PostgreSQL | Yes | Postgres + extension | No | `[pgvector]` |
| `elasticsearch` | Search-heavy stacks | Yes | ES 8.x | Yes | `[elasticsearch]` |
| `vectorchord` | High-perf Postgres | Yes | Postgres + extension | No | `[vectorchord]` |
| `chroma` | Rapid prototyping | Optional | None/Docker | No | `[chroma]` |
| `weaviate` | Knowledge graph | Yes | Docker/Cloud | Yes | `[weaviate]` |
| `redis` | Low-latency cache | Optional | Redis Stack | Yes | `[redis]` |
| `azure_search` | Azure ecosystem | Yes | Managed | Yes | `[azure-search]` |
| `lancedb` | Embedded/serverless | Yes | None | Optional | `[lancedb]` |

---

## Metadata Filtering

All ten backends store and return `CacheEntry.metadata`, so [metadata filters](metadata_filters.md) work everywhere. They differ only in **where** the filter is evaluated:

| Backend | Filter evaluation |
|---|---|
| `qdrant` | Native, all value types |
| `pgvector`, `vectorchord` | Native for strings (`jsonb @>`), Python for the rest |
| `elasticsearch` | Native for strings (`term` on a `flattened` field), Python for the rest |
| `chroma` | Native for strings and booleans, Python for the rest |
| `lancedb` | Narrows the scan natively for strings, Python decides |
| `memory`, `weaviate`, `azure_search`, `redis` | Python |

Results are verified in Python whatever the engine did, so the answer is the same everywhere — this table is about speed and recall, not correctness. A Python-side filter only sees the candidates the vector search returned, so it over-fetches (`Settings.metadata_filter_overfetch`) and can miss a match ranked below that window.

A custom backend opts in by storing `CacheEntry.metadata`, returning it on `CacheResult`, and setting `supports_metadata = True`; passing `filters` to one that has not raises `ConfigurationError`.

---

## `memory` — In-Memory

!!! info "Install"

    Built-in. No extra dependencies required.

    ```bash
    pip install medha-archai
    ```

The default backend. Stores all vectors in a Python dict keyed by entry ID. Data is lost when the process exits. Ideal for unit tests and local development.

```python
from medha import Settings

settings = Settings(backend_type="memory")
```

---

## `qdrant` — Qdrant

!!! info "Install"

    ```bash
    pip install "medha-archai[qdrant]"
    ```

Qdrant is the recommended production backend. It supports both an in-process mode (no server) and a full client-server deployment with gRPC/REST.

```python
from medha import Settings

# Local Docker instance
settings = Settings(
    backend_type="qdrant",
    qdrant_host="localhost",
    qdrant_port=6333,
)

# Qdrant Cloud
settings = Settings(
    backend_type="qdrant",
    qdrant_url="https://your-cluster.qdrant.tech",
    qdrant_api_key="your-api-key",
)
```

Start a local Qdrant instance with Docker:

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

---

## `pgvector` — PostgreSQL + pgvector

!!! info "Install"

    ```bash
    pip install "medha-archai[pgvector]"
    ```

Uses the `pgvector` extension for PostgreSQL. Best choice if you already run Postgres and want to consolidate infrastructure.

```python
from medha import Settings

settings = Settings(
    backend_type="pgvector",
    pg_dsn="postgresql://user:password@localhost:5432/mydb",
)
```

Install the extension in your database:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

---

## `elasticsearch` — Elasticsearch

!!! info "Install"

    ```bash
    pip install "medha-archai[elasticsearch]"
    ```

Uses Elasticsearch 8.x dense vector fields and kNN search. Suitable for stacks already using Elasticsearch for full-text search.

```python
from medha import Settings

settings = Settings(
    backend_type="elasticsearch",
    es_url="http://localhost:9200",
    # es_api_key="..." for Elastic Cloud
)
```

---

## `vectorchord` — VectorChord

!!! info "Install"

    ```bash
    pip install "medha-archai[vectorchord]"
    ```

VectorChord is a high-performance vector extension for PostgreSQL using IVF-PQ indexing. Recommended over pgvector for large-scale deployments (>1M entries).

```python
from medha import Settings

settings = Settings(
    backend_type="vectorchord",
    pg_dsn="postgresql://user:password@localhost:5432/mydb",
)
```

---

## `chroma` — Chroma

!!! info "Install"

    ```bash
    pip install "medha-archai[chroma]"
    ```

Chroma can run in-process (ephemeral) or as a server (persistent). Excellent for rapid prototyping and notebook environments.

```python
from medha import Settings

# In-process (ephemeral)
settings = Settings(backend_type="chroma")

# HTTP client (persistent)
settings = Settings(
    backend_type="chroma",
    chroma_host="localhost",
    chroma_port=8000,
)
```

---

## `weaviate` — Weaviate

!!! info "Install"

    ```bash
    pip install "medha-archai[weaviate]"
    ```

Weaviate is a knowledge-graph-oriented vector database with built-in schema management. Good for teams already using Weaviate or who need rich object properties alongside vectors.

```python
from medha import Settings

settings = Settings(
    backend_type="weaviate",
    weaviate_url="http://localhost:8080",
    # weaviate_api_key="..." for Weaviate Cloud
)
```

---

## `redis` — Redis Stack

!!! info "Install"

    ```bash
    pip install "medha-archai[redis]"
    ```

Uses Redis Stack's `RediSearch` module for vector similarity search. Delivers the lowest latency of any networked backend (< 1 ms P99 for small indexes). Persistence depends on your Redis configuration (RDB/AOF).

```python
from medha import Settings

settings = Settings(
    backend_type="redis",
    redis_url="redis://localhost:6379",
)
```

!!! note

    The `redis` backend uses Redis Stack, not vanilla Redis. Ensure `RediSearch` is available in your deployment.

---

## `azure_search` — Azure AI Search

!!! info "Install"

    ```bash
    pip install "medha-archai[azure-search]"
    ```

Fully managed vector search service on Azure. Recommended for teams running workloads in Azure that want zero infrastructure management.

```python
from medha import Settings

settings = Settings(
    backend_type="azure_search",
    azure_search_endpoint="https://your-service.search.windows.net",
    azure_search_api_key="your-admin-key",
)
```

---

## `lancedb` — LanceDB

!!! info "Install"

    ```bash
    pip install "medha-archai[lancedb]"
    ```

LanceDB is an embedded vector database that writes directly to disk (local or cloud object storage). No server required, making it ideal for serverless deployments or single-machine applications that need persistence.

```python
from medha import Settings

# Local filesystem
settings = Settings(
    backend_type="lancedb",
    lancedb_uri="./my_cache.lancedb",
)

# S3 / GCS / Azure Blob
settings = Settings(
    backend_type="lancedb",
    lancedb_uri="s3://my-bucket/cache",
)
```
