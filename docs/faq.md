# FAQ

---

## Which backend should I use in production?

**Qdrant** is the recommended default. It has first-class async support, a free managed cloud tier, and handles millions of vectors with low latency. If you already run PostgreSQL, **pgvector** is a strong alternative that eliminates a separate service — or **VectorChord** if you need higher throughput on large indexes (>1M entries). Use `memory` only for testing and local development; it loses all data on process exit.

---

## Can I use Medha without an API key?

Yes. The `FastEmbedAdapter` runs ONNX models locally — no API key, no internet access after the first model download, and no per-token cost. Install it with:

```bash
pip install "medha-archai[fastembed]"
```

Combined with the `memory` or `lancedb` backend, you can run a fully air-gapped Medha deployment.

---

## How do I disable semantic search and use exact-only?

Set `score_threshold_semantic` equal to `score_threshold_exact` in `Settings`. This makes Tier 3 (Semantic Match) unreachable, since Tier 2 (Exact Vector) will fire first for any score above the threshold, and Tier 3 only activates for scores in the range `[semantic, exact)`.

The simplest approach is to set both thresholds to `0.99`:

```python
settings = Settings(
    score_threshold_semantic=0.99,
    score_threshold_exact=0.99,
)
```

!!! note

    Medha validates that `score_threshold_semantic <= score_threshold_exact`. Setting them equal is permitted; setting semantic *above* exact raises `ConfigurationError`.

---

## What happens when the LLM schema changes?

If a database schema change makes a cached SQL query invalid (e.g., a column is renamed), you need to invalidate the affected entries. The cleanest approach is to group queries by table at store time via `template_id`:

```python
await cache.store(
    "How many users?",
    "SELECT COUNT(*) FROM users",
    template_id="users_table",
)
```

When the `users` table schema changes:

```python
await cache.invalidate_by_template("users_table")
```

All affected entries are removed in one call. Then re-warm the cache with corrected queries using `store_batch` or `warm_from_file`.

---

## Is Medha thread-safe?

Yes, at the async level. All public methods are `async` and safe to call concurrently from multiple coroutines within the same event loop. Medha uses asyncio locks internally to protect shared state (L1 cache, stats counters).

Medha is **not** designed for multi-process sharing of the in-memory backend — each process gets its own isolated in-memory store. For multi-process deployments, use a networked backend (Qdrant, pgvector, Redis, etc.) and a Redis-backed L1 cache (`l1_cache_type="redis"`).

---

## How do I switch embedders without losing cached data?

You cannot migrate cached vectors to a new embedding model without re-embedding everything, because vectors from different models occupy incompatible spaces.

The recommended migration path:

1. Create a new Medha collection with the new embedder:

    ```python
    async with Medha("cache_v2", embedder=new_embedder, settings=settings) as new_cache:
        ...
    ```

2. Export your existing entries:

    ```python
    async with Medha("cache_v1", embedder=old_embedder, settings=settings) as old_cache:
        df = await old_cache.export_to_dataframe()
    ```

3. Re-import into the new collection:

    ```python
    async with Medha("cache_v2", embedder=new_embedder, settings=settings) as new_cache:
        await new_cache.warm_from_dataframe(df, question_col="question", query_col="generated_query")
    ```

4. Switch traffic to `cache_v2` and delete `cache_v1`.

---

## What is the performance overhead of Medha?

On a cache **hit**, Medha adds:

- **< 0.1 ms** for an L1 cache hit (in-process dict lookup)
- **1–5 ms** for a template match (NER + cosine)
- **5–20 ms** for a vector search (network round-trip to backend + cosine)

On a cache **miss**, the overhead is the same as the miss path above, plus the time to embed the question (10–50 ms for ONNX, 50–200 ms for API embedders).

For comparison, a typical LLM call for Text-to-SQL takes **500 ms – 5 s** plus token cost. A 70% hit rate with Medha reduces average response time from, say, 2 s to under 100 ms for the majority of requests.

The biggest latency factor is the embedding call on miss. If your miss rate is high, consider pre-warming the cache with `warm_from_file` before going live.
