# Batch Operations

Medha provides batch methods for both **ingestion** (loading many entries at once) and **lookup** (searching for many questions in a single call).

---

## `search_batch` — Look Up Multiple Questions at Once

`search_batch()` embeds all questions in a single `aembed_batch()` round-trip, then runs the waterfall search for each question concurrently via `asyncio.gather()`. This is significantly faster than calling `search()` in a loop when you have many questions.

```python
async with Medha("demo", embedder=embedder, settings=settings) as cache:
    questions = [
        "How many active users?",
        "Total revenue this month",
        "List all products",
    ]
    results = await cache.search_batch(questions)

    for q, r in zip(questions, results):
        if r.strategy.value != "no_match":
            print(f"{q!r} → {r.generated_query} (strategy={r.strategy.value}, score={r.score:.3f})")
        else:
            print(f"{q!r} → cache miss")
```

Results are returned in the **same order** as the input questions. Each result is a `CacheResult` identical to what `search()` would return for that question individually.

**Sync wrapper:**

```python
results = cache.search_batch_sync(questions)
```

!!! tip "When to use `search_batch` vs `search`"

    Use `search_batch()` whenever you need to look up more than one question per request — for example, when warming a test harness, running a benchmark, or processing a batch of user queries in one pass. The primary saving is the embedding step: one `aembed_batch()` call instead of N individual `aembed()` calls.

---

Medha also provides several methods for efficiently loading large numbers of entries into the cache without calling `store()` in a loop.

---

## `store_batch`

Store a list of `(question, query)` tuples in a single call. Embeddings are computed concurrently up to `batch_embed_concurrency`:

```python
pairs = [
    ("How many users?", "SELECT COUNT(*) FROM users"),
    ("List all products", "SELECT * FROM products"),
    ("Active sessions count", "SELECT COUNT(*) FROM sessions WHERE active = true"),
]

async with Medha("demo", embedder=embedder, settings=settings) as cache:
    await cache.store_batch(pairs)
```

---

## `store_many`

Like `store_batch`, but accepts a list of dicts with full metadata (TTL, tags, response summary):

```python
entries = [
    {
        "question": "How many users?",
        "generated_query": "SELECT COUNT(*) FROM users",
        "tags": ["users", "aggregation"],
        "ttl_seconds": 3600,
    },
    {
        "question": "List all products",
        "generated_query": "SELECT * FROM products",
        "tags": ["products"],
    },
]

async with Medha("demo", embedder=embedder, settings=settings) as cache:
    await cache.store_many(entries)
```

---

## `warm_from_file`

Load entries from a JSON or JSONL file. Useful for seeding a new deployment from a curated query library.

=== "JSON"

    ```json
    [
      {
        "question": "How many active users?",
        "generated_query": "SELECT COUNT(*) FROM users WHERE active = true",
        "tags": ["users"]
      },
      {
        "question": "Total revenue this month",
        "generated_query": "SELECT SUM(amount) FROM orders WHERE MONTH(created_at) = MONTH(NOW())",
        "tags": ["revenue"]
      }
    ]
    ```

=== "JSONL"

    ```jsonl
    {"question": "How many active users?", "generated_query": "SELECT COUNT(*) FROM users WHERE active = true", "tags": ["users"]}
    {"question": "Total revenue this month", "generated_query": "SELECT SUM(amount) FROM orders WHERE MONTH(created_at) = MONTH(NOW())", "tags": ["revenue"]}
    ```

```python
async with Medha("demo", embedder=embedder, settings=settings) as cache:
    count = await cache.warm_from_file("queries.json")
    print(f"Loaded {count} entries")
```

!!! note

    File reads are restricted to `Settings.allowed_file_dir` when set, and files exceeding `Settings.max_file_size_mb` are rejected.

---

## `warm_from_dataframe`

Load entries from a pandas DataFrame with configurable column mapping:

```python
import pandas as pd

df = pd.DataFrame({
    "nl_question": ["How many users?", "List all products"],
    "sql_query": ["SELECT COUNT(*) FROM users", "SELECT * FROM products"],
    "category": ["users", "products"],
})

async with Medha("demo", embedder=embedder, settings=settings) as cache:
    count = await cache.warm_from_dataframe(
        df,
        question_col="nl_question",
        query_col="sql_query",
        tags_col="category",   # optional
    )
    print(f"Loaded {count} entries")
```

---

## `export_to_dataframe`

Export the entire cache collection to a pandas DataFrame for analysis or migration:

```python
async with Medha("demo", embedder=embedder, settings=settings) as cache:
    df = await cache.export_to_dataframe()
    print(df.columns)
    # ['id', 'question', 'generated_query', 'strategy', 'confidence', 'tags', 'created_at']
    df.to_csv("cache_export.csv", index=False)
```

---

## `dedup_collection`

Remove duplicate entries from the collection. Two entries are considered duplicates when their stored questions have cosine similarity above a configurable threshold:

```python
async with Medha("demo", embedder=embedder, settings=settings) as cache:
    removed = await cache.dedup_collection(similarity_threshold=0.99)
    print(f"Removed {removed} duplicate entries")
```

---

!!! tip "Tuning `batch_embed_concurrency`"

    The `batch_embed_concurrency` setting (default `8`) controls how many embedding requests run in parallel during batch ingestion. Increase it for cloud embedders with high rate limits; decrease it if you see rate-limit errors or memory pressure from large ONNX models.

    ```python
    settings = Settings(
        batch_embed_concurrency=16,  # double the default
        batch_upsert_size=200,       # larger upsert batches
    )
    ```

    For FastEmbedAdapter (ONNX in-process), `batch_embed_concurrency` governs thread-pool parallelism. For API-based embedders, it governs concurrent HTTP requests.
