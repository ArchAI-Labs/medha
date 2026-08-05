# Observability

Medha exposes hit/miss counters, per-strategy breakdowns, and latency percentiles through the `CacheStats` object and standard Python logging.

---

## `CacheStats` Fields

Retrieve statistics with `stats()`:

```python
async with Medha("demo", embedder=embedder, settings=settings) as cache:
    # ... store and search calls ...
    stats = await cache.stats()
```

| Field | Type | Description |
|---|---|---|
| `total_hits` | `int` | Number of successful cache lookups |
| `total_misses` | `int` | Number of cache misses |
| `hit_rate` | `float` | Fraction of requests that hit the cache |
| `avg_latency_ms` | `float` | Mean search latency across all requests |
| `p50_latency_ms` | `float` | Median search latency |
| `p95_latency_ms` | `float` | 95th-percentile search latency |
| `p99_latency_ms` | `float` | 99th-percentile search latency |
| `by_strategy` | `dict[str, StrategyStats]` | Per-tier breakdown, keyed by strategy value |

---

## Hit Rate

$$\text{hit\_rate} = \frac{\text{total\_hits}}{\text{total\_hits} + \text{total\_misses}}$$

A hit rate of 0.8 means 80% of LLM calls were avoided.

---

## Latency Percentiles

Search latency is tracked per request. Percentiles are computed over a rolling window:

```python
stats = await cache.stats()
print(f"P50: {stats.p50_latency_ms:.1f} ms")
print(f"P95: {stats.p95_latency_ms:.1f} ms")
print(f"P99: {stats.p99_latency_ms:.1f} ms")
```

Expected ranges (in-memory backend, FastEmbed):

| Tier | P50 | P95 |
|---|---|---|
| L1 Cache | < 0.1 ms | < 0.5 ms |
| Template Match | 1–3 ms | 5 ms |
| Exact / Semantic Vector | 5–15 ms | 20 ms |
| Fuzzy | 20–40 ms | 50 ms |

---

## Per-Strategy Breakdown

`stats.by_strategy` maps each strategy **value** (a `str`, e.g. `"l1_cache"` — not the `SearchStrategy` member itself) to a `StrategyStats` object:

```python
stats = await cache.stats()
for strategy, s in stats.by_strategy.items():
    print(f"{strategy}: hits={s.count}, avg={s.avg_latency_ms:.1f} ms")
```

`StrategyStats` carries `count` and `total_latency_ms`, plus a computed `avg_latency_ms` property.

Output example:

```
l1_cache: hits=142, avg=0.08 ms
template_match: hits=38, avg=2.1 ms
exact_vector_match: hits=21, avg=8.3 ms
semantic_match: hits=64, avg=11.2 ms
fuzzy_match: hits=5, avg=31.4 ms
```

---

## Persistent Statistics

!!! info "New in 0.5.0"

`CacheStats` is a per-process accumulator: without persistence, every restart resets the hit rate to zero. Since 0.5.0 Medha writes a `PersistedStats` snapshot into the backend every `stats_persist_interval` requests and reloads it on `start()`, so the counters describe the cache's whole history rather than the current process's uptime.

```python
settings = Settings(
    backend_type="lancedb",
    stats_persist_interval=100,   # flush every 100 requests (default)
)

async with Medha("demo", embedder=embedder, settings=settings) as cache:
    stats = await cache.stats()
    print(stats.total_requests, stats.hit_rate)   # includes previous runs
```

What is stored:

| Field | Description |
|---|---|
| `total_requests` | Every `search()` call |
| `total_hits` / `total_misses` / `total_errors` | Outcome counters |
| `hits_by_strategy` | Per-tier hit counts (`l1_cache`, `semantic_match`, …) |
| `last_reset_at` / `updated_at` | Timezone-aware UTC timestamps |
| `hit_rate` / `miss_rate` | Computed properties |

**Latency percentiles are not persisted.** They are sampled per process and merging them across restarts would be misleading, so after a restart latency describes the current process while the counters describe everything.

### Tuning the interval

Each flush is one small write. The interval trades write frequency against how many requests you lose on an unclean shutdown.

| `stats_persist_interval` | Write frequency | Worst-case loss |
|---|---|---|
| `1` | Every request | 0 requests |
| `100` *(default)* | Every 100 requests | Up to 99 |
| `1000` | Every 1000 requests | Up to 999 |

Writes run in a background task and are best-effort: a failure is logged and never propagates to the `search()` call that scheduled it. Persistence is skipped entirely when `collect_stats=False`.

### Backend support

All ten built-in backends implement `load_stats()` / `save_stats()`. On `VectorStorageBackend` the two methods are **not** abstract — they default to `return None` and a no-op, so a custom backend written against 0.4.x keeps working and simply opts out of stats persistence.

The `medha stats` CLI command reports the persisted snapshot; see the [CLI guide](cli.md).

---

## Logging

Use `setup_logging()` to configure the `medha` logger:

```python
from medha.logging import setup_logging

# Human-readable text format
setup_logging(level="INFO", format="text")

# Structured JSON for log aggregation (Datadog, CloudWatch, etc.)
setup_logging(level="INFO", format="json")
```

Or configure the `medha` logger directly:

```python
import logging

logging.getLogger("medha").setLevel(logging.DEBUG)
```

Key log events:

| Event | Level | Description |
|---|---|---|
| `cache.hit` | INFO | A search returned a cache hit |
| `cache.miss` | INFO | A search returned no result |
| `backend.init` | INFO | Backend connected successfully |
| `backend.error` | ERROR | Backend connection or query failed |
| `cleanup.run` | DEBUG | Background cleanup sweep started |
| `cleanup.deleted` | DEBUG | Number of expired entries removed |

---

## Prometheus Integration

Medha does not ship a Prometheus exporter, but `CacheStats` is easy to bridge:

```python
import asyncio
from prometheus_client import Counter, Histogram, start_http_server
from medha import Medha, Settings
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

hits_counter = Counter("medha_hits_total", "Cache hits", ["strategy"])
misses_counter = Counter("medha_misses_total", "Cache misses")
latency_hist = Histogram(
    "medha_search_latency_seconds",
    "Search latency",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)


async def search_with_metrics(cache, question: str):
    import time

    t0 = time.perf_counter()
    hit = await cache.search(question)
    elapsed = time.perf_counter() - t0
    latency_hist.observe(elapsed)

    if hit:
        hits_counter.labels(strategy=hit.strategy.name).inc()
    else:
        misses_counter.inc()

    return hit


async def main():
    start_http_server(8000)  # expose /metrics on :8000
    settings = Settings(backend_type="memory")
    async with Medha("demo", embedder=FastEmbedAdapter(), settings=settings) as cache:
        while True:
            await search_with_metrics(cache, "How many users?")
            await asyncio.sleep(1)


asyncio.run(main())
```

Access metrics at `http://localhost:8000/metrics`.
