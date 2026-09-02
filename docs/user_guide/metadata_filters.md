# Metadata Filters

Two questions that differ only by a date embed almost identically. "Revenue yesterday" and "revenue last Monday" are the same sentence with a different word in it, and the vector barely moves — so the semantic tier can answer one with the other's query, at high confidence, and nothing in the result says anything is wrong.

The same holds for any scope the embedding does not separate: a time window, a tenant, a currency, a region, a unit.

Metadata is the guardrail. The scope is stored beside the entry as structured data, and a search that declares which scope it needs never sees an entry carrying a different one.

```python
await cache.store(
    "revenue for the period",
    "SELECT SUM(amount) FROM sales WHERE day = '2026-08-12'",
    metadata={"resolved_date": "2026-08-12"},
)

hit = await cache.search(
    "revenue yesterday",
    filters={"resolved_date": "2026-08-12"},
)
```

Resolve the period in your application — that is where "yesterday" means something — then hand Medha the resolved value. If nothing is cached for that day, the search returns `NO_MATCH` and you generate the query as usual.

---

## Matching Rules

A filter is satisfied when **every** key it names is present on the entry with the same value.

| Entry metadata | Filter | Result |
|---|---|---|
| `{"date": "2026-08-12"}` | `{"date": "2026-08-12"}` | match |
| `{"date": "2026-08-12"}` | `{"date": "2026-08-13"}` | no match |
| `{"date": "2026-08-12", "tenant": "acme"}` | `{"date": "2026-08-12"}` | match — extra keys on the entry are ignored |
| `{"date": "2026-08-12"}` | `{"date": "2026-08-12", "tenant": "acme"}` | no match — every filter key must hold |
| `{}` | `{"date": "2026-08-12"}` | no match |

!!! warning "A missing key is a mismatch, never a wildcard"
    An entry stored without a scope does not answer a question that demands one. This is what makes the guardrail safe on a collection you have been filling for months: entries written before you adopted metadata carry none, so they never satisfy a filter. They keep answering unfiltered searches exactly as before.

    The flip side is that adopting filters on an existing collection returns `NO_MATCH` until those entries are re-stored with their scope.

Searching without `filters` is unchanged in every respect — same tiers, same keys, same results.

---

## What Can Go in Metadata

A flat mapping of scalars: `str`, `int`, `float`, `bool`.

```python
metadata={"resolved_date": "2026-08-12", "hour": 10, "ratio": 0.25, "draft": False}
```

| Rule | Limit |
|---|---|
| Keys | `^[A-Za-z_][A-Za-z0-9_.-]{0,63}$` |
| Keys per entry | 32 |
| String value length | 256 |
| Value types | `str`, `int`, `float`, `bool` — not `None`, not nested |

Nesting is rejected rather than flattened: ten backends store metadata ten different ways, and a flat map of scalars is what all of them accept. Keys are restricted because they end up in backend field names and filter expressions.

Two comparison details worth knowing:

- `True` never equals `1`. A boolean only matches a boolean.
- `10` and `10.0` are the same value, because not every backend preserves the distinction across a round-trip.

A value that breaks a rule raises `ValueError` from `store()` or `search()`, before anything is written or queried.

---

## Where Filters Apply

| Tier | Behaviour |
|---|---|
| 0 — L1 cache | Keyed by question **and** filters |
| 1 — Template match | Unaffected |
| 2 — Exact vector | Filtered |
| 3 — Semantic | Filtered |
| 4 — Fuzzy | Filtered |

The L1 cache is the one that needs explaining. It is keyed by the question alone and runs before every tier that knows about metadata — so an unfiltered search caching an answer for "revenue for the period" would hand that same answer to a filtered search asking for a different day, before any guardrail ran. Filtered lookups therefore get their own key, derived from the question and the filters together. Unfiltered searches keep the key they have always used.

Tier 1 is left alone deliberately: a template renders a fresh query from the question text rather than returning a stored entry, so it cannot answer with another scope's query.

!!! note "`store()` writes an unfiltered L1 entry"
    The L1 entry `store()` leaves behind is keyed for an unfiltered lookup, as it always was. The first filtered search for that question misses L1 and is served by the vector tier — one extra round trip, never a wrong scope.

---

## Strict and Soft

By default a filter mismatch removes the candidate entirely, so a scope with nothing cached returns `NO_MATCH`. That is the safe answer: the caller falls back to generating the query.

`Settings.metadata_filter_mode="soft"` keeps mismatching candidates but multiplies their confidence by `metadata_filter_soft_penalty` (default `0.5`), so they survive only if they still clear the tier's threshold.

```python
settings = Settings(
    metadata_filter_mode="soft",
    metadata_filter_soft_penalty=0.5,
)
```

Soft mode is for scopes where a near miss is still useful — a stale-but-related dashboard number, say. For a date or a tenant it is the wrong choice, which is why it is not the default.

---

## The Hit Reports Its Scope

`CacheHit.metadata` carries the scope of the entry that answered, so the caller can see which day, tenant or window was actually served:

```python
hit = await cache.search("revenue yesterday", filters={"resolved_date": "2026-08-12"})
print(hit.metadata)   # {'resolved_date': '2026-08-12'}
```

It is empty for a template hit, which has no stored entry behind it.

---

## Batch and Warming

`metadata` is accepted per item everywhere entries are written:

```python
await cache.store_batch([
    {"question": "revenue for the period", "generated_query": "...",
     "metadata": {"resolved_date": "2026-08-12"}},
])

await cache.store_many(rows)                       # same "metadata" key per row
await cache.warm_from_file("cache.jsonl")          # same key in the JSON
await cache.warm_from_dataframe(df, metadata_cols=["resolved_date", "tenant"])
```

`search_batch()` takes either one filter for the whole batch, or one per question — the usual shape when each question resolves to a different day:

```python
hits = await cache.search_batch(
    ["revenue yesterday", "revenue last Monday"],
    filters=[{"resolved_date": "2026-08-12"}, {"resolved_date": "2026-08-10"}],
)
```

A `None` in a slot searches that question unfiltered.

---

## Deduplication

`dedup_collection()` groups entries by query text **and** scope. The same SQL stored for two tenants is two entries, not a duplicate — collapsing them would leave the survivor answering for a scope it was never stored under.

---

## Backend Support

Every backend stores and returns metadata. They differ in whether the filter is evaluated by the storage engine or in Python afterwards:

| Backend | Filter evaluation |
|---|---|
| Qdrant | Native, all value types |
| pgvector, VectorChord | Native for string values (`jsonb @>`), Python for the rest |
| Elasticsearch | Native for string values (`term` on a `flattened` field), Python for the rest |
| Chroma | Native for strings and booleans, Python for the rest |
| LanceDB | Narrows the scan natively for string values, Python decides |
| InMemory, Weaviate, Azure AI Search, Redis | Python |

This is a performance distinction, not a correctness one: results are verified in Python whatever the engine did, so every backend returns the same answer.

A Python-side filter can only work with the candidates the vector search returned, so it over-fetches — `Settings.metadata_filter_overfetch` (default `10`) candidates per requested result. A match ranked below that window is missed. Raise it for collections where many entries share a question and differ only by scope.

!!! warning "A backend must round-trip metadata before it can be filtered"
    Passing `filters` to a backend that does not raises `ConfigurationError`. Every built-in backend supports it; a custom `VectorStorageBackend` opts in by storing `CacheEntry.metadata`, returning it on `CacheResult`, and setting `supports_metadata = True`.

### PostgreSQL: the optional index

`jsonb @>` answers from a GIN index, which is **not** created automatically — building one locks the table for the duration, and on an existing deployment it would index a column that is `{}` in every row written before the upgrade. Medha hands you the statement instead:

```python
print(backend.metadata_index_sql("my_cache"))
# CREATE INDEX CONCURRENTLY IF NOT EXISTS medha_my_cache_metadata_gin_idx
#     ON public.medha_my_cache USING gin (metadata);
```

Filtering works without it. Add it when a collection holds enough entries that actually carry metadata to make the scan hurt.

---

## From the CLI

```bash
medha search "revenue yesterday" --filter resolved_date=2026-08-12
medha search "revenue" -f date=2026-08-12 -f tenant=acme
```

Repeat `--filter` to require several. Values are compared as strings, so an entry stored with `{"hour": 10}` is not reachable from the command line — no coercion is attempted, because guessing would break a tenant genuinely named `10`.

---

## See Also

- [Multi-Tenancy](multi_tenancy.md) — tenant isolation, of which a `tenant` filter is one form
- [Templates](templates.md) — Tier 1, which intercepts parameterised questions before the vector tiers
- [Configuration](configuration.md) — the three `metadata_filter_*` settings
