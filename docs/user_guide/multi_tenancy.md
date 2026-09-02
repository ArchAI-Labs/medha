# Multi-Tenancy

A single Medha backend can serve multiple tenants, applications, or query languages simultaneously. Each **collection name** is a fully isolated namespace — different vector spaces, different entries, different thresholds.

This page covers four common patterns:

1. [Per-application isolation](#per-application-isolation) — multiple query languages on one backend
2. [Per-customer isolation](#per-customer-isolation) — SaaS: one namespace per customer
3. [Shared embedder](#shared-embedder) — one model in memory, many `Medha` instances
4. [Concurrent fan-out](#concurrent-fan-out) — parallel searches across tenants with `asyncio.gather`
5. [Per-tenant stats](#per-tenant-stats) — independent hit-rate monitoring per tenant
6. [Per-tenant backend selection](#per-tenant-backend-selection) — trial vs. paid tier isolation

---

## Per-Application Isolation

Two applications share the same backend but use different collection names. Their caches are completely independent — a query cached for the SQL app is never returned by the Cypher app.

```python
from medha import Medha, Settings
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

shared_settings = Settings(backend_type="memory")
shared_embedder = FastEmbedAdapter()

# SQL analytics app
medha_sql = Medha(
    collection_name="app_sql_analytics",
    embedder=shared_embedder,
    settings=shared_settings,
)
await medha_sql.start()
await medha_sql.store("How many users are there?", "SELECT COUNT(*) FROM users")

# Cypher graph explorer
medha_cypher = Medha(
    collection_name="app_cypher_graph",
    embedder=shared_embedder,
    settings=shared_settings,
)
await medha_cypher.start()
await medha_cypher.store("How many Person nodes?", "MATCH (p:Person) RETURN COUNT(p)")

# Same question in each app returns app-specific results
hit_sql    = await medha_sql.search("How many users are there?")
hit_cypher = await medha_cypher.search("How many users are there?")

print(hit_sql.generated_query)     # SELECT COUNT(*) FROM users
print(hit_cypher.generated_query)  # None — no match in the Cypher collection
```

---

## Per-Customer Isolation

In a SaaS platform each customer has their own data schema and question vocabulary. Namespacing collection names with the customer ID keeps each tenant's cache completely isolated — even when they ask the same question.

```python
TENANTS = {
    "tenant_acme": {
        "pairs": [
            ("How many users?",      "SELECT COUNT(*) FROM acme_users"),
            ("Total revenue",        "SELECT SUM(amount) FROM acme_invoices"),
        ],
        "settings": Settings(backend_type="memory", score_threshold_semantic=0.82),
    },
    "tenant_globex": {
        "pairs": [
            ("How many users?",      "SELECT COUNT(*) FROM globex_accounts"),  # same question, different table!
            ("Total revenue",        "SELECT SUM(revenue) FROM globex_deals"),
        ],
        "settings": Settings(backend_type="memory", score_threshold_semantic=0.87),
    },
}

shared_embedder = FastEmbedAdapter()
instances: dict[str, Medha] = {}

for tenant_id, config in TENANTS.items():
    m = Medha(
        collection_name=tenant_id,
        embedder=shared_embedder,
        settings=config["settings"],
    )
    await m.start()
    for question, sql in config["pairs"]:
        await m.store(question, sql)
    instances[tenant_id] = m

# Same question, different SQL per tenant
for tenant_id, m in instances.items():
    hit = await m.search("How many users do we have?")
    print(f"[{tenant_id}] {hit.generated_query}")
# [tenant_acme]   SELECT COUNT(*) FROM acme_users
# [tenant_globex] SELECT COUNT(*) FROM globex_accounts
```

### Or one collection with a tenant filter

A collection per tenant is the strongest isolation there is: separate vector spaces, separate thresholds, and a tenant's entries are unreachable from another instance by construction. It also means one `Medha` instance, one collection and one set of settings per tenant, which stops scaling somewhere around a few hundred of them.

[Metadata filters](metadata_filters.md) offer the other trade-off — one collection, the tenant carried on each entry:

```python
await cache.store("How many users?", "SELECT COUNT(*) FROM acme_users",
                  metadata={"tenant": "acme"})

hit = await cache.search("How many users do we have?", filters={"tenant": "acme"})
```

A search that declares a tenant only ever matches entries stored under it, and entries stored without one match no tenant at all. The isolation is enforced by the query rather than by the namespace, so it is only as strong as the caller passing the right filter — use collections when a mistake there would be a data leak, filters when the tenant count makes that impractical.

---

## Shared Embedder

Loading an embedding model costs ~200 MB. `FastEmbedAdapter` is stateless and safe to share across multiple `Medha` instances — you pay the loading cost once regardless of tenant count.

```python
N_TENANTS = 5
one_embedder = FastEmbedAdapter()  # loaded once

tenants = []
for i in range(N_TENANTS):
    m = Medha(
        collection_name=f"tenant_{i:03d}",
        embedder=one_embedder,           # same object reused
        settings=Settings(backend_type="memory"),
    )
    await m.start()
    await m.store(
        f"How many records in tenant {i}?",
        f"SELECT COUNT(*) FROM tenant_{i:03d}_records",
    )
    tenants.append(m)
```

---

## Concurrent Fan-Out

Because Medha is async-native, N concurrent tenant searches run in parallel and complete in roughly the time of a single search.

```python
import asyncio

# Fan out searches across all tenants simultaneously
results = await asyncio.gather(*[
    m.search(f"Revenue for customer group {i}")
    for i, m in enumerate(tenants)
])
```

---

## Per-Tenant Stats

Each `Medha` instance tracks its own stats independently. Use this to monitor per-tenant hit rates and identify tenants with low cache coverage.

```python
for tenant_id, m in instances.items():
    stats = await m.stats()
    print(f"[{tenant_id}] hit_rate={stats.hit_rate:.0%}  requests={stats.total_requests}")
```

See [Observability](observability.md) for the full `CacheStats` API.

---

## Per-Tenant Backend Selection

A common pattern is to use `backend_type="memory"` for **trial tenants** (zero infrastructure cost) and promote to a persistent backend on conversion.

```python
# Trial tenant — pure-Python InMemoryBackend, no infrastructure needed
trial_settings = Settings(backend_type="memory")
medha_trial = Medha("trial_tenant_001", embedder=shared_embedder, settings=trial_settings)

# Paid tenant — persistent Qdrant backend
paid_settings = Settings(backend_type="qdrant", qdrant_mode="docker", qdrant_host="localhost")
medha_paid = Medha("paid_tenant_acme", embedder=shared_embedder, settings=paid_settings)
```

| `backend_type` | Extra deps | Use case |
|---|---|---|
| `"memory"` | none (default) | Ephemeral tenants, CI, trial accounts |
| `"qdrant"` | `medha-archai[qdrant]` | Production — Docker or Cloud |
| `"pgvector"` | `medha-archai[pgvector]` | Teams already running PostgreSQL |
| `"elasticsearch"` | `medha-archai[elasticsearch]` | Teams on the Elastic stack |
| `"lancedb"` | `medha-archai[lancedb]` | Serverless / embedded deployments |

---

## Tenant Lifecycle

When a tenant is deactivated, close their `Medha` instance and drop their collection.

```python
# Deactivate
await medha_tenant.close()

# Delete data (Qdrant docker/cloud)
from qdrant_client import QdrantClient
client = QdrantClient(url="http://localhost:6333")
client.delete_collection("tenant_trial_001")
```

With `backend_type="memory"` closing the instance already releases all data.

---

!!! note "Full working example"
    The complete runnable notebook is available at
    [`demo/09_multi_tenant.ipynb`](https://github.com/ArchAI-Labs/medha/blob/main/demo/09_multi_tenant.ipynb).
