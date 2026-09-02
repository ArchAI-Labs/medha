# CLI

Medha ships a `medha` command for inspecting and maintaining a cache without writing a script — warming a collection from a file, expiring stale entries, exporting for analysis, or probing connectivity from a container healthcheck.

---

## Installation

```bash
pip install "medha-archai[cli]"
```

To also use `medha search` and `medha warm`, which need a real embedder:

```bash
pip install "medha-archai[cli,fastembed]"
```

Verify the install:

```bash
medha --help
```

---

## Configuration

The CLI takes **no connection flags**. It builds its `Settings` from `MEDHA_*` environment variables or a `.env` file in the working directory, exactly like the library — so a shell that can run your application can run `medha`:

| Variable | Default | Description |
|---|---|---|
| `MEDHA_BACKEND_TYPE` | `memory` | Which backend to connect to |
| `MEDHA_COLLECTION` | `default` | Collection name for all commands |
| `MEDHA_EMBEDDER_TYPE` | `_noop` | Embedder used by `search` and `warm` |
| `MEDHA_FASTEMBED_MODEL` | `BAAI/bge-small-en-v1.5` | FastEmbed model identifier |

Backend connection variables (`MEDHA_QDRANT_URL`, `MEDHA_PG_DSN`, …) follow the same pattern as the Python API. See [Configuration](configuration.md) for the full reference.

### Quick-start `.env` for Qdrant

```ini
MEDHA_BACKEND_TYPE=qdrant
MEDHA_QDRANT_URL=https://xyz.qdrant.tech
MEDHA_QDRANT_API_KEY=your-api-key
MEDHA_COLLECTION=prod_cache
MEDHA_EMBEDDER_TYPE=fastembed
```

### Embedder selection

`MEDHA_EMBEDDER_TYPE` picks the adapter:

| Value | Requires |
|---|---|
| `_noop` *(default)* | nothing — a placeholder that cannot embed |
| `fastembed` | `medha-archai[fastembed]`, optional `MEDHA_FASTEMBED_MODEL` |
| `openai` | `medha-archai[openai]`, `OPENAI_API_KEY` |
| `cohere` | `medha-archai[cohere]`, `COHERE_API_KEY` |
| `gemini` | `medha-archai[gemini]`, `GOOGLE_API_KEY` |
| `openai-compatible` | `medha-archai[openai]`, `MEDHA_OAI_COMPAT_BASE_URL` |
| `mistral` | `medha-archai[mistral]`, `MEDHA_MISTRAL_API_KEY` |

The default is deliberately `_noop`: most maintenance commands never embed anything, so they should not pay for loading a model. Only `search` and `warm` need a real embedder — `search` exits with code 1 and a message telling you what to set if it finds `_noop`.

---

## Commands

### `medha stats`

Report collection name, backend, entry counts, and — since 0.5.0 — hit rate and per-strategy breakdown read from the snapshot Medha persists into the backend.

```bash
medha stats
medha stats --collection my_cache
medha stats --json
```

```
Collection : default
Backend    : LanceDBBackend (lancedb)
Entries    : 128 (main)  4 (templates)
Requests   : 2400
Hit rate   : 71.3%
By strategy:
  L1       : 900
  Template : 120
  Exact    : 640
  Semantic : 52
  Fuzzy    : 0
```

Before any snapshot exists the counters are replaced by a single line:

```
Stats      : not yet persisted (run some searches first)
```

!!! note "Latency percentiles are still per-process"
    Entry counts and hit-rate counters come from the backend, so a fresh CLI invocation can report them. Latency percentiles are sampled per process and are deliberately **not** persisted — use [`stats()`](observability.md) from inside your application for those. With `--json`, `total_requests`, `hit_rate` and `hits_by_strategy` are `null` until the first snapshot is written.

---

### `medha search QUESTION`

Look up a question and print the best match. Requires a real embedder.

```bash
medha search "top 5 customers"
medha search "top 5 customers" -c my_cache --json
```

```
Strategy : semantic_match
Score    : 0.9214
Query    : SELECT * FROM customers ORDER BY revenue DESC LIMIT 5
```

Prints `No cache hit.` when nothing matches. Useful for answering "why did this question hit the wrong entry?" without instrumenting your application.

`--filter KEY=VALUE` (short `-f`) restricts the search to entries whose metadata carries that pair. Repeat it to require several:

```bash
medha search "revenue yesterday" --filter resolved_date=2026-08-12
medha search "revenue" -f date=2026-08-12 -f tenant=acme
```

Values are compared as strings, so an entry stored with `{"hour": 10}` is not reachable from the command line — no coercion is attempted, because guessing would break a tenant genuinely named `10`. `--json` reports the scope that answered under `metadata`. See [Metadata Filters](metadata_filters.md).

---

### `medha health`

Probe backend connectivity and embedder availability.

```bash
medha health
medha health --json
```

```
Backend  [OK]  128 entries
Embedder [OK]  BAAI/bge-small-en-v1.5  dim=384
Overall  : OK
```

Each component reports `OK`, `ERROR` or `SKIPPED`, and the command **exits 1 if either fails** — so it drops straight into a container healthcheck or a CI gate. The embedder probe reports `SKIPPED` when `MEDHA_EMBEDDER_TYPE=_noop`, and a skipped probe does not fail the check.

---

### `medha warm FILE`

Bulk-load entries from a JSON or JSONL file. Requires a real embedder — every entry is embedded on the way in.

```bash
MEDHA_EMBEDDER_TYPE=fastembed medha warm entries.jsonl
MEDHA_EMBEDDER_TYPE=fastembed medha warm entries.jsonl --collection sql_cache --ttl 86400
```

Each record needs at least `question` and `generated_query`; `response_summary` is optional.

```jsonl
{"question": "How many users?", "generated_query": "SELECT COUNT(*) FROM users"}
{"question": "List active orders", "generated_query": "SELECT * FROM orders WHERE active = 1"}
```

```
Progress: 2/2 entries stored.
Warmed 2 entries into 'default'.
```

`--ttl` applies the same expiry to all entries; `--batch-size` controls the upsert chunk size. File loading honours `MEDHA_ALLOWED_FILE_DIR` and `MEDHA_MAX_FILE_SIZE_MB` when set.

---

### `medha expire`

Delete all entries whose TTL has elapsed.

```bash
medha expire
medha expire --collection my_cache
```

Use this with a scheduler (cron, APScheduler) if `enable_background_cleanup` is disabled, or when you need immediate cleanup from outside the running process.

---

### `medha dedup`

Remove entries sharing the same `query_hash` (derived from the generated query string), keeping the most recently stored entry per hash. Requires `pandas`.

```bash
medha dedup
medha dedup --collection my_cache
```

---

### `medha invalidate QUESTION`

Remove the entry whose normalised question matches the argument.

```bash
medha invalidate "How many users are registered?"
```

Prints `Removed.`, or `Not found.` if no entry matches. Uses a plain text lookup — no embedder required.

---

### `medha invalidate-collection`

Delete every entry in the collection.

```bash
medha invalidate-collection --yes
```

The `--yes` flag is mandatory — without it the command refuses to run.

---

### `medha export`

Dump all entries to CSV (default) or JSON. Requires `pandas`.

```bash
medha export                                        # CSV to stdout
medha export --format csv --output cache.csv
medha export --format json --output cache.json
```

---

### `medha feedback QUESTION`

Record a correct or incorrect signal for a cached entry.

```bash
medha feedback "How many users are registered?" --correct
medha feedback "How many users are registered?" --incorrect
```

Prints `Feedback recorded.`, or `Entry not found.` if no entry matches. Works with the default `_noop` embedder, because it locates entries by exact normalised-question match rather than vector search. See [Feedback Loop](feedback.md) for the auto-invalidation behaviour.

---

### `medha logo`

Print the Medha lotus logo.

---

## Global Options

| Option | Description |
|---|---|
| `--collection TEXT` | Override the collection name (default: `MEDHA_COLLECTION`, or `default`). Accepted by every command that targets a collection; `search` and `health` also accept the short form `-c`. |
| `--json` | Machine-readable output. Available on `stats`, `search` and `health`. |
| `--help` | Show help for any command |

---

## Scripting

`--json` produces output for `jq`:

```bash
# fail a deploy gate if the cache is suspiciously empty
count=$(medha stats --json | jq '.entries')
[ "$count" -gt 100 ] || { echo "cache underpopulated"; exit 1; }

# alert when the hit rate drops (null until the first snapshot is persisted)
medha stats --json | jq -e '.hit_rate != null and .hit_rate > 0.5' >/dev/null \
  || echo "hit rate below target"

# nightly maintenance
medha expire && medha dedup
```

Commands exit non-zero on failure, so `&&` chaining and `set -e` behave as expected.

---

## See Also

* [Configuration](configuration.md) — full `MEDHA_*` variable reference
* [Feedback Loop](feedback.md) — `feedback_incorrect_threshold` and auto-invalidation
* [Observability](observability.md) — in-process metrics the CLI cannot see
* [Batch Operations](batch_operations.md) — the library equivalents of `warm`, `dedup` and `export`
* [Demo 26 — CLI](https://github.com/ArchAI-Labs/medha/blob/main/demo/26_cli.ipynb)
