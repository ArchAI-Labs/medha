# CLI

Medha ships a `medha` command for inspecting and maintaining a cache without writing a script — warming a collection from a file, expiring stale entries, exporting for analysis, or probing connectivity from a container healthcheck.

```bash
pip install "medha-archai[cli]"
```

---

## Configuration

The CLI takes **no connection flags**. It builds its `Settings` from the environment, exactly like the library, so a shell that can run your application can run `medha`:

```bash
export MEDHA_BACKEND_TYPE=qdrant
export MEDHA_QDRANT_URL=http://localhost:6333
export MEDHA_COLLECTION=production
export MEDHA_EMBEDDER_TYPE=fastembed
```

Every command accepts `--collection` / `-c` to override `MEDHA_COLLECTION` (which itself defaults to `default`).

See [Configuration](configuration.md) for the full variable list.

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

### Inspect

```bash
medha stats                      # entry count for the collection
medha stats --json               # {"collection": ..., "backend": ..., "count": N}
medha search "top 5 customers"   # look up a question, print the best match
medha search "top 5 customers" --json
medha health                     # probe backend + embedder
medha health --json
```

`search` prints the matching strategy, score, generated query, and summary — useful for answering "why did this question hit the wrong entry?" without instrumenting your app.

`health` probes the backend (via `count()`) and the embedder (via a real `aembed()` call), prints `OK` / `ERROR` / `SKIPPED` per component, and **exits 1 if either fails** — so it drops straight into a container healthcheck or a CI gate:

```bash
medha health --json || exit 1
```

The embedder probe reports `SKIPPED` when `MEDHA_EMBEDDER_TYPE=_noop`, and a skipped probe does not fail the check.

Since 0.5.0 `stats` also reports hit rate and a per-strategy breakdown, read from the snapshot Medha persists into the backend:

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

!!! note "Latency percentiles are still per-process"
    Entry counts and hit-rate counters come from the backend, so a fresh CLI invocation can report them. Latency percentiles are sampled per process and are deliberately **not** persisted — use [`stats()`](observability.md) from inside your application for those. Before any snapshot has been written the CLI prints `Stats : not yet persisted (run some searches first)`, and `--json` reports `null` for `total_requests`, `hit_rate`, and `hits_by_strategy`.

### Populate

```bash
medha warm entries.jsonl
medha warm entries.json --ttl 86400 --batch-size 500
```

Loads cache entries from a `.json` or `.jsonl` file. Requires a real embedder — every entry is embedded on the way in. `--ttl` applies the same expiry to all entries; `--batch-size` controls the upsert chunk size.

File loading honours `MEDHA_ALLOWED_FILE_DIR` and `MEDHA_MAX_FILE_SIZE_MB` when set.

### Maintain

```bash
medha expire                     # remove entries past their TTL
medha dedup                      # remove duplicate questions
medha invalidate "some question" # remove one entry
medha feedback "some question" --correct
medha feedback "some question" --incorrect
```

`feedback` works with the default `_noop` embedder, because it locates entries by exact normalized-question match rather than vector search. See [Feedback Loop](feedback.md).

### Export

```bash
medha export --format csv --output cache.csv
medha export --format json          # to stdout
```

### Destructive

```bash
medha invalidate-collection --yes
```

Drops the entire collection. The `--yes` flag is mandatory — without it the command refuses to run.

---

## Scripting

`--json` on `stats`, `search`, and `health` produces machine-readable output for `jq`:

```bash
# fail a deploy gate if the cache is suspiciously empty
count=$(medha stats --json | jq '.count')
[ "$count" -gt 100 ] || { echo "cache underpopulated"; exit 1; }

# nightly maintenance
medha expire && medha dedup
```

Commands exit non-zero on failure, so `&&` chaining and `set -e` behave as expected.

---

## Related

* [Configuration](configuration.md) — environment variables
* [Feedback Loop](feedback.md) — the `feedback` command in context
* [Observability](observability.md) — in-process metrics the CLI cannot see
* [Batch Operations](batch_operations.md) — the library equivalents of `warm`, `dedup`, and `export`

# CLI

Medha ships a command-line interface for administrative operations: inspecting collections, bulk-loading entries, expiring stale data, deduplicating, exporting, and recording feedback — all without writing Python code.

---

## Installation

```bash
pip install "medha-archai[cli]"
```

To also use `medha warm` (which requires an embedder):

```bash
pip install "medha-archai[cli,fastembed]"
```

Verify the install:

```bash
medha --help
```

---

## Configuration

All CLI commands read configuration from `MEDHA_*` environment variables or a `.env` file in the working directory. The most important ones:

| Variable | Default | Description |
|---|---|---|
| `MEDHA_BACKEND_TYPE` | `memory` | Which backend to connect to |
| `MEDHA_COLLECTION` | `default` | Collection name for all commands |
| `MEDHA_EMBEDDER_TYPE` | `_noop` | Embedder for `warm` (`fastembed`, `openai`, `cohere`, `gemini`) |
| `MEDHA_FASTEMBED_MODEL` | `BAAI/bge-small-en-v1.5` | FastEmbed model identifier |

Backend connection variables (`MEDHA_QDRANT_URL`, `MEDHA_PG_DSN`, etc.) follow the same pattern as the Python API. See [Configuration](configuration.md) for the full reference.

### Quick-start `.env` for Qdrant

```ini
MEDHA_BACKEND_TYPE=qdrant
MEDHA_QDRANT_URL=https://xyz.qdrant.tech
MEDHA_QDRANT_API_KEY=your-api-key
MEDHA_COLLECTION=prod_cache
MEDHA_EMBEDDER_TYPE=fastembed
```

---

## Commands

### `medha stats`

Print entry counts and backend type for a collection.

```bash
medha stats
medha stats --collection my_cache
```

```
Collection : default
Backend    : qdrant
Entries    : 1 204
Templates  : 37
```

!!! note

    `stats` reports structural counts only. In-process performance metrics (hit rate, latency percentiles) are not available from the CLI because `CacheStats` is a non-persistent in-memory accumulator on the `Medha` object.

---

### `medha warm FILE`

Bulk-load entries from a JSON or JSONL file. Requires a real embedder (`MEDHA_EMBEDDER_TYPE`).

```bash
MEDHA_EMBEDDER_TYPE=fastembed medha warm entries.jsonl
MEDHA_EMBEDDER_TYPE=fastembed medha warm entries.jsonl --collection sql_cache --ttl 86400
```

Each record must have at least `question` and `generated_query` keys. `response_summary` is optional.

```jsonl
{"question": "How many users?", "generated_query": "SELECT COUNT(*) FROM users"}
{"question": "List active orders", "generated_query": "SELECT * FROM orders WHERE active = 1"}
```

Output:

```
Progress: 2/2 entries stored.
Warmed 2 entries into 'default'.
```

---

### `medha expire`

Delete all entries whose TTL has elapsed.

```bash
medha expire
medha expire --collection my_cache
```

```
Expired 14 entries from 'my_cache'.
```

Use this with a scheduler (cron, APScheduler) if `enable_background_cleanup` is disabled or if you need immediate cleanup from outside the running process.

---

### `medha dedup`

Remove entries sharing the same `query_hash` (derived from the generated query string), keeping the most-recently stored entry per hash.

```bash
medha dedup
medha dedup --collection my_cache
```

```
Removed 3 duplicate entries from 'my_cache'.
```

Requires `pandas` (`pip install pandas`).

---

### `medha invalidate QUESTION`

Remove the entry whose normalised question matches the argument.

```bash
medha invalidate "How many users are registered?"
medha invalidate "How many users are registered?" --collection my_cache
```

```
Removed.
```

Prints `Not found.` if no entry matches. Uses a plain text lookup — no embedder required.

---

### `medha export`

Dump all entries in a collection to CSV (default) or JSON.

```bash
medha export                                        # CSV to stdout
medha export --format csv --output cache.csv        # CSV to file
medha export --format json --output cache.json      # JSON records
medha export --collection my_cache --format csv
```

Requires `pandas` (`pip install pandas`).

---

### `medha feedback QUESTION`

Record a correct or incorrect signal for a cached entry.

```bash
# Mark as correct
medha feedback "How many users are registered?" --correct

# Mark as incorrect
medha feedback "How many users are registered?" --no-correct
```

```
Feedback recorded.
```

Prints `Not found.` if no entry matches. Uses a plain text lookup — no embedder required.

See [Feedback Loop](feedback.md) for the full auto-invalidation behaviour.

---

### `medha logo`

Print the Medha lotus logo.

```bash
medha logo
```

---

## Global Options

All commands accept:

| Option | Description |
|---|---|
| `--collection TEXT` | Override the collection name (default: `MEDHA_COLLECTION` or `default`) |
| `--help` | Show help for any command |

---

## See Also

- [Feedback Loop](feedback.md) — `feedback_incorrect_threshold` and auto-invalidation
- [Batch Operations](batch_operations.md) — Python API equivalent of `warm` and `export`
- [Configuration](configuration.md) — full `MEDHA_*` variable reference
- [Demo 26 — CLI](https://github.com/ArchAI-Labs/medha/blob/main/demo/26_cli.ipynb)
