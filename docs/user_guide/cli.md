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
