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
