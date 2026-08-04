# Spec 12 — CLI (v0.4.0)

## Goal

Add a `medha` command-line tool for operational management of a running cache:
inspect entry counts, warm the cache from a file, invalidate entries, export
data, deduplicate, and purge expired entries — all without writing Python code.

The CLI is a thin async wrapper over the existing `Medha` public API.
It introduces no new core functionality and makes no changes to `core.py`,
`types.py`, or any backend.

---

## Package changes

### New optional extra — `[cli]`

```toml
# pyproject.toml
cli = ["typer>=0.12,<1", "rich>=13,<14"]
```

`rich` is listed separately so it does not bloat the core install; Typer uses
it when available for coloured output and tables.

Update the `all` and `all-no-chroma` meta-groups to include `cli`.

### Console scripts entrypoint

```toml
[project.scripts]
medha = "medha.cli:app"
```

### New module layout

```
src/medha/cli/
├── __init__.py       # exports app
├── _app.py           # Typer app, all commands
└── _noop_embedder.py # _NoOpEmbedder (private, CLI-only)
```

---

## `Settings` changes (config.py)

Three new fields. These are the only changes to an existing source file outside
`cli/`. All are additive and backward compatible.

```python
# Embedder selection (used by CLI factory only; Medha.__init__ is unchanged)
embedder_type: Literal["fastembed", "openai", "cohere", "gemini", "_noop"] = Field(
    default="_noop",
    description="Embedder to use. '_noop' is the default; real embedders require the matching extra.",
)

# Default collection name (used by CLI when --collection is not passed)
collection: str = Field(
    default="default",
    description="Default collection name for CLI commands. Set via MEDHA_COLLECTION.",
)

# FastEmbed model name (used by CLI when embedder_type='fastembed')
fastembed_model: str = Field(
    default="BAAI/bge-small-en-v1.5",
    description="FastEmbed model identifier. Set via MEDHA_FASTEMBED_MODEL.",
)
```

API keys for OpenAI, Cohere, and Gemini are **not** added to `Settings`.
`_resolve_embedder` reads the provider's standard env vars directly
(`OPENAI_API_KEY`, `COHERE_API_KEY`, `GOOGLE_API_KEY`) via `os.environ`.
This avoids duplicating secrets under a second name with the `MEDHA_` prefix
and is consistent with how those SDKs already work.

---

## `_NoOpEmbedder`

Lives in `src/medha/cli/_noop_embedder.py`. Not exported from `medha.__init__`.

```python
class _NoOpEmbedder(BaseEmbedder):
    """Placeholder embedder for CLI commands that do not embed text."""

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return "_noop"

    async def aembed(self, text: str) -> list[float]:
        raise RuntimeError(
            "This command requires a real embedder. "
            "Set MEDHA_EMBEDDER_TYPE (e.g. fastembed) and install the "
            "corresponding extra: pip install 'medha-archai[fastembed]'."
        )

    async def aembed_batch(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        raise RuntimeError(
            "This command requires a real embedder. "
            "Set MEDHA_EMBEDDER_TYPE (e.g. fastembed) and install the "
            "corresponding extra: pip install 'medha-archai[fastembed]'."
        )
```

### The `dimension=384` default and `start()` safety

`Medha.start()` calls `backend.initialize(collection, embedder.dimension)`.
For **existing** collections all built-in backends are idempotent on
`initialize()` — they skip creation if the collection already exists, regardless
of the dimension argument. Passing `dimension=384` from `_NoOpEmbedder` is
therefore safe for every admin command that targets an existing collection.

**Risk for non-existent collections**: if an admin command is run against a
collection that does not exist yet, `initialize()` will create it with
`dimension=384`. Any subsequent `store()` with a real embedder of a different
dimension will fail at the backend level. Mitigation: document this in the CLI
help text; no code guard needed in v0.4.0 (the scenario requires a user error).

---

## `stats` command — design constraint

`CacheStats` (hit rate, latency percentiles, per-strategy breakdown) is an
**in-process, non-persistent** accumulator on the `Medha` instance. A fresh CLI
invocation creates a new instance with zero stats — the performance metrics are
meaningless.

The `medha stats` command therefore reports **structural info** only, obtained
directly from the backend:

- Entry count in the main collection (`backend.count()`)
- Entry count in the template collection
- Backend type and collection name

This is intentional and must be documented in the command's help string.
Persistent stats (e.g. a dedicated stats collection in the backend) are out of
scope for v0.4.0.

---

## Commands

All commands accept:

| Option | Env var | Default | Description |
|--------|---------|---------|-------------|
| `--collection` | `MEDHA_COLLECTION` | `"default"` | Collection name |
| `--backend-type` | `MEDHA_BACKEND_TYPE` | `"memory"` | Backend type |

Backend connection options (`--qdrant-host`, `--pg-dsn`, etc.) are read from
`MEDHA_*` env vars via `Settings` — no extra CLI flags needed.

### `medha stats`

```
medha stats [--collection NAME]
```

Prints: collection name, backend type, entry count (main + template collection).
Does NOT print hit rate or latency (in-process stats, not available from CLI).

### `medha warm`

```
medha warm FILE [--collection NAME] [--ttl SECONDS] [--batch-size N]
```

- Requires a real embedder: fails early with a helpful message if
  `MEDHA_EMBEDDER_TYPE=_noop` (default).
- Accepts `.json` and `.jsonl` files (delegates to `Medha.warm_from_file()`).
- Prints a progress line every `--batch-size` entries.

### `medha invalidate`

```
medha invalidate QUESTION [--collection NAME]
```

Calls `Medha.invalidate(question)`. Prints "Removed" or "Not found".

### `medha invalidate-collection`

```
medha invalidate-collection [--collection NAME] [--yes]
```

Calls `Medha.invalidate_collection()`. Requires `--yes` to confirm (destructive).

### `medha expire`

```
medha expire [--collection NAME]
```

Calls `Medha.expire()`. Prints count of deleted entries.

### `medha dedup`

```
medha dedup [--collection NAME]
```

Calls `Medha.dedup_collection()`. Prints count of removed duplicates.
Requires `pandas` to be installed (same dependency as `export_to_dataframe()`);
prints an actionable error if missing.

### `medha export`

```
medha export [--collection NAME] [--output FILE] [--format csv|json]
```

Calls `Medha.export_to_dataframe()` then writes to file (or stdout if no
`--output`). Default format: `csv`.
Requires `pandas`; prints actionable error if missing.

### `medha feedback`

```
medha feedback QUESTION --correct | --incorrect [--collection NAME]
```

Calls `Medha.feedback(question, correct)`.
Prints `"Feedback recorded."` or `"Entry not found."` (when `feedback()` returns `False`).

Requires a real embedder only if `search_by_normalized_question()` triggers
embedding — which it does not (it is a plain text lookup). The `_NoOpEmbedder`
is therefore sufficient: this command works without setting `MEDHA_EMBEDDER_TYPE`.

`--correct` and `--incorrect` are mutually exclusive; Typer enforces this via
`typer.Option` with a shared callback. Omitting both exits with a usage error.

---

## Internal CLI factory

`_app.py` contains a shared async helper used by all commands:

```python
async def _build_medha(collection: str, settings: Settings) -> Medha:
    """Instantiate and start a Medha instance for the given collection."""
    embedder = _resolve_embedder(settings)
    m = Medha(
        collection_name=collection,
        embedder=embedder,
        settings=settings,
    )
    await m.start()
    return m
```

`collection` is always passed explicitly by the calling command (from the
`--collection` Typer option, which defaults to `settings.collection`).

`_resolve_embedder()` maps `settings.embedder_type` to the concrete adapter.
API keys for cloud embedders are read from the provider's standard env vars
via `os.environ` — not from `Settings` — to avoid renaming secrets:

```python
import os

def _resolve_embedder(settings: Settings) -> BaseEmbedder:
    et = settings.embedder_type
    if et == "_noop":
        from medha.cli._noop_embedder import _NoOpEmbedder
        return _NoOpEmbedder()
    if et == "fastembed":
        from medha.embeddings.fastembed_adapter import FastEmbedAdapter
        return FastEmbedAdapter(model_name=settings.fastembed_model)
    if et == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ConfigurationError("OPENAI_API_KEY env var is not set.")
        from medha.embeddings.openai_adapter import OpenAIAdapter
        return OpenAIAdapter(api_key=api_key)
    if et == "cohere":
        api_key = os.environ.get("COHERE_API_KEY", "")
        if not api_key:
            raise ConfigurationError("COHERE_API_KEY env var is not set.")
        from medha.embeddings.cohere_adapter import CohereAdapter
        return CohereAdapter(api_key=api_key)
    if et == "gemini":
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise ConfigurationError("GOOGLE_API_KEY env var is not set.")
        from medha.embeddings.gemini_adapter import GeminiAdapter
        return GeminiAdapter(api_key=api_key)
    raise ConfigurationError(f"Unknown embedder_type: '{et}'")
```

Typer commands are synchronous; each wraps the async logic with `asyncio.run()`.
This is safe for CLI use (no running event loop in a terminal process).

---

## Bugs and regression risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| `_NoOpEmbedder.aembed()` called by `warm` raises `RuntimeError` with no Typer formatting | Medium | Catch `RuntimeError` in the `warm` command and re-raise as `typer.BadParameter` with install hint |
| `_NoOpEmbedder` creates collection with wrong dimension if collection doesn't exist | Low | Documented in help text; no silent data corruption possible (backend schema error stops subsequent stores) |
| `medha stats` confuses users expecting hit-rate metrics | Medium | Explicit help string: "structural stats only; performance metrics require an in-process instance" |
| `medha dedup` and `medha export` fail if pandas not installed without clear message | Medium | Import-guard with `typer.BadParameter` pointing to `pip install pandas` |
| `asyncio.run()` inside Typer callback works fine in terminal; fails inside Jupyter or async test | Low | CLI is a terminal tool; document in test section that CLI tests must use `CliRunner`, not `asyncio` directly |
| `medha invalidate-collection` without `--yes` is a no-op but does not warn | Low | Print "use --yes to confirm" if flag is missing |
| Concurrency: multiple CLI processes targeting the same collection simultaneously | Low | Same risk as any client; backends handle concurrent writes via their own locking |
| Three new `Settings` fields (`embedder_type`, `collection`, `fastembed_model`) — existing env vars unaffected; defaults never break current `Medha(embedder=...)` call sites | None | Fields are read only by CLI factory |
| `medha feedback` calls `search_by_normalized_question()` which is a plain text lookup — no embedding needed, `_NoOpEmbedder` is safe | None | By design; documented in command help |
| `_resolve_embedder` raises `ConfigurationError` (not a Typer error) on missing API key — must be caught and re-raised as `typer.Exit(1)` with a readable message | Medium | Wrap `_resolve_embedder` call in each command with a `try/except ConfigurationError` |

---

## Tests

### Unit — `tests/unit/test_cli.py`

Uses `typer.testing.CliRunner` (sync, no `asyncio.run` in tests).
All backend operations mocked via `unittest.mock.AsyncMock`.

```
TestCliStats
  test_stats_prints_collection_and_count
  test_stats_unknown_backend_exits_nonzero

TestCliInvalidate
  test_invalidate_found_prints_removed
  test_invalidate_not_found_prints_not_found

TestCliInvalidateCollection
  test_invalidate_collection_requires_yes_flag
  test_invalidate_collection_with_yes_succeeds

TestCliExpire
  test_expire_prints_deleted_count
  test_expire_zero_deleted

TestCliDedup
  test_dedup_prints_removed_count
  test_dedup_missing_pandas_prints_actionable_error

TestCliExport
  test_export_csv_to_stdout
  test_export_json_to_file
  test_export_missing_pandas_prints_actionable_error

TestCliWarm
  test_warm_with_noop_embedder_prints_helpful_error
  test_warm_with_real_embedder_succeeds   (uses MockEmbedder via env var override)

TestNoOpEmbedder
  test_noop_embedder_dimension_property
  test_noop_embedder_aembed_raises_runtime_error
  test_noop_embedder_aembed_batch_raises_runtime_error

TestResolveEmbedder
  test_resolve_noop_returns_noop_embedder
  test_resolve_unknown_raises_configuration_error
  test_resolve_fastembed_returns_fastembed_adapter  (importorskip if not installed)
  test_resolve_openai_missing_api_key_raises_configuration_error
  test_resolve_cohere_missing_api_key_raises_configuration_error
  test_resolve_gemini_missing_api_key_raises_configuration_error

TestCliSettings
  test_settings_collection_default_is_default
  test_settings_collection_from_env_var            (MEDHA_COLLECTION=my_cache)
  test_settings_embedder_type_default_is_noop
  test_settings_fastembed_model_default

TestCliFeedback
  test_feedback_correct_prints_recorded
  test_feedback_incorrect_prints_recorded
  test_feedback_not_found_prints_not_found
  test_feedback_requires_correct_or_incorrect_flag
  test_feedback_correct_and_incorrect_mutually_exclusive
  test_feedback_works_with_noop_embedder           — no real embedder needed
```

### Integration — `tests/integration/test_cli_e2e.py`

End-to-end via `CliRunner` against a real `InMemoryBackend` (no mocking).
Requires `medha-archai[cli]` installed.

```
test_cli_stats_e2e              — start Medha, store 3 entries, run `medha stats`, verify count=3
test_cli_invalidate_e2e         — store entry, run `medha invalidate "question"`, verify count=0
test_cli_expire_e2e             — store entry with ttl=-1 (already expired), run expire, verify count=0
test_cli_warm_e2e               — create JSONL file, run `medha warm file.jsonl`, verify entries stored
test_cli_export_csv_e2e         — store 2 entries, export to CSV, verify file content
test_cli_feedback_e2e           — store entry, run `medha feedback "question" --incorrect`, verify counter incremented
```

---

## Demo

### `demo/26_cli.ipynb`

Sections:

1. **Installation** — `pip install "medha-archai[cli,fastembed]"`.
2. **Verify install** — `!medha --help` to show available commands.
3. **Prepare data** — store 5 entries programmatically into a local
   `InMemoryBackend` instance and save the collection to a JSONL file.
4. **`medha stats`** — show structural info.
5. **`medha warm`** — load the JSONL file; show entry count before and after.
6. **`medha expire` / `medha dedup`** — add entries with past TTL, run expire,
   add duplicates, run dedup.
7. **`medha invalidate`** — remove a specific entry by question text.
8. **`medha export`** — export to CSV and display with pandas.
9. **Environment variable table** — all `MEDHA_*` vars the CLI respects.

---

## CHANGELOG entry (0.4.0)

### Added
- `medha` CLI — install with `pip install "medha-archai[cli]"`.
  Commands: `stats`, `warm`, `invalidate`, `invalidate-collection`, `expire`,
  `dedup`, `export`, `feedback`.
- `Settings.embedder_type` — declarative embedder selection via
  `MEDHA_EMBEDDER_TYPE` env var. Accepted values: `fastembed`, `openai`,
  `cohere`, `gemini`, `_noop` (default). Used by the CLI factory; no change to
  the `Medha(embedder=...)` call signature.
- `Settings.collection` — default collection name for CLI commands via
  `MEDHA_COLLECTION` env var (default: `"default"`).
- `Settings.fastembed_model` — FastEmbed model name via `MEDHA_FASTEMBED_MODEL`
  env var (default: `"BAAI/bge-small-en-v1.5"`).
- `_NoOpEmbedder` (private, `medha.cli._noop_embedder`) — placeholder embedder
  used by CLI admin commands that do not perform vector search or storage.
  `medha feedback` works with `_NoOpEmbedder` because it uses a plain text
  lookup, not vector search.

### Notes
- `medha stats` reports structural information only (entry count). In-process
  performance metrics (hit rate, latency percentiles) are not available from the
  CLI because `CacheStats` is a non-persistent in-memory accumulator.
