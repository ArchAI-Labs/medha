# Playbook — Medha 0.4.0

Implementation guide for the two features specified in:
- `spec/11_feedback_loop.md`
- `spec/12_cli.md`

Each step is a self-contained prompt ready to give to Claude Code.
Run them in order; later steps assume earlier steps are complete.
After each step run `pytest tests/unit/ -x -q` unless the step only adds a
notebook or updates docs.

---

## Phase 1 — Feedback Loop

---

### Step 1.1 — Data models

**Files:** `src/medha/types.py`

```
Read spec/11_feedback_loop.md, section "Data model changes".

Add two new fields to CacheEntry in src/medha/types.py:
  feedback_correct:   int = Field(default=0, ge=0)
  feedback_incorrect: int = Field(default=0, ge=0)

Add the same two fields (same names, same defaults, no ge constraint needed) to
CacheResult in the same file.

Do not change any other field or method. Do not add comments.

Validate: run pytest tests/unit/test_types.py -x -q and confirm all tests pass.
Existing tests must not break — the new fields have defaults and are backward
compatible.
```

---

### Step 1.2 — Storage interface

**Files:** `src/medha/interfaces/storage.py`

```
Read spec/11_feedback_loop.md, section "Interface change — VectorStorageBackend".

Add the following abstract method to VectorStorageBackend in
src/medha/interfaces/storage.py, after the existing drop_collection() method
and before close():

  @abstractmethod
  async def update_feedback(
      self,
      collection_name: str,
      point_id: str,
      correct: bool,
  ) -> int:
      """Increment feedback_correct or feedback_incorrect for a stored entry.

      Args:
          collection_name: Target collection.
          point_id:        ID of the entry to update.
          correct:         True → increment feedback_correct;
                           False → increment feedback_incorrect.

      Returns:
          The new value of the incremented counter after the update.
          Returns 0 if the entry is not found (no exception raised).

      Raises:
          StorageError: If the update fails.
      """

Do not change any other method.

Validate: run python -c "from medha.interfaces.storage import VectorStorageBackend"
and confirm no import error. Then run pytest tests/unit/test_storage_interface.py -x -q
and confirm only the "partial implementation" test fails with TypeError mentioning
update_feedback (expected — the concrete backends don't implement it yet).
```

---

### Step 1.3 — Settings

**Files:** `src/medha/config.py`

```
Read spec/11_feedback_loop.md, section "Settings change".

Add one new field to the Settings class in src/medha/config.py, in the
"Cache lifecycle" section (after cleanup_interval_seconds):

  feedback_incorrect_threshold: int | None = Field(
      default=None,
      ge=1,
      description=(
          "When set, feedback(correct=False) automatically invalidates the entry "
          "once feedback_incorrect reaches this value. "
          "None disables auto-invalidation. Env var: MEDHA_FEEDBACK_INCORRECT_THRESHOLD."
      ),
  )

Do not change any other field or validator.

Validate: run pytest tests/unit/test_config.py -x -q — all tests must pass.
Also confirm: python -c "from medha.config import Settings; s = Settings(); print(s.feedback_incorrect_threshold)"
prints None.
```

---

### Step 1.4 — InMemoryBackend

**Files:** `src/medha/backends/memory.py`

```
Read spec/11_feedback_loop.md, section "Implementation per backend", row
InMemoryBackend.

Implement update_feedback() on InMemoryBackend in src/medha/backends/memory.py.

The method must:
1. Acquire self._lock.
2. Look up the entry by point_id in self._store[collection_name]["entries"].
3. If not found, log a warning (same style as update_usage_count) and return 0.
4. Increment payload["feedback_correct"] if correct=True, else payload["feedback_incorrect"].
   Both keys may be absent in older entries — treat missing as 0.
5. Return the new value of the incremented counter as int.

Place the method immediately after update_usage_count().

Also update _point_to_cache_result() at the bottom of the file to map
payload.get("feedback_correct", 0) and payload.get("feedback_incorrect", 0)
to the new CacheResult fields.

Also update upsert() to serialise the new CacheEntry fields:
  "feedback_correct":   entry.feedback_correct,
  "feedback_incorrect": entry.feedback_incorrect,

Validate: run pytest tests/unit/test_inmemory_backend.py -x -q — all must pass.
```

---

### Step 1.5 — PostgreSQL backends (pgvector + vectorchord)

**Files:** `src/medha/backends/_asyncpg_mixin.py`

```
Read spec/11_feedback_loop.md, section "Implementation per backend", rows
PgVectorBackend and VectorChordBackend.

Implement update_feedback() on _AsyncpgMixin in
src/medha/backends/_asyncpg_mixin.py, immediately after update_usage_count().

The method must:
1. Check self._pool is not None, else raise StorageError("Not connected...").
2. Build the column name: col = "feedback_correct" if correct else "feedback_incorrect".
3. Execute:
     UPDATE {schema}.{table}
     SET {col} = COALESCE({col}, 0) + 1
     WHERE id = $1::uuid
     RETURNING {col}
4. If the UPDATE returns a row, return the value of the returned column as int.
5. If no row returned (id not found), log a warning and return 0.

Use RETURNING to get the new value atomically without a second SELECT.

The method signature must be:
  async def update_feedback(self, collection_name: str, point_id: str, correct: bool) -> int:

The existing _table_name() and pg_schema from settings are already available
in the mixin — use them as in update_usage_count().

Validate: run pytest tests/unit/test_pgvector_backend.py tests/unit/test_vectorchord_backend.py -x -q
```

---

### Step 1.6 — Redis backend

**Files:** `src/medha/backends/redis_vector.py`

```
Read spec/11_feedback_loop.md, section "Implementation per backend", row
RedisVectorBackend.

Implement update_feedback() on RedisVectorBackend in
src/medha/backends/redis_vector.py, immediately after update_usage_count().

The method must:
1. Check self._client is not None.
2. Build the Redis hash key as in update_usage_count():
     col_key = _key_prefix(self._settings.redis_key_prefix, collection_name)
     key = f"{col_key}:{point_id}"
3. field = "feedback_correct" if correct else "feedback_incorrect"
4. Check the key exists (hexists on "original_question") — if not, return 0.
5. Use self._client.hincrby(key, field, 1) which returns the new value atomically.
6. Return the new value as int.

Signature:
  async def update_feedback(self, collection_name: str, id_: str, correct: bool) -> int:

Note: Redis uses id_ (not point_id) for consistency with the rest of this file.
The abstract interface uses point_id; the override is fine.

Validate: run pytest tests/unit/test_redis_vector_backend.py -x -q
```

---

### Step 1.7 — Elasticsearch backend

**Files:** `src/medha/backends/elasticsearch.py`

```
Read spec/11_feedback_loop.md, section "Implementation per backend", row
ElasticsearchBackend.

Implement update_feedback() on ElasticsearchBackend in
src/medha/backends/elasticsearch.py, immediately after update_usage_count().

Use the Elasticsearch Update API with a Painless inline script to increment
the counter atomically:

  script = {
      "source": (
          "if (ctx._source.containsKey(params.field)) "
          "{ ctx._source[params.field]++; } "
          "else { ctx._source[params.field] = 1; }"
      ),
      "params": {"field": "feedback_correct" if correct else "feedback_incorrect"},
  }
  resp = await self._client.update(
      index=index_name,
      id=point_id,
      body={"script": script},
      ignore=[404],
  )

If the document was not found (resp.get("result") == "not_found" or status 404),
return 0. Otherwise fetch the updated value with a follow-up get() call and
return it as int.

Signature:
  async def update_feedback(self, collection_name: str, point_id: str, correct: bool) -> int:

Use _index_name() helper (already present in the file) to build the index name.

Validate: run pytest tests/unit/test_elasticsearch_backend.py -x -q
```

---

### Step 1.8 — Qdrant backend

**Files:** `src/medha/backends/qdrant.py`

```
Read spec/11_feedback_loop.md, section "Implementation per backend", row
QdrantBackend.

Implement update_feedback() on QdrantBackend in src/medha/backends/qdrant.py,
immediately after update_usage_count().

Pattern is the same as update_usage_count() — read-modify-write via
client.retrieve() + client.set_payload():

1. Retrieve the point: await self.client.retrieve(collection_name, ids=[point_id], with_payload=True)
2. If empty, log warning and return 0.
3. payload = points[0].payload or {}
4. field = "feedback_correct" if correct else "feedback_incorrect"
5. new_val = payload.get(field, 0) + 1
6. await self.client.set_payload(collection_name, payload={field: new_val}, points=[point_id], wait=True)
7. Return new_val.

Wrap in try/except as in update_usage_count().

Signature:
  async def update_feedback(self, collection_name: str, point_id: str, correct: bool) -> int:

Validate: run pytest tests/unit/ -x -q -k "qdrant"
```

---

### Step 1.9 — Chroma and Weaviate backends

**Files:** `src/medha/backends/chroma.py`, `src/medha/backends/weaviate.py`

```
Read spec/11_feedback_loop.md, section "Implementation per backend", rows
ChromaBackend and WeaviateBackend.

Implement update_feedback() on both backends using read-modify-write.

--- ChromaBackend (src/medha/backends/chroma.py) ---
Place after update_usage_count().

1. Get the collection handle.
2. result = collection.get(ids=[id_], include=["metadatas"])
3. If not result["ids"], return 0.
4. metadata = result["metadatas"][0] or {}
5. field = "feedback_correct" if correct else "feedback_incorrect"
6. new_val = int(metadata.get(field, 0)) + 1
7. collection.update(ids=[id_], metadatas=[{**metadata, field: new_val}])
8. Return new_val.

Signature: async def update_feedback(self, collection_name: str, id_: str, correct: bool) -> int:

--- WeaviateBackend (src/medha/backends/weaviate.py) ---
Place after update_usage_count().

1. class_name = _class_name(self._settings.weaviate_collection_prefix, collection_name)
2. Use async Weaviate client to fetch the object (with_id=id_).
3. If not found, return 0.
4. field = "feedback_correct" if correct else "feedback_incorrect"
5. current = int(obj.properties.get(field, 0))
6. Update the object with {field: current + 1}.
7. Return current + 1.

Signature: async def update_feedback(self, collection_name: str, id_: str, correct: bool) -> int:

Validate: run pytest tests/unit/test_chroma_backend.py tests/unit/test_weaviate_backend.py -x -q
```

---

### Step 1.10 — Azure Search and LanceDB backends

**Files:** `src/medha/backends/azure_search.py`, `src/medha/backends/lancedb.py`

```
Read spec/11_feedback_loop.md, section "Implementation per backend", rows
AzureSearchBackend and LanceDBBackend.

--- AzureSearchBackend (src/medha/backends/azure_search.py) ---
Place after update_usage_count().

1. index_name = _index_name(self._settings.azure_search_index_name, collection_name)
2. Fetch the document: result = await client.get_document(key=point_id).
3. If not found (404), return 0.
4. field = "feedback_correct" if correct else "feedback_incorrect"
5. new_val = int(result.get(field, 0)) + 1
6. Merge document: await client.merge_or_upload_documents([{"id": point_id, field: new_val}])
7. Return new_val.

Signature: async def update_feedback(self, collection_name: str, point_id: str, correct: bool) -> int:

--- LanceDBBackend (src/medha/backends/lancedb.py) ---
Place after update_usage_count() (or after close() if update_usage_count is
not present).

1. table = await self._conn.open_table(_table_name(self._settings.lancedb_table_prefix, collection_name))
2. rows = await table.search().where(f"id = '{point_id}'").limit(1).to_pandas()
3. If rows is empty, return 0.
4. field = "feedback_correct" if correct else "feedback_incorrect"
5. current = int(rows.iloc[0].get(field, 0)) if field in rows.columns else 0
6. new_val = current + 1
7. await table.update(where=f"id = '{point_id}'", values={field: new_val})
8. Return new_val.

Signature: async def update_feedback(self, collection_name: str, point_id: str, correct: bool) -> int:

Validate: run pytest tests/unit/test_azure_search_backend.py tests/unit/test_lancedb_backend.py -x -q
```

---

### Step 1.11 — Core: `Medha.feedback()`

**Files:** `src/medha/core.py`

```
Read spec/11_feedback_loop.md, sections "API" and "core.py implementation".

Add the feedback() method to the Medha class in src/medha/core.py.
Place it after the invalidate_collection() method (around line 996) and before
the template management methods.

The method must implement this flow exactly:

  async def feedback(self, question: str, correct: bool) -> bool:
      1. normalized = normalize_question(question)
      2. result = await self._backend.search_by_normalized_question(
             self._collection_name, normalized
         )
      3. If result is None:
             log warning: "feedback: no entry found for '%s'", question[:50]
             return False
      4. new_count = await self._backend.update_feedback(
             self._collection_name, result.id, correct
         )
      5. If not correct and self._settings.feedback_incorrect_threshold is not None
             and new_count >= self._settings.feedback_incorrect_threshold:
             await self.invalidate(question)
             log info: "Auto-invalidated '%s' after %d incorrect feedbacks",
                       question[:50], new_count
      6. return True

Also add a sync wrapper after the other sync wrappers (near line 1660):

  def feedback_sync(self, question: str, correct: bool) -> bool:
      return self._run_sync(self.feedback(question, correct))

Do not change any other method.

Validate: run pytest tests/unit/test_core_waterfall.py tests/unit/test_invalidation.py -x -q
All existing tests must pass.
```

---

### Step 1.12 — Unit tests: feedback

**Files:** `tests/unit/test_feedback.py`, `tests/unit/test_storage_interface.py`

```
Read spec/11_feedback_loop.md, section "Tests".

1. Create tests/unit/test_feedback.py implementing all cases listed in that
   section under TestFeedbackTypes, TestInMemoryBackendUpdateFeedback, 
   TestMedhaFeedback, and TestMedhaFeedbackAutoInvalidation.

   Use the existing conftest.py fixtures: mock_embedder, medha_memory,
   inmemory_backend. The medha_memory fixture uses InMemoryBackend +
   MockEmbedder and is the right base for the Medha-level tests.

   For TestMedhaFeedbackAutoInvalidation, create a local fixture that sets
   feedback_incorrect_threshold in Settings, e.g.:

     @pytest.fixture
     async def medha_threshold(mock_embedder):
         from medha.backends.memory import InMemoryBackend
         from medha.core import Medha
         settings = Settings(
             backend_type="memory",
             score_threshold_exact=0.99,
             score_threshold_semantic=0.85,
             feedback_incorrect_threshold=3,
         )
         m = Medha("fb_threshold", mock_embedder, InMemoryBackend(), settings)
         await m.start()
         yield m
         await m.close()

   For TestFeedbackSettings, instantiate Settings directly and check field
   values and env var parsing.

2. Add three new test methods to TestBackendContract in
   tests/unit/test_storage_interface.py (see spec section "Contract test
   addition"). These must be inside the existing class and use the existing
   any_backend and make_entry_fixture fixtures.

Run: pytest tests/unit/test_feedback.py tests/unit/test_storage_interface.py -x -q
All new tests must pass. All pre-existing tests in test_storage_interface.py
must also still pass.
```

---

### Step 1.13 — Integration tests: feedback

**Files:** `tests/integration/test_feedback_e2e.py`

```
Read spec/11_feedback_loop.md, section "Integration — tests/integration/test_feedback_e2e.py".

Create tests/integration/test_feedback_e2e.py implementing all six test cases
listed there. Use the medha_memory fixture from conftest.py.

For test_auto_invalidation_with_l1:
- Call search() first to populate L1, then call feedback(correct=False) the
  required number of times, then call search() again and assert NO_MATCH —
  this confirms L1 was cleared by invalidate() inside feedback().

Run: pytest tests/integration/test_feedback_e2e.py -x -q
All six tests must pass.
```

---

### Step 1.14 — Demo notebook: feedback loop

**Files:** `demo/25_feedback_loop.ipynb`

```
Read spec/11_feedback_loop.md, section "Demo — demo/25_feedback_loop.ipynb".

Create demo/25_feedback_loop.ipynb with the six sections described.

Use InMemoryBackend + FastEmbedAdapter(model_name="BAAI/bge-small-en-v1.5").
The notebook must be fully self-contained and runnable with only
pip install "medha-archai[fastembed]".

Section structure:
1. Setup — imports, embedder, settings (no threshold), seed 5 pairs
2. Manual feedback mode — 5 searches, record correct/incorrect, show
   feedback counters in a DataFrame via export_to_dataframe()
3. Auto-invalidation mode — restart Medha with feedback_incorrect_threshold=3,
   simulate 3 incorrect feedbacks on one entry, confirm NO_MATCH
4. Mixed session — 10 searches with mixed feedback, show final state in DataFrame
5. Production notes — race condition caveat (Qdrant/Chroma/Weaviate/Azure/LanceDB),
   recommended threshold values (3-5), note that feedback_correct acting
   (e.g. score boost) is a future feature, mention feedback_sync() for sync contexts

Each section must have a markdown cell explaining what it demonstrates.
Do not use real LLM calls — use a mock_llm dict as in demo/13_framework_integrations.ipynb.
```

---

## Phase 2 — CLI

---

### Step 2.1 — Settings additions (CLI fields)

**Files:** `src/medha/config.py`

```
Read spec/12_cli.md, section "Settings changes (config.py)".

Add three new fields to the Settings class in src/medha/config.py.
Place them in a new comment block "# --- CLI ---" after the "# --- Timeouts ---"
section, just before the validators:

  embedder_type: Literal["fastembed", "openai", "cohere", "gemini", "_noop"] = Field(
      default="_noop",
      description=(
          "Embedder to use. '_noop' is the default (no embedding). "
          "Real embedders require the matching extra. "
          "Env var: MEDHA_EMBEDDER_TYPE."
      ),
  )

  collection: str = Field(
      default="default",
      description="Default collection name for CLI commands. Env var: MEDHA_COLLECTION.",
  )

  fastembed_model: str = Field(
      default="BAAI/bge-small-en-v1.5",
      description="FastEmbed model identifier used by the CLI. Env var: MEDHA_FASTEMBED_MODEL.",
  )

Do not change any existing field or validator.

Validate: run pytest tests/unit/test_config.py -x -q — all must pass.
Also confirm:
  python -c "from medha.config import Settings; s = Settings(); print(s.embedder_type, s.collection, s.fastembed_model)"
prints: _noop default BAAI/bge-small-en-v1.5
```

---

### Step 2.2 — CLI module: `_NoOpEmbedder` and package init

**Files:** `src/medha/cli/__init__.py`, `src/medha/cli/_noop_embedder.py`

```
Read spec/12_cli.md, sections "_NoOpEmbedder" and "New module layout".

1. Create src/medha/cli/__init__.py:
   from medha.cli._app import app
   __all__ = ["app"]

2. Create src/medha/cli/_noop_embedder.py with the _NoOpEmbedder class exactly
   as specified in the spec. Key points:
   - Inherits from BaseEmbedder
   - dimension property returns self._dimension (default 384)
   - model_name property returns "_noop"
   - aembed() raises RuntimeError with the install hint message
   - aembed_batch() raises RuntimeError with the same message
   - Class is NOT exported from medha.__init__ — it is private to the cli module

Do not create _app.py yet (that is the next step).

Validate:
  python -c "
  from medha.cli._noop_embedder import _NoOpEmbedder
  import asyncio
  e = _NoOpEmbedder()
  print(e.dimension, e.model_name)
  try:
      asyncio.run(e.aembed('test'))
  except RuntimeError as exc:
      print('OK:', str(exc)[:60])
  "
```

---

### Step 2.3 — CLI app

**Files:** `src/medha/cli/_app.py`

```
Read spec/12_cli.md, sections "Commands" and "Internal CLI factory".

Create src/medha/cli/_app.py implementing all eight commands as a Typer app.

Shared infrastructure in the file:
- _resolve_embedder(settings) function exactly as in the spec (reads OPENAI_API_KEY,
  COHERE_API_KEY, GOOGLE_API_KEY from os.environ for cloud embedders)
- _build_medha(collection, settings) async helper
- Each command wraps its async body with asyncio.run()
- ConfigurationError and RuntimeError from _resolve_embedder must be caught
  and re-raised as typer.Exit(code=1) after printing a readable error with
  typer.echo(..., err=True)

Commands to implement (signatures and behaviour as in spec):

  app = typer.Typer(help="Medha cache management CLI.")

  @app.command()
  def stats(collection, ...):
      # Reports: collection name, backend type, entry count (main + templates)
      # Does NOT report hit rate or latency — documents this in help string

  @app.command()
  def warm(file, collection, ttl, batch_size, ...):
      # Fails early if embedder_type == "_noop" with install hint
      # Delegates to Medha.warm_from_file()

  @app.command()
  def invalidate(question, collection, ...):
      # Calls Medha.invalidate(); prints "Removed." or "Not found."

  @app.command("invalidate-collection")
  def invalidate_collection(collection, yes, ...):
      # Requires --yes; prints warning and exits if omitted
      # Calls Medha.invalidate_collection()

  @app.command()
  def expire(collection, ...):
      # Calls Medha.expire(); prints "Deleted N expired entries."

  @app.command()
  def dedup(collection, ...):
      # Catches ImportError for pandas with actionable error
      # Calls Medha.dedup_collection()

  @app.command()
  def export(collection, output, format_, ...):
      # Catches ImportError for pandas with actionable error
      # Calls Medha.export_to_dataframe(); writes csv or json

  @app.command()
  def feedback(question, collection, correct, incorrect, ...):
      # --correct and --incorrect are mutually exclusive flags (bool options)
      # If neither is passed, exit with usage error
      # Calls Medha.feedback(); prints "Feedback recorded." or "Entry not found."
      # Works with _NoOpEmbedder (no embedding needed for this command)

All --collection options default to settings.collection (read from Settings,
which reads MEDHA_COLLECTION from env).

Validate: python -c "from medha.cli._app import app; print('OK')"
Then: medha --help (if [cli] extra is installed)
```

---

### Step 2.4 — Package changes

**Files:** `pyproject.toml`

```
Read spec/12_cli.md, sections "Package changes".

Make the following changes to pyproject.toml:

1. Add the [cli] optional dependency group after the [redis] group:
   cli = ["typer>=0.12,<1", "rich>=13,<14"]

2. Add the console scripts entrypoint (new [project.scripts] section):
   [project.scripts]
   medha = "medha.cli:app"

3. Update the [all] and [all-no-chroma] meta-groups to include "cli":
   all = ["medha-archai[...,cli]"]
   all-no-chroma = ["medha-archai[...,cli]"]

4. Add "cli" to the pytest markers list with description:
   "cli: tests that require the medha[cli] extra (typer, rich)"

Do not change anything else.

Validate: pip install -e ".[cli]" --quiet
Then: medha --help  (must show all eight commands)
```

---

### Step 2.5 — Unit tests: CLI

**Files:** `tests/unit/test_cli.py`

```
Read spec/12_cli.md, section "Unit — tests/unit/test_cli.py".

Create tests/unit/test_cli.py implementing all test classes and cases listed
in the spec.

Key implementation notes:
- Use typer.testing.CliRunner for all command tests — do NOT use asyncio.run
- Mock all Medha methods with unittest.mock.AsyncMock and patch medha.cli._app
  to inject a fake Medha instance
- For TestNoOpEmbedder: import _NoOpEmbedder directly and test without mocking
- For TestResolveEmbedder: patch os.environ for API key tests; use
  pytest.importorskip("fastembed") before the fastembed test
- For TestCliSettings: instantiate Settings() directly; monkeypatch os.environ
  for the env var tests
- For TestCliFeedback: the feedback command must work with the default
  _NoOpEmbedder — confirm no real embedder is required

Mark all tests that require typer with @pytest.mark.cli or use importorskip at
the top of the file:
  typer = pytest.importorskip("typer")
  from typer.testing import CliRunner

Run: pytest tests/unit/test_cli.py -x -q
All tests must pass.
```

---

### Step 2.6 — Integration tests: CLI

**Files:** `tests/integration/test_cli_e2e.py`

```
Read spec/12_cli.md, section "Integration — tests/integration/test_cli_e2e.py".

Create tests/integration/test_cli_e2e.py implementing the six e2e tests.

Key implementation notes:
- Use CliRunner (NOT asyncio.run) for all invocations
- Tests run against real InMemoryBackend — no mocking
- To share state between "setup" (Medha in-process) and "CLI invocation"
  (fresh process via CliRunner), use the InMemoryBackend's in-process store:
  - For stats/invalidate/expire: use a shared backend instance injected via
    patching, OR accept that the CLI creates its own instance and test the
    full roundtrip (store via CLI warm, then query via CLI stats)
  - The cleanest approach: use only CLI commands in each test — warm to add
    data, stats/invalidate/expire to operate on it, then verify via stats or
    another warm call
- test_cli_warm_e2e: write a temp JSONL file, run `medha warm`, then run
  `medha stats` and check the count output
- test_cli_feedback_e2e: run `medha warm` with one entry, run
  `medha feedback "question" --incorrect`, check output is "Feedback recorded."

Mark the file or class with @pytest.mark.cli.

Run: pytest tests/integration/test_cli_e2e.py -x -q -m cli
```

---

### Step 2.7 — Demo notebook: CLI

**Files:** `demo/26_cli.ipynb`

```
Read spec/12_cli.md, section "Demo — demo/26_cli.ipynb".

Create demo/26_cli.ipynb with the nine sections described.

Requirements:
- Use InMemoryBackend (MEDHA_BACKEND_TYPE=memory) throughout
- Use FastEmbedAdapter for the warm command; all other commands use _NoOpEmbedder
- All shell commands use the ! prefix (Jupyter shell magic)
- The notebook must be fully runnable with pip install "medha-archai[cli,fastembed]"

Section structure:
1. Installation cell: pip install "medha-archai[cli,fastembed]"
2. Verify: !medha --help
3. Prepare data — store 5 entries programmatically (Python) and save to a
   temp JSONL file
4. !medha stats — show output
5. !medha warm — load JSONL, show count before/after
6. !medha expire and !medha dedup — add stale/duplicate entries, show counts
7. !medha invalidate "question" — remove one entry
8. !medha export --format csv — pipe to pandas, display DataFrame
9. Environment variable reference table (all MEDHA_* vars the CLI uses)

Each section has a markdown cell explaining the command and what to expect.
```

---

## Phase 3 — Finalization

---

### Step 3.1 — CHANGELOG

**Files:** `CHANGELOG.md`

```
Read spec/11_feedback_loop.md section "CHANGELOG entry (0.4.0)" and
spec/12_cli.md section "CHANGELOG entry (0.4.0)".

Add a new [0.4.0] section at the top of CHANGELOG.md (before the [0.3.1]
section), using today's date.

The section must include:
- All items from both spec CHANGELOG entries merged under the correct headings:
  ### Breaking Changes
  ### Added
  ### Notes

Use the same formatting style as the existing [0.3.1] entry.
```

---

### Step 3.2 — README and version bump

**Files:** `README.md`, `pyproject.toml`, `src/medha/__init__.py` (if version is set there)

```
1. In pyproject.toml, change version = "0.3.1" to version = "0.4.0".

2. In README.md:
   a. In the Roadmap section, add two new checked items:
        * [x] Feedback loop — `Medha.feedback()` with optional auto-invalidation
              (`feedback_incorrect_threshold`).
        * [x] `medha` CLI — `pip install "medha-archai[cli]"`. Commands: stats,
              warm, invalidate, invalidate-collection, expire, dedup, export, feedback.
   b. In the Installation section, add the [cli] extra under "With a vector backend":
        # CLI management tool
        pip install "medha-archai[cli]"
   c. In the API Reference Summary table (Core section), add the new rows:
        | Medha.feedback(question, correct) | Record correct/incorrect signal |
        | Medha.feedback_sync              | Sync wrapper                     |
   d. In the Configuration & Types table, add:
        | Settings.feedback_incorrect_threshold | Auto-invalidate when incorrect count reaches N |
   e. Bump the breaking-change note in the Core section (or add a 0.4.0 note)
      that custom VectorStorageBackend subclasses must implement update_feedback().

3. Run the full test suite: pytest -x -q
   All tests must pass before considering 0.4.0 complete.
```

---

## Validation checklist

After all steps are complete:

```
pytest tests/unit/ -q                         # all unit tests
pytest tests/integration/test_feedback_e2e.py -q   # feedback integration
pytest tests/integration/test_cli_e2e.py -q -m cli # CLI integration
medha --help                                   # CLI is installed
medha stats --collection default               # CLI runs without error
python -c "
from medha import Medha
from medha.config import Settings
print(Settings().feedback_incorrect_threshold)  # None
print(Settings().embedder_type)                # _noop
"
```
