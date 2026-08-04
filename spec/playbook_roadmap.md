# Playbook — Medha Roadmap 0.4.1 → 0.5.0

Each section is the implementation guide for one release.
Each step is a self-contained prompt ready to give to Claude Code.
Run steps in order within a release; later steps assume earlier steps are complete.
After each step run `pytest tests/unit/ -x -q` unless the step only adds a notebook or
updates docs.

---

# Release 0.4.1 — CLI QoL + Version Fix

Scope: fix the stale `__version__` string, add three missing CLI commands (`search`,
`health`) and a `--json` flag for machine-readable output on `stats` and `search`.
No new public API, no breaking changes.

---

## Step 1.1 — Fix `__version__`

**Files:** `src/medha/__init__.py`

```
The file src/medha/__init__.py still has __version__ = "0.3.1".
The correct version is "0.4.0".

Change line 3:
  __version__ = "0.3.1"
to:
  __version__ = "0.4.0"

Do not change anything else in the file.

Validate:
  python -c "import medha; print(medha.__version__)"
must print: 0.4.0
```

---

## Step 1.2 — `medha search` command

**Files:** `src/medha/cli/_app.py`

```
Read src/medha/cli/_app.py to understand the existing command structure,
the _build_medha() helper, and the _resolve_embedder() factory.

Add a new `search` command to the Typer app in src/medha/cli/_app.py.
Place it immediately after the existing `stats` command.

Command signature:

  @app.command()
  def search(
      question: str = typer.Argument(..., help="Natural-language question to look up."),
      collection: str = typer.Option(None, "--collection", "-c",
          help="Collection to search. Defaults to MEDHA_COLLECTION (or 'default')."),
      json_output: bool = typer.Option(False, "--json", help="Print result as JSON."),
  ) -> None:
      """Search the cache for a question and print the best match."""

Behaviour:
1. Load Settings(). If collection is None use settings.collection.
2. If settings.embedder_type == "_noop", print to stderr:
     "Error: 'medha search' requires a real embedder.
      Set MEDHA_EMBEDDER_TYPE=fastembed (or openai/cohere/gemini) and install the
      matching extra."
   Then raise typer.Exit(code=1).
3. Resolve the embedder with _resolve_embedder(settings). Catch ConfigurationError
   and RuntimeError; print the message to stderr and raise typer.Exit(code=1).
4. Build Medha and call search:
     async def _run():
         async with _build_medha(collection, settings) as m:
             return await m.search(question)
     result = asyncio.run(_run())
   Note: _build_medha must be updated to support use as an async context manager
   (see step 1.2 note below — or add a standalone try/finally block here if the
   context manager is not yet implemented).
5. If json_output:
     import json
     print(json.dumps({
         "strategy": result.strategy.value,
         "score": result.score,
         "generated_query": result.generated_query,
         "response_summary": result.response_summary,
         "hit": result.strategy.value != "no_match",
     }, indent=2))
6. Else (human output):
   - If result.strategy.value == "no_match":
       typer.echo("No cache hit.")
   - Else:
       typer.echo(f"Strategy : {result.strategy.value}")
       typer.echo(f"Score    : {result.score:.4f}")
       typer.echo(f"Query    : {result.generated_query}")
       if result.response_summary:
           typer.echo(f"Summary  : {result.response_summary}")

Note on _build_medha: this helper currently creates a Medha instance, calls
start(), and returns it. To support the `async with` pattern used above, either:
  a) Turn _build_medha into an async context manager using @asynccontextmanager,
     which calls start() on enter and close() on exit, OR
  b) Use a plain try/finally block inline in the command.
Prefer option (a) — it will be reused by the context manager feature in 0.4.3.
Use contextlib.asynccontextmanager to implement it.

Validate:
  python -c "from medha.cli._app import app; print('import OK')"
  medha search --help   # must show the new command
```

---

## Step 1.3 — `--json` flag for `stats`

**Files:** `src/medha/cli/_app.py`

```
Read the existing `stats` command in src/medha/cli/_app.py.

Add a --json flag to the stats command:

  json_output: bool = typer.Option(False, "--json", help="Print result as JSON.")

Current stats output (text mode) remains unchanged:
  Collection : {collection}
  Backend    : {settings.backend_type}
  Entries    : {count}
  Templates  : {template_count}

New JSON mode (when --json is passed) outputs:
  {
    "collection": "...",
    "backend_type": "...",
    "entries": 42,
    "templates": 3
  }

The backend_count value in CacheStats is entries + templates.
To get the template count separately call:
  templates = await m._backend.count(m._collection_name + "_templates")
wrapped in a try/except StorageError (return 0 if the template collection
does not exist yet).

Do not change any other command.

Validate:
  medha stats --json   (must print valid JSON)
  medha stats          (must print human text as before)
```

---

## Step 1.4 — `medha health` command

**Files:** `src/medha/cli/_app.py`

```
Read src/medha/cli/_app.py to understand the command structure.

Add a `health` command to the Typer app. Place it last (before or after `feedback`).

Command signature:

  @app.command()
  def health(
      collection: str = typer.Option(None, "--collection", "-c",
          help="Collection to probe. Defaults to MEDHA_COLLECTION (or 'default')."),
      json_output: bool = typer.Option(False, "--json", help="Print result as JSON."),
  ) -> None:
      """Check connectivity to the configured backend and embedder."""

Behaviour:
1. Load Settings(). Resolve collection.
2. Probe the backend:
   a. Build the backend instance (same factory logic as _build_medha but do NOT
      call Medha — instantiate the backend directly from the factory used in
      Medha.__init__).
      Actually, simpler: call _build_medha and use the underlying m._backend.
   b. Call await backend.initialize() then await backend.count(collection).
   c. Record result: {"status": "ok", "entries": N} or {"status": "error", "detail": str(exc)}.
3. Probe the embedder:
   a. If settings.embedder_type == "_noop":
      record {"status": "skipped", "detail": "embedder_type is _noop"}.
   b. Else: call _resolve_embedder(settings); call await embedder.aembed(["health check"]).
      record {"status": "ok", "model": embedder.model_name, "dimension": embedder.dimension}
      or {"status": "error", "detail": str(exc)}.
4. overall = "ok" if both probes are "ok" or "skipped", else "error".
5. If json_output:
     print(json.dumps({
         "overall": overall,
         "backend": backend_probe,
         "embedder": embedder_probe,
     }, indent=2))
6. Else:
     typer.echo(f"Backend  [{backend_probe['status'].upper()}]  ...")
     typer.echo(f"Embedder [{embedder_probe['status'].upper()}]  ...")
     typer.echo(f"Overall  : {overall.upper()}")
7. If overall != "ok": raise typer.Exit(code=1).

Use asyncio.run() to run the async probes.

Validate:
  medha health --help
  medha health          (must print human-readable status)
  medha health --json   (must print valid JSON)
```

---

## Step 1.5 — Unit tests for new CLI commands

**Files:** `tests/unit/test_cli.py`

```
Read tests/unit/test_cli.py to understand the existing test structure,
the CliRunner usage, and how Medha is mocked.

Add test cases for the three new features to the existing file.
Use the same mocking approach already in the file (patch _build_medha or
patch Medha methods via unittest.mock.AsyncMock).

--- TestCliSearch ---

class TestCliSearch:

  def test_search_no_embedder_exits_1(runner, monkeypatch):
      # Set MEDHA_EMBEDDER_TYPE=_noop (default), invoke `search "question"`
      # Assert exit_code == 1 and error message mentions MEDHA_EMBEDDER_TYPE

  def test_search_hit_human_output(runner, monkeypatch):
      # Mock _resolve_embedder to return a mock embedder
      # Mock _build_medha (via asynccontextmanager) to return a Medha mock
      # whose search() returns a CacheResult with strategy=SearchStrategy.SEMANTIC,
      # score=0.92, generated_query="SELECT ...", response_summary=None
      # Assert output contains "semantic", "0.9200", "SELECT ..."

  def test_search_no_hit_human_output(runner, monkeypatch):
      # Mock search() to return CacheResult with strategy=SearchStrategy.NO_MATCH
      # Assert output is "No cache hit."

  def test_search_hit_json_output(runner, monkeypatch):
      # Same as test_search_hit but with --json flag
      # Parse output as JSON; assert keys: strategy, score, generated_query, hit
      # Assert hit == True

--- TestCliStatsJson ---

class TestCliStatsJson:

  def test_stats_json_output(runner, monkeypatch):
      # Mock _build_medha; mock backend.count to return 5
      # Invoke stats --json; parse output as JSON
      # Assert keys: collection, backend_type, entries
      # Assert entries == 5

  def test_stats_human_output_unchanged(runner, monkeypatch):
      # Existing behaviour: same mock; invoke stats (no --json)
      # Assert "Entries" appears in output

--- TestCliHealth ---

class TestCliHealth:

  def test_health_ok_human(runner, monkeypatch):
      # Mock backend.count to return 10; mock embedder.aembed to return [[0.1]*384]
      # Set MEDHA_EMBEDDER_TYPE=fastembed
      # Assert output contains "OK" and exit_code == 0

  def test_health_backend_error(runner, monkeypatch):
      # Mock backend.initialize() to raise StorageError("conn refused")
      # Assert exit_code == 1 and "ERROR" in output

  def test_health_noop_embedder_skipped(runner, monkeypatch):
      # Default _noop embedder
      # Assert embedder probe shows "SKIPPED"

  def test_health_json_output(runner, monkeypatch):
      # Mock all probes as OK; use --json
      # Parse output; assert overall == "ok"

Run: pytest tests/unit/test_cli.py -x -q -k "Search or StatsJson or Health"
All new tests must pass; existing tests must not break.
```

---

## Step 1.6 — Integration tests for new CLI commands

**Files:** `tests/integration/test_cli_e2e.py`

```
Read tests/integration/test_cli_e2e.py to understand the existing integration
test approach (CliRunner + real InMemoryBackend, no mocking of Medha itself).

Add three integration test functions to the existing file.

All three require the [cli] and [fastembed] extras:
  fastembed = pytest.importorskip("fastembed")
  typer_test = pytest.importorskip("typer.testing")

def test_search_e2e_hit(tmp_path):
    # 1. Write a JSONL file with one entry to tmp_path
    # 2. Set env vars: MEDHA_BACKEND_TYPE=memory, MEDHA_EMBEDDER_TYPE=fastembed
    # 3. Run `medha warm <file>` via CliRunner — assert exit_code == 0
    # Note: warm and search use different Medha instances (InMemoryBackend is
    #       in-process, so each CLI invocation gets a fresh empty store).
    #       To test a real hit, store the entry programmatically in-process
    #       and then call search via the public Python API rather than CLI,
    #       OR use LanceDB/persist-to-disk backend.
    #       Simplest: call medha.search_sync() directly in the test.
    # This test demonstrates CLI search integration by:
    #   a. Creating a Medha instance in-process with InMemoryBackend + FastEmbedAdapter
    #   b. Storing one entry
    #   c. Calling medha.search_sync("same question") directly (not via CLI)
    #   d. Asserting the result is a hit with the right generated_query
    # (CLI-level test via CliRunner for search is covered by unit tests above
    #  because InMemoryBackend state does not survive across process boundaries.)

def test_health_ok_e2e():
    # Set MEDHA_BACKEND_TYPE=memory, MEDHA_EMBEDDER_TYPE=fastembed
    # Run `medha health` via CliRunner
    # Assert exit_code == 0 and output contains "OK"

def test_health_json_e2e():
    # Same setup; run `medha health --json`
    # Parse JSON output; assert overall == "ok", backend.status == "ok"

Mark with @pytest.mark.cli.

Run: pytest tests/integration/test_cli_e2e.py -x -q -m cli -k "search_e2e or health"
```

---

## Step 1.7 — Update demo notebook

**Files:** `demo/26_cli.ipynb`

```
Read demo/26_cli.ipynb.

Add two new sections at the end of the notebook (before or after the
environment variables reference table):

Section: "medha search — query the cache from the shell"
  Markdown cell:
    "The `search` command lets you probe the cache directly from the terminal.
     It requires a real embedder (MEDHA_EMBEDDER_TYPE=fastembed or similar).
     Useful for debugging cache hits during development."
  Code cell (shell):
    !MEDHA_EMBEDDER_TYPE=fastembed medha search "who are the top customers?"
  Code cell (JSON output):
    !MEDHA_EMBEDDER_TYPE=fastembed medha search "who are the top customers?" --json

Section: "medha health — connectivity check"
  Markdown cell:
    "Use `health` before running a workload to verify that the backend and embedder
     are reachable. Exits with code 1 if any probe fails — useful in CI."
  Code cell:
    !medha health
  Code cell:
    !medha health --json

Do not remove or change existing cells.
Update the environment variables table to include any new MEDHA_* variables
introduced in 0.4.1 (none expected — search and health reuse existing env vars).
```

---

## Step 1.8 — CHANGELOG and version bump

**Files:** `CHANGELOG.md`, `pyproject.toml`, `src/medha/__init__.py`

```
1. In pyproject.toml change:
     version = "0.4.0"
   to:
     version = "0.4.1"

2. In src/medha/__init__.py change:
     __version__ = "0.4.0"
   to:
     __version__ = "0.4.1"
   (This was already set to "0.4.0" in step 1.1; now bump to "0.4.1".)

3. Add a new [0.4.1] section at the top of CHANGELOG.md (before the [0.4.0] section).
   Use today's date. Follow the same formatting style as the existing entries.

   ## [0.4.1] — <date>

   ### Fixed

   - `__version__` in `src/medha/__init__.py` corrected to `"0.4.0"` (was `"0.3.1"`
     due to a missed bump at release time).

   ### Added

   - `medha search <question>` CLI command — look up a question in the cache and
     print the best match (strategy, score, generated query). Requires a real embedder
     (`MEDHA_EMBEDDER_TYPE != _noop`). Supports `--json` for machine-readable output.

   - `medha health` CLI command — probe backend connectivity and embedder availability.
     Prints `OK` / `ERROR` / `SKIPPED` per component. Exits with code 1 if any probe
     fails. Supports `--json`.

   - `--json` flag on `medha stats` — outputs collection name, backend type, and entry
     count as a JSON object.

4. Run the full test suite: pytest -x -q
   All tests must pass before tagging 0.4.1.
```

---

# Release 0.4.2 — New Embedder Adapters

Scope: add two new embedding adapters — `OpenAICompatibleAdapter` (for Ollama, vLLM,
LocalAI, LM Studio) and `MistralAdapter` (Mistral API). Zero breaking changes.
Both adapters follow the exact same pattern as the existing OpenAI and Cohere adapters.

---

## Step 2.1 — Settings additions

**Files:** `src/medha/config.py`

```
Read src/medha/config.py to understand the existing settings structure.
Look at the OpenAI settings block and the Cohere settings block as reference.

Add two new settings blocks to the Settings class.

--- OpenAI-compatible block ---
Place after the existing OpenAI block (# --- OpenAI ---).
Comment: # --- OpenAI-compatible (Ollama, vLLM, LocalAI, LM Studio) ---

  oai_compat_base_url: str = Field(
      default="http://localhost:11434/v1",
      description=(
          "Base URL for any OpenAI-compatible embeddings endpoint. "
          "Env var: MEDHA_OAI_COMPAT_BASE_URL."
      ),
  )
  oai_compat_model: str = Field(
      default="nomic-embed-text",
      description=(
          "Model name to request from the OpenAI-compatible endpoint. "
          "Env var: MEDHA_OAI_COMPAT_MODEL."
      ),
  )
  oai_compat_api_key: SecretStr | None = Field(
      default=None,
      description=(
          "API key for the OpenAI-compatible endpoint (optional; many local servers "
          "accept any non-empty string). Env var: MEDHA_OAI_COMPAT_API_KEY."
      ),
  )

--- Mistral block ---
Place after the Cohere block (# --- Cohere ---).
Comment: # --- Mistral ---

  mistral_api_key: SecretStr | None = Field(
      default=None,
      description="Mistral API key. Env var: MEDHA_MISTRAL_API_KEY.",
  )
  mistral_model: str = Field(
      default="mistral-embed",
      description=(
          "Mistral embedding model identifier. "
          "Env var: MEDHA_MISTRAL_MODEL."
      ),
  )
  mistral_batch_size: int = Field(
      default=50,
      ge=1,
      le=512,
      description=(
          "Maximum number of texts per Mistral embed API request. "
          "Env var: MEDHA_MISTRAL_BATCH_SIZE."
      ),
  )

Do not change any other field or validator.

Validate: pytest tests/unit/test_config.py -x -q
Also: python -c "from medha.config import Settings; s = Settings(); print(s.oai_compat_base_url, s.mistral_model)"
must print: http://localhost:11434/v1 mistral-embed
```

---

## Step 2.2 — `OpenAICompatibleAdapter`

**Files:** `src/medha/embeddings/openai_compatible_adapter.py`

```
Read src/medha/embeddings/openai_adapter.py in full. The new adapter is nearly
identical but uses a configurable base_url.

Create src/medha/embeddings/openai_compatible_adapter.py implementing
OpenAICompatibleAdapter.

The class must:
1. Inherit from BaseEmbedder.
2. Accept __init__(self, settings: Settings) — read base_url, model, api_key from
   settings.oai_compat_base_url, settings.oai_compat_model, settings.oai_compat_api_key.
3. Use openai.AsyncOpenAI(base_url=base_url, api_key=api_key or "ollama") as the
   client (many local servers require a non-empty string as api_key; "ollama" is
   the conventional placeholder).
4. dimension property: return self._dimension (populated after first embed call, or
   set from a probe embed at init if self._probe_on_init=True; default: defer until
   first call, same pattern as OpenAIAdapter).
5. model_name property: return self._model.
6. aembed(text: str) -> list[float]: call self._aclient.embeddings.create with
   input=[text], model=self._model. Return embedding.
7. aembed_batch(texts: list[str]) -> list[list[float]]: chunk into batches of 20
   (same as OpenAIAdapter), call embeddings.create for each chunk, gather results.
8. Wrap API exceptions in EmbeddingError (same as OpenAIAdapter).

The only difference from OpenAIAdapter is that the client is constructed with
base_url set to settings.oai_compat_base_url.

The class uses the openai package — it goes into the existing [openai] optional
extra, not a new extra.

Validate:
  python -c "
  from medha.config import Settings
  from medha.embeddings.openai_compatible_adapter import OpenAICompatibleAdapter
  s = Settings(oai_compat_base_url='http://localhost:11434/v1', oai_compat_model='nomic-embed-text')
  e = OpenAICompatibleAdapter(s)
  print(e.model_name)  # nomic-embed-text
  "
  (No real network call needed for the import/init test.)
```

---

## Step 2.3 — `MistralAdapter`

**Files:** `src/medha/embeddings/mistral_adapter.py`

```
Read src/medha/embeddings/cohere_adapter.py in full. The Mistral adapter follows
the same batching pattern.

Create src/medha/embeddings/mistral_adapter.py implementing MistralAdapter.

The class must:
1. Inherit from BaseEmbedder.
2. Accept __init__(self, settings: Settings).
3. Import mistralai lazily inside the method to avoid ImportError when the
   [mistral] extra is not installed:
     try:
         from mistralai import Mistral
     except ImportError:
         raise ImportError("pip install 'medha-archai[mistral]'")
4. Build self._client = Mistral(api_key=settings.mistral_api_key.get_secret_value()
   if settings.mistral_api_key else None).
5. Store self._model = settings.mistral_model.
6. Store self._batch_size = settings.mistral_batch_size.
7. self._dimension: int | None = None  (populated on first embed).
8. dimension property: if self._dimension is None raise RuntimeError("not embedded yet").
   Return self._dimension.
9. model_name property: return self._model.
10. aembed(text: str) -> list[float]:
    result = await self._client.embeddings.create_async(
        model=self._model, inputs=[text]
    )
    vec = result.data[0].embedding
    self._dimension = len(vec)
    return vec
11. aembed_batch(texts: list[str]) -> list[list[float]]:
    results = []
    for chunk in _chunks(texts, self._batch_size):
        resp = await self._client.embeddings.create_async(
            model=self._model, inputs=chunk
        )
        results.extend(item.embedding for item in resp.data)
    if results and self._dimension is None:
        self._dimension = len(results[0])
    return results
12. Add a module-level helper:
    def _chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
13. Wrap exceptions in EmbeddingError.

Validate:
  python -c "
  from medha.config import Settings
  from medha.embeddings.mistral_adapter import MistralAdapter
  # import-only test (no API call)
  print('import OK')
  "
  Note: if mistralai is not installed, the import itself succeeds; only
  instantiation raises ImportError. Confirm this with a try/except test.
```

---

## Step 2.4 — Update `embedder_type` Literal and CLI factory

**Files:** `src/medha/config.py`, `src/medha/cli/_app.py`

```
Read src/medha/config.py (the embedder_type field) and src/medha/cli/_app.py
(the _resolve_embedder function).

1. In src/medha/config.py, extend the Literal for embedder_type:
   Change:
     Literal["fastembed", "openai", "cohere", "gemini", "_noop"]
   To:
     Literal["fastembed", "openai", "openai-compatible", "cohere", "gemini", "mistral", "_noop"]

2. In src/medha/cli/_app.py, extend _resolve_embedder(settings) to handle the
   two new values:

   elif settings.embedder_type == "openai-compatible":
       try:
           from medha.embeddings.openai_compatible_adapter import OpenAICompatibleAdapter
       except ImportError:
           raise ConfigurationError("pip install 'medha-archai[openai]'")
       return OpenAICompatibleAdapter(settings)

   elif settings.embedder_type == "mistral":
       try:
           from medha.embeddings.mistral_adapter import MistralAdapter
       except ImportError:
           raise ConfigurationError("pip install 'medha-archai[mistral]'")
       return MistralAdapter(settings)

   Place these two blocks before the final `else` branch that handles unknown types.

Do not change any other logic.

Validate:
  pytest tests/unit/test_config.py -x -q
  python -c "from medha.config import Settings; Settings(embedder_type='openai-compatible')"
  python -c "from medha.config import Settings; Settings(embedder_type='mistral')"
```

---

## Step 2.5 — Export from `__init__.py`

**Files:** `src/medha/__init__.py`

```
Read src/medha/__init__.py.

Add two new optional-import blocks for the new adapters, following the exact
same pattern used for CohereAdapter and GeminiAdapter.

Add after the GeminiAdapter block:

  try:
      from medha.embeddings.openai_compatible_adapter import OpenAICompatibleAdapter
      _optional.append("OpenAICompatibleAdapter")
  except ImportError:
      pass

  try:
      from medha.embeddings.mistral_adapter import MistralAdapter
      _optional.append("MistralAdapter")
  except ImportError:
      pass

Do not change __all__ directly — the _optional list is already included in __all__.

Validate:
  python -c "import medha; print([x for x in dir(medha) if 'Compatible' in x or 'Mistral' in x])"
  (will print the class names if the extras are installed, empty list otherwise)
```

---

## Step 2.6 — pyproject.toml optional extras

**Files:** `pyproject.toml`

```
Read pyproject.toml — specifically the [project.optional-dependencies] section.

1. Add a new [mistral] extra after the [gemini] group:
     mistral = ["mistralai>=1.0,<2"]

2. The [openai-compatible] adapter reuses the existing [openai] extra — no new group needed.
   Add a comment above the openai group:
     # Also used by OpenAICompatibleAdapter (Ollama, vLLM, LocalAI)

3. Update the [all] meta-group to include "mistral":
     Change the all line to add medha-archai[mistral] to the list.

4. Update [all-no-chroma] similarly.

Do not change any other dependency.

Validate:
  pip install -e ".[mistral]" --quiet
  python -c "import mistralai; print('mistralai OK')"
```

---

## Step 2.7 — Unit tests

**Files:** `tests/unit/test_openai_compatible_adapter.py`,
           `tests/unit/test_mistral_adapter.py`

```
Read tests/unit/test_openai_adapter.py and tests/unit/test_cohere_adapter.py
as reference for the test structure and mocking approach.

--- tests/unit/test_openai_compatible_adapter.py ---

class TestOpenAICompatibleAdapter:

  def test_model_name(mock_settings):
      # Settings with oai_compat_model="nomic-embed-text"
      # adapter = OpenAICompatibleAdapter(settings)
      # assert adapter.model_name == "nomic-embed-text"

  async def test_aembed_returns_vector(mock_oai_compat_client):
      # Mock openai.AsyncOpenAI; make embeddings.create return a fake response
      # Call await adapter.aembed("test question")
      # Assert result is a list of floats

  async def test_aembed_batch_chunked(mock_oai_compat_client):
      # Provide 25 texts (> 20 per chunk)
      # Assert embeddings.create was called twice (chunks of 20 + 5)
      # Assert len(result) == 25

  def test_uses_base_url(mock_settings):
      # Patch openai.AsyncOpenAI to capture constructor args
      # Settings with oai_compat_base_url="http://localhost:11434/v1"
      # Assert AsyncOpenAI was called with base_url="http://localhost:11434/v1"

  async def test_embedding_error_wrapped(mock_oai_compat_client):
      # Mock embeddings.create to raise openai.APIError
      # Assert EmbeddingError is raised

--- tests/unit/test_mistral_adapter.py ---

class TestMistralAdapter:

  def test_model_name(mock_settings):
      # Settings with mistral_model="mistral-embed"
      # assert adapter.model_name == "mistral-embed"

  async def test_aembed_returns_vector(mock_mistral_client):
      # Mock mistralai.Mistral; make embeddings.create_async return a fake response
      # Call await adapter.aembed("test")
      # Assert result is a list of floats; assert adapter.dimension is set

  async def test_aembed_batch_chunked(mock_mistral_client):
      # mistral_batch_size=5; provide 12 texts
      # Assert create_async called 3 times (chunks of 5, 5, 2)

  def test_import_error_when_mistralai_missing(monkeypatch):
      # monkeypatch sys.modules to hide "mistralai"
      # Assert ImportError raised on instantiation with "pip install" hint

  async def test_embedding_error_wrapped(mock_mistral_client):
      # Mock create_async to raise an exception
      # Assert EmbeddingError is raised

Skip tests that require the real package if not installed:
  pytest.importorskip("openai")    # in test_openai_compatible_adapter.py
  pytest.importorskip("mistralai") # in test_mistral_adapter.py

Run: pytest tests/unit/test_openai_compatible_adapter.py tests/unit/test_mistral_adapter.py -x -q
```

---

## Step 2.8 — CHANGELOG and version bump

**Files:** `CHANGELOG.md`, `pyproject.toml`, `src/medha/__init__.py`

```
1. In pyproject.toml change version = "0.4.1" to version = "0.4.2".
2. In src/medha/__init__.py change __version__ = "0.4.1" to __version__ = "0.4.2".

3. Add [0.4.2] section at the top of CHANGELOG.md (before [0.4.1]).
   Use today's date.

   ## [0.4.2] — <date>

   ### Added

   - **`OpenAICompatibleAdapter`**: embedder adapter for any OpenAI-compatible
     endpoint (Ollama, vLLM, LocalAI, LM Studio). Reuses the `[openai]` extra.
     Select with `MEDHA_EMBEDDER_TYPE=openai-compatible`;
     configure endpoint via `MEDHA_OAI_COMPAT_BASE_URL` (default:
     `http://localhost:11434/v1`) and `MEDHA_OAI_COMPAT_MODEL` (default:
     `nomic-embed-text`).

   - **`MistralAdapter`**: embedder adapter for the Mistral Embeddings API
     (`mistral-embed`, 1024 dimensions). Install with
     `pip install "medha-archai[mistral]"`. Select with
     `MEDHA_EMBEDDER_TYPE=mistral`; configure with `MEDHA_MISTRAL_API_KEY`.

   - `Settings.oai_compat_base_url`, `Settings.oai_compat_model`,
     `Settings.oai_compat_api_key` — configuration for `OpenAICompatibleAdapter`.

   - `Settings.mistral_api_key`, `Settings.mistral_model`,
     `Settings.mistral_batch_size` — configuration for `MistralAdapter`.

   - `embedder_type` now accepts `"openai-compatible"` and `"mistral"` in addition
     to the existing values.

4. Run: pytest -x -q
   All tests must pass.
```

---

# Release 0.4.3 — Developer Experience

Scope: three API improvements that do not change any existing behaviour —
async context manager support, `search_batch()` for multi-question lookups,
and startup validation in `start()`. No breaking changes.

---

## Step 3.1 — Async context manager on `Medha`

**Files:** `src/medha/core.py`

```
Read src/medha/core.py — specifically the start() and close() methods.

Add __aenter__ and __aexit__ methods to the Medha class.
Place them immediately after close() and before the first public search method.

  async def __aenter__(self) -> "Medha":
      await self.start()
      return self

  async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
      await self.close()

These are the only changes needed. Do not modify start() or close().

Validate:
  python -c "
  import asyncio
  from medha import Medha
  from medha.config import Settings
  from medha.embeddings.fastembed_adapter import FastEmbedAdapter

  async def test():
      settings = Settings(backend_type='memory')
      async with Medha('test', FastEmbedAdapter(), settings) as m:
          print('entered:', m._started)
      print('exited OK')

  asyncio.run(test())
  "

Also update the _build_medha helper in src/medha/cli/_app.py to use the context
manager instead of the current try/finally pattern (if it was added as a
try/finally in step 1.2):
  @asynccontextmanager
  async def _build_medha(collection: str, settings: Settings):
      embedder = _resolve_embedder(settings)
      backend = _make_backend(settings)   # use the factory already in Medha.__init__
      async with Medha(collection, embedder, backend, settings) as m:
          yield m

If _build_medha already uses @asynccontextmanager from step 1.2, just
verify it still works correctly — no change needed.

Run: pytest tests/unit/ -x -q
```

---

## Step 3.2 — `Medha.search_batch()`

**Files:** `src/medha/core.py`

```
Read src/medha/core.py — specifically the search() and aembed_batch() call in store_batch().

Add two new methods to the Medha class: search_batch() (async) and
search_batch_sync() (sync wrapper). Place them immediately after search() and
search_sync().

The goal of search_batch() is to embed all questions in a single aembed_batch()
call and then run the waterfall search for each question concurrently.

  async def search_batch(
      self,
      questions: list[str],
      collection_name: str | None = None,
  ) -> list[CacheResult]:
      """Search for multiple questions, embedding them in a single batch call.

      Returns results in the same order as the input questions.
      Each result follows the same waterfall search strategy as search().
      """
      if not questions:
          return []
      collection = collection_name or self._collection_name

      # 1. Embed all questions in one batch call (the main perf win).
      #    Use the same timeout logic as search(): wrap in asyncio.wait_for if
      #    settings.embedding_timeout is set.
      vectors = await self._embed_with_timeout(questions)

      # 2. Run waterfall search for each (question, vector) pair concurrently.
      tasks = [
          self._waterfall_search(q, v, collection)
          for q, v in zip(questions, vectors)
      ]
      results = await asyncio.gather(*tasks, return_exceptions=False)
      return list(results)

  def search_batch_sync(
      self,
      questions: list[str],
      collection_name: str | None = None,
  ) -> list[CacheResult]:
      return self._run_sync(self.search_batch(questions, collection_name))

Notes on implementation:
- _embed_with_timeout(texts: list[str]) is a new private helper that calls
  await self._embedder.aembed_batch(texts) wrapped in asyncio.wait_for if
  settings.embedding_timeout is set. Extract this logic from the existing
  search() method to avoid duplication.
- _waterfall_search(question, vector, collection) is the existing waterfall logic
  currently inlined in search(). Extract it as a private method.
- Refactor search() to call _embed_with_timeout([question])[0] and then
  _waterfall_search(question, vector, collection). The observable behaviour of
  search() must not change.
- Do not add the method to VectorStorageBackend — this is purely a Medha-level API.

Validate:
  python -c "
  import asyncio
  from medha import Medha
  from medha.config import Settings
  from medha.embeddings.fastembed_adapter import FastEmbedAdapter
  from medha.backends.memory import InMemoryBackend

  async def test():
      settings = Settings(backend_type='memory')
      async with Medha('batch_test', FastEmbedAdapter(), InMemoryBackend(), settings) as m:
          results = await m.search_batch(['what is the revenue?', 'show me all users'])
          assert len(results) == 2
          print('search_batch OK, strategies:', [r.strategy.value for r in results])

  asyncio.run(test())
  "

Run: pytest tests/unit/test_core_waterfall.py -x -q
Existing waterfall tests must not regress.
```

---

## Step 3.3 — Startup validation in `start()`

**Files:** `src/medha/core.py`, `src/medha/config.py`

```
Read src/medha/core.py — the start() method. Read src/medha/config.py.

--- Config change ---

Add one new field to Settings in src/medha/config.py, in the "Cache lifecycle" section:

  validate_on_start: bool = Field(
      default=True,
      description=(
          "When True, start() probes the backend with a count() call to verify "
          "connectivity before returning. Set to False to skip the probe "
          "(e.g. in unit tests where the backend is not yet initialized). "
          "Env var: MEDHA_VALIDATE_ON_START."
      ),
  )

--- Core change ---

In src/medha/core.py, at the end of start() (after await self._backend.initialize()
and before starting the cleanup task), add:

  if self._settings.validate_on_start:
      try:
          await self._backend.count(self._collection_name)
      except Exception as exc:
          raise StorageError(
              f"Backend connectivity check failed at start(): {exc}. "
              f"Set Settings(validate_on_start=False) to skip this check."
          ) from exc

No other changes to start().

Validate:
  pytest tests/unit/test_config.py -x -q
  python -c "
  import asyncio
  from medha import Medha
  from medha.config import Settings
  from medha.backends.memory import InMemoryBackend
  from medha.embeddings.fastembed_adapter import FastEmbedAdapter

  async def test():
      # Should succeed with InMemoryBackend
      async with Medha('v', FastEmbedAdapter(), InMemoryBackend(), Settings(backend_type='memory')) as m:
          print('start OK')

  asyncio.run(test())
  "
  # Also test that validate_on_start=False skips the probe:
  python -c "
  import asyncio
  from unittest.mock import AsyncMock, MagicMock
  # ... (manually verify via unit test)
  "

Run: pytest tests/unit/ -x -q
All existing tests must pass. Pay special attention to tests that call start() on
a mock backend — those tests may need to set validate_on_start=False in their
Settings fixture or mock the count() call.
```

---

## Step 3.4 — Unit tests

**Files:** `tests/unit/test_context_manager.py`,
           `tests/unit/test_search_batch.py`,
           `tests/unit/test_startup_validation.py`

```
Read tests/unit/conftest.py for available fixtures.
Read tests/unit/test_core_waterfall.py for the mocking approach.

--- tests/unit/test_context_manager.py ---

class TestAsyncContextManager:

  async def test_enter_calls_start(mock_embedder, mock_backend):
      # Create Medha with mock_backend
      # Use `async with medha_instance as m:`
      # Assert m is medha_instance
      # Assert mock_backend.initialize was called

  async def test_exit_calls_close(mock_embedder, mock_backend):
      # Use `async with medha_instance as m:`; do nothing
      # Assert mock_backend.close was called after the block

  async def test_exit_on_exception(mock_embedder, mock_backend):
      # Raise an exception inside the with block
      # Assert close() was still called (cleanup on error)
      # Assert the exception propagates normally

  def test_sync_usage_not_supported():
      # Confirm Medha does NOT implement __enter__ / __exit__
      # (sync context manager is intentionally not provided)
      m = Medha.__new__(Medha)
      assert not hasattr(m, '__enter__')

--- tests/unit/test_search_batch.py ---

class TestSearchBatch:

  async def test_empty_list_returns_empty(medha_memory):
      results = await medha_memory.search_batch([])
      assert results == []

  async def test_returns_same_count_as_input(medha_memory):
      results = await medha_memory.search_batch(["q1", "q2", "q3"])
      assert len(results) == 3

  async def test_order_preserved(medha_memory, mock_embedder):
      # Store two entries with distinct queries
      # search_batch(["question_1", "question_2"])
      # Assert results[0].generated_query corresponds to question_1's stored entry

  async def test_embedder_called_once_for_batch(medha_memory, mock_embedder):
      # Spy on mock_embedder.aembed_batch
      # Call search_batch(["q1", "q2", "q3"])
      # Assert aembed_batch was called exactly once with 3 items

  def test_search_batch_sync_wrapper(medha_memory_sync):
      # Use medha_memory_sync fixture or _run_sync pattern
      results = medha_memory.search_batch_sync(["q1"])
      assert isinstance(results, list)

--- tests/unit/test_startup_validation.py ---

class TestStartupValidation:

  async def test_count_called_on_start(mock_embedder, mock_backend):
      # Settings(validate_on_start=True)
      # Create Medha and call start()
      # Assert mock_backend.count was called

  async def test_count_not_called_when_disabled(mock_embedder, mock_backend):
      # Settings(validate_on_start=False)
      # Assert mock_backend.count was NOT called

  async def test_storage_error_on_connection_failure(mock_embedder, mock_backend):
      # mock_backend.count raises Exception("connection refused")
      # Settings(validate_on_start=True)
      # Assert StorageError is raised with "connectivity check" in the message

Run: pytest tests/unit/test_context_manager.py tests/unit/test_search_batch.py tests/unit/test_startup_validation.py -x -q
```

---

## Step 3.5 — Integration tests

**Files:** `tests/integration/test_context_manager_e2e.py`,
           `tests/integration/test_search_batch_e2e.py`

```
--- tests/integration/test_context_manager_e2e.py ---

Use FastEmbedAdapter + InMemoryBackend (real, not mocked).

class TestAsyncContextManagerE2E:

  async def test_full_lifecycle():
      async with Medha('ctx_test', FastEmbedAdapter(), InMemoryBackend(),
                       Settings(backend_type='memory')) as m:
          await m.store("revenue query", "SELECT SUM(revenue) FROM sales")
          result = await m.search("what is total revenue?")
          assert result.strategy.value != "no_match"
      # After the with block: close() was called, _started is False

  async def test_exception_in_block_still_closes():
      async with Medha(...) as m:
          backend_ref = m._backend
      # The backend was closed even though... actually test that close is called
      # even when an exception is raised inside the with block.
      closed = False
      original_close = backend_ref.close

      async def spy_close():
          nonlocal closed
          closed = True
          await original_close()

      backend_ref.close = spy_close
      try:
          async with Medha(...) as m:
              raise ValueError("test error")
      except ValueError:
          pass
      assert closed

--- tests/integration/test_search_batch_e2e.py ---

class TestSearchBatchE2E:

  async def test_batch_returns_hits_for_stored_questions():
      # Store 3 entries, search_batch with the same 3 questions
      # Assert all 3 results are hits (strategy != no_match)

  async def test_batch_miss_for_unknown_questions():
      # Store 1 entry, search_batch with 2 questions unrelated to stored entry
      # Assert 2 results are no_match

  async def test_batch_order_preserved():
      # Store 2 entries with different generated_queries
      # search_batch([question_for_entry_2, question_for_entry_1])
      # Assert results[0] corresponds to entry_2, results[1] to entry_1

  async def test_batch_empty_collection():
      results = await m.search_batch(["anything"])
      assert results[0].strategy.value == "no_match"

Run: pytest tests/integration/test_context_manager_e2e.py tests/integration/test_search_batch_e2e.py -x -q
```

---

## Step 3.6 — Demo notebook

**Files:** `demo/27_async_and_batch.ipynb`

```
Create demo/27_async_and_batch.ipynb.

This notebook requires: pip install "medha-archai[fastembed]"

Section structure:

1. Async context manager (markdown + code)
   - Markdown: "Starting with 0.4.3, Medha implements the async context manager
     protocol. Use `async with Medha(...) as m:` instead of calling
     `start()` / `close()` manually. The context manager guarantees cleanup even
     if an exception occurs inside the block."
   - Code: show the async with pattern, store 3 entries, search one, print result.

2. search_batch() (markdown + code)
   - Markdown: "search_batch() embeds all questions in a single aembed_batch()
     call, then runs the waterfall search for each question concurrently.
     Use it when you need to look up many questions at once — e.g. warming a
     test harness or running a benchmark."
   - Code: store 5 entries, call search_batch() with 5 questions, print a
     DataFrame with question, strategy, score, generated_query.

3. Performance note (markdown + code)
   - Benchmark: compare 5 sequential search() calls vs one search_batch() call.
   - Show timing with %timeit or time.perf_counter.
   - Expected: batch is faster because the embedding step is merged into one
     API call.

4. startup validation (markdown + code)
   - Markdown: "start() now probes the backend with count() before returning.
     This surfaces connectivity issues early. Disable with
     Settings(validate_on_start=False) in unit tests or CI environments where
     the backend may not be reachable."
   - Code: demonstrate ConfigurationError / StorageError on bad DSN
     (e.g. wrong Qdrant URL), then show validate_on_start=False to skip.

Use InMemoryBackend + FastEmbedAdapter throughout (no external services).
Each section must have a markdown cell explaining what it demonstrates.
```

---

## Step 3.7 — CHANGELOG and version bump

**Files:** `CHANGELOG.md`, `pyproject.toml`, `src/medha/__init__.py`

```
1. Bump version to "0.4.3" in pyproject.toml and src/medha/__init__.py.

2. Add [0.4.3] section at the top of CHANGELOG.md.

   ## [0.4.3] — <date>

   ### Added

   - **Async context manager**: `Medha` now implements `__aenter__` / `__aexit__`.
     Use `async with Medha(...) as m:` instead of calling `start()` and `close()`
     manually. `close()` is guaranteed to run even if an exception occurs.

   - **`Medha.search_batch(questions)`** — look up a list of questions in a single
     call. All questions are embedded via one `aembed_batch()` round-trip; waterfall
     searches run concurrently via `asyncio.gather()`. Returns results in input order.
     Sync wrapper: `search_batch_sync()`.

   - **Startup validation**: `start()` now calls `count()` on the collection after
     `initialize()` to verify backend connectivity. Raises `StorageError` if the probe
     fails. Disable with `Settings(validate_on_start=False)` (or env var
     `MEDHA_VALIDATE_ON_START=false`).

   - `Settings.validate_on_start` (default `True`) — toggle backend probe in `start()`.

3. Run: pytest -x -q
   All tests must pass.
```

---

# Release 0.5.0 — Persistent Stats + Feedback Boosting

Scope: two significant new features that extend the storage interface
(`load_stats` / `save_stats`) and the waterfall search logic (score adjustment).

**This release contains no breaking changes.** Unlike 0.4.0 — which made
`update_feedback()` abstract and broke every custom backend — the two new
storage methods ship as non-abstract defaults (`return None` / no-op), so
existing `VectorStorageBackend` subclasses keep instantiating and working
untouched. Stats persistence simply stays off for backends that do not override
them. All ten built-in backends do override them (Steps 4.3, 4.4).
Feedback boosting is likewise opt-in: `feedback_boost_factor` defaults to `0.0`,
which reproduces 0.4.3 scoring exactly.

> **Security baseline (prerequisite already merged on top of 0.4.3).** This
> release builds on the hardening shipped after 0.4.3: the shared
> `medha.backends._escape.quote_sql_literal` helper, validated `pg_schema` and
> table/index identifiers, `VectorChordBackend` `vc_lists` validation, and
> `SECURITY.md`. Step 4.4 adds **new** identifier interpolation and filter
> clauses to *every* backend, so that new code MUST reuse this baseline — the
> shared escaper, each backend's existing name-sanitiser, the validated schema,
> and bind parameters — and must never reintroduce inline `'` → `''` escaping.
> See the per-backend and "For all backends" notes in Step 4.4.

---

## Step 4.1 — `PersistedStats` data model

**Files:** `src/medha/types.py`

```
Read src/medha/types.py to understand the existing type definitions.
Read the existing CacheStats model in full.

Add a new Pydantic model PersistedStats to src/medha/types.py.
Place it immediately before CacheStats.

  class PersistedStats(BaseModel):
      """Snapshot of cache performance metrics stored in the backend.

      Persisted after every Settings.stats_persist_interval requests.
      Loaded on start() so metrics survive process restarts.
      """
      total_requests: int = 0
      total_hits: int = 0
      total_misses: int = 0
      total_errors: int = 0
      hits_by_strategy: dict[str, int] = Field(default_factory=dict)
      last_reset_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
      updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

  NOTE: use `lambda: datetime.now(timezone.utc)`, NOT `datetime.utcnow` —
  utcnow() is deprecated from Python 3.12 and returns a naive datetime.
  This matches the existing CacheStats model, which already uses
  `default_factory=lambda: datetime.now(timezone.utc)`. `timezone` is already
  imported in types.py.

      @property
      def hit_rate(self) -> float:
          return self.total_hits / self.total_requests if self.total_requests else 0.0

      @property
      def miss_rate(self) -> float:
          return self.total_misses / self.total_requests if self.total_requests else 0.0

Also add PersistedStats to the __all__ list in src/medha/__init__.py (it is a
public type, exportable alongside CacheStats).

Validate:
  python -c "
  from medha.types import PersistedStats
  s = PersistedStats(total_requests=10, total_hits=7)
  print(s.hit_rate)  # 0.7
  "
  pytest tests/unit/test_types.py -x -q
```

---

## Step 4.2 — Storage interface: `load_stats` / `save_stats`

**Files:** `src/medha/interfaces/storage.py`

```
Read src/medha/interfaces/storage.py in full.

Add two new methods to VectorStorageBackend. They are NOT abstract: both ship
with a working default so that existing custom subclasses keep instantiating
unchanged. Stats persistence is an opt-in capability, not part of the minimum
backend contract.

Place them after update_feedback() and before close() — i.e. after the last
@abstractmethod and immediately before the existing non-abstract connect()
helper, so the "optional overrides" block stays together.

  async def load_stats(self, collection_name: str) -> PersistedStats | None:
      """Load persisted statistics for the collection, or None if not yet saved.

      Default implementation returns None (stats persistence not supported).
      Backends that support it store the stats as a JSON blob under the key
      f"_medha_stats_{collection_name}" (in a dedicated metadata location
      appropriate for the backend — not in the main vector index).

      Returns:
          PersistedStats if previously saved, None otherwise.

      Raises:
          StorageError: If the load fails (not if simply absent).
      """
      return None

  async def save_stats(
      self,
      collection_name: str,
      stats: PersistedStats,
  ) -> None:
      """Persist statistics for the collection.

      Default implementation is a no-op (stats persistence not supported).

      Args:
          collection_name: Target collection.
          stats:           Snapshot to persist.

      Raises:
          StorageError: If the save fails.
      """
      return

RATIONALE (do not "simplify" this back to @abstractmethod): 0.4.0 already broke
every custom backend by adding update_feedback() as abstract. Repeating that in
0.5.0 for an optional feature is not worth a second breaking change — the pair
degrades cleanly, since core treats "no persisted stats" as a first-run cold
start anyway (see Step 4.6). All ten built-in backends still override both
(Steps 4.3 and 4.4); the defaults exist for third-party subclasses only.

Import PersistedStats at the top of the file:
  from medha.types import PersistedStats

Validate:
  python -c "from medha.interfaces.storage import VectorStorageBackend; print('OK')"
  pytest tests/unit/test_storage_interface.py -x -q
  ALL tests must still pass — including test_partial_implementation_fails, whose
  TypeError comes from the genuinely abstract methods it omits (search, upsert,
  scroll, count, delete, close), not from the two new ones. If that test fails,
  you made the new methods abstract by mistake.

Also add a contract test to tests/unit/test_storage_interface.py asserting the
defaults are inherited rather than required:

  async def test_stats_methods_are_optional(self):
      """A backend that does not override load/save_stats still instantiates."""
      # build a minimal concrete subclass implementing ONLY the abstract methods
      # (reuse the existing helper/fixture pattern in this file), then:
      assert await backend.load_stats("c") is None
      await backend.save_stats("c", PersistedStats())  # must not raise
```

---

## Step 4.3 — `load_stats` / `save_stats` on InMemoryBackend

**Files:** `src/medha/backends/memory.py`

```
Read src/medha/backends/memory.py. Look at how __init__ sets up self._store
and how other metadata (like usage counts) is handled.

Implement load_stats() and save_stats() on InMemoryBackend.

Strategy: store persisted stats in a separate dict self._meta_store keyed by
collection name. No serialization needed (in-memory).

1. In __init__, add:
     self._meta_store: dict[str, PersistedStats] = {}

2. Add the two methods after save_stats:

   async def load_stats(self, collection_name: str) -> PersistedStats | None:
       async with self._lock:
           return self._meta_store.get(collection_name)

   async def save_stats(self, collection_name: str, stats: PersistedStats) -> None:
       async with self._lock:
           self._meta_store[collection_name] = stats

Import PersistedStats at the top of the file.

Note: InMemoryBackend stats do NOT survive process restarts (by design — it is
an in-memory backend). Persistence behaviour is meaningful for disk/network backends.
The interface is implemented for contract compliance and for testing.

Validate:
  python -c "
  import asyncio
  from medha.backends.memory import InMemoryBackend
  from medha.types import PersistedStats

  async def test():
      b = InMemoryBackend()
      await b.initialize()
      assert await b.load_stats('col') is None
      s = PersistedStats(total_requests=5, total_hits=3)
      await b.save_stats('col', s)
      loaded = await b.load_stats('col')
      assert loaded.total_requests == 5
      print('OK')

  asyncio.run(test())
  "
  pytest tests/unit/test_inmemory_backend.py -x -q
```

---

## Step 4.4 — `load_stats` / `save_stats` on all remaining backends

**Files:** `src/medha/backends/qdrant.py`, `src/medha/backends/_asyncpg_mixin.py`,
           `src/medha/backends/elasticsearch.py`, `src/medha/backends/chroma.py`,
           `src/medha/backends/weaviate.py`, `src/medha/backends/redis_vector.py`,
           `src/medha/backends/azure_search.py`, `src/medha/backends/lancedb.py`

```
Read src/medha/backends/memory.py to understand the expected contract.
For each backend below, implement load_stats() and save_stats() using the
backend's native key-value or metadata storage mechanism.

The stats are stored as a JSON string under the key:
  f"_medha_stats_{collection_name}"

--- _AsyncpgMixin (covers PgVectorBackend and VectorChordBackend) ---
Add load_stats / save_stats to _asyncpg_mixin.py.
Use a separate PostgreSQL table: {pg_schema}._medha_stats
  CREATE TABLE IF NOT EXISTS {schema}._medha_stats (
      collection_name TEXT PRIMARY KEY,
      stats_json      TEXT NOT NULL,
      updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
  )
Create this table in initialize() (call CREATE TABLE IF NOT EXISTS).
  load_stats: SELECT stats_json WHERE collection_name = $1; parse with PersistedStats.model_validate_json()
  save_stats: INSERT ... ON CONFLICT DO UPDATE SET stats_json = $2, updated_at = now()
  SECURITY: the only interpolated identifier is {schema} — use the validated
  self._settings.pg_schema (already checked against _SAFE_IDENTIFIER_RE). The
  table name is a fixed literal; collection_name and stats_json are bind params
  ($1/$2), never interpolated. If you create this table inside
  VectorChordBackend.initialize(), keep the existing _validate_vc_lists() call.

--- QdrantBackend ---
Use Qdrant's payload-only collection for metadata or a separate "medha_meta"
collection. Simpler: use a dedicated Qdrant collection named
f"{collection_name}__meta" with a single point (id=0) whose payload contains
{"stats_json": "<json string>"}.
  load_stats: retrieve point 0 from the meta collection; parse stats_json field.
             If collection does not exist or point missing, return None.
  save_stats: upsert point 0 with {"stats_json": stats.model_dump_json()}.

--- ElasticsearchBackend ---
Use a dedicated index f"{index_name}__meta" with document id "_stats".
  load_stats: GET _meta/_doc/_stats; parse "_source.stats_json".
  save_stats: INDEX _meta/_doc/_stats with {"stats_json": stats.model_dump_json()}.

--- ChromaBackend ---
Use a dedicated ChromaDB collection f"{chroma_collection_name}__meta".
Store a single document with id="_stats" and metadata={"stats_json": "<json>"}.
  load_stats: collection.get(ids=["_stats"]); parse metadatas[0]["stats_json"].
  save_stats: collection.upsert(ids=["_stats"], metadatas=[{"stats_json": ...}]).

--- WeaviateBackend ---
Use a dedicated Weaviate class f"{class_name}Meta" with property "statsJson".
Store a single object with a fixed UUID derived from the collection name:
  import uuid; meta_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, collection_name))
  load_stats: fetch the object by meta_id; parse statsJson property.
  save_stats: upsert the object.

--- RedisVectorBackend ---
Use a plain Redis string key (not a hash): f"{key_prefix}:__stats:{collection_name}"
  load_stats: GET the key; if None return None; else PersistedStats.model_validate_json(value).
  save_stats: SET the key to stats.model_dump_json().

--- AzureSearchBackend ---
Use a dedicated Azure AI Search index f"{index_name}-meta".
Store a single document with key="stats_{collection_name}".
  load_stats: get_document(key); parse "stats_json" field.
  save_stats: merge_or_upload_documents([{"id": key, "stats_json": ...}]).

--- LanceDBBackend ---
Use a dedicated LanceDB table f"{table_prefix}_meta" (sanitise the name with the
existing _table_name regex, do not f-string raw input).
Store a single row with id=collection_name and a "stats_json" text column.
  load_stats: search table where id == collection_name; parse stats_json.
  save_stats: upsert the row.
  SECURITY: build the .where() filter as f"id = '{quote_sql_literal(collection_name)}'"
  using quote_sql_literal from medha.backends._escape (the same helper the other
  LanceDB lookups now use). Do NOT reintroduce inline .replace("'", "''").

For all backends:
  - Import PersistedStats at the top of each file.
  - Wrap exceptions in StorageError.
  - load_stats must return None (not raise) when data is absent.
  - SECURITY: derive the meta collection/index/table/key name from collection_name
    with the SAME sanitiser the backend already uses (_table_name, _index_name,
    _az_index_name, _safe_name, uuid5, …) — never a raw f-string of untrusted input.
  - SECURITY: for any single-quoted SQL/OData filter literal, use quote_sql_literal()
    from medha.backends._escape; do NOT inline `'` -> `''`. Prefer document-key /
    bind-parameter APIs over filter strings where the backend offers them.
  - Stats are serialised with PersistedStats.model_dump_json() and parsed with
    PersistedStats.model_validate_json() (JSON only — no pickle), consistent with
    SECURITY.md.

Validate each backend individually:
  pytest tests/unit/test_<backend>_backend.py -x -q
After all backends: pytest tests/unit/ -x -q
```

---

## Step 4.5 — Settings: `stats_persist_interval`

**Files:** `src/medha/config.py`

```
Read src/medha/config.py — specifically the stats-related settings block.

Add one new field to Settings, in the "# --- Stats ---" section:

  stats_persist_interval: int = Field(
      default=100,
      ge=1,
      description=(
          "Persist CacheStats to the backend after every N requests. "
          "Set to 1 to persist on every request (accurate but slower). "
          "Set to a large number to reduce write frequency. "
          "Env var: MEDHA_STATS_PERSIST_INTERVAL."
      ),
  )

Do not change any other field.

Validate: pytest tests/unit/test_config.py -x -q
```

---

## Step 4.6 — Core: load stats on `start()`, persist on `search()`

**Files:** `src/medha/core.py`

```
Read src/medha/core.py — specifically start(), search(), and the internal stats
accumulation logic (look for self._stats or the CacheStats update calls).

Make two targeted changes to src/medha/core.py:

--- Change A: load persisted stats in start() ---

At the end of start() (after the connectivity check added in 0.4.3, before the
background cleanup task), add:

  if self._settings.collect_stats:
      try:
          persisted = await self._backend.load_stats(self._collection_name)
          if persisted is not None:
              self._load_persisted_stats(persisted)
      except StorageError as exc:
          _logger.warning("Could not load persisted stats: %s", exc)

Add a private method _load_persisted_stats(persisted: PersistedStats) that
copies the counts from persisted into the in-memory stats accumulators.
Map fields: total_requests, total_hits, total_misses, total_errors, hits_by_strategy.

--- Change B: persist stats periodically in search() ---

After updating the in-memory stats at the end of search() (look for the block
that increments hit/miss counters), add:

  if (
      self._settings.collect_stats
      and self._settings.stats_persist_interval > 0
      and self._request_count % self._settings.stats_persist_interval == 0
  ):
      asyncio.create_task(self._persist_stats_task())

Where self._request_count is an int counter incremented on each search() call
(add it to __init__ as self._request_count: int = 0 if not already present).

Add _persist_stats_task() as a private async method:

  async def _persist_stats_task(self) -> None:
      try:
          snapshot = self._build_persisted_stats()
          await self._backend.save_stats(self._collection_name, snapshot)
      except Exception as exc:
          _logger.warning("Could not persist stats: %s", exc)

Add _build_persisted_stats() as a private method that reads the in-memory
accumulators and returns a PersistedStats instance.

Import PersistedStats at the top of core.py.

Validate:
  pytest tests/unit/test_core_waterfall.py tests/unit/test_stats.py -x -q
  All existing tests must pass.
```

---

## Step 4.7 — Update `medha stats` CLI to show persisted metrics

**Files:** `src/medha/cli/_app.py`

```
Read the stats command in src/medha/cli/_app.py.

Currently stats shows: collection, backend_type, entries, templates.
Update it to also show persisted metrics if available.

Change:

  async with _build_medha(collection, settings) as m:
      count = await m._backend.count(collection)
      # template count (existing logic)
      persisted = await m._backend.load_stats(collection)

If persisted is not None:
  - Human output: add lines:
      Requests : {persisted.total_requests}
      Hit rate : {persisted.hit_rate:.1%}
      By strategy:
        L1        : {persisted.hits_by_strategy.get('l1_cache', 0)}
        Template  : {persisted.hits_by_strategy.get('template', 0)}
        Exact     : {persisted.hits_by_strategy.get('exact', 0)}
        Semantic  : {persisted.hits_by_strategy.get('semantic', 0)}
        Fuzzy     : {persisted.hits_by_strategy.get('fuzzy', 0)}
  - JSON output: add to the JSON dict:
      "total_requests": persisted.total_requests,
      "hit_rate": persisted.hit_rate,
      "hits_by_strategy": persisted.hits_by_strategy,

If persisted is None:
  - Human output: add line: "Stats    : not yet persisted (run some searches first)"
  - JSON output: "total_requests": null, "hit_rate": null

Import PersistedStats in the CLI file if needed.

Validate:
  medha stats --help
  medha stats         (human output shows Requests line)
  medha stats --json  (JSON has hit_rate key)
```

---

## Step 4.8 — `Settings.feedback_boost_factor`

**Files:** `src/medha/config.py`

```
Read src/medha/config.py — specifically the feedback settings block.

Add one new field to Settings, in the "# --- Feedback ---" section,
after feedback_incorrect_threshold:

  feedback_boost_factor: float = Field(
      default=0.0,
      ge=0.0,
      le=1.0,
      description=(
          "When > 0, the similarity score of a cached result is multiplied by "
          "(1 + feedback_boost_factor * trust), where trust = feedback_correct / "
          "(feedback_correct + feedback_incorrect). "
          "A trust of 1.0 (all positive feedback) boosts the score by feedback_boost_factor. "
          "Default 0.0 disables boosting (backward compatible). "
          "Env var: MEDHA_FEEDBACK_BOOST_FACTOR."
      ),
  )

Do not change any other field.

Validate: pytest tests/unit/test_config.py -x -q
python -c "from medha.config import Settings; s = Settings(); print(s.feedback_boost_factor)"
must print: 0.0
```

---

## Step 4.9 — Feedback score boosting in waterfall search

**Files:** `src/medha/core.py`

```
Read src/medha/core.py — specifically the _waterfall_search() private method
(introduced in step 3.2 of 0.4.3) or the inline waterfall search logic.
Find the point where semantic similarity results are evaluated against the
score threshold (score_threshold_semantic).

Add a private helper method _apply_feedback_boost() to the Medha class:

  def _apply_feedback_boost(
      self,
      score: float,
      feedback_correct: int,
      feedback_incorrect: int,
  ) -> float:
      if self._settings.feedback_boost_factor == 0.0:
          return score
      total = feedback_correct + feedback_incorrect
      if total == 0:
          return score
      trust = feedback_correct / total
      boosted = score * (1.0 + self._settings.feedback_boost_factor * trust)
      return min(1.0, boosted)

Apply the boost in the semantic search step of the waterfall:
After retrieving candidates from the backend with similarity search (the step
that checks score >= settings.score_threshold_semantic), for each candidate call:
  adjusted_score = self._apply_feedback_boost(
      result.score,
      result.feedback_correct,
      result.feedback_incorrect,
  )
  if adjusted_score >= self._settings.score_threshold_semantic:
      # use this result (update result.score to adjusted_score for transparency)

Also apply the boost in the fuzzy fallback step for consistency.

Do NOT apply the boost to L1 cache hits, exact hash hits, or template matches
(those are already perfect matches; boosting is only meaningful for similarity-based ranking).

Validate:
  python -c "
  from medha.core import Medha
  from medha.config import Settings
  m = Medha.__new__(Medha)
  m._settings = Settings(feedback_boost_factor=0.5)
  # With 4 correct, 1 incorrect: trust = 0.8
  # boost = 0.5 * 0.8 = 0.4
  # 0.85 * 1.4 = 1.19 -> clamped to 1.0
  assert m._apply_feedback_boost(0.85, 4, 1) == 1.0
  # With factor=0.0: no change
  m._settings = Settings(feedback_boost_factor=0.0)
  assert m._apply_feedback_boost(0.85, 100, 0) == 0.85
  print('OK')
  "
  pytest tests/unit/test_core_waterfall.py tests/unit/test_feedback.py -x -q
```

---

## Step 4.10 — Weaviate E2E test

**Files:** `tests/integration/test_weaviate_e2e.py`

```
Read tests/integration/test_qdrant_backend.py and tests/integration/test_end_to_end.py
as reference for E2E test structure.

Create tests/integration/test_weaviate_e2e.py.

This test requires a running Weaviate instance. Skip automatically if not available:

  import os
  import pytest
  pytestmark = pytest.mark.skipif(
      not os.environ.get("WEAVIATE_TEST_URL"),
      reason="WEAVIATE_TEST_URL not set"
  )

The test cases should mirror the existing test_end_to_end.py flow but use
WeaviateBackend instead of InMemoryBackend:

class TestWeaviateE2E:

  @pytest.fixture
  async def weaviate_medha(mock_embedder):
      from medha.backends.weaviate import WeaviateBackend
      settings = Settings(
          backend_type="weaviate",
          weaviate_mode="local",
          weaviate_host=os.environ["WEAVIATE_TEST_URL"],
          score_threshold_exact=0.99,
          score_threshold_semantic=0.85,
      )
      backend = WeaviateBackend(settings)
      m = Medha("weaviate_e2e_test", mock_embedder, backend, settings)
      await m.start()
      yield m
      await m.invalidate_collection("weaviate_e2e_test")
      await m.close()

  async def test_store_and_search_semantic(weaviate_medha):
  async def test_template_match(weaviate_medha):
  async def test_l1_cache_hit(weaviate_medha):
  async def test_invalidation(weaviate_medha):
  async def test_feedback_update(weaviate_medha):
  async def test_load_save_stats(weaviate_medha):  # new in 0.5.0

Mark all tests with @pytest.mark.slow @pytest.mark.integration.

Run: WEAVIATE_TEST_URL=http://localhost:8080 pytest tests/integration/test_weaviate_e2e.py -x -q
```

---

## Step 4.11 — Azure AI Search E2E test

**Files:** `tests/integration/test_azure_search_e2e.py`

```
Read tests/integration/test_weaviate_e2e.py (just created) for the skip pattern.

Create tests/integration/test_azure_search_e2e.py.

Skip if AZURE_SEARCH_ENDPOINT is not set:
  pytestmark = pytest.mark.skipif(
      not os.environ.get("AZURE_SEARCH_ENDPOINT"),
      reason="AZURE_SEARCH_ENDPOINT not set"
  )

class TestAzureSearchE2E:

  @pytest.fixture
  async def azure_medha(mock_embedder):
      from medha.backends.azure_search import AzureSearchBackend
      settings = Settings(
          backend_type="azure-search",
          azure_search_endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
          azure_search_api_key=os.environ.get("AZURE_SEARCH_API_KEY"),
          score_threshold_semantic=0.85,
      )
      backend = AzureSearchBackend(settings)
      m = Medha("azure_e2e_test", mock_embedder, backend, settings)
      await m.start()
      yield m
      await m.invalidate_collection("azure_e2e_test")
      await m.close()

Same test cases as the Weaviate E2E above:
  async def test_store_and_search_semantic(azure_medha)
  async def test_template_match(azure_medha)
  async def test_l1_cache_hit(azure_medha)
  async def test_invalidation(azure_medha)
  async def test_feedback_update(azure_medha)
  async def test_load_save_stats(azure_medha)

Mark with @pytest.mark.slow @pytest.mark.integration.

Run: AZURE_SEARCH_ENDPOINT=https://... AZURE_SEARCH_API_KEY=... pytest tests/integration/test_azure_search_e2e.py -x -q
```

---

## Step 4.12 — Unit tests for new 0.5.0 features

**Files:** `tests/unit/test_persisted_stats.py`,
           `tests/unit/test_feedback_boost.py`

```
--- tests/unit/test_persisted_stats.py ---

class TestPersistedStatsModel:

  def test_default_hit_rate_is_zero():
      s = PersistedStats()
      assert s.hit_rate == 0.0

  def test_hit_rate_calculation():
      s = PersistedStats(total_requests=10, total_hits=7)
      assert s.hit_rate == pytest.approx(0.7)

  def test_miss_rate_sum_not_required_to_be_1():
      # miss_rate = misses/requests; errors are neither hits nor misses
      s = PersistedStats(total_requests=10, total_hits=7, total_misses=2, total_errors=1)
      assert s.miss_rate == pytest.approx(0.2)

class TestInMemoryBackendStats:

  async def test_load_returns_none_before_save(inmemory_backend):
      result = await inmemory_backend.load_stats("col")
      assert result is None

  async def test_save_and_load_roundtrip(inmemory_backend):
      s = PersistedStats(total_requests=5, total_hits=3, hits_by_strategy={"semantic": 3})
      await inmemory_backend.save_stats("col", s)
      loaded = await inmemory_backend.load_stats("col")
      assert loaded.total_requests == 5
      assert loaded.hits_by_strategy["semantic"] == 3

  async def test_save_overwrites_previous(inmemory_backend):
      await inmemory_backend.save_stats("col", PersistedStats(total_requests=1))
      await inmemory_backend.save_stats("col", PersistedStats(total_requests=2))
      loaded = await inmemory_backend.load_stats("col")
      assert loaded.total_requests == 2

class TestStorageInterfaceStats:
  # Add to existing TestBackendContract in test_storage_interface.py
  # (update that file rather than creating a new one)

  async def test_load_stats_returns_none_initially(any_backend):
  async def test_save_stats_and_load_roundtrip(any_backend):
  async def test_stats_methods_are_optional():
      # Confirm the NON-breaking contract from Step 4.2: a subclass implementing
      # only the abstract methods (no load_stats/save_stats override) still
      # instantiates, load_stats() returns None and save_stats() is a no-op.
      # Do NOT assert TypeError here — these two methods are deliberately not
      # abstract.

--- tests/unit/test_feedback_boost.py ---

class TestApplyFeedbackBoost:

  def test_boost_disabled_by_default():
      # Settings(feedback_boost_factor=0.0); call _apply_feedback_boost
      # Assert score unchanged regardless of feedback counts

  def test_no_feedback_no_change():
      # Settings(feedback_boost_factor=0.5)
      # feedback_correct=0, feedback_incorrect=0 → trust=0 → score unchanged

  def test_full_positive_feedback():
      # feedback_correct=10, feedback_incorrect=0 → trust=1.0
      # score=0.8, factor=0.5 → boosted = 0.8 * 1.5 = 1.2 → clamped to 1.0

  def test_mixed_feedback():
      # feedback_correct=3, feedback_incorrect=1 → trust=0.75
      # score=0.85, factor=0.4 → boost = 0.4 * 0.75 = 0.3
      # boosted = 0.85 * 1.3 = 1.105 → clamped to 1.0

  def test_all_negative_feedback():
      # feedback_correct=0, feedback_incorrect=5 → trust=0.0
      # Score unchanged (no downgrade)

  async def test_boosted_result_passes_threshold(medha_memory):
      # Store an entry and set feedback_correct=10
      # Normally the score might be 0.82 (just below threshold 0.85)
      # With feedback_boost_factor=0.5 the boosted score should exceed threshold
      # Verify the entry is returned (strategy != no_match)
      pass  # This is a behavioral integration test — see integration tests

Run: pytest tests/unit/test_persisted_stats.py tests/unit/test_feedback_boost.py -x -q
Also update tests/unit/test_storage_interface.py to add the two new contract tests.
```

---

## Step 4.13 — Integration tests for new 0.5.0 features

**Files:** `tests/integration/test_persisted_stats_e2e.py`,
           `tests/integration/test_feedback_boost_e2e.py`

```
--- tests/integration/test_persisted_stats_e2e.py ---

class TestPersistedStatsE2E:

  async def test_stats_survive_restart(tmp_path):
      # Use LanceDB (persists to disk) or just InMemoryBackend (conceptual test)
      # Approach with InMemory: same backend instance shared between two Medha instances
      backend = InMemoryBackend()
      settings = Settings(
          backend_type="memory",
          stats_persist_interval=1,  # persist on every request
      )
      m1 = Medha("stats_test", FastEmbedAdapter(), backend, settings)
      await m1.start()
      await m1.store("what is revenue?", "SELECT SUM(revenue) FROM sales")
      await m1.search("what is the total revenue?")   # creates a hit
      await m1.close()

      # Simulate restart by creating a new Medha instance with the SAME backend
      m2 = Medha("stats_test", FastEmbedAdapter(), backend, Settings(
          backend_type="memory", stats_persist_interval=1
      ))
      await m2.start()
      stats = await m2.stats("stats_test")
      assert stats.total_hits >= 1, "Stats must survive across Medha instances"
      await m2.close()

  async def test_cli_stats_shows_hit_rate(tmp_path, monkeypatch):
      # Run a search via Python API against a shared InMemoryBackend
      # Then invoke `medha stats --json` via CliRunner
      # Assert hit_rate is present in the JSON output (not null)
      # Note: CLI creates its own Medha instance, so this test requires
      # backend state to be loaded from persisted stats.
      # Simplest: test that the stats JSON key "hit_rate" exists (value may be null
      # if the CLI backend is fresh).
      pass  # Document the limitation in a comment.

--- tests/integration/test_feedback_boost_e2e.py ---

class TestFeedbackBoostE2E:

  async def test_boost_raises_low_score_above_threshold():
      # Use a low score_threshold_semantic (e.g. 0.5) to ensure semantic search works
      # Store an entry, call feedback(question, correct=True) 5 times
      # Search with a slightly different question (one that normally scores 0.52, below 0.6)
      # With feedback_boost_factor=0.5 and trust=1.0, 0.52 * 1.5 = 0.78 → above threshold
      # Assert the result is a hit
      # Note: exact score values depend on the embedder model.
      # Use a threshold/boost combination that is robust.
      pass

  async def test_no_boost_when_factor_is_zero():
      # Same setup; Settings(feedback_boost_factor=0.0)
      # Assert the result is still a miss (score unchanged)
      pass

Run: pytest tests/integration/test_persisted_stats_e2e.py tests/integration/test_feedback_boost_e2e.py -x -q
```

---

## Step 4.14 — Demo notebook: persistent stats and feedback boosting

**Files:** `demo/28_persistent_stats.ipynb`, `demo/29_feedback_boosting.ipynb`

```
--- demo/28_persistent_stats.ipynb ---

Requires: pip install "medha-archai[fastembed,lancedb]"

Section structure:
1. The problem — markdown: "CacheStats is in-memory; every process restart resets
   it. From 0.5.0, Medha persists a PersistedStats snapshot to the backend after
   every N requests (configurable via Settings.stats_persist_interval)."
2. Setup with LanceDB (persists to disk):
   - Settings(backend_type="lancedb", stats_persist_interval=5)
   - Store 10 entries, run 20 searches.
3. Simulate restart — create a new Medha instance pointing to the same LanceDB path.
   Call medha.stats() and show that total_requests and hit_rate survived.
4. `medha stats --json` shell output — show the enriched stats with hit_rate.
5. Tuning stats_persist_interval — markdown table:
   | interval | write frequency | accuracy |
   | 1        | every request   | exact    |
   | 100      | every 100 req   | ± 100    |
   | 1000     | every 1000 req  | ± 1000   |

--- demo/29_feedback_boosting.ipynb ---

Requires: pip install "medha-archai[fastembed]"

Section structure:
1. The problem — markdown: "The feedback API introduced in 0.4.0 accumulates
   correct/incorrect signals but does not change search results. From 0.5.0,
   positive feedback boosts the similarity score of highly-rated entries."
2. Setup — Settings(feedback_boost_factor=0.3, score_threshold_semantic=0.75)
3. Before feedback — store 1 entry; show that a slightly different question
   barely misses the threshold (or just hits it).
4. After feedback — call feedback(question, correct=True) 5 times.
   Run the same search again; show score comparison in a table.
5. Trust formula — markdown:
   "trust = feedback_correct / (feedback_correct + feedback_incorrect)
    adjusted_score = min(1.0, score * (1 + factor * trust))"
6. Choosing the factor — guidance: 0.0 = disabled, 0.1 = subtle, 0.5 = aggressive.
   Recommendation: start with 0.2 and measure hit rate change.
7. Relationship with auto-invalidation — both features compose:
   incorrect signals can remove bad entries; correct signals boost good ones.

Use InMemoryBackend + FastEmbedAdapter throughout.
Each section must have a markdown explanation cell.
```

---

## Step 4.15 — CHANGELOG and version bump

**Files:** `CHANGELOG.md`, `pyproject.toml`, `src/medha/__init__.py`

```
1. Bump version to "0.5.0" in pyproject.toml and src/medha/__init__.py.

2. Add [0.5.0] section at the top of CHANGELOG.md.

   NOTE: CHANGELOG.md already carries an `## [Unreleased]` section holding the
   four `### Security` bullets below (added right after 0.4.3, when that work was
   merged). Do NOT duplicate them: rename `## [Unreleased]` to `## [0.5.0] — <date>`
   and add the `### Added` section above the existing `### Security` one.

   ## [0.5.0] — <date>

   ### Added

   - **Persistent `CacheStats`**: Medha now persists a `PersistedStats` snapshot to
     the backend after every `Settings.stats_persist_interval` requests (default 100).
     Snapshots are loaded on `start()`, so hit-rate and request-count metrics survive
     process restarts. `medha stats` (CLI) now shows hit rate and per-strategy breakdown
     when persisted data is available.

   - **`PersistedStats`** model (`medha.types`): Pydantic snapshot with
     `total_requests`, `total_hits`, `total_misses`, `total_errors`,
     `hits_by_strategy`, `last_reset_at`, `updated_at`, and computed
     `hit_rate` / `miss_rate` properties.

   - **`VectorStorageBackend.load_stats()`** and **`save_stats()`**: two new
     methods implemented by all ten built-in backends. They are **not** abstract —
     they ship with a `return None` / no-op default, so existing custom backends
     keep working unchanged and simply opt out of stats persistence.

   - **`Settings.stats_persist_interval`** (default `100`, env
     `MEDHA_STATS_PERSIST_INTERVAL`): how often (in requests) the in-memory stats
     are flushed to the backend.

   - **Feedback score boosting**: when `Settings.feedback_boost_factor > 0`, the
     similarity score of a result is adjusted upward proportionally to the fraction
     of positive feedback it has received. Formula:
     `adjusted = min(1.0, score × (1 + factor × trust))` where
     `trust = feedback_correct / (feedback_correct + feedback_incorrect)`.
     Default is `0.0` (disabled — fully backward compatible).

   - **`Settings.feedback_boost_factor`** (`float`, default `0.0`, range `[0.0, 1.0]`,
     env `MEDHA_FEEDBACK_BOOST_FACTOR`).

   - **Weaviate E2E test** (`tests/integration/test_weaviate_e2e.py`): full end-to-end
     test for `WeaviateBackend`, skipped when `WEAVIATE_TEST_URL` is not set.

   - **Azure AI Search E2E test** (`tests/integration/test_azure_search_e2e.py`):
     full end-to-end test for `AzureSearchBackend`, skipped when
     `AZURE_SEARCH_ENDPOINT` is not set.

   ### Security

   - **FIPS compatibility**: `question_hash()` / `query_hash()` now call
     `hashlib.md5(..., usedforsecurity=False)` (the digests are cache keys, not a
     security control). This unblocks execution on FIPS-enabled hosts (where bare
     `md5()` raises) and silences static analysers. Digest values are unchanged —
     existing caches stay compatible.

   - **Centralised filter escaping**: single-quoted SQL/OData literals in
     `AzureSearchBackend` (OData `$filter`) and `LanceDBBackend` (DataFusion
     `where`) now go through one audited helper,
     `medha.backends._escape.quote_sql_literal`, instead of scattered inline
     `'` → `''` replacements. New `load_stats`/`save_stats` filter clauses reuse it.

   - **VectorChord input validation**: `VectorChordBackend.initialize()` validates
     `vc_lists` before it is interpolated into the `CREATE INDEX … WITH (lists=…)`
     DDL, rejecting any non-integer passed via `**kwargs` (which bypasses Pydantic)
     with `StorageInitializationError`. The `Settings` path (already typed
     `list[int]`) is unchanged.

   - **`SECURITY.md`** added — documents the trust model: stored queries are
     returned verbatim, templates/`parameter_patterns` are trusted config (ReDoS),
     file loading honours `allowed_file_dir` / `max_file_size_mb`, secrets use
     `SecretStr`, and backend identifiers are validated/sanitised.

3. Run: pytest -x -q
   All tests must pass before tagging 0.5.0.
```

---

## Validation checklist (full roadmap)

Run these commands after each release is complete:

```
# 0.4.1
pytest tests/unit/ -q
pytest tests/integration/test_cli_e2e.py -q -m cli
medha --help
medha search --help
medha health --json
python -c "import medha; print(medha.__version__)"  # 0.4.1

# 0.4.2
pytest tests/unit/test_openai_compatible_adapter.py tests/unit/test_mistral_adapter.py -q
python -c "from medha.config import Settings; Settings(embedder_type='openai-compatible')"
python -c "from medha.config import Settings; Settings(embedder_type='mistral')"
python -c "import medha; print(medha.__version__)"  # 0.4.2

# 0.4.3
pytest tests/unit/test_context_manager.py tests/unit/test_search_batch.py tests/unit/test_startup_validation.py -q
pytest tests/integration/test_context_manager_e2e.py tests/integration/test_search_batch_e2e.py -q
python -c "
import asyncio
from medha import Medha
from medha.backends.memory import InMemoryBackend
from medha.config import Settings
from medha.embeddings.fastembed_adapter import FastEmbedAdapter

async def test():
    async with Medha('t', FastEmbedAdapter(), InMemoryBackend(), Settings(backend_type='memory')) as m:
        results = await m.search_batch(['q1', 'q2'])
        assert len(results) == 2
    print('0.4.3 OK')

asyncio.run(test())
"

# 0.5.0
pytest tests/unit/test_persisted_stats.py tests/unit/test_feedback_boost.py -q
pytest tests/integration/test_persisted_stats_e2e.py tests/integration/test_feedback_boost_e2e.py -q
python -c "from medha.types import PersistedStats; s = PersistedStats(total_requests=10, total_hits=7); print(s.hit_rate)"
# must print: 0.7
python -c "import medha; print(medha.__version__)"  # 0.5.0
```
