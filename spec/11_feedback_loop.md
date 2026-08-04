# Spec 11 — Feedback Loop (v0.4.0)

## Goal

Allow callers to mark a cache hit as correct or incorrect.
Feedback is stored as two counters (`feedback_correct`, `feedback_incorrect`) on
the `CacheEntry` payload. When `Settings.feedback_incorrect_threshold` is set,
`feedback()` automatically invalidates the entry once the incorrect count reaches
the threshold — making the feature immediately useful without requiring manual
intervention.

---

## API

### `Medha.feedback(question, correct)`

```python
async def feedback(
    self,
    question: str,
    correct: bool,
) -> bool:
    """Record feedback for a previously cached question.

    Locates the entry by exact normalized-question match, increments
    feedback_correct or feedback_incorrect, and — if
    Settings.feedback_incorrect_threshold is set and the incorrect count
    has reached it — automatically invalidates the entry.

    Args:
        question: The original natural-language question.
        correct:  True → the cached query was correct; False → it was wrong.

    Returns:
        True  if the entry was found and updated.
        False if no entry exists for the question (expired, invalidated, or
              never stored).
    """
```

Usage:

```python
# Basic: just record feedback
hit = await cache.search("How many users are registered?")
await cache.feedback("How many users are registered?", correct=True)
await cache.feedback("How many users are registered?", correct=False)

# With auto-invalidation: entry is removed once incorrect count reaches 3
settings = Settings(feedback_incorrect_threshold=3)
cache = Medha("my_cache", embedder=embedder, settings=settings)
```

---

## Data model changes

### `CacheEntry` (types.py)

Two new fields with `default=0`. Fully backward compatible: existing serialised
entries without these keys deserialise to 0.

```python
feedback_correct:   int = Field(default=0, ge=0)
feedback_incorrect: int = Field(default=0, ge=0)
```

### `CacheResult` (types.py)

Same two fields, same defaults. Lets callers inspect accumulated feedback
counters without a separate lookup.

```python
feedback_correct:   int = Field(default=0)
feedback_incorrect: int = Field(default=0)
```

---

## `Settings` change (config.py)

One new field:

```python
feedback_incorrect_threshold: int | None = Field(
    default=None,
    ge=1,
    description=(
        "When set, feedback(correct=False) automatically invalidates the entry "
        "once feedback_incorrect reaches this value. None disables auto-invalidation."
    ),
)
```

Env var: `MEDHA_FEEDBACK_INCORRECT_THRESHOLD`. Default `None` preserves existing
behaviour for users who do not opt in.

---

## Interface change — `VectorStorageBackend`

New abstract method in `interfaces/storage.py`.
**Return type is `int`** (the new counter value after the update):

```python
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
```

Returning the new counter value avoids a second lookup in `core.py` for the
threshold check. The convention "return 0 when not found" mirrors the existing
`update_usage_count` behaviour (silent no-op on missing ID).

### Breaking change scope

Any custom `VectorStorageBackend` subclass not included in the project will
raise `TypeError` at instantiation until it implements `update_feedback()`.
This is the same pattern as 0.3.1 (`find_expired`, `find_by_query_hash`,
`find_by_template_id`, `drop_collection`). Document in CHANGELOG.

### Implementation per backend

| Backend              | Mechanism                                                  | Atomic |
|----------------------|------------------------------------------------------------|--------|
| `InMemoryBackend`    | Dict increment under `asyncio.Lock`; returns new value     | Yes    |
| `QdrantBackend`      | Retrieve → `set_payload(count+1)`; returns written value   | No     |
| `PgVectorBackend`    | `UPDATE … SET … + 1 RETURNING feedback_correct/incorrect`  | Yes    |
| `VectorChordBackend` | Same SQL via `_AsyncpgMixin`                               | Yes    |
| `ElasticsearchBackend` | Painless script `ctx._source.f++`; `_update` API        | Yes    |
| `ChromaBackend`      | Read metadata → write `count+1`; returns written value     | No     |
| `WeaviateBackend`    | `update_object()` read-modify-write; returns written value | No     |
| `RedisVectorBackend` | `HINCRBY` → returns new value directly                     | Yes    |
| `AzureSearchBackend` | Merge document with `count+1`; returns written value       | No     |
| `LanceDBBackend`     | `update()` with expression; returns written value          | No     |

Non-atomic backends (Qdrant, Chroma, Weaviate, Azure, LanceDB) use the same
read-modify-write pattern already present in `update_usage_count`, which ships
in production without issues. Feedback is a human-in-the-loop operation:
concurrent calls on the same entry in the same millisecond are not a realistic
scenario. In the worst case a lost update shifts the threshold trigger by one
call — the entry will be invalidated on the next feedback instead.

The auto-invalidation path is safe under any concurrency level because
`invalidate()` is idempotent (returns `False` if the entry is already gone,
no error raised).

---

## `core.py` implementation

### Flow

```
feedback(question, correct)
  1. normalize_question(question)
  2. backend.search_by_normalized_question(collection, normalized)
       → None  → log warning, return False
  3. new_count = await backend.update_feedback(collection, result.id, correct)
  4. If not correct
         and settings.feedback_incorrect_threshold is not None
         and new_count >= settings.feedback_incorrect_threshold:
       await self.invalidate(question)   # idempotent, L1 also cleared
       log info "Auto-invalidated '{question}' after {new_count} incorrect feedbacks"
  5. return True
```

Step 4 uses `self.invalidate()` (already implemented) which handles both
backend deletion and L1 cache eviction in one call.

### Template collection

Template entries live in `__medha_templates_{collection}`. Feedback always
targets the **main** collection. Template-matched results are stored in the
main collection via `store()` after the first template hit, so the lookup
succeeds for templates too once the question has been stored at least once.

If the template match is the very first hit (entry not yet in main collection),
`search_by_normalized_question()` returns `None` and `feedback()` returns
`False`. This is documented behaviour, not a bug.

### L1 cache

No direct L1 interaction for feedback storage. When auto-invalidation fires,
`invalidate()` clears the L1 entry as part of its normal flow.

---

## Known limitations

### 1. Non-atomic increment on five backends

Qdrant, Chroma, Weaviate, Azure AI Search, and LanceDB use read-modify-write.
Under concurrent feedback the counter may be slightly off. This is the same
trade-off already accepted for `update_usage_count`. For the human-in-the-loop
use case it is not a practical concern.

The auto-invalidation consequence: the threshold trigger may fire one call late
(if a write was lost) or one call early (impossible — lost write means a lower
count). The entry will still be invalidated within one extra `feedback()` call.

### 2. Feedback on template-first hits

If a question is served exclusively via template matching and has never been
through `store()`, there is no corresponding main-collection entry.
`feedback()` returns `False`. Callers should handle this gracefully; it is not
an error.

---

## Bugs and regression risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Custom backends break at instantiation (new abstract method) | High (external users) | Document as breaking change; `TypeError` message is self-explanatory |
| `CacheEntry.feedback_*` absent in entries written by 0.3.x | Low | Pydantic v2 defaults to 0 — tested explicitly |
| `update_feedback` returns `int` not `None` — callers that ignore return value are unaffected | None | Return type change is additive |
| Auto-invalidation fires under race condition one call early/late | Negligible | Idempotent `invalidate()`; documented |
| `feedback_incorrect_threshold=1` invalidates on the very first negative signal | Intended | User choice; no guard needed |
| `update_usage_count` inconsistency (not in ABC but contract-tested) | Pre-existing | Out of scope; leave for a dedicated cleanup |

---

## Tests

### Unit — `tests/unit/test_feedback.py`

```
TestFeedbackTypes
  test_cache_entry_feedback_defaults_zero
  test_cache_result_feedback_defaults_zero
  test_cache_entry_backward_compat_no_feedback_fields
      — CacheEntry built from dict without feedback keys → defaults to 0

TestInMemoryBackendUpdateFeedback
  test_update_feedback_correct_returns_new_count
  test_update_feedback_incorrect_returns_new_count
  test_update_feedback_accumulates_and_returns_correct_count
  test_update_feedback_missing_id_returns_zero_no_exception

TestMedhaFeedback
  test_feedback_correct_returns_true
  test_feedback_incorrect_returns_true
  test_feedback_returns_false_when_not_found
  test_feedback_after_invalidate_returns_false
  test_feedback_counters_visible_in_cache_result
  test_feedback_on_l1_hit_updates_backend

TestMedhaFeedbackAutoInvalidation
  test_no_auto_invalidation_when_threshold_is_none
  test_auto_invalidation_fires_at_threshold
      — threshold=3, call feedback(False) three times, entry gone after third
  test_auto_invalidation_does_not_fire_below_threshold
      — threshold=3, two incorrect feedbacks → entry still present
  test_auto_invalidation_clears_l1
      — after auto-invalidation, search returns NO_MATCH (L1 cleared)
  test_auto_invalidation_is_idempotent
      — threshold=1, call feedback(False) twice → no exception on second call
  test_correct_feedback_never_triggers_invalidation
      — threshold=1, call feedback(True) → entry not removed

TestFeedbackSettings
  test_feedback_incorrect_threshold_none_by_default
  test_feedback_incorrect_threshold_accepts_positive_int
  test_feedback_incorrect_threshold_rejects_zero
  test_feedback_incorrect_threshold_from_env_var
```

### Contract test addition — `tests/unit/test_storage_interface.py`

Add to `TestBackendContract` (runs on all backends in `_CONTRACT_BACKENDS`):

```
test_update_feedback_correct_returns_one
    — upsert entry, call update_feedback(correct=True) → returns 1
test_update_feedback_incorrect_returns_one
    — upsert entry, call update_feedback(correct=False) → returns 1
test_update_feedback_accumulates
    — 2 × correct, 1 × incorrect → correct=2, incorrect=1 visible after scroll
test_update_feedback_missing_id_returns_zero
    — update_feedback on non-existent id → returns 0, no exception
```

### Integration — `tests/integration/test_feedback_e2e.py`

Uses `medha_memory` fixture (InMemoryBackend + MockEmbedder):

```
test_full_feedback_loop_correct
test_full_feedback_loop_incorrect
test_feedback_not_found_returns_false
test_feedback_counters_cumulative
test_auto_invalidation_end_to_end
    — Settings(feedback_incorrect_threshold=2), store, 2× feedback(False),
      search → NO_MATCH
test_auto_invalidation_with_l1
    — same as above but search before feedback to populate L1; confirm L1 cleared
```

---

## Demo

### `demo/25_feedback_loop.ipynb`

1. **Setup** — InMemoryBackend + FastEmbedAdapter, seed 5 question-query pairs.
2. **Manual feedback mode** — record correct/incorrect without threshold;
   show counters in a DataFrame via `export_to_dataframe()`.
3. **Auto-invalidation mode** — restart with
   `Settings(feedback_incorrect_threshold=3)`; simulate 3 incorrect feedbacks
   on one entry; confirm the entry disappears from search.
4. **Mixed session** — 10 searches, mixed correct/incorrect feedback,
   show which entries survive and which are auto-invalidated.
5. **Production notes** — race condition caveat for non-Postgres/Redis backends,
   recommended threshold values, note that acting on `feedback_correct` counters
   (e.g. boosting confidence) is a future feature.

---

## CHANGELOG entry (0.4.0)

### Added
- `Medha.feedback(question, correct)` — record a correct/incorrect signal for a
  cached question. Returns `True` if found and updated, `False` if not found.
- `Settings.feedback_incorrect_threshold` (`int | None`, default `None`,
  env `MEDHA_FEEDBACK_INCORRECT_THRESHOLD`) — when set, entries are
  automatically invalidated once `feedback_incorrect` reaches the threshold.
- `CacheEntry.feedback_correct` / `CacheEntry.feedback_incorrect` — new fields,
  default 0, backward compatible with 0.3.x entries.
- `CacheResult.feedback_correct` / `CacheResult.feedback_incorrect` — counters
  visible in backend search results.
- `VectorStorageBackend.update_feedback(collection, point_id, correct) -> int`
  — new abstract method returning the new counter value; implemented by all ten
  built-in backends.

### Breaking Changes
- **Custom `VectorStorageBackend` subclasses** must implement
  `update_feedback(collection_name, point_id, correct) -> int` or raise
  `TypeError` at instantiation.
