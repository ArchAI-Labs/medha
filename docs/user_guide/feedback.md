# Feedback Loop

A cache hit is not automatically a *correct* answer. Semantic similarity can match a question whose stored query is subtly wrong for the new phrasing — and, without a correction signal, that entry keeps being served forever.

The feedback loop lets your application report whether a served query was right, and optionally retires entries that keep getting it wrong.

---

## Recording Feedback

```python
hit = await cache.search("How many orders were placed last month?")

if hit.found:
    rows = run_query(hit.generated_query)
    # ... user (or a validator) judges the result ...
    await cache.feedback(hit.question, correct=True)
```

`feedback()` looks the entry up by **exact normalized-question match** — not by vector similarity — increments the matching counter, and returns a `bool`:

| Return | Meaning |
|---|---|
| `True` | Entry found and counter updated |
| `False` | No entry for that question (expired, invalidated, or never stored) |

A sync wrapper is available as `feedback_sync()`.

!!! warning "Pass the question, not the query"
    The lookup key is the question text. Pass the same string you passed to `search()` (or `hit.question`), not `hit.generated_query`.

Because the lookup is a plain text match rather than a vector search, `feedback()` does not need a working embedder — this is why the CLI can record feedback without loading a model.

---

## Counters

Every cache entry carries two counters, both defaulting to `0`:

| Field | Description |
|---|---|
| `feedback_correct` | Times the stored query was reported correct |
| `feedback_incorrect` | Times it was reported wrong |

They appear on `CacheEntry` (stored form) and on `CacheResult` (backend search results), so you can inspect them when exporting or auditing a collection:

```python
df = await cache.export_to_dataframe()
print(df[["question", "feedback_correct", "feedback_incorrect"]])
```

Entries written by Medha 0.3.x and earlier are read back with both counters at `0` — the schema change is backward compatible.

---

## Auto-Invalidation

Set `feedback_incorrect_threshold` to have Medha retire an entry once it accumulates enough negative reports:

```python
from medha.config import Settings

settings = Settings(feedback_incorrect_threshold=3)
```

Or via environment:

```bash
export MEDHA_FEEDBACK_INCORRECT_THRESHOLD=3
```

With this set, the third `feedback(question, correct=False)` call increments the counter to 3, triggers `invalidate(question)`, and logs the removal at `INFO`. The next `search()` for that question misses, and your application regenerates the query with the LLM.

The default is `None` — counters accumulate but nothing is ever auto-removed. Start there, watch the counters for a while, and enable the threshold once you know what a realistic error rate looks like for your workload.

!!! note "Only negative feedback triggers invalidation"
    The threshold is checked exclusively on `correct=False` calls. A high `feedback_correct` count never protects an entry from being invalidated, and never removes one.

---

## Score Boosting

!!! info "New in 0.5.0"

Auto-invalidation acts on negative feedback. Score boosting is its counterpart: it makes entries that keep being confirmed *easier to retrieve*.

```python
settings = Settings(feedback_boost_factor=0.25)
```

When the factor is above zero, a candidate's similarity score is adjusted before it is compared against the threshold:

```
trust    = feedback_correct / (feedback_correct + feedback_incorrect)
adjusted = min(1.0, score × (1 + feedback_boost_factor × trust))
```

An entry with only positive feedback (`trust = 1.0`) gets the full boost; one with a mixed record gets a proportional share. Entries with **no feedback, or negative feedback only, are left untouched** — boosting never lowers a score. Use `feedback_incorrect_threshold` to act on negative signal.

### Where it applies

| Tier | Boosted? | Why |
|---|---|---|
| L1 cache | No | Exact question hash — already a certainty |
| Template | No | Pattern match, not a similarity score |
| Exact | No | Hash-equivalent match |
| **Semantic** | **Yes** | Cosine similarity ranking |
| **Fuzzy** | **Yes** | Levenshtein ratio ranking |

Two things change in the semantic tier when boosting is on. Candidates are retrieved down to `score_threshold_semantic / (1 + factor)` — the lowest score a boost could possibly lift over the line, so recall widens by exactly that much and no more. And candidates are **re-ranked** by adjusted score, so a marginally less similar entry with a strong track record can outrank a closer one nobody has confirmed.

### Choosing a factor

| Factor | Effect |
|---|---|
| `0.0` *(default)* | Disabled — scoring is identical to 0.4.x |
| `0.1` | Subtle; breaks ties between near-equal candidates |
| `0.2` – `0.3` | Recommended starting range |
| `0.5`+ | Aggressive; a trusted entry can absorb a large similarity gap |

Start at `0.2` and compare the hit rate before and after — [persistent stats](observability.md#persistent-statistics) make that comparison survive restarts. Raise it only if the extra hits are genuinely correct: the failure mode of a high factor is a well-rated entry answering questions it should not.

The two feedback features compose. `feedback_boost_factor` promotes what works, `feedback_incorrect_threshold` retires what does not:

```python
settings = Settings(
    feedback_boost_factor=0.25,
    feedback_incorrect_threshold=3,
)
```

Set via environment as `MEDHA_FEEDBACK_BOOST_FACTOR` (a float in `[0.0, 1.0]`).

---

## From the CLI

```bash
medha feedback "How many orders were placed last month?" --correct
medha feedback "How many orders were placed last month?" --incorrect
```

See the [CLI](cli.md) page for setup and the full command list.

---

## Choosing a Signal

The hard part is deciding what counts as "correct". Some options, roughly in order of cost:

* **Execution success** — the query ran without an error. Cheap, catches schema drift and broken SQL, but a query can run fine and still answer the wrong question.
* **User action** — the user accepted the result, kept the dashboard, or did not immediately rephrase. A rephrasing within a few seconds is a strong negative signal.
* **Explicit rating** — a thumbs up/down in the UI. Accurate but sparse; most users never click.
* **LLM-as-judge** — a second model compares the question to the generated query. Most expensive, and it reintroduces the LLM call you were caching to avoid — so reserve it for sampling, not every request.

Mixing a cheap always-on signal (execution success) with a sparse high-quality one (explicit rating) tends to work better than either alone.

---

## Related

* [Invalidation](invalidation.md) — manual and bulk removal
* [Observability](observability.md) — hit rate and per-strategy metrics
* [TTL & Lifecycle](ttl_and_lifecycle.md) — time-based expiry

# Feedback Loop

The feedback loop lets your application report whether a cached query was correct or incorrect. Medha accumulates these signals per entry and can automatically invalidate entries that exceed an error threshold.

---

## Recording Feedback

Call `feedback()` with the original question and a boolean indicating correctness:

```python
async with Medha("demo", embedder=embedder, settings=settings) as cache:
    await cache.store(
        "How many active users do we have?",
        "SELECT COUNT(*) FROM users WHERE active = true",
    )

    # User confirmed the query was correct
    await cache.feedback("How many active users do we have?", correct=True)

    # User reported the query was wrong
    await cache.feedback("How many active users do we have?", correct=False)
```

`feedback()` returns `True` if the entry was found and updated, `False` if no entry matched (expired, invalidated, or never stored).

Feedback is resolved by exact normalised-question lookup — the same mechanism as `invalidate()`. It does **not** perform a vector search.

---

## Reading Feedback Counters

After recording feedback, the counters are visible on the `CacheResult` returned by `search()`:

```python
hit = await cache.search("How many active users do we have?")
if hit:
    print(hit.feedback_correct)    # number of correct signals
    print(hit.feedback_incorrect)  # number of incorrect signals
```

They are also stored on the underlying `CacheEntry` and persist across restarts for durable backends (Qdrant, pgvector, etc.).

---

## Auto-Invalidation on Error Threshold

Set `feedback_incorrect_threshold` in `Settings` to automatically remove an entry once its incorrect count reaches the limit:

```python
settings = Settings(
    backend_type="qdrant",
    feedback_incorrect_threshold=3,  # invalidate after 3 incorrect signals
)

async with Medha("demo", embedder=embedder, settings=settings) as cache:
    await cache.store("How many orders exist?", "SELECT COUNT(*) FROM orders")

    await cache.feedback("How many orders exist?", correct=False)
    await cache.feedback("How many orders exist?", correct=False)
    await cache.feedback("How many orders exist?", correct=False)  # triggers invalidation

    # Entry is gone — next search returns NO_MATCH
    hit = await cache.search("How many orders exist?")
    print(hit.strategy)  # SearchStrategy.NO_MATCH
```

!!! note "Correct feedback never invalidates"

    Only incorrect signals count toward the threshold. Any number of correct feedbacks leave the entry untouched.

When auto-invalidation fires, both the vector backend entry and the L1 cache entry are removed atomically. Calling `feedback()` again after invalidation returns `False` without raising an exception.

---

## Threshold via Environment Variable

```bash
export MEDHA_FEEDBACK_INCORRECT_THRESHOLD=5
```

Set to `None` (the default) to disable auto-invalidation entirely — counters accumulate but no entry is ever removed automatically.

---

## Behaviour Reference

| Scenario | Return value | Side effect |
|---|---|---|
| Entry found, `correct=True` | `True` | `feedback_correct` incremented by 1 |
| Entry found, `correct=False`, below threshold | `True` | `feedback_incorrect` incremented by 1 |
| Entry found, `correct=False`, threshold reached | `True` | Entry invalidated from backend and L1 |
| Entry not found | `False` | No change |
| Entry already invalidated, called again | `False` | No change |

---

## Typical Integration Pattern

```python
async def handle_user_correction(question: str, was_correct: bool, cache: Medha) -> None:
    updated = await cache.feedback(question, correct=was_correct)
    if not updated:
        # Entry expired or was never cached — nothing to update
        return
    if not was_correct:
        # Optionally log for audit
        logger.warning("Incorrect cache hit reported for: %s", question[:80])
```

---

## See Also

- [Invalidation](invalidation.md) — manual removal strategies
- [TTL & Lifecycle](ttl_and_lifecycle.md) — time-based expiry
- [Configuration](configuration.md) — `feedback_incorrect_threshold` setting
- [Demo 25 — Feedback Loop](https://github.com/ArchAI-Labs/medha/blob/main/demo/25_feedback_loop.ipynb)
