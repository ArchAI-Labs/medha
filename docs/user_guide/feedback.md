# Feedback Loop

A cache hit is not automatically a *correct* answer. Semantic similarity can match a question whose stored query is subtly wrong for the new phrasing — and, without a correction signal, that entry keeps being served forever.

The feedback loop lets your application report whether a served query was right. Medha accumulates these signals per entry, can retire entries that keep getting it wrong, and (since 0.5.0) can promote the ones that keep being confirmed.

---

## Recording Feedback

```python
from medha.types import SearchStrategy

question = "How many orders were placed last month?"
hit = await cache.search(question)

if hit.strategy is not SearchStrategy.NO_MATCH:
    rows = run_query(hit.generated_query)
    # ... the user (or a validator) judges the result ...
    await cache.feedback(question, correct=True)
```

`feedback()` resolves the entry by **exact normalised-question lookup** — the same mechanism as `invalidate()`, not a vector search — increments the matching counter, and returns a `bool`:

| Return | Meaning |
|---|---|
| `True` | Entry found and counter updated |
| `False` | No entry for that question (expired, invalidated, or never stored) |

A sync wrapper is available as `feedback_sync()`.

!!! warning "Pass the question, not the query"
    The lookup key is the question text, so keep the original string around — `CacheHit` does not carry it back. Pass the same value you passed to `search()`, never `hit.generated_query`.

Because the lookup is a plain text match, `feedback()` needs no working embedder — which is why the CLI can record feedback without loading a model.

!!! warning "A question with several entries has no defined feedback target"
    `store()` takes the query as a separate argument, so the same question can
    be stored with two different queries. Both entries then carry the same
    normalized question **and the same vector** — the embedding comes from the
    question alone — so they tie at score 1.0, and which one answers a
    `search()` is decided by the order the backend happens to return ties in.

    `feedback()` marks exactly one of them, and not necessarily the one that
    answered: it resolves the question through its own lookup, independently
    of the search that produced the hit. `invalidate()` is the exception — it
    removes all of them.

    Until entries can be told apart, keep one query per question: overwrite by
    calling `invalidate()` before `store()` rather than storing a second
    variant. Distinguishable entries are tracked in
    [issue #36](https://github.com/ArchAI-Labs/medha/issues/36).

---

## Counters

Every cache entry carries two counters, both defaulting to `0`:

| Field | Description |
|---|---|
| `feedback_correct` | Times the stored query was reported correct |
| `feedback_incorrect` | Times it was reported wrong |

They live on `CacheEntry` (the stored form) and on `CacheResult` (backend search results), and persist in the backend, so they survive restarts on every durable backend.

!!! note "They are not on `CacheHit`"
    The object returned by `search()` carries the query, confidence, strategy and expiry — not the feedback counters. To inspect them, export the collection:

    ```python
    df = await cache.export_to_dataframe()
    print(df[["original_question", "feedback_correct", "feedback_incorrect"]])
    ```

Entries written by Medha 0.3.x and earlier are read back with both counters at `0` — the schema change is backward compatible.

---

## Auto-Invalidation

Set `feedback_incorrect_threshold` to have Medha retire an entry once it accumulates enough negative reports:

```python
settings = Settings(feedback_incorrect_threshold=3)

async with Medha("demo", embedder=embedder, settings=settings) as cache:
    await cache.store("How many orders exist?", "SELECT COUNT(*) FROM orders")

    await cache.feedback("How many orders exist?", correct=False)
    await cache.feedback("How many orders exist?", correct=False)
    await cache.feedback("How many orders exist?", correct=False)  # triggers invalidation

    hit = await cache.search("How many orders exist?")
    print(hit.strategy)  # SearchStrategy.NO_MATCH
```

Or via environment:

```bash
export MEDHA_FEEDBACK_INCORRECT_THRESHOLD=3
```

When auto-invalidation fires, the vector backend entry and the L1 cache entry are both removed, the removal is logged at `INFO`, and the next `search()` misses so your application regenerates the query with the LLM. Calling `feedback()` again afterwards returns `False` without raising.

The default is `None` — counters accumulate but nothing is ever auto-removed. Start there, watch the counters for a while, and enable the threshold once you know what a realistic error rate looks like for your workload.

!!! note "Only negative feedback triggers invalidation"
    The threshold is checked exclusively on `correct=False` calls. Any number of correct feedbacks leaves the entry untouched, and a high `feedback_correct` count never protects an entry from being invalidated.

### Behaviour reference

| Scenario | Return value | Side effect |
|---|---|---|
| Entry found, `correct=True` | `True` | `feedback_correct` incremented by 1 |
| Entry found, `correct=False`, below threshold | `True` | `feedback_incorrect` incremented by 1 |
| Entry found, `correct=False`, threshold reached | `True` | Entry invalidated from backend and L1 |
| Entry not found | `False` | No change |
| Entry already invalidated, called again | `False` | No change |

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

Start at `0.2` and compare the hit rate before and after — [persistent stats](observability.md) make that comparison survive restarts. Raise it only if the extra hits are genuinely correct: the failure mode of a high factor is a well-rated entry answering questions it should not.

The two feedback features compose. `feedback_boost_factor` promotes what works, `feedback_incorrect_threshold` retires what does not:

```python
settings = Settings(
    feedback_boost_factor=0.25,
    feedback_incorrect_threshold=3,
)
```

Set via environment as `MEDHA_FEEDBACK_BOOST_FACTOR` (a float in `[0.0, 1.0]`).

---

## Integration Pattern

```python
async def handle_user_correction(question: str, was_correct: bool, cache: Medha) -> None:
    updated = await cache.feedback(question, correct=was_correct)
    if not updated:
        # Entry expired or was never cached — nothing to update
        return
    if not was_correct:
        logger.warning("Incorrect cache hit reported for: %s", question[:80])
```

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

## See Also

* [Invalidation](invalidation.md) — manual and bulk removal
* [Observability](observability.md) — hit rate and per-strategy metrics
* [TTL & Lifecycle](ttl_and_lifecycle.md) — time-based expiry
* [Configuration](configuration.md) — `feedback_incorrect_threshold` and `feedback_boost_factor`
* [Demo 25 — Feedback Loop](https://github.com/ArchAI-Labs/medha/blob/main/demo/25_feedback_loop.ipynb)
