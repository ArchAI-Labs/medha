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
