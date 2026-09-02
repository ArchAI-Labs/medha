"""The shared L1 must round-trip every ``CacheHit`` field.

Regression guard for the bug where ``RedisL1Cache._serialise`` was a
hand-written field whitelist that omitted ``expires_at``.  Entries written to
the distributed L1 came back with ``expires_at=None``, so the expiry check in
``Medha._check_l1_cache`` never fired: on a Redis-backed L1 an expired entry
kept being served until the Redis-level ``ttl`` — one global value, and only
when configured — removed the key.

The serialisers are pure functions, so they are tested directly: no Redis
server, no driver, no event loop.  ``redis_adapter`` imports the driver inside
``__init__``, so this module is importable without the optional dependency.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from medha.l1_cache.redis_adapter import _deserialise, _serialise
from medha.types import CacheHit, SearchStrategy

EXPIRES_AT = datetime(2026, 9, 2, 10, 30, tzinfo=timezone.utc)


def _full_hit() -> CacheHit:
    """A hit with every field set to a non-default value."""
    return CacheHit(
        generated_query="SELECT count(*) FROM users",
        response_summary="42 users",
        confidence=0.93,
        strategy=SearchStrategy.SEMANTIC_MATCH,
        template_used="count_entities",
        expires_at=EXPIRES_AT,
    )


def test_roundtrip_preserves_the_whole_model():
    """Equality over the model catches any field the serialiser drops."""
    hit = _full_hit()
    assert _deserialise(_serialise(hit)) == hit


def test_expires_at_survives_the_roundtrip():
    """The field the whitelist dropped, named so the regression is obvious."""
    restored = _deserialise(_serialise(_full_hit()))
    assert restored.expires_at == EXPIRES_AT, (
        f"expires_at dropped by the serialiser (got {restored.expires_at!r})"
    )


def test_no_field_is_omitted():
    """Guards against a whitelist creeping back in.

    Asserting against ``model_fields`` means a field added to ``CacheHit``
    later fails here if it is not serialised, instead of silently vanishing
    on the distributed L1 the way ``expires_at`` did.
    """
    payload = json.loads(_serialise(_full_hit()))
    assert set(payload) == set(CacheHit.model_fields), (
        f"serialised keys {sorted(payload)} != model fields "
        f"{sorted(CacheHit.model_fields)}"
    )


def test_defaults_roundtrip():
    hit = CacheHit()
    restored = _deserialise(_serialise(hit))
    assert restored == hit
    assert restored.strategy is SearchStrategy.NO_MATCH
    assert restored.expires_at is None


@pytest.mark.parametrize(
    "strategy",
    list(SearchStrategy),
    ids=lambda s: s.value,
)
def test_every_strategy_roundtrips(strategy):
    hit = CacheHit(generated_query="SELECT 1", strategy=strategy)
    assert _deserialise(_serialise(hit)).strategy is strategy


def test_legacy_payload_still_loads():
    """A key written by 0.5.0 must survive the upgrade.

    The old format had exactly these five keys and no ``expires_at``; the
    missing key has to fall back to the model default rather than raise,
    otherwise every pre-upgrade key in a shared Redis becomes a hard miss.
    """
    legacy = json.dumps({
        "generated_query": "SELECT count(*) FROM users",
        "response_summary": "42 users",
        "confidence": 0.93,
        "strategy": "semantic_match",
        "template_used": "count_entities",
    })
    restored = _deserialise(legacy)
    assert restored.generated_query == "SELECT count(*) FROM users"
    assert restored.confidence == pytest.approx(0.93)
    assert restored.strategy is SearchStrategy.SEMANTIC_MATCH
    assert restored.template_used == "count_entities"
    assert restored.expires_at is None


def test_expired_hit_is_recognisable_after_roundtrip():
    """The end the bug broke: core compares expires_at against now(UTC)."""
    expired = CacheHit(
        generated_query="SELECT 1",
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    restored = _deserialise(_serialise(expired))
    assert restored.expires_at is not None
    assert restored.expires_at <= datetime.now(timezone.utc)
