"""Structured metadata attached to cache entries, and the filters over it.

Two questions that differ only by a scoping dimension the embedding does not
separate — a date, a time window, a tenant, a currency — sit almost on top of
each other in vector space, so the semantic tier can answer one with the
other's query at high confidence.  Metadata is the guardrail: the scope is
stored beside the entry as structured data, and a search that declares the
scope it needs never sees an entry carrying a different one.

Values are deliberately flat and scalar.  Ten backends store them in ten
different ways (a Qdrant payload, a JSONB column, a Chroma metadata value, a
Redis hash field), and every one of those accepts a flat map of scalars.
Nesting would have to be flattened, encoded, or rejected per backend, and the
filter dialects diverge even further.  A flat map is the intersection.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any

from medha.types import CacheResult, MetadataDict

logger = logging.getLogger(__name__)

# Ceiling on the over-fetch a Python-side filter is allowed to ask for. A
# filter matching nothing in the top-N would otherwise scale the fetch with the
# caller's limit without ever finding anything.
MAX_FILTER_FETCH = 1000

# Keys travel into backend field names (``metadata.resolved_date`` as a Qdrant
# payload key, a JSONB path, a RediSearch tag) and into filter expressions, so
# they are restricted to a character set every backend accepts unescaped.
MAX_KEYS = 32
MAX_KEY_LENGTH = 64
MAX_VALUE_LENGTH = 256

_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]{0,63}$")


def validate_metadata(metadata: Any, *, label: str = "metadata") -> MetadataDict:
    """Return *metadata* as a validated flat map, or raise ``ValueError``.

    ``None`` and ``{}`` both return ``{}``: an entry without metadata and one
    that explicitly declares none are the same entry.

    Args:
        metadata: The raw mapping supplied by the caller.
        label: Name used in error messages (``"metadata"`` or ``"filters"``).

    Returns:
        A new dict, safe to store.

    Raises:
        ValueError: If the shape, a key, or a value is not storable.
    """
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise ValueError(f"{label} must be a dict, got {type(metadata).__name__}")
    if not metadata:
        return {}
    if len(metadata) > MAX_KEYS:
        raise ValueError(
            f"{label} has {len(metadata)} keys, exceeds the maximum of {MAX_KEYS}"
        )

    validated: MetadataDict = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise ValueError(f"{label} keys must be strings, got {type(key).__name__}")
        if not _KEY_RE.match(key):
            raise ValueError(
                f"{label} key {key!r} is invalid: must match "
                f"^[A-Za-z_][A-Za-z0-9_.-]{{0,{MAX_KEY_LENGTH - 1}}}$"
            )
        # bool is a subclass of int, so it has to be accepted before the
        # numeric check rather than after it.
        if isinstance(value, bool):
            validated[key] = value
        elif isinstance(value, str):
            if len(value) > MAX_VALUE_LENGTH:
                raise ValueError(
                    f"{label} value for {key!r} is {len(value)} characters, "
                    f"exceeds the maximum of {MAX_VALUE_LENGTH}"
                )
            validated[key] = value
        elif isinstance(value, int):
            validated[key] = value
        elif isinstance(value, float):
            # NaN never equals itself, so an entry stored with one could never
            # be matched by a filter carrying the same value; infinities do not
            # survive JSON round-trips through several backends.
            if value != value or value in (float("inf"), float("-inf")):
                raise ValueError(f"{label} value for {key!r} must be a finite number")
            validated[key] = value
        else:
            raise ValueError(
                f"{label} value for {key!r} must be str, int, float or bool, "
                f"got {type(value).__name__}"
            )
    return validated


def canonical_json(metadata: MetadataDict) -> str:
    """Serialise *metadata* to a stable JSON string.

    Keys are sorted and separators are tight, so two equal maps always produce
    the same bytes — which is what makes the fingerprint below usable as a
    cache key and as part of a deduplication identity.
    """
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def metadata_fingerprint(metadata: MetadataDict) -> str:
    """Return a short stable digest of *metadata* (empty string when empty).

    Used to namespace L1 cache keys by the filters a search declared, so a
    filtered lookup can never be served an entry stored for a different scope.
    """
    if not metadata:
        return ""
    return hashlib.md5(
        canonical_json(metadata).encode("utf-8"), usedforsecurity=False
    ).hexdigest()


def dumps_metadata(metadata: MetadataDict | None) -> str:
    """Serialise metadata for a backend field that holds a JSON string.

    Empty metadata serialises to ``""`` rather than ``"{}"``: backends with a
    fixed schema backfill missing text columns with the empty string, so rows
    written before this feature and rows written without metadata read back
    identically.
    """
    if not metadata:
        return ""
    return canonical_json(metadata)


def loads_metadata(raw: Any) -> MetadataDict:
    """Rebuild metadata from a backend field, tolerating anything unusable.

    A row can carry an empty string (no metadata, or written by an older
    version), ``None``, or — if something outside medha wrote it — a JSON
    document that is not a flat object.  None of those is worth failing a
    search over: the entry simply has no metadata, and a filtered search will
    not match it.
    """
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        parsed: Any = raw
    else:
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            logger.debug("Ignoring unparseable metadata payload: %r", raw)
            return {}
    if not isinstance(parsed, dict):
        logger.debug("Ignoring non-object metadata payload: %r", parsed)
        return {}
    return {
        key: value
        for key, value in parsed.items()
        if isinstance(key, str) and isinstance(value, (str, int, float, bool))
    }


def _values_equal(left: Any, right: Any) -> bool:
    """Exact-match comparison for a single metadata value.

    ``True == 1`` in Python, which would let a filter of ``{"active": True}``
    match an entry storing ``1``.  Booleans are therefore only ever equal to
    booleans.  Ints and floats stay interchangeable (``1 == 1.0``) because
    several backends cannot preserve the distinction across a round-trip.
    """
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    return bool(left == right)


def split_filters(
    filters: MetadataDict | None,
    pushable: tuple[type, ...] = (str,),
) -> tuple[MetadataDict, MetadataDict]:
    """Split *filters* into the part a backend can push down, and the rest.

    A pushed-down constraint is evaluated by the storage engine before it picks
    its top-N, which is the whole point: it is the only way a match ranked
    below the unfiltered top-N can be found at all. But a dialect that
    evaluates a constraint even slightly differently from
    :func:`metadata_matches` would drop rows that ought to match, and nothing
    downstream could tell — the caller would just see a cache miss.

    So the default is the narrow, provable case: string equality, which every
    dialect here means the same way. Numbers and booleans stay in the residual
    and are checked in Python. That costs an index lookup on those keys and
    nothing in correctness, and the scoping dimensions this feature exists for
    — a date, a time window, a tenant, a region, a currency — are strings.

    A backend whose engine is known to agree on more types widens *pushable*;
    Qdrant does, and its filtering is covered by tests against a real instance.

    Returns:
        ``(pushable, residual)``. Either may be empty.
    """
    if not filters:
        return {}, {}
    down: MetadataDict = {}
    rest: MetadataDict = {}
    for key, value in filters.items():
        if isinstance(value, pushable):
            down[key] = value
        else:
            rest[key] = value
    return down, rest


def filter_fetch_size(limit: int, residual: MetadataDict, overfetch: int) -> int:
    """How many rows to retrieve so *limit* survive the Python-side filter.

    With nothing left to check in Python, the engine already returned matches
    only and ``limit`` is exactly right. Otherwise the fetch is widened, since
    an unknown share of the rows is about to be discarded.
    """
    if not residual:
        return limit
    return min(max(limit * overfetch, limit), MAX_FILTER_FETCH)


def verify_filters(
    results: list[CacheResult],
    filters: MetadataDict | None,
    limit: int,
) -> list[CacheResult]:
    """Keep the results that satisfy every filter, capped at *limit*.

    Run after every backend query, including one whose engine has already
    applied the filter. There it is a no-op that costs a dict comparison per
    returned row, and it is what makes a native filter unable to return a wrong
    answer: a dialect that is too permissive, or that disagrees on a type, is
    corrected here. Only a dialect that is too *strict* can still do damage,
    which is what :func:`split_filters` is careful about.
    """
    if not filters:
        return results[:limit]
    return [r for r in results if metadata_matches(r.metadata, filters)][:limit]


def metadata_matches(metadata: MetadataDict, filters: MetadataDict | None) -> bool:
    """Whether an entry's *metadata* satisfies *filters*.

    Exact equality on every filter key (AND).  A key the entry does not carry
    is a mismatch, never a wildcard: an entry stored without a scope must not
    answer a question that demands one.  Keys the entry carries and the filter
    does not are ignored.

    Empty or absent filters match everything, so an unfiltered search behaves
    exactly as it did before metadata existed.
    """
    if not filters:
        return True
    for key, expected in filters.items():
        if key not in metadata:
            return False
        if not _values_equal(metadata[key], expected):
            return False
    return True
