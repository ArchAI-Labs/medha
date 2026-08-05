"""Text normalization: lowercasing, whitespace cleanup, synonym replacement."""

from __future__ import annotations

import hashlib
import logging
import re

logger = logging.getLogger(__name__)

_DEFAULT_REPLACEMENTS: list[tuple[str, str]] = [
    (r"\bfirst\s+(\d+)\b", r"top \1"),
    # Longer phrases first to avoid partial matches (same principle as
    # "list all the" before "list all").
    (r"\bcan\s+you\s+tell\s+me\b", "get"),
    (r"\bcould\s+you\s+(show|get|find|list)\b", r"\1"),
    (r"\bi\s+want\s+to\s+know\b", "get"),
    (r"\bget\s+me\s+the\b", "get"),
    (r"\bget\s+me\b", "get"),
    (r"\bshow\s+me\s+the\b", "show"),
    (r"\bshow\s+me\b", "show"),
    (r"\blist\s+all\s+the\b", "list"),
    (r"\blist\s+all\b", "list"),
    (r"\bwho\s+are\s+the\b", "list"),
    (r"\bwhat\s+are\s+the\b", "list"),
    (r"\btell\s+me\s+about\b", "get"),
    (r"\bfind\s+out\b", "find"),
    (r"\bplease\b", ""),
]


def normalize_question(
    question: str,
    extra_replacements: list[tuple[str, str]] | None = None,
) -> str:
    """Normalize a natural-language question into canonical form.

    Steps:
        1. Strip and collapse whitespace.
        2. Lowercase.
        3. Apply replacement patterns (default + extras).
        4. Remove trailing punctuation.
        5. Final whitespace cleanup.

    Args:
        question: Raw user input.
        extra_replacements: Additional (pattern, replacement) pairs appended
            after the defaults.

    Returns:
        Normalized string. Returns "" for empty/whitespace-only input.
    """
    text = " ".join(question.split())
    if not text:
        logger.debug("Normalization: empty input")
        return ""

    text = text.lower()

    replacements = _DEFAULT_REPLACEMENTS
    if extra_replacements:
        replacements = replacements + list(extra_replacements)

    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)

    text = re.sub(r"[.!?]+$", "", text)

    text = " ".join(text.split())
    logger.debug("Normalized: '%s' -> '%s'", question[:60], text[:60])
    return text


def question_hash(question: str) -> str:
    """Return the MD5 hex digest of a normalized question.

    Used as the L1 cache key (Tier 0).

    Args:
        question: Raw (un-normalized) question. Will be normalized internally.

    Returns:
        32-character hex string.
    """
    normalized = normalize_question(question)
    return hashlib.md5(normalized.encode("utf-8"), usedforsecurity=False).hexdigest()


def query_hash(query: str) -> str:
    """Return the MD5 hex digest of a generated query string.

    Used for deduplication in the payload.

    Args:
        query: The SQL/Cypher/GraphQL query string.

    Returns:
        32-character hex string.
    """
    return hashlib.md5(query.strip().encode("utf-8"), usedforsecurity=False).hexdigest()
