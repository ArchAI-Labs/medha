"""Unit tests for the shared SQL/OData string-literal escaper."""

from medha.backends._escape import quote_sql_literal


def test_no_quotes_unchanged():
    assert quote_sql_literal("abc_123-xyz") == "abc_123-xyz"


def test_single_quote_is_doubled():
    assert quote_sql_literal("O'Brien") == "O''Brien"


def test_multiple_quotes_each_doubled():
    assert quote_sql_literal("a'b'c") == "a''b''c"


def test_injection_attempt_is_neutralised():
    # Classic break-out attempt: after escaping, every quote is doubled, so it
    # can no longer terminate the surrounding literal.
    escaped = quote_sql_literal("' OR '1'='1")
    assert escaped == "'' OR ''1''=''1"
    # No lone (unpaired) single quote survives.
    assert "'" not in escaped.replace("''", "")


def test_backslash_is_preserved():
    # Neither OData nor DataFusion treat backslash as an escape character, so it
    # must pass through unchanged (doubling the quote is the only escaping).
    assert quote_sql_literal("a\\'b") == "a\\''b"


def test_empty_string():
    assert quote_sql_literal("") == ""


def test_unicode_is_preserved():
    assert quote_sql_literal("café ☕ '") == "café ☕ ''"
