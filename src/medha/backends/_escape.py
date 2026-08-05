"""Shared string-literal escaping for backend filter expressions.

Some backends build filter strings that embed user-influenced values inside
single-quoted string literals because their drivers do not expose bind
parameters for filter expressions:

- Azure AI Search OData ``$filter`` (e.g. ``query_hash eq '...'``)
- LanceDB / DataFusion ``where`` clauses (e.g. ``query_hash = '...'``)

In both OData and DataFusion SQL, a single quote inside a string literal is
escaped by doubling it (``'`` -> ``''``); backslash is **not** an escape
character, so doubling the quote is sufficient and complete.

Centralising the logic here keeps a single audited, tested implementation so a
new ``.where()``/``filter`` call cannot silently forget to escape its input.
"""


def quote_sql_literal(value: str) -> str:
    """Escape ``value`` for use inside a single-quoted SQL/OData string literal.

    Doubles every single quote (``'`` -> ``''``). The returned string does
    **not** include the surrounding quotes — the caller adds those::

        f"name = '{quote_sql_literal(user_input)}'"

    Args:
        value: The raw string to embed in a single-quoted literal.

    Returns:
        The escaped string, safe to place between two single quotes.
    """
    return value.replace("'", "''")
