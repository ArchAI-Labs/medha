"""LanceDB must reconcile a table created by an older version of medha.

``initialize()`` calls ``create_table(name, schema=..., exist_ok=True)``, and
``exist_ok=True`` opens a table that already exists while **ignoring** the
schema it was handed. A table therefore keeps exactly the columns it was
created with: every column added to ``_build_schema`` since then is absent, and
the first upsert supplying one fails on the whole batch — at write time, far
from the cause.

0.5.0 added the two feedback counters to the Arrow schema, so a table created
by 0.4.x is the concrete instance of this.

pyarrow alone is enough to exercise it: deciding *which* columns are missing
and *what* to backfill them with is pure schema arithmetic, and ``lancedb``
itself is only needed for the single call that applies the result.
"""

import pytest

pa = pytest.importorskip("pyarrow")

from medha.backends.lancedb import (  # noqa: E402
    _backfill_expression,
    _build_schema,
    _missing_fields,
)
from medha.exceptions import StorageInitializationError  # noqa: E402

DIM = 8
COUNTERS = ("feedback_correct", "feedback_incorrect")


def _schema_before_0_5_0(dimension: int = DIM) -> "pa.Schema":
    """The table shape as 0.4.x created it: today's schema minus the counters."""
    return pa.schema([f for f in _build_schema(dimension) if f.name not in COUNTERS])


# ---------------------------------------------------------------------------
# Which columns are missing
# ---------------------------------------------------------------------------


def test_current_schema_needs_no_migration():
    current = _build_schema(DIM)
    assert _missing_fields(current, current) == []


def test_detects_columns_added_after_the_table_was_created():
    missing = _missing_fields(_schema_before_0_5_0(), _build_schema(DIM))
    assert [f.name for f in missing] == list(COUNTERS)


def test_extra_columns_in_the_table_are_left_alone():
    """A table written by a *newer* version must not be touched."""
    newer = pa.schema(list(_build_schema(DIM)) + [pa.field("from_the_future", pa.string())])
    assert _missing_fields(newer, _build_schema(DIM)) == []


# ---------------------------------------------------------------------------
# What the new columns are backfilled with
# ---------------------------------------------------------------------------


def test_backfill_matches_what_the_reader_assumes():
    assert _backfill_expression(pa.field("n", pa.int64())) == "0"
    assert _backfill_expression(pa.field("f", pa.float64())) == "0.0"
    assert _backfill_expression(pa.field("s", pa.string())) == "''"


def test_every_field_in_the_current_schema_has_a_backfill():
    """Guards the next column added to ``_build_schema``."""
    for field in _build_schema(DIM):
        if field.name == "vector":
            continue  # never backfilled: a row without a vector is not a row
        assert _backfill_expression(field), f"no backfill expression for {field.name}"


# ---------------------------------------------------------------------------
# Applying the migration
# ---------------------------------------------------------------------------


class _FakeTable:
    """Stands in for an AsyncTable: schema() is a coroutine method."""

    def __init__(self, schema: "pa.Schema", add_columns_fails: bool = False) -> None:
        self._schema = schema
        self._fails = add_columns_fails
        self.added: dict[str, str] | None = None

    async def schema(self) -> "pa.Schema":
        return self._schema

    async def add_columns(self, transforms: dict[str, str]) -> None:
        if self._fails:
            raise RuntimeError("add_columns is not available on this version")
        self.added = transforms


class _FakeSyncTable(_FakeTable):
    """Stands in for the sync table, which exposes schema as a property."""

    def __init__(self, schema: "pa.Schema") -> None:
        super().__init__(schema)
        self.schema = schema  # type: ignore[assignment]


def _backend():
    """A backend instance without running __init__, which requires lancedb."""
    from medha.backends.lancedb import LanceDBBackend

    return LanceDBBackend.__new__(LanceDBBackend)


async def test_reconcile_adds_the_missing_columns():
    table = _FakeTable(_schema_before_0_5_0())

    await _backend()._reconcile_schema(table, "t", _build_schema(DIM))

    assert table.added == {"feedback_correct": "0", "feedback_incorrect": "0"}


async def test_reconcile_is_a_noop_when_the_schema_already_matches():
    table = _FakeTable(_build_schema(DIM))

    await _backend()._reconcile_schema(table, "t", _build_schema(DIM))

    assert table.added is None


async def test_reconcile_reads_a_schema_exposed_as_a_property():
    table = _FakeSyncTable(_schema_before_0_5_0())

    await _backend()._reconcile_schema(table, "t", _build_schema(DIM))

    assert table.added == {"feedback_correct": "0", "feedback_incorrect": "0"}


async def test_reconcile_names_the_columns_when_it_cannot_add_them():
    """The floor: fail at initialize() with the cause, not at the first upsert."""
    table = _FakeTable(_schema_before_0_5_0(), add_columns_fails=True)

    with pytest.raises(StorageInitializationError) as excinfo:
        await _backend()._reconcile_schema(table, "my_table", _build_schema(DIM))

    message = str(excinfo.value)
    assert "my_table" in message
    for name in COUNTERS:
        assert name in message
