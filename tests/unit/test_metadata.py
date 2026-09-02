"""Unit tests for medha.utils.metadata — validation, encoding, matching."""

import pytest

from medha.utils.metadata import (
    MAX_KEYS,
    MAX_VALUE_LENGTH,
    canonical_json,
    dumps_metadata,
    loads_metadata,
    metadata_fingerprint,
    metadata_matches,
    validate_metadata,
)


class TestValidateMetadata:
    def test_none_and_empty_are_the_same_thing(self):
        assert validate_metadata(None) == {}
        assert validate_metadata({}) == {}

    def test_accepts_every_scalar_type(self):
        meta = {"date": "2026-08-12", "hour": 10, "ratio": 0.5, "draft": False}
        assert validate_metadata(meta) == meta

    def test_returns_a_copy(self):
        original = {"tenant": "acme"}
        validated = validate_metadata(original)
        validated["tenant"] = "other"
        assert original == {"tenant": "acme"}

    def test_rejects_non_dict(self):
        with pytest.raises(ValueError, match="must be a dict"):
            validate_metadata([("date", "2026-08-12")])

    def test_rejects_non_string_key(self):
        with pytest.raises(ValueError, match="keys must be strings"):
            validate_metadata({1: "x"})

    @pytest.mark.parametrize("key", ["", "1date", "with space", "quo'te", "a" * 65])
    def test_rejects_unsafe_key(self, key):
        with pytest.raises(ValueError, match="is invalid"):
            validate_metadata({key: "x"})

    @pytest.mark.parametrize("key", ["date", "_date", "resolved.date", "tenant-id", "a" * 64])
    def test_accepts_safe_key(self, key):
        assert validate_metadata({key: "x"}) == {key: "x"}

    def test_rejects_nested_value(self):
        with pytest.raises(ValueError, match="must be str, int, float or bool"):
            validate_metadata({"window": {"from": "10:00"}})

    def test_rejects_none_value(self):
        # None cannot be matched exactly and does not survive every backend.
        with pytest.raises(ValueError, match="must be str, int, float or bool"):
            validate_metadata({"date": None})

    def test_rejects_oversized_value(self):
        with pytest.raises(ValueError, match="exceeds the maximum"):
            validate_metadata({"note": "x" * (MAX_VALUE_LENGTH + 1)})

    def test_rejects_too_many_keys(self):
        with pytest.raises(ValueError, match="exceeds the maximum"):
            validate_metadata({f"k{i}": i for i in range(MAX_KEYS + 1)})

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_non_finite_number(self, value):
        with pytest.raises(ValueError, match="finite number"):
            validate_metadata({"ratio": value})

    def test_label_appears_in_the_message(self):
        with pytest.raises(ValueError, match="^filters key"):
            validate_metadata({"bad key": "x"}, label="filters")


class TestEncoding:
    def test_canonical_json_is_key_order_independent(self):
        assert canonical_json({"a": 1, "b": 2}) == canonical_json({"b": 2, "a": 1})

    def test_fingerprint_is_stable_and_empty_for_no_metadata(self):
        assert metadata_fingerprint({}) == ""
        assert metadata_fingerprint({"a": 1, "b": 2}) == metadata_fingerprint({"b": 2, "a": 1})
        assert metadata_fingerprint({"a": 1}) != metadata_fingerprint({"a": 2})

    def test_empty_metadata_dumps_to_empty_string(self):
        # Backends with a fixed schema backfill missing text columns with "",
        # so an old row and a new row without metadata must read the same.
        assert dumps_metadata({}) == ""
        assert dumps_metadata(None) == ""
        assert loads_metadata("") == {}

    def test_round_trip(self):
        meta = {"date": "2026-08-12", "hour": 10, "ratio": 0.5, "draft": True}
        assert loads_metadata(dumps_metadata(meta)) == meta

    def test_loads_accepts_a_dict_as_is(self):
        # Qdrant and the in-memory backend hand back a nested object rather
        # than an encoded string.
        assert loads_metadata({"date": "2026-08-12"}) == {"date": "2026-08-12"}

    @pytest.mark.parametrize("raw", [None, "", "not json", "[1,2]", '"text"', "123"])
    def test_loads_tolerates_unusable_payloads(self, raw):
        assert loads_metadata(raw) == {}

    def test_loads_drops_values_it_cannot_represent(self):
        assert loads_metadata('{"ok":"x","nested":{"a":1},"list":[1]}') == {"ok": "x"}


class TestMetadataMatches:
    def test_no_filters_matches_anything(self):
        assert metadata_matches({}, None) is True
        assert metadata_matches({}, {}) is True
        assert metadata_matches({"date": "2026-08-12"}, None) is True

    def test_exact_match_on_every_key(self):
        meta = {"date": "2026-08-12", "tenant": "acme"}
        assert metadata_matches(meta, {"date": "2026-08-12"}) is True
        assert metadata_matches(meta, {"date": "2026-08-12", "tenant": "acme"}) is True
        assert metadata_matches(meta, {"date": "2026-08-13"}) is False

    def test_all_filters_must_hold(self):
        meta = {"date": "2026-08-12", "tenant": "acme"}
        assert metadata_matches(meta, {"date": "2026-08-12", "tenant": "other"}) is False

    def test_missing_key_is_a_mismatch_not_a_wildcard(self):
        # An entry stored without a scope must not answer a scoped question.
        assert metadata_matches({}, {"date": "2026-08-12"}) is False
        assert metadata_matches({"tenant": "acme"}, {"date": "2026-08-12"}) is False

    def test_extra_keys_on_the_entry_are_ignored(self):
        meta = {"date": "2026-08-12", "tenant": "acme"}
        assert metadata_matches(meta, {"date": "2026-08-12"}) is True

    def test_bool_is_not_an_integer(self):
        assert metadata_matches({"flag": True}, {"flag": 1}) is False
        assert metadata_matches({"flag": 1}, {"flag": True}) is False
        assert metadata_matches({"flag": True}, {"flag": True}) is True

    def test_int_and_float_stay_interchangeable(self):
        # Not every backend preserves the distinction across a round-trip.
        assert metadata_matches({"hour": 10}, {"hour": 10.0}) is True
