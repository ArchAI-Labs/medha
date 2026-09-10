"""Unit tests for medha.utils.nlp."""

from unittest.mock import MagicMock, patch

import pytest

from medha.exceptions import ParameterExtractionError
from medha.types import QueryTemplate
from medha.utils.nlp import ParameterExtractor, keyword_overlap_score


@pytest.fixture
def extractor():
    return ParameterExtractor(use_spacy=False)


def _make_gliner_extractor(predict_return=None, side_effect=None):
    """Return a ParameterExtractor with a mocked GLiNER model (no real install needed)."""
    ext = ParameterExtractor(use_spacy=False, use_gliner=False)
    mock_gliner = MagicMock()
    if side_effect is not None:
        mock_gliner.predict_entities.side_effect = side_effect
    else:
        mock_gliner.predict_entities.return_value = predict_return or []
    ext._gliner = mock_gliner
    ext._gliner_available = True
    return ext


@pytest.fixture
def count_template():
    return QueryTemplate(
        intent="top_n",
        template_text="Show top {count} {entity}",
        query_template="SELECT * FROM {entity} LIMIT {count}",
        parameters=["count", "entity"],
        priority=1,
        parameter_patterns={
            "count": r"\b(\d+)\b",
            "entity": r"\b(users|products|orders|employees)\b",
        },
    )


@pytest.fixture
def dept_template():
    return QueryTemplate(
        intent="filter_by_department",
        template_text="List employees in {department}",
        query_template="SELECT * FROM employees WHERE dept = '{department}'",
        parameters=["department"],
        priority=2,
    )


class TestExtractViaRegex:
    def test_extract_number_via_regex(self, extractor, count_template):
        params = extractor.extract("Show top 5 products", count_template)
        assert params["count"] == "5"

    def test_extract_entity_via_regex(self, extractor, count_template):
        params = extractor.extract("Show top 5 users", count_template)
        assert params["entity"] == "users"


class TestExtractViaHeuristics:
    def test_extract_via_heuristics(self, extractor, dept_template):
        # Use lowercase prefix so only "Engineering" is a capitalized word
        params = extractor.extract("list employees in Engineering", dept_template)
        assert params["department"] == "Engineering"


class TestRenderQuery:
    def test_render_query(self, extractor, count_template):
        params = {"count": "10", "entity": "products"}
        result = extractor.render_query(count_template, params)
        assert result == "SELECT * FROM products LIMIT 10"

    def test_render_query_unfilled_placeholder(self, extractor, count_template):
        with pytest.raises(ParameterExtractionError):
            extractor.render_query(count_template, {"count": "10"})


class TestSanitizeValue:
    def test_sanitize_value(self):
        assert ParameterExtractor._sanitize_value("hello") == "hello"
        assert ParameterExtractor._sanitize_value("it's") == "its"
        assert ParameterExtractor._sanitize_value("DROP;--") == "DROP--"


class TestPatternVerifiedRendering:
    """Issue #39: a pattern-verified value must render intact or raise —
    it must never be silently corrupted by the generic sanitizer."""

    @pytest.fixture
    def time_range_template(self):
        return QueryTemplate(
            intent="meetings_in_range",
            template_text="Show meetings between {time_range}",
            query_template="SELECT * FROM meetings WHERE time_range = '{time_range}'",
            parameters=["time_range"],
            parameter_patterns={"time_range": r"\b(\d{2}:\d{2}-\d{2}:\d{2})\b"},
        )

    @pytest.fixture
    def date_template(self):
        return QueryTemplate(
            intent="events_on_date",
            template_text="Show events on {event_date}",
            query_template="SELECT * FROM events WHERE event_date = '{event_date}'",
            parameters=["event_date"],
            parameter_patterns={"event_date": r"\b(\d{2}/\d{2}/\d{4})\b"},
        )

    def test_time_range_extracted_and_rendered_intact(
        self, extractor, time_range_template
    ):
        """The reported case: 10:00-12:00 must survive extraction + render."""
        params = extractor.extract(
            "Show meetings between 10:00-12:00", time_range_template
        )
        assert params["time_range"] == "10:00-12:00"

        result = extractor.render_query(time_range_template, params)
        assert "10:00-12:00" in result
        assert "1000-1200" not in result

    def test_non_iso_date_extracted_and_rendered_intact(self, extractor, date_template):
        params = extractor.extract("Show events on 10/08/2026", date_template)
        assert params["event_date"] == "10/08/2026"

        result = extractor.render_query(date_template, params)
        assert "10/08/2026" in result

    def test_render_query_raises_instead_of_corrupting_unverified_value(
        self, extractor, dept_template
    ):
        """`department` has no parameter_patterns, so a value sanitization
        would alter (the '&') must raise rather than silently render 'RD'."""
        with pytest.raises(ParameterExtractionError):
            extractor.render_query(dept_template, {"department": "R&D"})

    def test_value_not_matching_declared_pattern_still_sanitized(
        self, extractor, time_range_template
    ):
        """A value that doesn't fit the declared pattern gets no free pass:
        it is sanitized like any NER/heuristic value, and corruption raises."""
        with pytest.raises(ParameterExtractionError):
            extractor.render_query(
                time_range_template, {"time_range": "10:00-12:00; DROP"}
            )

    def test_heuristic_and_ner_values_still_sanitized_when_clean(
        self, extractor, dept_template
    ):
        """Plain alphanumeric values from heuristics/NER keep working unchanged."""
        result = extractor.render_query(dept_template, {"department": "Engineering"})
        assert "Engineering" in result

    def test_existing_regex_templates_render_exactly_as_before(
        self, extractor, count_template
    ):
        params = {"count": "10", "entity": "products"}
        result = extractor.render_query(count_template, params)
        assert result == "SELECT * FROM products LIMIT 10"


class TestUnresolvedRelativeExpression:
    """Issue #44: a value extracted verbatim (typically via a template's own
    ``parameter_patterns``) must never render if it is still a relative time
    marker like "yesterday" — that path bypasses `_sanitize_value()`
    entirely, so without this check the tier would render a query built on
    a word instead of a date."""

    @pytest.fixture
    def day_template(self):
        return QueryTemplate(
            intent="sales_by_day",
            template_text="Show sales for {day}",
            query_template="SELECT SUM(amount) FROM sales WHERE day = '{day}'",
            parameters=["day"],
            parameter_patterns={
                "day": r"\b(yesterday|today|tomorrow|last week|\d{4}-\d{2}-\d{2})\b"
            },
        )

    @pytest.mark.parametrize(
        "value",
        [
            "yesterday",
            "Yesterday",
            "today",
            "tomorrow",
            "tonight",
            "last week",
            "next week",
            "this month",
            "last month",
            "next year",
            "this Monday",
            "last Friday",
            "next Sunday",
            "3 days ago",
            "in 2 weeks",
        ],
    )
    def test_relative_expression_refuses_to_render(
        self, extractor, day_template, value
    ):
        with pytest.raises(ParameterExtractionError):
            extractor.render_query(day_template, {"day": value})

    def test_resolved_date_renders_exactly_as_before(self, extractor, day_template):
        result = extractor.render_query(day_template, {"day": "2026-08-12"})
        assert result == "SELECT SUM(amount) FROM sales WHERE day = '2026-08-12'"

    def test_unrelated_capitalized_value_still_renders(self, extractor, dept_template):
        """A value that merely shares no words with the relative-marker set
        is unaffected — the check is whole-value, not substring."""
        result = extractor.render_query(dept_template, {"department": "Engineering"})
        assert "Engineering" in result


class TestKeywordOverlapScore:
    def test_keyword_overlap_score(self):
        score = keyword_overlap_score("show top employees", "Show top {count} {entity}")
        assert score > 0

    def test_no_overlap(self):
        score = keyword_overlap_score("unrelated xyz", "Show top {count} {entity}")
        assert score == 0.0

    def test_empty_template(self):
        score = keyword_overlap_score("anything", "")
        assert score == 0.0


class TestNoSpacyFallback:
    def test_no_spacy_fallback(self):
        ext = ParameterExtractor(use_spacy=False)
        assert ext.spacy_available is False
        # Should still work via regex + heuristics
        template = QueryTemplate(
            intent="top_n",
            template_text="Show top {count} {entity}",
            query_template="SELECT * FROM {entity} LIMIT {count}",
            parameters=["count", "entity"],
            parameter_patterns={
                "count": r"\b(\d+)\b",
                "entity": r"\b(users|products)\b",
            },
        )
        params = ext.extract("Show top 5 products", template)
        assert params == {"count": "5", "entity": "products"}


class TestExtractEntities:
    def test_extract_numbers(self, extractor):
        entities = extractor.extract_entities("Show top 5 products limit 10")
        assert "number" in entities
        assert "5" in entities["number"]
        assert "10" in entities["number"]

    def test_extract_capitalized_names(self, extractor):
        # Use lowercase "find" so only "John Smith" is capitalized
        entities = extractor.extract_entities("find John Smith in the database")
        assert "person" in entities
        assert "John Smith" in entities["person"]

    def test_extract_empty_text(self, extractor):
        entities = extractor.extract_entities("no entities here")
        assert "number" not in entities


class TestExtractNoParameters:
    def test_extract_no_params_required(self, extractor):
        template = QueryTemplate(
            intent="list_all",
            template_text="Show all records",
            query_template="SELECT * FROM records",
        )
        params = extractor.extract("Show all records", template)
        assert params == {}


class TestExtractionFailure:
    def test_missing_params_raises(self, extractor):
        template = QueryTemplate(
            intent="multi",
            template_text="{a} and {b}",
            query_template="SELECT {a}, {b}",
            parameters=["a", "b"],
        )
        with pytest.raises(ParameterExtractionError):
            extractor.extract("nothing matches here", template)


class TestEntityCache:
    def test_same_text_returns_cached_result(self, extractor):
        """extract_entities() on the same text must not re-parse."""
        call_count = 0
        original_findall = __import__("re").findall


        def counting_findall(pattern, string, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_findall(pattern, string, *args, **kwargs)

        import medha.utils.nlp as nlp_module
        original = nlp_module.re.findall
        nlp_module.re.findall = counting_findall

        try:
            extractor.extract_entities("find 5 users")
            call_count = 0
            extractor.extract_entities("find 5 users")  # should hit cache
            assert call_count == 0, "Cache miss: re.findall was called on repeated input"
        finally:
            nlp_module.re.findall = original

    def test_different_texts_are_cached_separately(self, extractor):
        r1 = extractor.extract_entities("show 5 records")
        r2 = extractor.extract_entities("show 10 records")
        assert r1 != r2
        assert "5" in r1.get("number", [])
        assert "10" in r2.get("number", [])

    def test_cache_is_populated_after_first_call(self, extractor):
        extractor._entity_cache.clear()
        extractor.extract_entities("unique question 42")
        assert "unique question 42" in extractor._entity_cache

    def test_cache_evicts_when_full(self, extractor):
        extractor._entity_cache.clear()
        # Fill cache to max
        for i in range(ParameterExtractor._ENTITY_CACHE_MAXSIZE):
            extractor.extract_entities(f"question number {i}")
        assert len(extractor._entity_cache) == ParameterExtractor._ENTITY_CACHE_MAXSIZE
        # One more entry triggers eviction (clear-on-full)
        extractor.extract_entities("overflow question")
        assert len(extractor._entity_cache) == 1

    def test_cache_hit_returns_same_object(self, extractor):
        extractor._entity_cache.clear()
        result1 = extractor.extract_entities("stable text 99")
        result2 = extractor.extract_entities("stable text 99")
        assert result1 == result2


class TestGLiNERInit:
    def test_gliner_disabled_by_default(self):
        ext = ParameterExtractor(use_spacy=False)
        assert ext.gliner_available is False

    def test_gliner_not_installed_graceful(self):
        """use_gliner=True with package missing must not raise."""
        with patch.dict("sys.modules", {"gliner": None}):
            ext = ParameterExtractor(use_spacy=False, use_gliner=True)
        assert ext.gliner_available is False

    def test_gliner_available_after_mock_injection(self):
        ext = _make_gliner_extractor()
        assert ext.gliner_available is True

    def test_default_model_constant(self):
        assert ParameterExtractor._GLINER_DEFAULT_MODEL == "urchade/gliner_medium-v2.1"


class TestExtractViaGLiNER:
    @pytest.fixture
    def person_template(self):
        return QueryTemplate(
            intent="find_person",
            template_text="Find employee {person}",
            query_template="MATCH (p:Person {{name: '{person}'}}) RETURN p",
            parameters=["person"],
        )

    @pytest.fixture
    def multi_template(self):
        return QueryTemplate(
            intent="org_project",
            template_text="Show {org} project {project}",
            query_template="SELECT * FROM projects WHERE org='{org}' AND name='{project}'",
            parameters=["org", "project"],
        )

    def test_extracts_entity_by_param_name(self, person_template):
        ext = _make_gliner_extractor(
            predict_return=[{"label": "person", "text": "Alice", "score": 0.97}]
        )
        params = ext.extract("Find employee Alice", person_template)
        assert params == {"person": "Alice"}

    def test_uses_template_params_as_labels(self, multi_template):
        """GLiNER must receive exactly template.parameters as labels."""
        ext = _make_gliner_extractor(
            predict_return=[
                {"label": "org", "text": "Acme", "score": 0.95},
                {"label": "project", "text": "Apollo", "score": 0.93},
            ]
        )
        ext.extract("Show Acme project Apollo", multi_template)
        called_labels = ext._gliner.predict_entities.call_args[0][1]
        assert set(called_labels) == set(multi_template.parameters)

    def test_ignores_entity_with_unknown_label(self, person_template):
        """Entities whose label is not in template.parameters must be dropped."""
        ext = _make_gliner_extractor(
            predict_return=[
                {"label": "irrelevant_label", "text": "foo", "score": 0.99},
            ]
        )
        # All lowercase → no capitalized word for heuristics either → must raise
        with pytest.raises(ParameterExtractionError):
            ext.extract("find employee foo", person_template)

    def test_takes_first_entity_per_label(self, person_template):
        """When GLiNER returns multiple spans for the same label, keep the first."""
        ext = _make_gliner_extractor(
            predict_return=[
                {"label": "person", "text": "Alice", "score": 0.97},
                {"label": "person", "text": "Bob", "score": 0.85},
            ]
        )
        params = ext.extract("Find employee Alice or Bob", person_template)
        assert params["person"] == "Alice"

    def test_prediction_error_falls_through_to_heuristics(self, person_template):
        """If predict_entities raises, cascade continues to heuristics."""
        ext = _make_gliner_extractor(side_effect=RuntimeError("model error"))
        # Lowercase prefix so "Alice" is the only capitalized word — heuristics catch it
        params = ext.extract("find employee Alice", person_template)
        assert params == {"person": "Alice"}

    def test_empty_prediction_falls_through(self, person_template):
        """Empty GLiNER result must fall through to heuristics."""
        ext = _make_gliner_extractor(predict_return=[])
        params = ext.extract("find employee Alice", person_template)
        assert params == {"person": "Alice"}

    def test_no_parameters_template_skips_gliner(self):
        """Template with no parameters must skip GLiNER entirely."""
        ext = _make_gliner_extractor()
        template = QueryTemplate(
            intent="list_all",
            template_text="Show all records",
            query_template="SELECT * FROM records",
        )
        params = ext.extract("Show all records", template)
        assert params == {}
        ext._gliner.predict_entities.assert_not_called()


class TestCascadeWithGLiNER:
    @pytest.fixture
    def mixed_template(self):
        return QueryTemplate(
            intent="top_n_person",
            template_text="Show top {count} records for {person}",
            query_template="SELECT TOP {count} * FROM records WHERE person='{person}'",
            parameters=["count", "person"],
            parameter_patterns={"count": r"\b(\d+)\b"},
        )

    def test_regex_fills_count_gliner_fills_person(self, mixed_template):
        """Regex extracts count, GLiNER fills the remaining person param."""
        ext = _make_gliner_extractor(
            predict_return=[{"label": "person", "text": "Alice", "score": 0.96}]
        )
        params = ext.extract("Show top 5 records for Alice", mixed_template)
        assert params["count"] == "5"
        assert params["person"] == "Alice"

    def test_gliner_called_before_spacy(self, mixed_template):
        """When both enabled, GLiNER predict_entities must be called before spaCy."""
        call_order = []

        ext = ParameterExtractor(use_spacy=False, use_gliner=False)

        mock_gliner = MagicMock()
        mock_gliner.predict_entities.side_effect = lambda q, labels: (
            call_order.append("gliner") or [{"label": "person", "text": "Alice", "score": 0.9}]
        )
        ext._gliner = mock_gliner
        ext._gliner_available = True

        mock_nlp = MagicMock()
        mock_nlp.return_value.ents = []
        ext._nlp = mock_nlp
        ext._spacy_available = True

        ext.extract("Show top 5 records for Alice", mixed_template)
        assert call_order[0] == "gliner"

    def test_gliner_result_not_overwritten_by_spacy(self, mixed_template):
        """A param already filled by GLiNER must not be overwritten by spaCy."""
        ext = _make_gliner_extractor(
            predict_return=[{"label": "person", "text": "GlinerAlice", "score": 0.99}]
        )

        mock_doc = MagicMock()
        spacy_ent = MagicMock()
        spacy_ent.label_ = "PERSON"
        spacy_ent.text = "SpacyBob"
        mock_doc.ents = [spacy_ent]
        ext._nlp = MagicMock(return_value=mock_doc)
        ext._spacy_available = True

        params = ext.extract("Show top 5 records for GlinerAlice", mixed_template)
        assert params["person"] == "GlinerAlice"
