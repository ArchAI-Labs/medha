"""Unit tests for feedback-driven score boosting (0.5.0).

Covers the pure formula (_apply_feedback_boost) and its two consumers in the
waterfall: the semantic tier and the fuzzy fallback.
"""

from unittest.mock import AsyncMock

import pytest

from medha.config import Settings
from medha.core import Medha
from medha.types import CacheResult, SearchStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _medha_with(settings: Settings, results: list[CacheResult]) -> Medha:
    """Build a bare Medha wired to a stub backend returning ``results``.

    The stub mimics a real backend: it filters by score_threshold and returns
    matches sorted by descending raw score.
    """
    m = Medha.__new__(Medha)
    m._settings = settings
    m._collection_name = "boost_test"

    async def _search(collection_name, vector, limit=5, score_threshold=0.0):
        matching = [r for r in results if r.score >= score_threshold]
        matching.sort(key=lambda r: r.score, reverse=True)
        return matching[:limit]

    m._backend = AsyncMock()
    m._backend.search = AsyncMock(side_effect=_search)
    return m


def _result(
    id: str,
    score: float,
    query: str = "SELECT 1",
    correct: int = 0,
    incorrect: int = 0,
    question: str = "count all orders",
) -> CacheResult:
    return CacheResult(
        id=id,
        score=score,
        original_question=question,
        normalized_question=question,
        generated_query=query,
        query_hash="h" + id,
        feedback_correct=correct,
        feedback_incorrect=incorrect,
    )


# ---------------------------------------------------------------------------
# The formula
# ---------------------------------------------------------------------------


class TestApplyFeedbackBoost:
    def test_boost_disabled_by_default(self):
        m = Medha.__new__(Medha)
        m._settings = Settings()

        assert m._settings.feedback_boost_factor == 0.0
        assert m._apply_feedback_boost(0.85, 100, 0) == 0.85
        assert m._apply_feedback_boost(0.85, 0, 100) == 0.85

    def test_no_feedback_no_change(self):
        m = Medha.__new__(Medha)
        m._settings = Settings(feedback_boost_factor=0.5)

        assert m._apply_feedback_boost(0.85, 0, 0) == 0.85

    def test_full_positive_feedback_clamps_at_one(self):
        m = Medha.__new__(Medha)
        m._settings = Settings(feedback_boost_factor=0.5)

        # trust = 1.0 → 0.8 * 1.5 = 1.2 → clamped
        assert m._apply_feedback_boost(0.8, 10, 0) == 1.0

    def test_mixed_feedback(self):
        m = Medha.__new__(Medha)
        m._settings = Settings(feedback_boost_factor=0.4)

        # trust = 3/4 = 0.75 → 0.5 * (1 + 0.4 * 0.75) = 0.5 * 1.3 = 0.65
        assert m._apply_feedback_boost(0.5, 3, 1) == pytest.approx(0.65)

    def test_all_negative_feedback_leaves_score_untouched(self):
        m = Medha.__new__(Medha)
        m._settings = Settings(feedback_boost_factor=0.5)

        # trust = 0.0 → the boost never penalises, it only rewards
        assert m._apply_feedback_boost(0.85, 0, 5) == 0.85

    def test_boost_is_monotonic_in_trust(self):
        m = Medha.__new__(Medha)
        m._settings = Settings(feedback_boost_factor=0.3)

        low = m._apply_feedback_boost(0.5, 1, 3)
        mid = m._apply_feedback_boost(0.5, 2, 2)
        high = m._apply_feedback_boost(0.5, 3, 1)
        assert low < mid < high

    def test_result_never_exceeds_one(self):
        m = Medha.__new__(Medha)
        m._settings = Settings(feedback_boost_factor=1.0)

        assert m._apply_feedback_boost(0.99, 50, 0) == 1.0


class TestFeedbackBoostFactorSetting:
    def test_default_is_zero(self):
        assert Settings().feedback_boost_factor == 0.0

    def test_rejects_negative(self):
        with pytest.raises(ValueError):
            Settings(feedback_boost_factor=-0.1)

    def test_rejects_above_one(self):
        with pytest.raises(ValueError):
            Settings(feedback_boost_factor=1.1)

    def test_env_var_is_honoured(self, monkeypatch):
        monkeypatch.setenv("MEDHA_FEEDBACK_BOOST_FACTOR", "0.25")
        assert Settings().feedback_boost_factor == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Semantic tier
# ---------------------------------------------------------------------------


class TestSemanticTierBoost:
    async def test_disabled_boost_keeps_retrieval_threshold(self):
        settings = Settings(score_threshold_semantic=0.85, feedback_boost_factor=0.0)
        m = _medha_with(settings, [_result("a", 0.90)])

        await m._search_semantic([0.1] * 8)

        assert m._backend.search.await_args.kwargs["score_threshold"] == pytest.approx(0.85)

    async def test_enabled_boost_lowers_retrieval_threshold(self):
        """Retrieval must dip below the threshold so good entries can be rescued."""
        settings = Settings(score_threshold_semantic=0.90, feedback_boost_factor=0.5)
        m = _medha_with(settings, [_result("a", 0.95)])

        await m._search_semantic([0.1] * 8)

        # 0.90 / (1 + 0.5) = 0.60 — the tightest bound that can still reach 0.90
        assert m._backend.search.await_args.kwargs["score_threshold"] == pytest.approx(0.60)

    async def test_sub_threshold_entry_rescued_by_positive_feedback(self):
        settings = Settings(score_threshold_semantic=0.85, feedback_boost_factor=0.5)
        m = _medha_with(settings, [_result("a", 0.60, "SELECT rescued", correct=10)])

        hit = await m._search_semantic([0.1] * 8)

        assert hit is not None
        assert hit.strategy == SearchStrategy.SEMANTIC_MATCH
        assert hit.generated_query == "SELECT rescued"

    async def test_same_entry_is_a_miss_without_boost(self):
        settings = Settings(score_threshold_semantic=0.85, feedback_boost_factor=0.0)
        m = _medha_with(settings, [_result("a", 0.60, "SELECT rescued", correct=10)])

        assert await m._search_semantic([0.1] * 8) is None

    async def test_sub_threshold_entry_not_rescued_without_feedback(self):
        settings = Settings(score_threshold_semantic=0.85, feedback_boost_factor=0.5)
        m = _medha_with(settings, [_result("a", 0.60)])

        assert await m._search_semantic([0.1] * 8) is None

    async def test_negative_feedback_entry_not_rescued(self):
        settings = Settings(score_threshold_semantic=0.85, feedback_boost_factor=0.5)
        m = _medha_with(settings, [_result("a", 0.60, correct=0, incorrect=8)])

        assert await m._search_semantic([0.1] * 8) is None

    async def test_trusted_entry_outranks_closer_untrusted_one(self):
        """Boosting re-ranks: a well-rated entry can beat a marginally closer one."""
        settings = Settings(score_threshold_semantic=0.85, feedback_boost_factor=0.5)
        m = _medha_with(
            settings,
            [
                _result("closer", 0.90, "SELECT closer"),
                _result("trusted", 0.88, "SELECT trusted", correct=10),
            ],
        )

        hit = await m._search_semantic([0.1] * 8)

        assert hit is not None
        assert hit.generated_query == "SELECT trusted"

    async def test_ranking_unchanged_when_boost_disabled(self):
        settings = Settings(score_threshold_semantic=0.85, feedback_boost_factor=0.0)
        m = _medha_with(
            settings,
            [
                _result("closer", 0.90, "SELECT closer"),
                _result("trusted", 0.88, "SELECT trusted", correct=10),
            ],
        )

        hit = await m._search_semantic([0.1] * 8)

        assert hit is not None
        assert hit.generated_query == "SELECT closer"

    async def test_confidence_reflects_boosted_score(self):
        settings = Settings(score_threshold_semantic=0.85, feedback_boost_factor=0.5)
        m = _medha_with(settings, [_result("a", 0.60, correct=10)])

        hit = await m._search_semantic([0.1] * 8)

        # boosted 0.60 * 1.5 = 0.90, then the standard 0.9x semantic penalty
        assert hit is not None
        assert hit.confidence == pytest.approx(0.81)

    async def test_empty_results_return_none(self):
        settings = Settings(score_threshold_semantic=0.85, feedback_boost_factor=0.5)
        m = _medha_with(settings, [])

        assert await m._search_semantic([0.1] * 8) is None


# ---------------------------------------------------------------------------
# Fuzzy tier
# ---------------------------------------------------------------------------


class TestFuzzyTierBoost:
    """The fuzzy ratio is normalised to 0..1 before the boost is applied."""

    QUESTION = "count all orders"
    STORED = "count all the orders placed"  # rapidfuzz ratio ≈ 74.4

    def _medha(self, factor: float, correct: int, incorrect: int = 0) -> Medha:
        settings = Settings(
            score_threshold_fuzzy=80.0,
            score_threshold_fuzzy_prefilter=0.0,
            feedback_boost_factor=factor,
        )
        return _medha_with(
            settings,
            [
                _result(
                    "f",
                    0.70,
                    "SELECT count(*) FROM orders",
                    correct=correct,
                    incorrect=incorrect,
                    question=self.STORED,
                )
            ],
        )

    def test_raw_ratio_is_below_threshold(self):
        """Guards the calibration the rescue tests below depend on."""
        fuzz = pytest.importorskip("rapidfuzz").fuzz
        assert 70.0 < fuzz.ratio(self.QUESTION, self.STORED) < 80.0

    async def test_no_hit_without_boost(self):
        pytest.importorskip("rapidfuzz")
        m = self._medha(factor=0.0, correct=10)

        assert await m._search_fuzzy(self.QUESTION, [0.1] * 8) is None

    async def test_positive_feedback_rescues_fuzzy_match(self):
        pytest.importorskip("rapidfuzz")
        m = self._medha(factor=0.5, correct=10)

        hit = await m._search_fuzzy(self.QUESTION, [0.1] * 8)

        assert hit is not None
        assert hit.strategy == SearchStrategy.FUZZY_MATCH
        assert hit.generated_query == "SELECT count(*) FROM orders"

    async def test_negative_feedback_does_not_rescue(self):
        pytest.importorskip("rapidfuzz")
        m = self._medha(factor=0.5, correct=0, incorrect=10)

        assert await m._search_fuzzy(self.QUESTION, [0.1] * 8) is None

    async def test_confidence_stays_normalised(self):
        """Confidence is a 0..1 ratio, not the raw 0..100 rapidfuzz score."""
        pytest.importorskip("rapidfuzz")
        m = self._medha(factor=0.0, correct=0)
        m._settings = Settings(
            score_threshold_fuzzy=70.0,
            score_threshold_fuzzy_prefilter=0.0,
            feedback_boost_factor=0.0,
        )

        hit = await m._search_fuzzy(self.QUESTION, [0.1] * 8)

        assert hit is not None
        assert 0.70 < hit.confidence < 0.80
