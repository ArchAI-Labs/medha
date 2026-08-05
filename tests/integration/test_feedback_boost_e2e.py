"""End-to-end tests for feedback score boosting (0.5.0).

These tests use a hand-built embedder with a known geometry instead of a real
model, so the similarity score is exact and the assertions do not drift when an
embedding model changes.  The stored question and the search question sit at a
cosine similarity of exactly 0.60 — below the 0.85 semantic threshold, so the
entry is only reachable once positive feedback boosts it.
"""

import pytest

from medha.backends.memory import InMemoryBackend
from medha.config import Settings
from medha.core import Medha
from medha.interfaces.embedder import BaseEmbedder
from medha.types import SearchStrategy
from medha.utils.normalization import normalize_question

STORED_QUESTION = "what is the total revenue for last quarter"
SIMILAR_QUESTION = "give me quarterly income figures"
STORED_QUERY = "SELECT SUM(revenue) FROM sales WHERE quarter = 'Q4'"

# cos(theta) between the two vectors below is exactly 0.60
_STORED_VECTOR = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
_SIMILAR_VECTOR = [0.6, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
_UNRELATED_VECTOR = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class GeometricEmbedder(BaseEmbedder):
    """Maps the two known questions to vectors 0.60 apart; anything else is orthogonal."""

    def __init__(self) -> None:
        self._map = {
            normalize_question(STORED_QUESTION): _STORED_VECTOR,
            normalize_question(SIMILAR_QUESTION): _SIMILAR_VECTOR,
        }

    @property
    def dimension(self) -> int:
        return 8

    @property
    def model_name(self) -> str:
        return "geometric-test-embedder"

    async def aembed(self, text: str) -> list[float]:
        return self._map.get(normalize_question(text), _UNRELATED_VECTOR)

    async def aembed_batch(self, texts: list[str], **kwargs: object) -> list[list[float]]:
        return [await self.aembed(t) for t in texts]


async def _make_medha(boost_factor: float) -> Medha:
    backend = InMemoryBackend()
    await backend.connect()
    settings = Settings(
        backend_type="memory",
        score_threshold_exact=0.99,
        score_threshold_semantic=0.85,
        score_threshold_fuzzy=80.0,
        feedback_boost_factor=boost_factor,
        l1_cache_max_size=0,  # keep every search on the vector path
    )
    m = Medha("boost_e2e", GeometricEmbedder(), backend, settings)
    await m.start()
    return m


async def _give_positive_feedback(m: Medha, times: int) -> None:
    for _ in range(times):
        assert await m.feedback(STORED_QUESTION, correct=True) is True


class TestFeedbackBoostE2E:
    async def test_baseline_score_is_below_threshold(self):
        """Without feedback the entry is unreachable — the premise of these tests."""
        m = await _make_medha(boost_factor=0.0)
        try:
            await m.store(STORED_QUESTION, STORED_QUERY)

            result = await m.search(SIMILAR_QUESTION)

            assert result.strategy == SearchStrategy.NO_MATCH
        finally:
            await m.close()

    async def test_boost_raises_low_score_above_threshold(self):
        m = await _make_medha(boost_factor=0.5)
        try:
            await m.store(STORED_QUESTION, STORED_QUERY)
            await _give_positive_feedback(m, 5)

            result = await m.search(SIMILAR_QUESTION)

            # trust = 1.0 → 0.60 * (1 + 0.5) = 0.90 ≥ 0.85
            assert result.strategy == SearchStrategy.SEMANTIC_MATCH
            assert result.generated_query == STORED_QUERY
        finally:
            await m.close()

    async def test_no_boost_when_factor_is_zero(self):
        m = await _make_medha(boost_factor=0.0)
        try:
            await m.store(STORED_QUESTION, STORED_QUERY)
            await _give_positive_feedback(m, 5)

            result = await m.search(SIMILAR_QUESTION)

            assert result.strategy == SearchStrategy.NO_MATCH
        finally:
            await m.close()

    async def test_boost_too_small_still_misses(self):
        """0.60 * (1 + 0.1) = 0.66 — still short of the 0.85 threshold."""
        m = await _make_medha(boost_factor=0.1)
        try:
            await m.store(STORED_QUESTION, STORED_QUERY)
            await _give_positive_feedback(m, 5)

            result = await m.search(SIMILAR_QUESTION)

            assert result.strategy == SearchStrategy.NO_MATCH
        finally:
            await m.close()

    async def test_negative_feedback_does_not_boost(self):
        m = await _make_medha(boost_factor=0.5)
        try:
            await m.store(STORED_QUESTION, STORED_QUERY)
            for _ in range(5):
                await m.feedback(STORED_QUESTION, correct=False)

            result = await m.search(SIMILAR_QUESTION)

            # trust = 0.0 → score unchanged → still a miss
            assert result.strategy == SearchStrategy.NO_MATCH
        finally:
            await m.close()

    async def test_mixed_feedback_scales_the_boost(self):
        """trust = 3/5 = 0.6 → 0.60 * (1 + 0.5*0.6) = 0.78, still below 0.85."""
        m = await _make_medha(boost_factor=0.5)
        try:
            await m.store(STORED_QUESTION, STORED_QUERY)
            for _ in range(3):
                await m.feedback(STORED_QUESTION, correct=True)
            for _ in range(2):
                await m.feedback(STORED_QUESTION, correct=False)

            result = await m.search(SIMILAR_QUESTION)

            assert result.strategy == SearchStrategy.NO_MATCH
        finally:
            await m.close()

    async def test_exact_question_still_hits_without_feedback(self):
        """Boosting must not disturb the tiers above semantic."""
        m = await _make_medha(boost_factor=0.5)
        try:
            await m.store(STORED_QUESTION, STORED_QUERY)

            result = await m.search(STORED_QUESTION)

            assert result.strategy != SearchStrategy.NO_MATCH
            assert result.generated_query == STORED_QUERY
        finally:
            await m.close()

    async def test_unrelated_question_is_never_boosted_into_a_hit(self):
        """An orthogonal question stays a miss no matter how trusted the entry is."""
        m = await _make_medha(boost_factor=1.0)
        try:
            await m.store(STORED_QUESTION, STORED_QUERY)
            await _give_positive_feedback(m, 10)

            result = await m.search("completely different topic about weather patterns")

            assert result.strategy == SearchStrategy.NO_MATCH
        finally:
            await m.close()

    async def test_confidence_reports_the_boosted_score(self):
        m = await _make_medha(boost_factor=0.5)
        try:
            await m.store(STORED_QUESTION, STORED_QUERY)
            await _give_positive_feedback(m, 5)

            result = await m.search(SIMILAR_QUESTION)

            # boosted 0.90, then the standard 0.9x semantic penalty
            assert result.confidence == pytest.approx(0.81, abs=1e-6)
        finally:
            await m.close()
