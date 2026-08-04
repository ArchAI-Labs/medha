"""End-to-end tests for persistent cache statistics (0.5.0).

Uses InMemoryBackend shared across two Medha instances to simulate a process
restart: the backend outlives the Medha object, which is exactly the property
that disk/network backends provide across real restarts.
"""

import asyncio
import json

import pytest

from medha.backends.memory import InMemoryBackend
from medha.config import Settings
from medha.core import Medha
from medha.types import PersistedStats


def _settings(**overrides) -> Settings:
    base = dict(
        backend_type="memory",
        score_threshold_exact=0.99,
        score_threshold_semantic=0.85,
        stats_persist_interval=1,  # persist on every request
    )
    base.update(overrides)
    return Settings(**base)


class TestPersistedStatsE2E:
    """Restart tests rely on close() flushing — no manual draining.

    These deliberately do NOT await Medha._stats_persist_tasks by hand: a real
    application never does, so a test that drained the queue itself would pass
    even if shutdown dropped every pending snapshot.
    """

    async def test_stats_survive_restart(self, mock_embedder):
        backend = InMemoryBackend()
        await backend.connect()

        m1 = Medha("stats_test", mock_embedder, backend, _settings())
        await m1.start()
        await m1.store("what is revenue?", "SELECT SUM(revenue) FROM sales")
        await m1.search("what is revenue?")  # exact hit
        await m1.search("something entirely unrelated to revenue")  # miss
        await m1.close()

        # Simulate a restart: brand-new Medha, same backend.
        m2 = Medha("stats_test", mock_embedder, backend, _settings())
        await m2.start()
        try:
            stats = await m2.stats("stats_test")
            assert stats.total_requests == 2, "stats must survive across Medha instances"
            assert stats.total_hits >= 1
        finally:
            await m2.close()

    async def test_counters_accumulate_across_restarts(self, mock_embedder):
        backend = InMemoryBackend()
        await backend.connect()

        m1 = Medha("acc_test", mock_embedder, backend, _settings())
        await m1.start()
        await m1.search("first")
        await m1.close()

        m2 = Medha("acc_test", mock_embedder, backend, _settings())
        await m2.start()
        try:
            await m2.search("second")
            stats = await m2.stats("acc_test")
            assert stats.total_requests == 2
        finally:
            await m2.close()

    async def test_hit_rate_reflects_persisted_snapshot(self, mock_embedder):
        backend = InMemoryBackend()
        await backend.connect()

        m1 = Medha("rate_test", mock_embedder, backend, _settings())
        await m1.start()
        await m1.store("cached question", "SELECT 1")
        await m1.search("cached question")  # hit
        await m1.search("uncached question here")  # miss
        await m1.close()

        persisted = await backend.load_stats("rate_test")
        assert persisted is not None
        assert persisted.total_requests == 2
        assert persisted.total_hits == 1
        assert persisted.hit_rate == pytest.approx(0.5)

    async def test_interval_controls_write_frequency(self, mock_embedder):
        backend = InMemoryBackend()
        await backend.connect()

        m = Medha("interval_test", mock_embedder, backend, _settings(stats_persist_interval=3))
        await m.start()
        try:
            await m.search("one")
            await m.search("two")
            # Below the interval no task is scheduled at all, so this is not racy.
            assert await backend.load_stats("interval_test") is None, "not yet at the interval"

            await m.search("three")
            # This is the one test that must observe the *periodic* write while
            # the process is still alive — the single thing close()'s flush
            # cannot demonstrate — so it awaits the scheduled task directly.
            await asyncio.gather(*m._stats_persist_tasks)
            persisted = await backend.load_stats("interval_test")
            assert persisted is not None
            assert persisted.total_requests == 3
        finally:
            await m.close()

    async def test_close_flushes_below_the_interval(self, mock_embedder):
        """A process that never reaches the interval must still persist on close.

        Regression: with the default interval of 100, a short-lived or
        low-traffic process used to persist nothing at all — search() schedules
        snapshots only on interval boundaries, and close() did not flush.
        """
        backend = InMemoryBackend()
        await backend.connect()

        m = Medha("short_lived", mock_embedder, backend, _settings(stats_persist_interval=100))
        await m.start()
        for i in range(5):  # far below the interval: no task is ever scheduled
            await m.search(f"question {i}")
        assert await backend.load_stats("short_lived") is None, "nothing written mid-life"
        await m.close()

        persisted = await backend.load_stats("short_lived")
        assert persisted is not None, "close() must flush the pending stats"
        assert persisted.total_requests == 5

    async def test_close_captures_an_in_flight_snapshot(self, mock_embedder):
        """close() drains scheduled tasks before shutting the backend down.

        Regression: closing while a snapshot task was still in flight raced the
        backend teardown, and the write failed with "Not connected".
        """
        backend = InMemoryBackend()
        await backend.connect()

        m = Medha("in_flight", mock_embedder, backend, _settings(stats_persist_interval=1))
        await m.start()
        await m.search("only question")
        assert m._stats_persist_tasks, "a task should still be pending at close time"
        await m.close()

        persisted = await backend.load_stats("in_flight")
        assert persisted is not None
        assert persisted.total_requests == 1

    async def test_close_without_requests_writes_nothing(self, mock_embedder):
        """An idle instance must not overwrite an existing snapshot with zeros."""
        backend = InMemoryBackend()
        await backend.connect()
        await backend.save_stats("idle_test", PersistedStats(total_requests=42, total_hits=40))

        m = Medha("idle_test", mock_embedder, backend, _settings())
        await m.start()
        await m.close()  # no search() at all

        persisted = await backend.load_stats("idle_test")
        assert persisted is not None
        assert persisted.total_requests == 42, "an idle instance must not clobber the snapshot"

    async def test_stats_are_isolated_per_collection(self, mock_embedder):
        backend = InMemoryBackend()
        await backend.connect()

        m_a = Medha("coll_a", mock_embedder, backend, _settings())
        await m_a.start()
        await m_a.search("question for a")
        await m_a.close()

        assert await backend.load_stats("coll_b") is None
        stats_a = await backend.load_stats("coll_a")
        assert stats_a is not None and stats_a.total_requests == 1

    async def test_corrupt_snapshot_does_not_block_start(self, mock_embedder):
        """A backend that fails to load stats must degrade to a cold start."""
        from medha.exceptions import StorageError

        backend = InMemoryBackend()
        await backend.connect()

        async def _boom(collection_name):
            raise StorageError("stats blob unreadable")

        backend.load_stats = _boom

        m = Medha("corrupt_test", mock_embedder, backend, _settings())
        await m.start()  # must not raise
        try:
            stats = await m.stats("corrupt_test")
            assert stats.total_requests == 0
        finally:
            await m.close()


@pytest.mark.cli
class TestCliStatsShowsPersistedMetrics:
    """`medha stats` surfaces the persisted snapshot when one exists.

    These tests are synchronous: the CLI drives its own asyncio.run(), so the
    backend is created and seeded inside that same loop (InMemoryBackend's lock
    binds to the first loop that uses it).
    """

    def _run_cli(self, args, seed: PersistedStats | None):
        from contextlib import asynccontextmanager
        from unittest.mock import patch

        typer_testing = pytest.importorskip("typer.testing")
        from medha.cli._app import app

        settings = _settings()

        @asynccontextmanager
        async def _fake_build(collection, _settings_arg):
            backend = InMemoryBackend()
            await backend.connect()
            await backend.initialize(collection, 384)
            if seed is not None:
                await backend.save_stats(collection, seed)
            m = Medha(collection, _FixedDimEmbedder(), backend, settings)
            await m.start()
            try:
                yield m
            finally:
                await m.close()

        with patch("medha.cli._app._build_medha", new=_fake_build):
            return typer_testing.CliRunner().invoke(app, args)

    def test_json_reports_hit_rate(self):
        seed = PersistedStats(
            total_requests=10,
            total_hits=8,
            hits_by_strategy={"semantic_match": 5, "l1_cache": 3},
        )

        result = self._run_cli(["stats", "--collection", "cli_stats", "--json"], seed)

        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["total_requests"] == 10
        assert data["hit_rate"] == pytest.approx(0.8)
        assert data["hits_by_strategy"]["semantic_match"] == 5

    def test_human_output_reports_hit_rate(self):
        seed = PersistedStats(total_requests=4, total_hits=3)

        result = self._run_cli(["stats", "--collection", "cli_stats_h"], seed)

        assert result.exit_code == 0, result.output
        assert "Requests" in result.output
        assert "75.0%" in result.output

    def test_json_reports_null_when_never_persisted(self):
        result = self._run_cli(["stats", "--collection", "cli_stats_empty", "--json"], None)

        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["hit_rate"] is None
        assert data["total_requests"] is None

    def test_human_output_notes_missing_snapshot(self):
        result = self._run_cli(["stats", "--collection", "cli_stats_none"], None)

        assert result.exit_code == 0, result.output
        assert "not yet persisted" in result.output


class _FixedDimEmbedder:
    """Minimal embedder for CLI tests — never actually queried for a search."""

    @property
    def dimension(self) -> int:
        return 384

    @property
    def model_name(self) -> str:
        return "fixed-dim"

    async def aembed(self, text: str) -> list[float]:
        return [0.0] * 384

    async def aembed_batch(self, texts: list[str], **kwargs) -> list[list[float]]:
        return [[0.0] * 384 for _ in texts]
