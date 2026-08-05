"""Integration tests for automatic TTL cleanup via start()/close()."""


import pytest

from medha.backends.memory import InMemoryBackend
from medha.config import Settings
from medha.core import Medha


@pytest.fixture
def mock_embedder(mock_embedder):  # reuse conftest fixture
    return mock_embedder


class TestCleanupTask:
    async def test_cleanup_task_started_by_start(self, mock_embedder):
        """start() must create a background cleanup task when cleanup_interval_seconds is set."""
        settings = Settings(
            backend_type="memory",
            cleanup_interval_seconds=60,
            l1_cache_max_size=0,
        )
        m = Medha(
            collection_name="cleanup_start_test",
            embedder=mock_embedder,
            backend=InMemoryBackend(),
            settings=settings,
        )
        await m.start()
        try:
            assert m._cleanup_task is not None
            assert not m._cleanup_task.done()
        finally:
            await m.close()

    async def test_cleanup_task_not_started_when_disabled(self, mock_embedder):
        """start() must NOT create a cleanup task when cleanup_interval_seconds is None."""
        settings = Settings(
            backend_type="memory",
            cleanup_interval_seconds=None,
            l1_cache_max_size=0,
        )
        m = Medha(
            collection_name="no_cleanup_test",
            embedder=mock_embedder,
            backend=InMemoryBackend(),
            settings=settings,
        )
        await m.start()
        try:
            assert m._cleanup_task is None
        finally:
            await m.close()

    async def test_cleanup_task_cancelled_by_close(self, mock_embedder):
        """close() must cancel the cleanup task without raising."""
        settings = Settings(
            backend_type="memory",
            cleanup_interval_seconds=60,
            l1_cache_max_size=0,
        )
        m = Medha(
            collection_name="cleanup_close_test",
            embedder=mock_embedder,
            backend=InMemoryBackend(),
            settings=settings,
        )
        await m.start()
        task = m._cleanup_task
        assert task is not None

        await m.close()  # must not raise

        assert task.done()
        assert task.cancelled()
        assert m._cleanup_task is None

    async def test_cleanup_removes_expired_entries_automatically(self, mock_embedder):
        """After cleanup fires, expired entries must be removed from the backend."""
        settings = Settings(
            backend_type="memory",
            cleanup_interval_seconds=60,  # interval is long but we call expire() manually
            l1_cache_max_size=0,
        )
        backend = InMemoryBackend()
        m = Medha(
            collection_name="auto_cleanup_test",
            embedder=mock_embedder,
            backend=backend,
            settings=settings,
        )
        await m.start()
        try:
            # Store one expired entry (ttl=-1 → already past)
            await m.store("expired entry", "SELECT exp", ttl=-1)
            # Store one immortal entry
            await m.store("immortal entry", "SELECT immortal")

            count_before = await backend.count("auto_cleanup_test")
            assert count_before == 2

            # Manually trigger expire (simulates what _cleanup_loop does)
            deleted = await m.expire()
            assert deleted == 1

            count_after = await backend.count("auto_cleanup_test")
            assert count_after == 1
        finally:
            await m.close()

    async def test_context_manager_cancels_cleanup_task(self, mock_embedder):
        """Using Medha as async context manager must cancel cleanup on exit."""
        settings = Settings(
            backend_type="memory",
            cleanup_interval_seconds=60,
            l1_cache_max_size=0,
        )
        task_ref: list = []

        async with Medha(
            collection_name="ctx_cleanup_test",
            embedder=mock_embedder,
            backend=InMemoryBackend(),
            settings=settings,
        ) as m:
            task_ref.append(m._cleanup_task)

        task = task_ref[0]
        assert task is not None
        assert task.done()
        assert task.cancelled()
