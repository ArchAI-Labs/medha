"""Unit tests for Medha async context manager protocol."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from medha.config import Settings
from medha.core import Medha


@pytest.fixture
def mock_backend():
    backend = MagicMock()
    backend.initialize = AsyncMock()
    backend.connect = AsyncMock()
    backend.count = AsyncMock(return_value=0)
    backend.close = AsyncMock()
    backend.load_stats = AsyncMock(return_value=None)
    backend.save_stats = AsyncMock()
    return backend


@pytest.fixture
def settings():
    return Settings(backend_type="memory", validate_on_start=True)


class TestAsyncContextManager:

    async def test_enter_calls_start(self, mock_embedder, mock_backend, settings):
        medha = Medha("test", mock_embedder, mock_backend, settings)
        async with medha as m:
            assert m is medha
            mock_backend.initialize.assert_awaited()

    async def test_exit_calls_close(self, mock_embedder, mock_backend, settings):
        medha = Medha("test", mock_embedder, mock_backend, settings)
        async with medha:
            pass
        mock_backend.close.assert_awaited_once()

    async def test_exit_on_exception(self, mock_embedder, mock_backend, settings):
        medha = Medha("test", mock_embedder, mock_backend, settings)
        with pytest.raises(ValueError, match="test error"):
            async with medha:
                raise ValueError("test error")
        mock_backend.close.assert_awaited_once()

    def test_sync_usage_not_supported(self):
        m = Medha.__new__(Medha)
        assert not hasattr(m, "__enter__")
