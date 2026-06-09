"""Unit tests for Settings.validate_on_start and the start() connectivity probe."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from medha.config import Settings
from medha.core import Medha
from medha.exceptions import StorageError


@pytest.fixture
def mock_backend():
    backend = MagicMock()
    backend.initialize = AsyncMock()
    backend.connect = AsyncMock()
    backend.count = AsyncMock(return_value=0)
    backend.close = AsyncMock()
    return backend


class TestStartupValidation:

    async def test_count_called_on_start(self, mock_embedder, mock_backend):
        settings = Settings(validate_on_start=True)
        m = Medha("test_validate", mock_embedder, mock_backend, settings)
        await m.start()
        await m.close()
        called_with = [c.args[0] for c in mock_backend.count.call_args_list]
        assert "test_validate" in called_with

    async def test_count_not_called_when_disabled(self, mock_embedder, mock_backend):
        # Make legacy check raise StorageError so it's silently swallowed
        mock_backend.count.side_effect = StorageError("not found")
        settings = Settings(validate_on_start=False)
        m = Medha("test_no_validate", mock_embedder, mock_backend, settings)
        await m.start()
        await m.close()
        called_with = [c.args[0] for c in mock_backend.count.call_args_list]
        assert "test_no_validate" not in called_with

    async def test_storage_error_on_connection_failure(self, mock_embedder, mock_backend):
        collection_name = "test_conn_fail"

        def count_side_effect(coll):
            if coll == collection_name:
                raise Exception("connection refused")
            return 0  # legacy check passes normally

        mock_backend.count.side_effect = count_side_effect
        settings = Settings(validate_on_start=True)
        m = Medha(collection_name, mock_embedder, mock_backend, settings)
        with pytest.raises(StorageError, match="connectivity check"):
            await m.start()
