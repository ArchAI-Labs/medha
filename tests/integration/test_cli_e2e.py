"""E2E integration tests for medha.cli against a real InMemoryBackend (Spec 12 — CLI v0.4.0)."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

typer = pytest.importorskip("typer")
from typer.testing import CliRunner

from medha.backends.memory import InMemoryBackend
from medha.cli._app import app
from medha.config import Settings
from medha.core import Medha

runner = CliRunner()


# Prevent InMemoryBackend.close() from clearing _store between CLI invocations.
# __aexit__ must be patched on the CLASS because Python looks up dunder methods
# on the type, not the instance.
async def _noop_aexit(self, exc_type, exc_val, exc_tb):
    return False


@pytest.fixture
def e2e_medha(mock_embedder):
    """Started Medha+InMemoryBackend instance, function-scoped, for CLI e2e tests."""
    settings = Settings(backend_type="memory")
    backend = InMemoryBackend()
    m = Medha(collection_name="cli_e2e", embedder=mock_embedder, backend=backend, settings=settings)
    asyncio.run(m.start())
    yield m
    asyncio.run(m.close())


def _invoke(m: Medha, *args: str, env: dict | None = None):
    """Run a CLI command with _build_medha patched to return m and __aexit__ patched to no-op."""
    with patch.object(Medha, "__aexit__", _noop_aexit):
        with patch("medha.cli._app._build_medha", new=AsyncMock(return_value=m)):
            return runner.invoke(app, list(args), env=env)


@pytest.mark.cli
class TestCliE2E:
    def test_cli_stats_e2e(self, e2e_medha):
        """Store 3 entries in-process; medha stats must report count=3."""
        async def _store():
            for i in range(3):
                await e2e_medha.store(f"How many things {i}", f"SELECT {i}")

        asyncio.run(_store())

        result = _invoke(e2e_medha, "stats", "--collection", "cli_e2e")

        assert result.exit_code == 0
        assert "3" in result.output
        assert "cli_e2e" in result.output

    def test_cli_invalidate_e2e(self, e2e_medha):
        """Store one entry, invalidate it via CLI, verify Removed and count drops to 0."""
        question = "What is the user count"

        async def _store():
            await e2e_medha.store(question, "SELECT COUNT(*) FROM users")

        asyncio.run(_store())

        result = _invoke(e2e_medha, "invalidate", question)
        assert result.exit_code == 0
        assert "Removed" in result.output

        stats = _invoke(e2e_medha, "stats", "--collection", "cli_e2e")
        assert stats.exit_code == 0
        assert "0" in stats.output

    def test_cli_expire_e2e(self, e2e_medha):
        """Store an already-expired entry (ttl=-1); medha expire must delete it and report count=1."""
        async def _store():
            await e2e_medha.store("Expired question", "SELECT expired", ttl=-1)

        asyncio.run(_store())

        result = _invoke(e2e_medha, "expire", "--collection", "cli_e2e")

        assert result.exit_code == 0
        assert "1" in result.output

    def test_cli_warm_e2e(self, e2e_medha, tmp_path):
        """Write a JSONL file, warm via CLI, then verify stats shows the stored count."""
        entries = [
            {"question": f"warm question {i}", "generated_query": f"SELECT {i}"}
            for i in range(3)
        ]
        jsonl_file = tmp_path / "warm.jsonl"
        jsonl_file.write_text("\n".join(json.dumps(e) for e in entries))

        # Bypass the _noop early-exit check; _build_medha is still patched so
        # our MockEmbedder-backed instance is used for the actual warm.
        result = _invoke(
            e2e_medha,
            "warm",
            str(jsonl_file),
            env={"MEDHA_EMBEDDER_TYPE": "openai", "OPENAI_API_KEY": "test-key"},
        )
        assert result.exit_code == 0

        stats = _invoke(e2e_medha, "stats", "--collection", "cli_e2e")
        assert stats.exit_code == 0
        assert "3" in stats.output

    def test_cli_export_csv_e2e(self, e2e_medha, tmp_path):
        """Store 2 entries; export to CSV via CLI; verify file exists and contains the questions."""
        pytest.importorskip("pandas")

        async def _store():
            await e2e_medha.store("Export question one", "SELECT 1")
            await e2e_medha.store("Export question two", "SELECT 2")

        asyncio.run(_store())

        out_file = tmp_path / "export.csv"
        result = _invoke(
            e2e_medha,
            "export",
            "--collection", "cli_e2e",
            "--output", str(out_file),
        )

        assert result.exit_code == 0
        assert out_file.exists()
        content = out_file.read_text()
        assert "Export question" in content

    def test_cli_feedback_e2e(self, e2e_medha):
        """Store one entry; record --incorrect feedback via CLI; output must be 'Feedback recorded.'"""
        question = "How many products are there"

        async def _store():
            await e2e_medha.store(question, "SELECT COUNT(*) FROM products")

        asyncio.run(_store())

        result = _invoke(e2e_medha, "feedback", question, "--incorrect")

        assert result.exit_code == 0
        assert "Feedback recorded." in result.output
