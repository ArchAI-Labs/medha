"""Unit tests for medha.cli commands and supporting components (Spec 12 — CLI v0.4.0)."""

from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

typer = pytest.importorskip("typer")
from typer.testing import CliRunner

from medha.cli._app import _resolve_embedder, app
from medha.cli._noop_embedder import _NoOpEmbedder
from medha.config import Settings
from medha.exceptions import ConfigurationError

runner = CliRunner()


def _make_mock_medha(**overrides):
    """Create a mock Medha instance with async context manager support and sane defaults."""
    m = MagicMock()
    m.__aenter__ = AsyncMock(return_value=m)
    m.__aexit__ = AsyncMock(return_value=False)
    m._collection_name = "default"
    m._template_collection = "default_templates"
    m._backend = MagicMock()
    m._backend.count = AsyncMock(return_value=0)
    for k, v in overrides.items():
        setattr(m, k, v)
    return m


@pytest.mark.cli
class TestCliStats:
    def test_stats_prints_collection_and_count(self):
        mock_medha = _make_mock_medha()
        mock_medha._backend.count = AsyncMock(side_effect=[5, 2])

        with patch("medha.cli._app._build_medha", new=AsyncMock(return_value=mock_medha)):
            result = runner.invoke(app, ["stats", "--collection", "test_coll"])

        assert result.exit_code == 0
        assert "test_coll" in result.output
        assert "5" in result.output
        assert "2" in result.output

    def test_stats_unknown_backend_exits_nonzero(self):
        with patch(
            "medha.cli._app._build_medha",
            new=AsyncMock(side_effect=ConfigurationError("bad backend")),
        ):
            result = runner.invoke(app, ["stats"])

        assert result.exit_code != 0


@pytest.mark.cli
class TestCliInvalidate:
    def test_invalidate_found_prints_removed(self):
        mock_medha = _make_mock_medha()
        mock_medha.invalidate = AsyncMock(return_value=True)

        with patch("medha.cli._app._build_medha", new=AsyncMock(return_value=mock_medha)):
            result = runner.invoke(app, ["invalidate", "my question"])

        assert result.exit_code == 0
        assert "Removed" in result.output

    def test_invalidate_not_found_prints_not_found(self):
        mock_medha = _make_mock_medha()
        mock_medha.invalidate = AsyncMock(return_value=False)

        with patch("medha.cli._app._build_medha", new=AsyncMock(return_value=mock_medha)):
            result = runner.invoke(app, ["invalidate", "no such question"])

        assert result.exit_code == 0
        assert "Not found" in result.output


@pytest.mark.cli
class TestCliInvalidateCollection:
    def test_invalidate_collection_requires_yes_flag(self):
        result = runner.invoke(app, ["invalidate-collection"])
        assert result.exit_code != 0

    def test_invalidate_collection_with_yes_succeeds(self):
        mock_medha = _make_mock_medha()
        mock_medha.invalidate_collection = AsyncMock(return_value=10)

        with patch("medha.cli._app._build_medha", new=AsyncMock(return_value=mock_medha)):
            result = runner.invoke(app, ["invalidate-collection", "--yes"])

        assert result.exit_code == 0
        assert "10" in result.output


@pytest.mark.cli
class TestCliExpire:
    def test_expire_prints_deleted_count(self):
        mock_medha = _make_mock_medha()
        mock_medha.expire = AsyncMock(return_value=7)

        with patch("medha.cli._app._build_medha", new=AsyncMock(return_value=mock_medha)):
            result = runner.invoke(app, ["expire"])

        assert result.exit_code == 0
        assert "7" in result.output

    def test_expire_zero_deleted(self):
        mock_medha = _make_mock_medha()
        mock_medha.expire = AsyncMock(return_value=0)

        with patch("medha.cli._app._build_medha", new=AsyncMock(return_value=mock_medha)):
            result = runner.invoke(app, ["expire"])

        assert result.exit_code == 0
        assert "0" in result.output


@pytest.mark.cli
class TestCliDedup:
    def test_dedup_prints_removed_count(self):
        pytest.importorskip("pandas")
        mock_medha = _make_mock_medha()
        mock_medha.dedup_collection = AsyncMock(return_value=3)

        with patch("medha.cli._app._build_medha", new=AsyncMock(return_value=mock_medha)):
            result = runner.invoke(app, ["dedup"])

        assert result.exit_code == 0
        assert "3" in result.output

    def test_dedup_missing_pandas_prints_actionable_error(self):
        with patch.dict(sys.modules, {"pandas": None}):
            result = runner.invoke(app, ["dedup"])

        assert result.exit_code != 0
        assert "pandas" in result.output


@pytest.mark.cli
class TestCliExport:
    def test_export_csv_to_stdout(self):
        pd = pytest.importorskip("pandas")
        mock_medha = _make_mock_medha()
        mock_df = pd.DataFrame({"question": ["q1", "q2"], "query": ["SELECT 1", "SELECT 2"]})
        mock_medha.export_to_dataframe = AsyncMock(return_value=mock_df)

        with patch("medha.cli._app._build_medha", new=AsyncMock(return_value=mock_medha)):
            result = runner.invoke(app, ["export"])

        assert result.exit_code == 0
        assert "question" in result.output
        assert "q1" in result.output

    def test_export_json_to_file(self, tmp_path):
        pd = pytest.importorskip("pandas")
        out_file = tmp_path / "out.json"
        mock_medha = _make_mock_medha()
        mock_df = pd.DataFrame({"question": ["q1"], "query": ["SELECT 1"]})
        mock_medha.export_to_dataframe = AsyncMock(return_value=mock_df)

        with patch("medha.cli._app._build_medha", new=AsyncMock(return_value=mock_medha)):
            result = runner.invoke(app, ["export", "--format", "json", "--output", str(out_file)])

        assert result.exit_code == 0
        assert out_file.exists()
        assert "q1" in out_file.read_text()

    def test_export_missing_pandas_prints_actionable_error(self):
        with patch.dict(sys.modules, {"pandas": None}):
            result = runner.invoke(app, ["export"])

        assert result.exit_code != 0
        assert "pandas" in result.output


@pytest.mark.cli
class TestCliWarm:
    def test_warm_with_noop_embedder_prints_helpful_error(self, tmp_path):
        warm_file = tmp_path / "data.jsonl"
        warm_file.write_text('{"question": "q1", "query": "SELECT 1"}\n')

        with patch.dict(os.environ, {"MEDHA_EMBEDDER_TYPE": "_noop"}):
            result = runner.invoke(app, ["warm", str(warm_file)])

        assert result.exit_code != 0
        assert "embedder" in result.output.lower()

    def test_warm_with_real_embedder_succeeds(self, tmp_path):
        warm_file = tmp_path / "data.jsonl"
        warm_file.write_text('{"question": "q1", "query": "SELECT 1"}\n')

        mock_medha = _make_mock_medha()
        mock_medha.warm_from_file = AsyncMock(return_value=1)

        env_overrides = {"MEDHA_EMBEDDER_TYPE": "openai", "OPENAI_API_KEY": "test-key"}
        with patch("medha.cli._app._build_medha", new=AsyncMock(return_value=mock_medha)):
            with patch.dict(os.environ, env_overrides):
                result = runner.invoke(app, ["warm", str(warm_file)])

        assert result.exit_code == 0
        assert "1" in result.output


class TestNoOpEmbedder:
    def test_noop_embedder_dimension_property(self):
        e = _NoOpEmbedder()
        assert e.dimension == 384

    def test_noop_embedder_aembed_raises_runtime_error(self):
        e = _NoOpEmbedder()
        with pytest.raises(RuntimeError, match="real embedder"):
            asyncio.run(e.aembed("test"))

    def test_noop_embedder_aembed_batch_raises_runtime_error(self):
        e = _NoOpEmbedder()
        with pytest.raises(RuntimeError, match="real embedder"):
            asyncio.run(e.aembed_batch(["test"]))


class TestResolveEmbedder:
    def test_resolve_noop_returns_noop_embedder(self):
        s = Settings(embedder_type="_noop")
        result = _resolve_embedder(s)
        assert isinstance(result, _NoOpEmbedder)

    def test_resolve_unknown_raises_configuration_error(self):
        s = MagicMock(spec=Settings)
        s.embedder_type = "unknown"
        with pytest.raises(ConfigurationError, match="Unknown embedder_type"):
            _resolve_embedder(s)

    def test_resolve_fastembed_returns_fastembed_adapter(self):
        pytest.importorskip("fastembed")
        from medha.embeddings.fastembed_adapter import FastEmbedAdapter

        s = Settings(embedder_type="fastembed")
        result = _resolve_embedder(s)
        assert isinstance(result, FastEmbedAdapter)

    def test_resolve_openai_missing_api_key_raises_configuration_error(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        s = Settings(embedder_type="openai")
        with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
            _resolve_embedder(s)

    def test_resolve_cohere_missing_api_key_raises_configuration_error(self, monkeypatch):
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        s = Settings(embedder_type="cohere")
        with pytest.raises(ConfigurationError, match="COHERE_API_KEY"):
            _resolve_embedder(s)

    def test_resolve_gemini_missing_api_key_raises_configuration_error(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        s = Settings(embedder_type="gemini")
        with pytest.raises(ConfigurationError, match="GOOGLE_API_KEY"):
            _resolve_embedder(s)


class TestCliSettings:
    def test_settings_collection_default_is_default(self):
        s = Settings()
        assert s.collection == "default"

    def test_settings_collection_from_env_var(self, monkeypatch):
        monkeypatch.setenv("MEDHA_COLLECTION", "my_cache")
        s = Settings()
        assert s.collection == "my_cache"

    def test_settings_embedder_type_default_is_noop(self):
        s = Settings()
        assert s.embedder_type == "_noop"

    def test_settings_fastembed_model_default(self):
        s = Settings()
        assert s.fastembed_model == "BAAI/bge-small-en-v1.5"


@pytest.mark.cli
class TestCliFeedback:
    def test_feedback_correct_prints_recorded(self):
        mock_medha = _make_mock_medha()
        mock_medha.feedback = AsyncMock(return_value=True)

        with patch("medha.cli._app._build_medha", new=AsyncMock(return_value=mock_medha)):
            result = runner.invoke(app, ["feedback", "my question", "--correct"])

        assert result.exit_code == 0
        assert "Feedback recorded" in result.output

    def test_feedback_incorrect_prints_recorded(self):
        mock_medha = _make_mock_medha()
        mock_medha.feedback = AsyncMock(return_value=True)

        with patch("medha.cli._app._build_medha", new=AsyncMock(return_value=mock_medha)):
            result = runner.invoke(app, ["feedback", "my question", "--incorrect"])

        assert result.exit_code == 0
        assert "Feedback recorded" in result.output

    def test_feedback_not_found_prints_not_found(self):
        mock_medha = _make_mock_medha()
        mock_medha.feedback = AsyncMock(return_value=False)

        with patch("medha.cli._app._build_medha", new=AsyncMock(return_value=mock_medha)):
            result = runner.invoke(app, ["feedback", "my question", "--correct"])

        assert result.exit_code == 0
        assert "Entry not found" in result.output

    def test_feedback_requires_correct_or_incorrect_flag(self):
        result = runner.invoke(app, ["feedback", "my question"])
        assert result.exit_code != 0

    def test_feedback_correct_and_incorrect_mutually_exclusive(self):
        result = runner.invoke(app, ["feedback", "my question", "--correct", "--incorrect"])
        assert result.exit_code != 0

    def test_feedback_works_with_noop_embedder(self, monkeypatch):
        monkeypatch.delenv("MEDHA_EMBEDDER_TYPE", raising=False)
        mock_medha = _make_mock_medha()
        mock_medha.feedback = AsyncMock(return_value=True)

        with patch("medha.cli._app._build_medha", new=AsyncMock(return_value=mock_medha)):
            result = runner.invoke(app, ["feedback", "my question", "--correct"])

        assert result.exit_code == 0
        assert "Feedback recorded" in result.output
