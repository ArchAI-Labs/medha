"""Unit tests for Medha security and input-validation features."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from medha.config import Settings
from medha.types import SearchStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_medha(settings: Settings | None = None, embedder=None, backend=None):
    from medha.core import Medha

    if embedder is None:
        embedder = MagicMock()
        embedder.dimension = 4
        embedder.aembed = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4])
        embedder.aembed_batch = AsyncMock(return_value=[[0.1, 0.2, 0.3, 0.4]])

    if backend is None:
        from medha.backends.memory import InMemoryBackend
        backend = InMemoryBackend()

    s = settings or Settings(backend_type="memory")
    m = Medha(collection_name="sec_test", embedder=embedder, backend=backend, settings=s)
    return m


# ---------------------------------------------------------------------------
# max_question_length
# ---------------------------------------------------------------------------


class TestMaxQuestionLength:
    async def test_search_rejects_oversized_question(self):
        """Questions longer than max_question_length must return ERROR strategy."""
        settings = Settings(backend_type="memory", max_question_length=100)
        m = _make_medha(settings=settings)
        await m.start()
        try:
            oversized = "x" * 101
            hit = await m.search(oversized)
            assert hit.strategy == SearchStrategy.ERROR
        finally:
            await m.close()

    async def test_search_accepts_question_at_limit(self):
        """Questions exactly at max_question_length must not be rejected for length."""
        settings = Settings(backend_type="memory", max_question_length=100)
        m = _make_medha(settings=settings)
        await m.start()
        try:
            exact = "x" * 100
            hit = await m.search(exact)
            assert hit.strategy != SearchStrategy.ERROR
        finally:
            await m.close()

    async def test_store_rejects_oversized_question(self):
        """store() with oversized question must raise ValueError."""
        settings = Settings(backend_type="memory", max_question_length=64)
        m = _make_medha(settings=settings)
        await m.start()
        try:
            oversized = "q" * 65
            with pytest.raises(ValueError, match="max_question_length"):
                await m.store(oversized, "SELECT 1")
        finally:
            await m.close()

    async def test_store_accepts_question_at_limit(self):
        """store() with question exactly at limit must succeed."""
        settings = Settings(backend_type="memory", max_question_length=64)
        m = _make_medha(settings=settings)
        await m.start()
        try:
            exact = "q" * 64
            ok = await m.store(exact, "SELECT 1")
            assert ok is True
        finally:
            await m.close()

    async def test_default_limit_allows_normal_questions(self):
        """Default max_question_length (8192) must not reject ordinary questions."""
        m = _make_medha()
        await m.start()
        try:
            hit = await m.search("How many users are there")
            assert hit.strategy != SearchStrategy.ERROR
        finally:
            await m.close()

    async def test_very_long_question_with_default_limit(self):
        """A 10 000-char question exceeds the default 8192 limit → ERROR."""
        settings = Settings(backend_type="memory")
        m = _make_medha(settings=settings)
        await m.start()
        try:
            long_q = "a" * 10_000
            hit = await m.search(long_q)
            assert hit.strategy == SearchStrategy.ERROR
        finally:
            await m.close()


# ---------------------------------------------------------------------------
# allowed_file_dir
# ---------------------------------------------------------------------------


class TestAllowedFileDir:
    async def test_warm_from_file_allowed_path(self, tmp_path):
        """warm_from_file must succeed when the file is inside allowed_file_dir."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        data_file = allowed_dir / "data.jsonl"
        entry = {"question": "How many users?", "generated_query": "SELECT COUNT(*) FROM users"}
        data_file.write_text(json.dumps(entry) + "\n", encoding="utf-8")

        settings = Settings(backend_type="memory", allowed_file_dir=str(allowed_dir))
        m = _make_medha(settings=settings)
        await m.start()
        try:
            count = await m.warm_from_file(str(data_file))
            assert count == 1
        finally:
            await m.close()

    async def test_warm_from_file_outside_allowed_dir_raises(self, tmp_path):
        """warm_from_file must raise when the file is outside allowed_file_dir."""
        from medha.exceptions import MedhaError

        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        outside_file = tmp_path / "outside.jsonl"
        entry = {"question": "q", "generated_query": "SELECT 1"}
        outside_file.write_text(json.dumps(entry) + "\n", encoding="utf-8")

        settings = Settings(backend_type="memory", allowed_file_dir=str(allowed_dir))
        m = _make_medha(settings=settings)
        await m.start()
        try:
            with pytest.raises(MedhaError):
                await m.warm_from_file(str(outside_file))
        finally:
            await m.close()

    async def test_warm_from_file_no_restriction(self, tmp_path):
        """When allowed_file_dir is None (default), any path is accepted."""
        data_file = tmp_path / "data.jsonl"
        entry = {"question": "q", "generated_query": "SELECT 1"}
        data_file.write_text(json.dumps(entry) + "\n", encoding="utf-8")

        settings = Settings(backend_type="memory", allowed_file_dir=None)
        m = _make_medha(settings=settings)
        await m.start()
        try:
            count = await m.warm_from_file(str(data_file))
            assert count == 1
        finally:
            await m.close()

    async def test_load_templates_from_file_outside_allowed_dir_raises(self, tmp_path):
        """load_templates_from_file must raise when file is outside allowed_file_dir."""
        from medha.exceptions import MedhaError

        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        template = {
            "intent": "count",
            "template_text": "How many {entity}",
            "query_template": "SELECT COUNT(*) FROM {entity}",
            "parameters": ["entity"],
        }
        outside_file = tmp_path / "templates.json"
        outside_file.write_text(json.dumps([template]), encoding="utf-8")

        settings = Settings(backend_type="memory", allowed_file_dir=str(allowed_dir))
        m = _make_medha(settings=settings)
        await m.start()
        try:
            with pytest.raises(MedhaError):
                await m.load_templates_from_file(str(outside_file))
        finally:
            await m.close()


# ---------------------------------------------------------------------------
# max_file_size_mb
# ---------------------------------------------------------------------------


class TestMaxFileSizeMb:
    async def test_warm_from_file_rejects_oversized_file(self, tmp_path):
        """warm_from_file must raise when the file exceeds max_file_size_mb."""
        from medha.exceptions import MedhaError

        data_file = tmp_path / "big.jsonl"
        # Write ~2MB of data (max=1MB)
        entry = {"question": "q", "generated_query": "SELECT 1"}
        line = json.dumps(entry)
        data_file.write_text((line + "\n") * 50_000, encoding="utf-8")

        settings = Settings(backend_type="memory", max_file_size_mb=1)
        m = _make_medha(settings=settings)
        await m.start()
        try:
            with pytest.raises(MedhaError):
                await m.warm_from_file(str(data_file))
        finally:
            await m.close()

    async def test_warm_from_file_accepts_file_within_size_limit(self, tmp_path):
        """warm_from_file must succeed when the file is within max_file_size_mb."""
        data_file = tmp_path / "small.jsonl"
        entry = {"question": "How many users?", "generated_query": "SELECT COUNT(*) FROM users"}
        data_file.write_text(json.dumps(entry) + "\n", encoding="utf-8")

        settings = Settings(backend_type="memory", max_file_size_mb=10)
        m = _make_medha(settings=settings)
        await m.start()
        try:
            count = await m.warm_from_file(str(data_file))
            assert count == 1
        finally:
            await m.close()


# ---------------------------------------------------------------------------
# Path traversal prevention
# ---------------------------------------------------------------------------


class TestPathTraversalPrevention:
    async def test_path_traversal_via_dotdot_is_blocked(self, tmp_path):
        """A path with ../ that escapes allowed_file_dir must be rejected."""
        from medha.exceptions import MedhaError

        allowed_dir = tmp_path / "safe"
        allowed_dir.mkdir()
        # Place a file one level above (outside allowed_dir)
        outside_file = tmp_path / "secret.jsonl"
        outside_file.write_text('{"question": "q", "generated_query": "SELECT 1"}\n')

        settings = Settings(backend_type="memory", allowed_file_dir=str(allowed_dir))
        m = _make_medha(settings=settings)
        await m.start()
        try:
            traversal_path = str(allowed_dir / ".." / "secret.jsonl")
            with pytest.raises(MedhaError):
                await m.warm_from_file(traversal_path)
        finally:
            await m.close()
