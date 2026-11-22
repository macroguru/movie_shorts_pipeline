"""
Тесты кэша анализа и учёта использования AI-вызовов.

AnalysisCache строится на фейковом конфиге с tmp_path — реальная temp/
папка проекта не затрагивается.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cache import AnalysisCache, make_cache_key
from usage import UsageTracker, estimate_tokens


def make_cache_config(tmp_path: Path, enabled: bool = True) -> SimpleNamespace:
    """Конфиг для AnalysisCache: temp_dir в tmp_path, кэш включён/выключен."""
    return SimpleNamespace(temp_dir=tmp_path, cache_enabled=enabled)


@pytest.fixture
def cache(tmp_path):
    return AnalysisCache(make_cache_config(tmp_path / "temp"))


class TestMakeCacheKey:
    def test_stable(self):
        assert make_cache_key("analyze", "hash1", {"a": 1}) == make_cache_key(
            "analyze", "hash1", {"a": 1}
        )

    def test_param_order_insensitive(self):
        # Порядок ключей параметров не влияет на ключ
        assert make_cache_key("k", "h", {"a": 1, "b": 2}) == make_cache_key(
            "k", "h", {"b": 2, "a": 1}
        )

    def test_params_sensitive(self):
        assert make_cache_key("k", "h", {"a": 1}) != make_cache_key("k", "h", {"a": 2})

    def test_kind_and_hash_sensitive(self):
        assert make_cache_key("k1", "h", {}) != make_cache_key("k2", "h", {})
        assert make_cache_key("k", "h1", {}) != make_cache_key("k", "h2", {})


class TestAnalysisCache:
    def test_put_get_roundtrip(self, cache):
        cache.put("analyze", "abc", {"clips": 3}, '{"result": true}')
        assert cache.get("analyze", "abc", {"clips": 3}) == '{"result": true}'

    def test_params_mismatch_is_miss(self, cache):
        cache.put("analyze", "abc", {"clips": 3}, "value-A")
        assert cache.get("analyze", "abc", {"clips": 5}) is None

    def test_kind_mismatch_is_miss(self, cache):
        cache.put("analyze", "abc", {}, "value-A")
        assert cache.get("transcript", "abc", {}) is None

    def test_overwrite_same_key(self, cache):
        cache.put("k", "h", {}, "first")
        cache.put("k", "h", {}, "second")
        assert cache.get("k", "h", {}) == "second"

    def test_clear(self, cache):
        cache.put("k", "h", {}, "value")
        cache.clear()
        assert cache.get("k", "h", {}) is None

    def test_db_file_created(self, tmp_path):
        temp = tmp_path / "temp"
        AnalysisCache(make_cache_config(temp))
        assert (temp / "analysis_cache.db").exists()

    def test_disabled_mode_always_misses(self, tmp_path):
        cache_off = AnalysisCache(make_cache_config(tmp_path / "t2", enabled=False))

        cache_off.put("k", "h", {}, "значение")  # no-op, ничего не пишет

        assert cache_off.get("k", "h", {}) is None
        assert not (tmp_path / "t2" / "analysis_cache.db").exists()

    def test_close_is_idempotent(self, cache):
        cache.put("k", "h", {}, "v")
        cache.close()
        cache.close()  # повторный close безопасен
        # После close кэш отдаёт None, а не падает
        assert cache.get("k", "h", {}) is None


class TestEstimateTokens:
    def test_empty_is_zero(self):
        assert estimate_tokens("") == 0
        assert estimate_tokens(None) == 0

    def test_rough_quarter(self):
        assert estimate_tokens("x" * 40) == 10

    def test_short_text_min_one(self):
        assert estimate_tokens("абв") == 1


class TestUsageTracker:
    def test_record_writes_jsonl(self, tmp_path):
        path = tmp_path / "usage" / "usage.jsonl"
        tracker = UsageTracker(path)

        tracker.record(
            kind="text", provider="gemini", model="m1",
            prompt_tokens=10, completion_tokens=5,
        )
        tracker.record(
            kind="vision", provider="openai_compatible", model="m2",
            prompt_tokens=100, completion_tokens=50,
        )

        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

        first = json.loads(lines[0])
        assert first["kind"] == "text"
        assert first["provider"] == "gemini"
        assert first["prompt_tokens"] == 10
        assert first["completion_tokens"] == 5
        assert "ts" in first

    def test_session_summary_totals(self, tmp_path):
        tracker = UsageTracker(tmp_path / "usage.jsonl")

        tracker.record("text", "gemini", "m1", 10, 5)
        tracker.record("text", "gemini", "m1", 20, 15)
        tracker.record("vision", "gemini", "v1", 100, 50)

        summary = tracker.session_summary()

        assert summary["calls_total"] == 3
        assert summary["tokens_total"] == (10 + 5) + (20 + 15) + (100 + 50)
        assert summary["by_model"]["gemini/m1"] == {
            "calls": 2, "prompt_tokens": 30, "completion_tokens": 20,
        }
        assert len(summary["recent"]) == 3

    def test_recent_capped(self, tmp_path):
        tracker = UsageTracker(tmp_path / "usage.jsonl")
        for i in range(60):
            tracker.record("text", "gemini", "m", 1, 1)

        summary = tracker.session_summary()
        assert summary["calls_total"] == 60
        assert len(summary["recent"]) == 50  # RECENT_LIMIT

    def test_tracker_dir_created(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "usage.jsonl"
        tracker = UsageTracker(path)
        tracker.record("text", "gemini", "m", 1, 1)
        assert path.exists()
