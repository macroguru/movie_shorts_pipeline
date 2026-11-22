"""
Общие фикстуры тестов Movie Shorts Pipeline.

Тесты не должны зависеть от сети и реального .env: провайдеры подменяются
фейками, а конфиги собираются из SimpleNamespace.
"""

import sys
from pathlib import Path

import pytest

# Корень проекта — чтобы импорты config/core/... работали из любого запуска pytest
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Регистрируем маркер, чтобы pytest не ругался на unknown mark
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: интеграционные тесты с ffmpeg/ffprobe")


class FakeTextProvider:
    """
    Фейковый TextProvider: возвращает заготовленные ответы по очереди.

    Args:
        responses: Список ответов complete(); после исчерпания — последний

    Attributes:
        calls: Список (prompt, system) всех вызовов
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, prompt: str, system=None) -> str:
        self.calls.append((prompt, system))
        if not self.responses:
            raise AssertionError("FakeTextProvider: ответов больше нет")
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]

    @property
    def call_count(self) -> int:
        return len(self.calls)


@pytest.fixture
def fake_text_provider():
    """Фабрика фейковых текстовых провайдеров: fake_text_provider(["ответ"])."""
    return FakeTextProvider


class FakeUsageTracker:
    """Тихий учёт использования в памяти — без записи на диск."""

    def __init__(self):
        self.entries = []

    def record(self, kind, provider, model, prompt_tokens, completion_tokens):
        self.entries.append({
            "kind": kind,
            "provider": provider,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        })

    def session_summary(self):
        return {"calls_total": len(self.entries)}


@pytest.fixture
def fake_usage_tracker():
    """Фейковый UsageTracker — чтобы тесты не писали temp/usage.jsonl."""
    return FakeUsageTracker()
