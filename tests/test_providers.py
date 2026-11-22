"""
Тесты фабрики AI-провайдеров: выбор реализации по конфигу без сети.

Реальные HTTP-вызовы подменяются monkeypatch-ом httpx.post.
"""

import json
from types import SimpleNamespace

import httpx
import pytest

import providers
from providers import (
    GeminiTextProvider,
    GeminiVisionProvider,
    OpenAICompatibleTextProvider,
    OpenAICompatibleVisionProvider,
    ProviderError,
    TextProvider,
    VisionProvider,
    create_text_provider,
    create_vision_provider,
)


def make_config(**overrides) -> SimpleNamespace:
    """Минимальный конфиг фабрики провайдеров (как Config, но без .env)."""
    base = dict(
        ai_text_provider="gemini",
        ai_text_model="test-text-model",
        ai_text_api_key="test-key",
        ai_text_base_url=None,
        ai_vision_provider="gemini",
        ai_vision_model="test-vision-model",
        ai_vision_api_key="test-key",
        ai_vision_base_url=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class FakeResponse:
    """Минимальный ответ httpx для подмены."""

    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    @property
    def text(self):
        return json.dumps(self._payload)


COMPLETION_PAYLOAD = {
    "choices": [{"message": {"content": "  Готовый ответ модели.  "}}],
    "usage": {"prompt_tokens": 12, "completion_tokens": 7},
}


class TestCreateTextProvider:
    def test_gemini_factory(self, fake_usage_tracker):
        provider = create_text_provider(make_config(), tracker=fake_usage_tracker)

        assert isinstance(provider, GeminiTextProvider)
        assert isinstance(provider, TextProvider)  # Protocol выполняется
        assert provider.api_key == "test-key"
        assert provider.model == "test-text-model"

    def test_gemini_requires_key(self, fake_usage_tracker):
        config = make_config(ai_text_api_key="")

        with pytest.raises(ProviderError, match="API-ключ"):
            create_text_provider(config, tracker=fake_usage_tracker)

    def test_openai_compatible_factory(self, fake_usage_tracker):
        config = make_config(
            ai_text_provider="openai_compatible",
            ai_text_base_url="http://localhost:8000/v1",
        )
        provider = create_text_provider(config, tracker=fake_usage_tracker)

        assert isinstance(provider, OpenAICompatibleTextProvider)
        assert provider.base_url == "http://localhost:8000/v1"

    def test_openai_compatible_requires_base_url(self, fake_usage_tracker):
        config = make_config(ai_text_provider="openai_compatible")

        with pytest.raises(ProviderError, match="BASE_URL"):
            create_text_provider(config, tracker=fake_usage_tracker)

    def test_unknown_provider_raises(self, fake_usage_tracker):
        config = make_config(ai_text_provider="claude_desktop")

        with pytest.raises(ProviderError, match="Неизвестный"):
            create_text_provider(config, tracker=fake_usage_tracker)


class TestOpenAICompatibleComplete:
    def _provider(self, tracker):
        config = make_config(
            ai_text_provider="openai_compatible",
            ai_text_base_url="http://localhost:8000/v1/",
        )
        return create_text_provider(config, tracker=tracker)

    def test_complete_returns_text(self, monkeypatch, fake_usage_tracker):
        provider = self._provider(fake_usage_tracker)
        calls = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            calls["url"] = url
            calls["headers"] = headers
            calls["json"] = json
            return FakeResponse(COMPLETION_PAYLOAD)

        monkeypatch.setattr(providers.httpx, "post", fake_post)

        text = provider.complete("Вопрос про фильм", system="Ты продюсер")

        # Ответ обрезается по краям
        assert text == "Готовый ответ модели."
        assert calls["url"] == "http://localhost:8000/v1/chat/completions"
        assert calls["headers"]["Authorization"] == "Bearer test-key"
        assert calls["json"]["model"] == "test-text-model"
        assert calls["json"]["messages"][0]["role"] == "system"
        assert calls["json"]["messages"][-1]["content"] == "Вопрос про фильм"

    def test_complete_records_usage(self, monkeypatch, fake_usage_tracker):
        provider = self._provider(fake_usage_tracker)
        monkeypatch.setattr(
            providers.httpx, "post",
            lambda *a, **kw: FakeResponse(COMPLETION_PAYLOAD),
        )

        provider.complete("Привет")

        assert fake_usage_tracker.session_summary()["calls_total"] == 1
        entry = fake_usage_tracker.entries[0]
        assert entry["provider"] == "openai_compatible"
        assert entry["prompt_tokens"] == 12
        assert entry["completion_tokens"] == 7

    def test_http_error_no_silent_fallback(self, monkeypatch, fake_usage_tracker):
        provider = self._provider(fake_usage_tracker)

        def broken_post(*args, **kwargs):
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(providers.httpx, "post", broken_post)

        with pytest.raises(ProviderError, match="connection refused"):
            provider.complete("Привет")
        # Фоллбэка на Gemini не произошло: записей в учёте нет
        assert fake_usage_tracker.session_summary()["calls_total"] == 0

    def test_http_500_raises(self, monkeypatch, fake_usage_tracker):
        provider = self._provider(fake_usage_tracker)
        monkeypatch.setattr(
            providers.httpx, "post",
            lambda *a, **kw: FakeResponse({"error": "boom"}, status_code=500),
        )

        with pytest.raises(ProviderError, match="HTTP 500"):
            provider.complete("Привет")

    def test_empty_choices_raises(self, monkeypatch, fake_usage_tracker):
        provider = self._provider(fake_usage_tracker)
        monkeypatch.setattr(
            providers.httpx, "post",
            lambda *a, **kw: FakeResponse({"choices": []}),
        )

        with pytest.raises(ProviderError):
            provider.complete("Привет")


class TestCreateVisionProvider:
    def test_none_disables_vision(self, fake_usage_tracker):
        config = make_config(ai_vision_provider="none")
        assert create_vision_provider(config) is None

    def test_gemini_factory(self, fake_usage_tracker):
        provider = create_vision_provider(
            make_config(), tracker=fake_usage_tracker
        )
        assert isinstance(provider, GeminiVisionProvider)
        assert isinstance(provider, VisionProvider)  # Protocol выполняется

    def test_openai_compatible_factory(self, fake_usage_tracker):
        config = make_config(
            ai_vision_provider="openai_compatible",
            ai_vision_base_url="http://localhost:8000/v1",
        )
        provider = create_vision_provider(config, tracker=fake_usage_tracker)
        assert isinstance(provider, OpenAICompatibleVisionProvider)

    def test_openai_compatible_requires_base_url(self, fake_usage_tracker):
        config = make_config(ai_vision_provider="openai_compatible")
        with pytest.raises(ProviderError, match="BASE_URL"):
            create_vision_provider(config, tracker=fake_usage_tracker)

    def test_unknown_provider_raises(self, fake_usage_tracker):
        config = make_config(ai_vision_provider="midjourney")
        with pytest.raises(ProviderError, match="Неизвестный"):
            create_vision_provider(config, tracker=fake_usage_tracker)
