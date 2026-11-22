"""
Абстракция AI-провайдеров: текст и зрение через Gemini или OpenAI-совместимый API.

Фабрика create_text_provider / create_vision_provider выбирает реализацию
по config. Тихого переключения между провайдерами нет: любая ошибка
оборачивается в ProviderError с исходным сообщением. Каждый вызов
записывается в UsageTracker (токены из ответа API или оценка по символам/4).
"""

import base64
import json
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import httpx

from config import Config
from usage import UsageTracker, estimate_tokens

# Общие таймауты HTTP-вызовов
HTTP_TIMEOUT = 120.0


class ProviderError(Exception):
    """Ошибка провайдера — исходное сообщение API сохраняется как есть."""
    pass


@runtime_checkable
class TextProvider(Protocol):
    """Провайдер текстового завершения. Возвращает сырой текст; JSON разбирает вызывающий код."""

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        ...


@runtime_checkable
class VisionProvider(Protocol):
    """Провайдер анализа изображений. Возвращает сырой текст."""

    def analyze_images(self, prompt: str, image_paths: list) -> str:
        ...


class _BaseProvider:
    """Общее: модель, трекер учёта. Ключ API задаётся в наследниках."""

    def __init__(self, model: str, tracker: UsageTracker):
        self.model = model
        self.usage = tracker

    def _record(self, kind: str, provider: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.usage.record(
            kind=kind,
            provider=provider,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


# ---------- Gemini ----------

class GeminiTextProvider(_BaseProvider):
    """Текст через google.genai (см. прежний паттерн в ai_engine.py)."""

    def __init__(self, api_key: str, model: str, tracker: UsageTracker):
        super().__init__(model, tracker)
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        """Ленивая инициализация клиента genai."""
        if self._client is None:
            try:
                from google import genai
            except ImportError as e:
                raise ProviderError(f"google-genai не установлен: {e}")
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        """Один вызов генерации текста."""
        client = self._get_client()

        from google.genai import types as genai_types

        config = genai_types.GenerateContentConfig(
            system_instruction=system,
        ) if system else None

        try:
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
        except Exception as e:
            raise ProviderError(f"Gemini API error: {e}")

        text = (response.text or "").strip()
        if not text:
            raise ProviderError("Gemini вернул пустой ответ")

        # Токены из usage_metadata, при отсутствии — оценка по символам
        usage_meta = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage_meta, "prompt_token_count", None)
        completion_tokens = getattr(usage_meta, "candidates_token_count", None)
        if prompt_tokens is None:
            prompt_tokens = estimate_tokens(prompt)
        if completion_tokens is None:
            completion_tokens = estimate_tokens(text)

        self._record("text", "gemini", prompt_tokens, completion_tokens)
        return text


class GeminiVisionProvider(GeminiTextProvider):
    """Зрение через google.genai: изображения как части содержимого."""

    def analyze_images(self, prompt: str, image_paths: list) -> str:
        """Один вызов анализа кадров. Изображения читаются с диска."""
        client = self._get_client()

        from google.genai import types as genai_types

        parts = []
        for path in image_paths:
            data = Path(path).read_bytes()
            mime = "image/jpeg" if Path(path).suffix.lower() in (".jpg", ".jpeg") else "image/png"
            parts.append(genai_types.Part.from_bytes(data=data, mime_type=mime))
        parts.append(prompt)

        try:
            response = client.models.generate_content(
                model=self.model,
                contents=parts,
            )
        except Exception as e:
            raise ProviderError(f"Gemini API error: {e}")

        text = (response.text or "").strip()
        if not text:
            raise ProviderError("Gemini вернул пустой ответ")

        usage_meta = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage_meta, "prompt_token_count", None)
        completion_tokens = getattr(usage_meta, "candidates_token_count", None)
        if prompt_tokens is None:
            prompt_tokens = estimate_tokens(prompt) + len(image_paths) * 258
        if completion_tokens is None:
            completion_tokens = estimate_tokens(text)

        self._record("vision", "gemini", prompt_tokens, completion_tokens)
        return text


# ---------- OpenAI-совместимый (chat/completions) ----------

class OpenAICompatibleTextProvider(_BaseProvider):
    """Текст через {base_url}/chat/completions (OpenAI chat format) на httpx."""

    def __init__(self, api_key: str, model: str, base_url: str, tracker: UsageTracker):
        super().__init__(model, tracker)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _chat(self, messages: list, json_mode: bool = False) -> tuple:
        """
        Один POST в chat/completions.

        Returns:
            (текст ответа, usage-словарь от API или None)
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "messages": messages}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=HTTP_TIMEOUT,
            )
        except httpx.HTTPError as e:
            raise ProviderError(f"HTTP error: {e}")

        if response.status_code >= 400:
            raise ProviderError(f"HTTP {response.status_code}: {response.text[:500]}")

        try:
            data = response.json()
        except ValueError as e:
            raise ProviderError(f"Невалидный JSON в ответе API: {e}")

        choices = data.get("choices") or []
        if not choices:
            raise ProviderError("API вернул пустой список choices")

        text = (choices[0].get("message") or {}).get("content") or ""
        text = text.strip()
        if not text:
            raise ProviderError("API вернул пустой ответ")

        return text, data.get("usage")

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        """Один вызов генерации текста; JSON-режим не включается принудительно."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        text, usage = self._chat(messages)

        prompt_tokens = (usage or {}).get("prompt_tokens")
        completion_tokens = (usage or {}).get("completion_tokens")
        if prompt_tokens is None:
            prompt_tokens = estimate_tokens(prompt)
        if completion_tokens is None:
            completion_tokens = estimate_tokens(text)

        self._record("text", "openai_compatible", prompt_tokens, completion_tokens)
        return text


class OpenAICompatibleVisionProvider(OpenAICompatibleTextProvider):
    """Зрение через chat/completions: изображения как base64 data-URL в контенте."""

    def analyze_images(self, prompt: str, image_paths: list) -> str:
        """Один вызов анализа кадров (vision-сообщение с image_url частями)."""
        content = [{"type": "text", "text": prompt}]

        for path in image_paths:
            data = Path(path).read_bytes()
            b64 = base64.b64encode(data).decode("ascii")
            mime = "image/jpeg" if Path(path).suffix.lower() in (".jpg", ".jpeg") else "image/png"
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })

        messages = [{"role": "user", "content": content}]
        text, usage = self._chat(messages)

        prompt_tokens = (usage or {}).get("prompt_tokens")
        completion_tokens = (usage or {}).get("completion_tokens")
        if prompt_tokens is None:
            prompt_tokens = estimate_tokens(prompt) + len(image_paths) * 258
        if completion_tokens is None:
            completion_tokens = estimate_tokens(text)

        self._record("vision", "openai_compatible", prompt_tokens, completion_tokens)
        return text


# ---------- Фабрики ----------

def create_text_provider(config: Config, tracker: UsageTracker = None) -> TextProvider:
    """
    Создать текстовый провайдер по config.ai_text_provider.

    Args:
        config: Конфигурация проекта
        tracker: Учёт использования (по умолчанию общий UsageTracker)

    Returns:
        Реализация TextProvider

    Raises:
        ProviderError: неизвестный провайдер, нет ключа или base_url
    """
    tracker = tracker or UsageTracker()

    if config.ai_text_provider == "gemini":
        if not config.ai_text_api_key:
            raise ProviderError("Не задан API-ключ текстового провайдера (AI_TEXT_API_KEY / GEMINI_API_KEY)")
        return GeminiTextProvider(config.ai_text_api_key, config.ai_text_model, tracker)

    if config.ai_text_provider == "openai_compatible":
        if not config.ai_text_base_url:
            raise ProviderError("AI_TEXT_BASE_URL не задан — нужен адрес OpenAI-совместимого API")
        if not config.ai_text_api_key:
            raise ProviderError("Не задан API-ключ текстового провайдера (AI_TEXT_API_KEY)")
        return OpenAICompatibleTextProvider(
            config.ai_text_api_key, config.ai_text_model, config.ai_text_base_url, tracker
        )

    raise ProviderError(f"Неизвестный текстовый провайдер: {config.ai_text_provider}")


def create_vision_provider(config: Config, tracker: UsageTracker = None) -> Optional[VisionProvider]:
    """
    Создать vision-провайдер по config.ai_vision_provider.

    Args:
        config: Конфигурация проекта
        tracker: Учёт использования (по умолчанию общий UsageTracker)

    Returns:
        Реализация VisionProvider или None при "none"

    Raises:
        ProviderError: неизвестный провайдер, нет ключа или base_url
    """
    if config.ai_vision_provider == "none":
        return None

    tracker = tracker or UsageTracker()

    if config.ai_vision_provider == "gemini":
        if not config.ai_vision_api_key:
            raise ProviderError("Не задан API-ключ vision-провайдера (AI_VISION_API_KEY / GEMINI_API_KEY)")
        return GeminiVisionProvider(config.ai_vision_api_key, config.ai_vision_model, tracker)

    if config.ai_vision_provider == "openai_compatible":
        if not config.ai_vision_base_url:
            raise ProviderError("AI_VISION_BASE_URL не задан — нужен адрес OpenAI-совместимого API")
        if not config.ai_vision_api_key:
            raise ProviderError("Не задан API-ключ vision-провайдера (AI_VISION_API_KEY)")
        return OpenAICompatibleVisionProvider(
            config.ai_vision_api_key, config.ai_vision_model, config.ai_vision_base_url, tracker
        )

    raise ProviderError(f"Неизвестный vision-провайдер: {config.ai_vision_provider}")
