"""
Модуль учёта использования AI-провайдеров: вызовы, токены, сводка за сессию.

Все вызовы провайдеров записываются в JSONL-файл temp/usage.jsonl —
по одной строке на вызов. Сводка за сессию собирается из памяти.
"""

import json
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Сколько последних вызовов хранится в сводке сессии
RECENT_LIMIT = 50


def estimate_tokens(text: str) -> int:
    """
    Грубая оценка количества токенов по длине текста (примерно 4 символа на токен).

    Args:
        text: Текст запроса или ответа

    Returns:
        Оценка числа токенов (не меньше 0)
    """
    if not text:
        return 0
    return max(len(text) // 4, 1)


class UsageTracker:
    """
    Потокобезопасный учёт вызовов AI-провайдеров.

    Каждая запись дописывается в temp/usage.jsonl, сводка за сессию
    собирается в памяти (вызовы по провайдеру/модели, суммарные токены,
    последние вызовы).
    """

    def __init__(self, path: Path = None):
        """
        Args:
            path: Путь к JSONL-файлу учёта (по умолчанию temp/usage.jsonl)
        """
        self.path = Path(path) if path is not None else Path("temp") / "usage.jsonl"
        self._lock = threading.Lock()
        # (provider, model) -> [calls, prompt_tokens, completion_tokens]
        self._totals = defaultdict(lambda: [0, 0, 0])
        self._recent = []

    def record(
        self,
        kind: str,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """
        Записать один вызов провайдера.

        Args:
            kind: Тип вызова (text | vision) — для сводки
            provider: Имя провайдера (gemini | openai_compatible)
            model: Название модели
            prompt_tokens: Токены запроса (или оценка)
            completion_tokens: Токены ответа (или оценка)
        """
        entry = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "kind": kind,
            "provider": provider,
            "model": model,
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
        }

        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except OSError:
                # Учёт не должен ломать основной пайплайн
                pass

            totals = self._totals[(provider, model)]
            totals[0] += 1
            totals[1] += entry["prompt_tokens"]
            totals[2] += entry["completion_tokens"]

            self._recent.append(entry)
            if len(self._recent) > RECENT_LIMIT:
                self._recent = self._recent[-RECENT_LIMIT:]

    def session_summary(self) -> dict:
        """
        Сводка использования за сессию.

        Returns:
            Словарь: calls_total, по (provider, model) — вызовы и токены,
            и последние вызовы (не более RECENT_LIMIT)
        """
        with self._lock:
            by_model = {}
            calls_total = 0
            tokens_total = 0

            for (provider, model), (calls, prompt, completion) in self._totals.items():
                by_model[f"{provider}/{model}"] = {
                    "calls": calls,
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                }
                calls_total += calls
                tokens_total += prompt + completion

            return {
                "calls_total": calls_total,
                "tokens_total": tokens_total,
                "by_model": by_model,
                "recent": list(self._recent),
            }
