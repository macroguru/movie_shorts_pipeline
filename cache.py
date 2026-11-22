"""
Кэш результатов AI-анализа на SQLite.

Хранит ответы провайдеров и транскрипты по ключу
sha256(canonical JSON (kind, content_hash, params)).
При выключенном кэше (config.cache_enabled=False) — полный no-op.
"""

import hashlib
import json
import sqlite3
import threading
from pathlib import Path


def make_cache_key(kind: str, content_hash: str, params: dict) -> str:
    """
    Ключ кэша: sha256 канонического JSON от (kind, content_hash, params).

    Args:
        kind: Тип записи (analyze, analyze_stream, transcript, ...)
        content_hash: Хэш исходного контента (например, SHA256 субтитров)
        params: Параметры запроса, влияющие на результат

    Returns:
        hex-строка SHA256
    """
    canonical = json.dumps(
        [kind, content_hash, params],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AnalysisCache:
    """
    Потокобезопасный SQLite-кэш анализа (temp/analysis_cache.db).

    Таблица: key (PK), value (TEXT), created_at. Ключ — из make_cache_key.
    """

    def __init__(self, config):
        """
        Args:
            config: Конфигурация проекта (используется cache_enabled и temp_dir)
        """
        self.config = config
        self.enabled = bool(getattr(config, "cache_enabled", False))
        self.db_path = None
        self._lock = threading.Lock()
        self._conn = None

        # temp_dir может отсутствовать (например, в тестовых фейках конфига):
        # тогда кэш работает как полный no-op
        temp_dir = getattr(config, "temp_dir", None)
        if not self.enabled or temp_dir is None:
            self.enabled = False
            return

        self.db_path = Path(temp_dir) / "analysis_cache.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False
        )
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS cache ("
                "key TEXT PRIMARY KEY, value TEXT NOT NULL, created_at TEXT NOT NULL)"
            )
            self._conn.commit()

    def get(self, kind: str, content_hash: str, params: dict) -> str | None:
        """
        Получить значение из кэша.

        Args:
            kind: Тип записи
            content_hash: Хэш исходного контента
            params: Параметры запроса

        Returns:
            Сохранённое значение или None (в том числе когда кэш выключен)
        """
        if not self.enabled or self._conn is None:
            return None

        key = make_cache_key(kind, content_hash, params)
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM cache WHERE key = ?", (key,)
            ).fetchone()

        return row[0] if row else None

    def put(self, kind: str, content_hash: str, params: dict, value: str) -> None:
        """
        Сохранить значение в кэше (no-op при выключенном кэше).

        Args:
            kind: Тип записи
            content_hash: Хэш исходного контента
            params: Параметры запроса
            value: Значение для сохранения (строка, обычно JSON)
        """
        if not self.enabled or self._conn is None:
            return

        key = make_cache_key(kind, content_hash, params)
        from datetime import datetime

        created_at = datetime.now().isoformat(timespec="seconds")
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, created_at) VALUES (?, ?, ?)",
                (key, value, created_at),
            )
            self._conn.commit()

    def clear(self) -> None:
        """Очистить кэш (no-op при выключенном кэше)."""
        if not self.enabled or self._conn is None:
            return

        with self._lock:
            self._conn.execute("DELETE FROM cache")
            self._conn.commit()

    def close(self) -> None:
        """Закрыть соединение с базой (безопасно вызывать повторно)."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
