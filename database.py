"""
Модуль работы с SQLite базой данных.
Хранит информацию о фильмах и опубликованных постах.
"""

import sqlite3
import random
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any


class Database:
    """Класс для работы с базой данных фильмов и постов."""

    def __init__(self, db_path: str = None):
        """
        Инициализация базы данных.

        Args:
            db_path: Путь к файлу базы данных (по умолчанию movies_posts.db в корне проекта)
        """
        if db_path is None:
            db_path = Path(__file__).parent / "movies_posts.db"

        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Получить соединение с базой данных."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Инициализация схемы базы данных."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS movies_posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                year INTEGER,
                genre TEXT,
                tg_msg_id INTEGER,
                clip_paths TEXT,
                poster_path TEXT,
                created_at TEXT NOT NULL,
                video_path TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_code ON movies_posts(code)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at ON movies_posts(created_at)
        """)

        conn.commit()
        conn.close()

    def get_unique_code(self) -> str:
        """
        Генерация уникального 4-значного кода.

        Returns:
            Строка с уникальным кодом от 1000 до 9999
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        max_attempts = 100
        for _ in range(max_attempts):
            code = str(random.randint(1000, 9999))

            cursor.execute("SELECT code FROM movies_posts WHERE code = ?", (code,))
            if cursor.fetchone() is None:
                conn.close()
                return code

        conn.close()
        raise ValueError("Не удалось сгенерировать уникальный код после 100 попыток")

    def is_code_exists(self, code: str) -> bool:
        """
        Проверка существования кода в базе.

        Args:
            code: Код для проверки

        Returns:
            True если код существует, False иначе
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT 1 FROM movies_posts WHERE code = ?", (code,))
        exists = cursor.fetchone() is not None

        conn.close()
        return exists

    def save_movie(
        self,
        code: str,
        title: str,
        year: int,
        genre: str = None,
        tg_msg_id: int = None,
        clip_paths: str = None,
        poster_path: str = None,
        video_path: str = None
    ) -> int:
        """
        Сохранение информации о фильме.

        Args:
            code: Уникальный код фильма
            title: Название фильма
            year: Год выпуска
            genre: Жанр фильма
            tg_msg_id: ID сообщения в Telegram
            clip_paths: Пути к клипам (JSON строка)
            poster_path: Путь к постеру
            video_path: Путь к исходному видео

        Returns:
            ID сохраненной записи
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        created_at = datetime.now().isoformat()

        cursor.execute("""
            INSERT INTO movies_posts
            (code, title, year, genre, tg_msg_id, clip_paths, poster_path, created_at, video_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (code, title, year, genre, tg_msg_id, clip_paths, poster_path, created_at, video_path))

        record_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return record_id

    def get_by_code(self, code: str) -> Optional[Dict[str, Any]]:
        """
        Получение записи по коду фильма.

        Args:
            code: Код фильма

        Returns:
            Словарь с данными записи или None если не найдено
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM movies_posts WHERE code = ?", (code,))
        row = cursor.fetchone()

        conn.close()

        if row is None:
            return None

        return dict(row)

    def get_all(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Получение всех записей.

        Args:
            limit: Максимальное количество записей

        Returns:
            Список словарей с данными записей
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM movies_posts
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def update_telegram_msg_id(self, code: str, tg_msg_id: int):
        """
        Обновление ID сообщения в Telegram.

        Args:
            code: Код фильма
            tg_msg_id: ID сообщения в Telegram
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE movies_posts
            SET tg_msg_id = ?
            WHERE code = ?
        """, (tg_msg_id, code))

        conn.commit()
        conn.close()

    def update_clip_paths(self, code: str, clip_paths: str):
        """
        Обновление путей к клипам.

        Args:
            code: Код фильма
            clip_paths: Пути к клипам (JSON строка)
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE movies_posts
            SET clip_paths = ?
            WHERE code = ?
        """, (clip_paths, code))

        conn.commit()
        conn.close()

    def delete_by_code(self, code: str) -> bool:
        """
        Удаление записи по коду.

        Args:
            code: Код фильма для удаления

        Returns:
            True если запись удалена, False если не найдена
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM movies_posts WHERE code = ?", (code,))
        deleted = cursor.rowcount > 0

        conn.commit()
        conn.close()

        return deleted

    def get_stats(self) -> Dict[str, Any]:
        """
        Получение статистики по базе данных.

        Returns:
            Словарь со статистикой
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) as total FROM movies_posts")
        total = cursor.fetchone()["total"]

        cursor.execute("SELECT COUNT(*) as published FROM movies_posts WHERE tg_msg_id IS NOT NULL")
        published = cursor.fetchone()["published"]

        conn.close()

        return {
            "total_movies": total,
            "published": published,
            "not_published": total - published
        }
