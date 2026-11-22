"""
Publisher модуль для публикации постов в Telegram-канал.
Отправляет карточки фильмов с постерами и текстом.
"""

import httpx
import socket
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Dict, Any

from config import Config


@contextmanager
def ipv4_only():
    """
    Временный запрет IPv6 для DNS-резолва.

    У части провайдеров IPv6-маршрут до api.telegram.org не работает:
    соединение виснет до таймаута (~21 c) и только потом откатывается
    на IPv4. Принудительный IPv4 убирает эту задержку.
    """
    original = socket.getaddrinfo

    def resolve_ipv4(*args, **kwargs):
        return [entry for entry in original(*args, **kwargs) if entry[0] == socket.AF_INET]

    socket.getaddrinfo = resolve_ipv4
    try:
        yield
    finally:
        socket.getaddrinfo = original


class PublisherError(Exception):
    """Исключение для ошибок публикации."""
    pass


class Publisher:
    """Класс для публикации в Telegram."""

    def __init__(self, config: Config):
        """
        Инициализация паблишера.

        Args:
            config: Конфигурация проекта
        """
        self.config = config
        self.bot_token = config.telegram_bot_token
        self.channel_id = config.telegram_channel_id
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    def publish_to_telegram(
        self,
        code: str,
        title: str,
        year: int,
        post_data: Dict[str, str],
        poster_path: str,
        caption: Optional[str] = None
    ) -> int:
        """
        Публикация поста в Telegram-канал.

        Args:
            code: Код фильма
            title: Название фильма
            year: Год выпуска
            post_data: Данные поста (genre, synopsis, hook_fact)
            poster_path: Путь к изображению постера
            caption: Готовый текст поста (для стримов); по умолчанию собирается для фильма

        Returns:
            message_id опубликованного сообщения
        """
        poster_path = Path(poster_path)

        if not poster_path.exists():
            raise PublisherError(f"Постер не найден: {poster_path}")

        # Готовая подпись (стримы) или стандартный текст для фильма
        caption = caption or self._build_caption(code, title, year, post_data)

        # Отправляем фото с текстом
        message_id = self._send_photo(poster_path, caption)

        print(f"Пост опубликован в Telegram. Message ID: {message_id}")

        return message_id

    def _build_caption(
        self,
        code: str,
        title: str,
        year: int,
        post_data: Dict[str, str]
    ) -> str:
        """
        Построение текста поста для Telegram.

        Args:
            code: Код фильма
            title: Название фильма
            year: Год выпуска
            post_data: Данные поста

        Returns:
            Форматированный текст поста
        """
        genre = post_data.get("genre", "Не указан")
        synopsis = post_data.get("synopsis", "")
        hook_fact = post_data.get("hook_fact", "")

        # Формируем текст с HTML-разметкой для Telegram
        caption = (
            f"🎬 <b>{title} ({year})</b>\n"
            f"🎭 <b>Жанр:</b> {genre}\n\n"
            f"{synopsis}\n\n"
            f"💡 <i>{hook_fact}</i>\n\n"
            f"🔍 <b>Код фильма:</b> <code>{code}</code>\n\n"
            f"#код{code} #кино #фильмы"
        )

        return caption

    def _send_photo(self, photo_path: Path, caption: str) -> int:
        """
        Отправка фото в Telegram.

        Args:
            photo_path: Путь к изображению
            caption: Текст подписи

        Returns:
            message_id отправленного сообщения
        """
        url = f"{self.base_url}/sendPhoto"

        # Определяем chat_id
        chat_id = self._normalize_channel_id()

        try:
            with ipv4_only(), httpx.Client(timeout=60.0) as client:
                with open(photo_path, "rb") as photo_file:
                    files = {
                        "photo": (photo_path.name, photo_file, "image/jpeg")
                    }
                    data = {
                        "chat_id": chat_id,
                        "caption": caption,
                        "parse_mode": "HTML"
                    }

                    response = client.post(url, files=files, data=data)
                    response.raise_for_status()

                    result = response.json()

                    if not result.get("ok"):
                        raise PublisherError(
                            f"Telegram API error: {result.get('description', 'Unknown error')}"
                        )

                    return result["result"]["message_id"]

        except httpx.HTTPError as e:
            raise PublisherError(f"Ошибка HTTP при отправке в Telegram: {e}")

    def _normalize_channel_id(self) -> str:
        """
        Нормализация ID канала.

        Returns:
            ID канала в правильном формате
        """
        channel_id = self.channel_id.strip()

        # Если начинается с @, оставляем как есть
        if channel_id.startswith("@"):
            return channel_id

        # Если числовой ID, убеждаемся что есть -100 префикс
        if channel_id.startswith("-100"):
            return channel_id

        # Добавляем -100 если нужно
        if channel_id.isdigit():
            return f"-100{channel_id}"

        return channel_id

    def send_video(
        self,
        video_path: str,
        caption: str = ""
    ) -> int:
        """
        Отправка видео в Telegram.

        Args:
            video_path: Путь к видеофайлу
            caption: Текст подписи

        Returns:
            message_id отправленного сообщения
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise PublisherError(f"Видео не найдено: {video_path}")

        url = f"{self.base_url}/sendVideo"
        chat_id = self._normalize_channel_id()

        # Проверяем размер файла (Telegram лимит 50MB для ботов)
        max_size = 50 * 1024 * 1024  # 50 MB
        if video_path.stat().st_size > max_size:
            raise PublisherError(
                f"Размер видео превышает лимит Telegram (50MB): "
                f"{video_path.stat().st_size / 1024 / 1024:.1f}MB"
            )

        try:
            with ipv4_only(), httpx.Client(timeout=120.0) as client:
                with open(video_path, "rb") as video_file:
                    files = {
                        "video": (video_path.name, video_file, "video/mp4")
                    }
                    data = {
                        "chat_id": chat_id,
                        "caption": caption,
                        "parse_mode": "HTML",
                        "supports_streaming": True
                    }

                    response = client.post(url, files=files, data=data)
                    response.raise_for_status()

                    result = response.json()

                    if not result.get("ok"):
                        raise PublisherError(
                            f"Telegram API error: {result.get('description', 'Unknown error')}"
                        )

                    return result["result"]["message_id"]

        except httpx.HTTPError as e:
            raise PublisherError(f"Ошибка HTTP при отправке видео: {e}")

    def edit_message_caption(
        self,
        message_id: int,
        new_caption: str
    ) -> bool:
        """
        Редактирование подписи сообщения.

        Args:
            message_id: ID сообщения для редактирования
            new_caption: Новая подпись

        Returns:
            True если успешно
        """
        url = f"{self.base_url}/editMessageCaption"
        chat_id = self._normalize_channel_id()

        data = {
            "chat_id": chat_id,
            "message_id": message_id,
            "caption": new_caption,
            "parse_mode": "HTML"
        }

        try:
            with ipv4_only(), httpx.Client(timeout=30.0) as client:
                response = client.post(url, json=data)
                response.raise_for_status()

                result = response.json()
                return result.get("ok", False)

        except httpx.HTTPError as e:
            raise PublisherError(f"Ошибка при редактировании сообщения: {e}")

    def delete_message(self, message_id: int) -> bool:
        """
        Удаление сообщения из канала.

        Args:
            message_id: ID сообщения для удаления

        Returns:
            True если успешно
        """
        url = f"{self.base_url}/deleteMessage"
        chat_id = self._normalize_channel_id()

        data = {
            "chat_id": chat_id,
            "message_id": message_id
        }

        try:
            with ipv4_only(), httpx.Client(timeout=30.0) as client:
                response = client.post(url, json=data)
                response.raise_for_status()

                result = response.json()
                return result.get("ok", False)

        except httpx.HTTPError as e:
            raise PublisherError(f"Ошибка при удалении сообщения: {e}")

    def get_channel_info(self) -> Dict[str, Any]:
        """
        Получение информации о канале.

        Returns:
            Информация о канале
        """
        url = f"{self.base_url}/getChat"
        chat_id = self._normalize_channel_id()

        data = {"chat_id": chat_id}

        try:
            with ipv4_only(), httpx.Client(timeout=30.0) as client:
                response = client.post(url, json=data)
                response.raise_for_status()

                result = response.json()

                if not result.get("ok"):
                    raise PublisherError(
                        f"Telegram API error: {result.get('description', 'Unknown error')}"
                    )

                chat = result.get("result", {})
                return {
                    "id": chat.get("id"),
                    "title": chat.get("title"),
                    "username": chat.get("username"),
                    "type": chat.get("type"),
                    "members_count": chat.get("members_count")
                }

        except httpx.HTTPError as e:
            raise PublisherError(f"Ошибка при получении информации о канале: {e}")

    def test_connection(self) -> bool:
        """
        Проверка соединения с Telegram API.

        Returns:
            True если соединение успешно
        """
        url = f"{self.base_url}/getMe"

        try:
            with ipv4_only(), httpx.Client(timeout=10.0) as client:
                response = client.get(url)
                response.raise_for_status()

                result = response.json()
                return result.get("ok", False)

        except httpx.HTTPError:
            return False
