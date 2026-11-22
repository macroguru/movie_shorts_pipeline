"""
YouTube Publisher модуль для загрузки клипов в YouTube Shorts.

Использует официальный YouTube Data API v3: OAuth2-вход в браузере один раз,
дальше токен хранится в youtube_token.json и обновляется сам.
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

from config import Config

# Клипы пайплайна всегда вертикальные и короче минуты — YouTube сам
# распознаёт их как Shorts, отдельного флага не нужно.
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

SETUP_INSTRUCTION = (
    "YouTube не настроен. Разовая настройка:\n"
    "1. https://console.cloud.google.com — создать проект\n"
    "2. APIs & Services → включить YouTube Data API v3\n"
    "3. OAuth consent screen → External → добавить свой Google-аккаунт в Test users\n"
    "4. Credentials → Create Credentials → OAuth client ID → Desktop app\n"
    "5. Скачать JSON и сохранить как client_secrets.json в папке проекта"
)


class YouTubeError(Exception):
    """Исключение для ошибок публикации в YouTube."""
    pass


class YouTubePublisher:
    """Класс для загрузки клипов в YouTube Shorts."""

    def __init__(self, config: Config):
        """
        Инициализация паблишера.

        Args:
            config: Конфигурация проекта
        """
        self.config = config

    def is_configured(self) -> bool:
        """
        Проверка, что OAuth-секреты на месте.

        Returns:
            True, если client_secrets.json существует
        """
        return self.config.youtube_client_secrets.exists()

    def get_service(self):
        """
        Авторизация в YouTube API: токен с диска или вход через браузер.

        Returns:
            Объект сервиса youtube API

        Raises:
            YouTubeError: если нет client_secrets.json или вход не удался
        """
        client_secrets = self.config.youtube_client_secrets
        token_path = self.config.youtube_token

        if not client_secrets.exists():
            raise YouTubeError(SETUP_INSTRUCTION)

        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
        except ImportError as e:
            raise YouTubeError(
                "Библиотеки Google не установлены. Выполните: "
                "pip install google-api-python-client google-auth-oauthlib"
            ) from e

        credentials = None

        if token_path.exists():
            try:
                credentials = Credentials.from_authorized_user_file(str(token_path), SCOPES)
            except (ValueError, json.JSONDecodeError) as e:
                print(f"Токен повреждён, потребуется повторный вход: {e}")
                credentials = None

        if credentials and credentials.expired and credentials.refresh_token:
            try:
                credentials.refresh(Request())
            except Exception as e:  # RefreshError и прочие — проще перелогиниться
                print(f"Не удалось обновить токен ({e}), потребуется повторный вход")
                credentials = None

        if not credentials or not credentials.valid:
            flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets), SCOPES)
            credentials = flow.run_local_server(port=0)
            token_path.write_text(credentials.to_json(encoding="utf-8"), encoding="utf-8")
            print(f"Токен сохранён: {token_path}")

        from googleapiclient.discovery import build

        return build("youtube", "v3", credentials=credentials, cache_discovery=False)

    def upload_clip(
        self,
        clip_path: str,
        title: str,
        description: str,
        tags: List[str],
        log: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Загрузка одного клипа в YouTube.

        Args:
            clip_path: Путь к вертикальному клипу
            title: Название видео
            description: Описание видео
            tags: Список тегов (без #)
            log: Куда писать прогресс

        Returns:
            video_id загруженного видео

        Raises:
            YouTubeError: если файл не найден или загрузка не удалась
        """
        from googleapiclient.errors import HttpError
        from googleapiclient.http import MediaFileUpload

        emit = log or print
        clip = Path(clip_path)

        if not clip.exists():
            raise YouTubeError(f"Клип не найден: {clip}")

        body = {
            "snippet": {
                "title": title,
                "description": description,
                "tags": tags,
                "categoryId": self.config.youtube_category_id,
            },
            "status": {
                "privacyStatus": self.config.youtube_privacy,
                "selfDeclaredMadeForKids": False,
            },
        }

        media = MediaFileUpload(
            str(clip),
            chunksize=8 * 1024 * 1024,
            resumable=True,
            mimetype="video/mp4",
        )

        try:
            service = self.get_service()
            request = service.videos().insert(
                part="snippet,status",
                body=body,
                media_body=media,
            )

            response = None
            while response is None:
                status, response = request.next_chunk()
                if status and status.progress() >= 0.25:
                    emit(f"  Загрузка {clip.name}: {int(status.progress() * 100)}%")

            video_id = response["id"]
            emit(f"  Загружено: https://youtube.com/shorts/{video_id}")
            return video_id

        except HttpError as e:
            raise YouTubeError(f"Ошибка YouTube API ({e.resp.status}): {e._get_reason() if hasattr(e, '_get_reason') else e}")

    def upload_movie(
        self,
        title: str,
        year: Optional[int],
        code: str,
        clip_paths: List[str],
        description: str,
        tags: List[str],
        code_in_title: bool = True,
        log: Optional[Callable[[str], None]] = None
    ) -> List[Dict[str, str]]:
        """
        Загрузка всех клипов фильма.

        Args:
            title: Название фильма
            year: Год выпуска
            code: Код фильма
            clip_paths: Пути к клипам
            description: Описание (одно на все клипы)
            tags: Теги (без #)
            code_in_title: Писать ли код в названии ролика
                (для клипов Twitch — False)
            log: Куда писать прогресс

        Returns:
            Список {"clip", "video_id", "url"} по каждому загруженному клипу
        """
        emit = log or print

        year_line = f" ({year})" if year else ""
        uploaded = []

        for i, clip_path in enumerate(clip_paths, 1):
            emit(f"Клип {i}/{len(clip_paths)}: {Path(clip_path).name}")
            if code_in_title:
                video_title = f"{title}{year_line} — Код {code} #Shorts [{i}]"
            else:
                video_title = f"{title}{year_line} #Shorts [{i}]"
            video_id = self.upload_clip(
                clip_path=clip_path,
                title=video_title,
                description=description,
                tags=tags,
                log=emit,
            )
            uploaded.append({
                "clip": clip_path,
                "video_id": video_id,
                "url": f"https://youtube.com/shorts/{video_id}",
            })

        return uploaded