"""
Модуль конфигурации для Movie Shorts Pipeline.
Загружает и валидирует настройки из .env файла.
"""

import os
import shutil
from pathlib import Path
from dotenv import load_dotenv


class ConfigError(Exception):
    """Исключение для ошибок конфигурации."""
    pass


class Config:
    """Класс конфигурации с валидацией настроек."""

    def __init__(self, env_path: str = None):
        """
        Инициализация конфигурации.

        Args:
            env_path: Путь к .env файлу (по умолчанию .env в директории проекта)
        """
        if env_path is None:
            env_path = Path(__file__).parent / ".env"

        load_dotenv(env_path)

        # Обязательные настройки
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_channel_id = os.getenv("TELEGRAM_CHANNEL_ID")

        # Опциональные настройки со значениями по умолчанию
        self.output_dir = Path(os.getenv("OUTPUT_DIR", Path(__file__).parent / "output"))
        self.temp_dir = Path(os.getenv("TEMP_DIR", Path(__file__).parent / "temp"))
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.whisper_device = os.getenv("WHISPER_DEVICE", "cpu")
        self.whisper_model = os.getenv("WHISPER_MODEL", "small")

        # Кодировщик видео: auto (NVENC, если есть GPU), nvenc или cpu
        self.video_encoder = os.getenv("VIDEO_ENCODER", "auto").strip().lower()

        # YouTube Shorts (фича необязательна): OAuth-секреты, токен, статус и категория видео
        self.youtube_client_secrets = Path(
            os.getenv("YOUTUBE_CLIENT_SECRETS", Path(__file__).parent / "client_secrets.json")
        )
        self.youtube_token = Path(
            os.getenv("YOUTUBE_TOKEN", Path(__file__).parent / "youtube_token.json")
        )
        self.youtube_privacy = os.getenv("YOUTUBE_PRIVACY", "public").strip().lower()
        self.youtube_category_id = os.getenv("YOUTUBE_CATEGORY_ID", "24").strip()

        # Папка текущего фильма (заполняется через set_movie_context)
        self.movie_dir = None

        self._validate()
        self._ensure_directories()
        self._check_ffmpeg()

    def _validate(self):
        """Проверка обязательных настроек."""
        errors = []

        if not self.gemini_api_key:
            errors.append("GEMINI_API_KEY не указан в .env файле")

        if not self.telegram_bot_token:
            errors.append("TELEGRAM_BOT_TOKEN не указан в .env файле")

        if not self.telegram_channel_id:
            errors.append("TELEGRAM_CHANNEL_ID не указан в .env файле")

        if errors:
            raise ConfigError("\n".join(errors))

        # Валидация формата channel_id
        channel_id = self.telegram_channel_id.strip()
        if not (channel_id.startswith("@") or (channel_id.startswith("-100") and len(channel_id) > 10)):
            print(f"Предупреждение: TELEGRAM_CHANNEL_ID='{channel_id}' может быть в неверном формате")
            print("Используйте @channel_name или числовой ID вида -100xxxxxxxxxx")

    def _ensure_directories(self):
        """Создание необходимых директорий."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _check_ffmpeg(self):
        """Проверка наличия FFmpeg в системе."""
        if shutil.which("ffmpeg") is None:
            raise ConfigError(
                "FFmpeg не найден в системе!\n"
                "Установите FFmpeg:\n"
                "  Windows: winget install ffmpeg или скачайте с https://ffmpeg.org/download.html\n"
                "  macOS: brew install ffmpeg\n"
                "  Linux: sudo apt install ffmpeg или sudo dnf install ffmpeg"
            )

        if shutil.which("ffprobe") is None:
            raise ConfigError(
                "ffprobe не найден в системе (входит в состав FFmpeg).\n"
                "Установите FFmpeg полностью."
            )

    @property
    def temp_subs_path(self) -> Path:
        """Путь к файлу субтитров во временной директории."""
        return self.temp_dir / "subs.srt"

    def set_movie_context(self, code: str, title: str, year: int = 0) -> Path:
        """
        Задать папку текущего фильма. Все клипы, постер и описание
        будут сохранены внутри неё.

        Args:
            code: Код фильма
            title: Название фильма
            year: Год выпуска

        Returns:
            Путь к папке фильма
        """
        safe_title = "".join(c if c not in '<>:"/\\|?*' else "_" for c in title).strip(" .")
        folder_name = f"{code}_{safe_title}_{year}" if year else f"{code}_{safe_title}"
        self.movie_dir = self.output_dir / folder_name
        self.movie_dir.mkdir(parents=True, exist_ok=True)
        return self.movie_dir

    def get_clip_output_path(self, code: str, clip_index: int) -> Path:
        """Получить путь для выходного клипа (в папке фильма, если она задана)."""
        base = self.movie_dir or self.output_dir
        return base / f"clip_{code}_{clip_index:02d}.mp4"

    def get_poster_path(self, code: str) -> Path:
        """Получить путь для постера (в папке фильма, если она задана)."""
        base = self.movie_dir or self.output_dir
        return base / f"poster_{code}.jpg"

    def get_description_path(self, code: str) -> Path:
        """Получить путь для файла с описанием публикации."""
        base = self.movie_dir or self.output_dir
        return base / f"description_{code}.txt"

    def __str__(self) -> str:
        return (
            f"Config:\n"
            f"  Gemini Model: {self.gemini_model}\n"
            f"  Whisper Model: {self.whisper_model} ({self.whisper_device})\n"
            f"  Output Dir: {self.output_dir}\n"
            f"  Temp Dir: {self.temp_dir}\n"
            f"  Channel: {self.telegram_channel_id}"
        )
