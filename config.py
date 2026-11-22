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

        # ---------- AI-провайдеры (текст и зрение настраиваются отдельно) ----------
        # Текстовый провайдер: gemini | openai_compatible
        self.ai_text_provider = os.getenv("AI_TEXT_PROVIDER", "gemini").strip().lower()
        self.ai_text_model = os.getenv("AI_TEXT_MODEL", "") or self.gemini_model
        # Ключ: отдельный или общий GEMINI_API_KEY
        self.ai_text_api_key = os.getenv("AI_TEXT_API_KEY", "") or self.gemini_api_key
        # Базовый URL — только для openai_compatible
        self.ai_text_base_url = os.getenv("AI_TEXT_BASE_URL", "").strip() or None

        # Vision-провайдер: gemini | openai_compatible | none (vision отключён)
        self.ai_vision_provider = os.getenv("AI_VISION_PROVIDER", "gemini").strip().lower()
        self.ai_vision_model = os.getenv("AI_VISION_MODEL", "") or self.gemini_model
        self.ai_vision_api_key = os.getenv("AI_VISION_API_KEY", "") or self.gemini_api_key
        self.ai_vision_base_url = os.getenv("AI_VISION_BASE_URL", "").strip() or None

        # ---------- Highlight-движок ----------
        self.highlight_min_duration = float(os.getenv("HIGHLIGHT_MIN_DURATION", "20"))
        self.highlight_max_duration = float(os.getenv("HIGHLIGHT_MAX_DURATION", "60"))
        self.highlight_workers = int(os.getenv("HIGHLIGHT_WORKERS", "3"))
        # Визуальная проверка кандидатов: auto (если есть vision-провайдер) | off
        self.highlight_vision_check = os.getenv("HIGHLIGHT_VISION_CHECK", "auto").strip().lower()

        # ---------- Монтаж ----------
        # Частота кадров выхода: 30 или 60
        try:
            self.output_fps = int(os.getenv("OUTPUT_FPS", "30"))
        except ValueError:
            self.output_fps = 30
        # Нормализация аудио (EBU R128 loudnorm)
        self.audio_normalize = os.getenv("AUDIO_NORMALIZE", "1").strip().lower() in ("1", "true", "yes", "on")
        # Шумоподавление (afftdn) — по умолчанию выключено
        self.noise_reduction = os.getenv("NOISE_REDUCTION", "0").strip().lower() in ("1", "true", "yes", "on")
        # Стиль субтитров: bold | boxed | karaoke
        self.subtitle_style = os.getenv("SUBTITLE_STYLE", "bold").strip().lower()
        # Пресет монтажа: auto (по типу источника) | movie | stream | clip | podcast | gaming | off
        self.editing_preset = os.getenv("EDITING_PRESET", "auto").strip().lower()

        # ---------- Кэш анализа ----------
        self.cache_enabled = os.getenv("ANALYSIS_CACHE", "1").strip().lower() in ("1", "true", "yes", "on")

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

        # Telegram — необязателен: приложение работает и без публикации,
        # публиковать можно позже (publish_result проверит ключи сам)
        if not self.telegram_bot_token or not self.telegram_channel_id:
            print("Предупреждение: TELEGRAM_BOT_TOKEN/TELEGRAM_CHANNEL_ID не заданы — "
                  "публикация в Telegram будет недоступна")

        if errors:
            raise ConfigError("\n".join(errors))

    @property
    def telegram_configured(self) -> bool:
        """Достаточно ли ключей для публикации в Telegram."""
        return bool(self.telegram_bot_token and self.telegram_channel_id)

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
