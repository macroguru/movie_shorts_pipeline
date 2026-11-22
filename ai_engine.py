"""
AI Engine модуль для анализа субтитров через AI-провайдер (Gemini или
OpenAI-совместимый API через providers.py). Находит вирусные сцены и
генерирует текст поста. Ответы кэшируются в AnalysisCache.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from config import Config
from providers import ProviderError, create_text_provider
from cache import AnalysisCache
from usage import UsageTracker


@dataclass
class Clip:
    """Данные о клипе."""
    start: str      # HH:MM:SS
    end: str        # HH:MM:SS
    hook_description: str


@dataclass
class PostData:
    """Данные для поста."""
    genre: str
    synopsis: str
    hook_fact: str


class AIEngineError(Exception):
    """Исключение для ошибок AI Engine."""
    pass


class AIEngine:
    """Класс для анализа субтитров через Gemini."""

    SYSTEM_PROMPT = """Ты продюсер вирусного киноконтента для TikTok и Reels. Твоя задача — найти сцены с наивысшим градусом конфликта, напряжения, неожиданного твиста или черного юмора.

КРИТЕРИИ ВЫБОРА СЦЕН:
- Сцена должна захватывать зрителя в первые 3 секунды
- Сцена должна обрываться на клиффхэнгере
- Длительность сцены: от 40 до 75 секунд
- Приоритет: эмоциональные пики, кульминационные моменты, шокирующие revelations

ВЕРНИ СТРОГИЙ JSON в следующем формате:
{{
    "clips": [
        {{
            "start": "HH:MM:SS",
            "end": "HH:MM:SS",
            "hook_description": "Краткое описание момента (1-2 предложения)"
        }}
    ],
    "post_data": {{
        "genre": "Жанр через слэш (напр. Триллер / Детектив)",
        "synopsis": "Интригующий синопсис без спойлеров на 3-4 предложения",
        "hook_fact": "Короткий цепляющий факт о фильме или рейтинг"
    }}
}}

ВАЖНО:
- Верни ровно {clips_count} клипов
- Используй формат времени HH:MM:SS (без миллисекунд)
- Не добавляй никаких пояснений вне JSON
- Убедись что JSON валидный"""

    # Промпт для режима стрима: моменты вместо сцен фильма, другие постовые данные
    STREAM_SYSTEM_PROMPT = """Ты редактор вирусных нарезок стримов для TikTok, YouTube Shorts и Reels. Тебе дают таймкоды субтитров записи стрима: летсплей, подкаст, реакции, общение с чатом.

КРИТЕРИИ ВЫБОРА МОМЕНТОВ:
- Момент понятен без знания контекста всего стрима
- Цепляет зрителя в первые 3 секунды
- Имеет завершённый мини-сюжет: завязка, развитие, развязка (реакция, панчлайн, итог)
- Длительность момента: от 20 до 60 секунд
- Моменты не должны пересекаться друг с другом

ВЕРНИ СТРОГИЙ JSON в следующем формате:
{{
    "clips": [
        {{
            "start": "HH:MM:SS",
            "end": "HH:MM:SS",
            "hook_description": "Краткое описание момента (1-2 предложения)"
        }}
    ],
    "post_data": {{
        "title": "Тема стрима или момента — короткая фраза для описания",
        "context": "Контекст для описания публикации: что происходит в этих моментах (2-3 предложения)",
        "hashtags": "3-7 дополнительных хэштегов через пробел без решётки (напр. летсплей подкаст реакции)"
    }}
}}

ВАЖНО:
- Верни ровно {clips_count} клипов{duration_line}
- Используй формат времени HH:MM:SS (без миллисекунд)
- Не добавляй никаких пояснений вне JSON
- Убедись что JSON валидный"""

    def __init__(self, config: Config):
        """
        Инициализация AI Engine.

        Args:
            config: Конфигурация проекта
        """
        self.config = config
        self._client = None
        self._cache = AnalysisCache(config)
        # temp_dir может отсутствовать в тестовых фейках конфига:
        # учёт важен только для реальных вызовов API
        temp_dir = getattr(config, "temp_dir", None)
        if temp_dir is not None:
            self._usage = UsageTracker(Path(temp_dir) / "usage.jsonl")
        else:
            self._usage = UsageTracker()

    @property
    def client(self):
        """
        Ленивая инициализация клиента Gemini.

        Оставлен для обратной совместимости: внутренние вызовы идут через
        текстовый провайдер (providers.py).
        """
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.config.gemini_api_key)
            except ImportError:
                raise AIEngineError(
                    "google-genai не установлен. Выполните: pip install google-genai"
                )
        return self._client

    def resolve_title(self, raw_title: str, year: int = None) -> str:
        """
        Определение официального русского названия фильма через Gemini.

        Args:
            raw_title: Название из имени файла (обычно латиницей)
            year: Год выпуска, если известен

        Returns:
            Официальное русское название

        Raises:
            AIEngineError: если Gemini не ответил
        """
        year_line = f"\nГод выпуска: {year}." if year else ""
        prompt = (
            "Определи официальное русское название фильма по названию из имени файла релиза.\n"
            f'Название из имени файла: "{raw_title}"{year_line}\n\n'
            "Ответь ТОЛЬКО русским названием без кавычек, пояснений и года.\n"
            "Если фильм неизвестен, верни исходное название как есть."
        )

        try:
            response = self._call_gemini(prompt)
        except AIEngineError:
            raise

        title = response.strip().strip('"«»')

        if not title:
            raise AIEngineError("Провайдер вернул пустое название")

        return title

    def analyze_subtitles(
        self,
        subs_path: str = None,
        clips_count: int = 3
    ) -> Dict[str, Any]:
        """
        Анализ субтитров и поиск вирусных сцен.

        Args:
            subs_path: Путь к SRT файлу (по умолчанию из config)
            clips_count: Количество клипов для поиска

        Returns:
            Словарь с клипами и данными для поста
        """
        if subs_path is None:
            subs_path = self.config.temp_subs_path

        subs_path = Path(subs_path)

        if not subs_path.exists():
            raise AIEngineError(f"SRT файл не найден: {subs_path}")

        # Читаем субтитры
        srt_content = subs_path.read_text(encoding="utf-8")

        # Парсим и готовим текст для анализа
        subtitles_text = self._prepare_subtitles_for_analysis(srt_content)

        # Формируем промпт
        prompt = self._build_prompt(subtitles_text, clips_count)

        # Отправляем запрос (с кэшем) и разбираем ответ
        response = self._cached_call(
            kind="analyze",
            content_hash=self._subs_hash(srt_content),
            params={"clips_count": clips_count},
            prompt=prompt,
        )
        return self._parse_response(response)

    def analyze_subtitles_stream(
        self,
        subs_path: str = None,
        clips_count: int = 3,
        min_duration: float = 20.0,
        max_duration: float = 60.0
    ) -> Dict[str, Any]:
        """
        Анализ субтитров стрима и поиск вирусных моментов.

        Args:
            subs_path: Путь к SRT файлу (по умолчанию из config)
            clips_count: Количество моментов для поиска
            min_duration: Минимальная длительность момента в секундах
            max_duration: Максимальная длительность момента в секундах

        Returns:
            Словарь с клипами и данными для поста
            (post_data: title, context, hashtags)
        """
        if subs_path is None:
            subs_path = self.config.temp_subs_path

        subs_path = Path(subs_path)

        if not subs_path.exists():
            raise AIEngineError(f"SRT файл не найден: {subs_path}")

        # Читаем субтитры
        srt_content = subs_path.read_text(encoding="utf-8")

        # Парсим и готовим текст для анализа
        subtitles_text = self._prepare_subtitles_for_analysis(srt_content)

        # Формируем промпт
        prompt = self._build_prompt(
            subtitles_text,
            clips_count,
            stream=True,
            min_duration=min_duration,
            max_duration=max_duration,
        )

        # Отправляем запрос (с кэшем) и разбираем ответ
        response = self._cached_call(
            kind="analyze_stream",
            content_hash=self._subs_hash(srt_content),
            params={
                "clips_count": clips_count,
                "min_duration": min_duration,
                "max_duration": max_duration,
            },
            prompt=prompt,
        )
        return self._parse_response(response, stream=True)

    def _prepare_subtitles_for_analysis(self, srt_content: str) -> str:
        """
        Подготовка субтитров для анализа.

        Args:
            srt_content: Сырое содержимое SRT файла

        Returns:
            Текст субтитров с таймкодами для анализа
        """
        lines = srt_content.strip().split("\n")
        result = []

        i = 0
        while i < len(lines):
            # Пропускаем номер блока
            if lines[i].strip().isdigit():
                i += 1
                if i >= len(lines):
                    break

                # Читаем таймкод
                time_line = lines[i].strip()
                if " --> " not in time_line:
                    i += 1
                    continue

                start_time = time_line.split(" --> ")[0].strip()
                # Конвертируем в HH:MM:SS
                start_time = self._normalize_time(start_time)

                i += 1

                # Читаем текст до пустой строки
                text_lines = []
                while i < len(lines) and lines[i].strip():
                    text_lines.append(lines[i].strip())
                    i += 1

                text = " ".join(text_lines)
                result.append(f"[{start_time}] {text}")
            else:
                i += 1

        return "\n".join(result)

    def _normalize_time(self, time_str: str) -> str:
        """
        Нормализация времени в формат HH:MM:SS.

        Args:
            time_str: Строка времени (HH:MM:SS,mmm или HH:MM:SS.mmm)

        Returns:
            Нормализованная строка HH:MM:SS
        """
        # Убираем миллисекунды
        time_str = time_str.replace(",", ".").split(".")[0]
        return time_str

    def _build_prompt(
        self,
        subtitles_text: str,
        clips_count: int,
        stream: bool = False,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None
    ) -> str:
        """
        Построение промпта для Gemini.

        Args:
            subtitles_text: Текст субтитров с таймкодами
            clips_count: Количество клипов
            stream: True — промпт для стрима, False — для фильма
            min_duration: Минимальная длительность момента (только стрим)
            max_duration: Максимальная длительность момента (только стрим)

        Returns:
            Полный промпт для API
        """
        if stream:
            system_prompt = self.STREAM_SYSTEM_PROMPT.format(
                clips_count=clips_count,
                duration_line=f" ({int(min_duration or 20)}-{int(max_duration or 60)} секунд)" if min_duration else "",
            )
            source_label = "СУБТИТРЫ СТРИМА:"
        else:
            system_prompt = self.SYSTEM_PROMPT.format(clips_count=clips_count)
            source_label = "СУБТИТРЫ ФИЛЬМА:"

        user_prompt = f"""{source_label}
{subtitles_text[:50000]}  # Ограничиваем размер

Найди {clips_count} самых вирусных {'моментов' if stream else 'сцен'} согласно критериям выше.
Верни ТОЛЬКО валидный JSON без пояснений."""

        return f"{system_prompt}\n\n{user_prompt}"

    def _call_gemini(self, prompt: str) -> str:
        """
        Вызов текстового провайдера (Gemini или OpenAI-совместимый).

        Имя метода сохранено для обратной совместимости; запрос идёт через
        create_text_provider(config).

        Args:
            prompt: Промпт для отправки

        Returns:
            Ответ модели

        Raises:
            AIEngineError: на ошибке провайдера (исходное сообщение сохранено)
        """
        try:
            provider = create_text_provider(self.config, self._usage)
        except ProviderError as e:
            raise AIEngineError(f"Ошибка провайдера: {e}")

        try:
            return provider.complete(prompt)
        except ProviderError as e:
            raise AIEngineError(f"Ошибка AI API: {e}")

    def _subs_hash(self, srt_content: str) -> str:
        """
        SHA256 содержимого субтитров — часть ключа кэша анализа.

        Args:
            srt_content: Содержимое SRT файла

        Returns:
            hex-строка SHA256
        """
        return hashlib.sha256(srt_content.encode("utf-8")).hexdigest()

    def _cached_call(
        self,
        kind: str,
        content_hash: str,
        params: dict,
        prompt: str,
    ) -> str:
        """
        Вызов провайдера с кэшированием ответа в AnalysisCache.

        Args:
            kind: Тип записи кэша (analyze | analyze_stream)
            content_hash: Хэш субтитров
            params: Параметры запроса (влияют на результат)
            prompt: Готовый промпт

        Returns:
            Ответ модели (из кэша или свежий)

        Raises:
            AIEngineError: на ошибке провайдера
        """
        # Хэш промпта входит в параметры: разные промпты при том же хэше
        # субтитров (например, после правки критериев) не должны делить кэш
        cache_params = dict(params)
        cache_params["prompt_sha"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        cached = self._cache.get(kind, content_hash, cache_params)
        if cached:
            return cached

        response = self._call_gemini(prompt)
        self._cache.put(kind, content_hash, cache_params, response)
        return response

    def summarize_stream_moments(self, highlights: list, title: str = "") -> dict:
        """
        Данные поста для подборки моментов стрима: один вызов провайдера.

        Принимает Highlight-объекты (highlights.py) или словари — читает
        reason/transcript/keywords.

        Args:
            highlights: Найденные моменты (Highlight или dict с reason/transcript/keywords)
            title: Название стрима или стримера

        Returns:
            {"title", "context", "hashtags"} — стримовый формат post_data

        Raises:
            AIEngineError: на ошибке провайдера или невалидном JSON
        """
        if not highlights:
            raise AIEngineError("Список моментов пуст — нечего описывать")

        moments_lines = []
        for i, moment in enumerate(highlights, 1):
            if isinstance(moment, dict):
                reason = moment.get("reason", "")
                transcript = moment.get("transcript", "")
                keywords = moment.get("keywords", [])
                start = moment.get("start_seconds", moment.get("start_sec", 0))
                end = moment.get("end_seconds", moment.get("end_sec", 0))
            else:
                reason = getattr(moment, "reason", "")
                transcript = getattr(moment, "transcript", "")
                keywords = getattr(moment, "keywords", [])
                start = getattr(moment, "start_seconds", 0)
                end = getattr(moment, "end_seconds", 0)

            minutes, seconds = divmod(max(int(start), 0), 60)
            timecode = f"{minutes:02d}:{seconds:02d}"
            hours, rest = divmod(minutes, 60)
            if hours:
                timecode = f"{hours}:{rest:02d}:{seconds:02d}"

            end_minutes, end_seconds_int = divmod(max(int(end), 0), 60)
            end_timecode = f"{end_minutes:02d}:{end_seconds_int:02d}"
            end_hours, end_rest = divmod(end_minutes, 60)
            if end_hours:
                end_timecode = f"{end_hours}:{end_rest:02d}:{end_seconds_int:02d}"

            line = f"{i}. [{timecode}-{end_timecode}] {reason}"
            if keywords:
                line += f" (теги: {', '.join(keywords[:5])})"
            if transcript:
                line += f"\n   Реплики: {transcript[:500]}"
            moments_lines.append(line)

        moments_text = "\n".join(moments_lines)[:50000]
        title_line = f'Название стрима: "{title}".\n' if title else ""

        prompt = (
            f"{title_line}Список найденных ярких моментов записи стрима:\n\n"
            f"{moments_text}\n\n"
            "На основе этих моментов подготовь данные для публикации подборки.\n"
            "ВЕРНИ СТРОГИЙ JSON:\n"
            '{"title": "Тема подборки — короткая фраза", '
            '"context": "Что происходит в подборке, 2-3 предложения, без спойлеров", '
            '"hashtags": "3-7 хэштегов через пробел без решётки"}\n'
            "Без пояснений вне JSON, валидный JSON."
        )

        response = self._call_gemini(prompt)

        data = None
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
            except json.JSONDecodeError:
                data = None
        if data is None:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError as e:
                    raise AIEngineError(f"Невалидный JSON от провайдера: {e}")

        if not isinstance(data, dict):
            raise AIEngineError("Не удалось извлечь JSON из ответа провайдера")

        missing = {"title", "context", "hashtags"} - set(data.keys())
        if missing:
            raise AIEngineError(f"В ответе отсутствуют поля: {', '.join(sorted(missing))}")

        return {
            "title": str(data["title"]).strip(),
            "context": str(data["context"]).strip(),
            "hashtags": str(data["hashtags"]).strip(),
        }

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Парсинг ответа от Gemini.

        Args:
            response: Сырой ответ от API

        Returns:
            Распарсенные данные
        """
        # Извлекаем JSON из ответа (может быть в markdown блоке)
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)

        if json_match:
            json_str = json_match.group(1)
        else:
            # Пытаемся найти JSON напрямую
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise AIEngineError("Не удалось извлечь JSON из ответа Gemini")

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise AIEngineError(f"Невалидный JSON от Gemini: {e}")

        # Валидируем структуру
        if "clips" not in data:
            raise AIEngineError("В ответе отсутствует поле 'clips'")

        if "post_data" not in data:
            raise AIEngineError("В ответе отсутствует поле 'post_data'")

        # Валидируем клипы
        for clip in data["clips"]:
            if not all(k in clip for k in ["start", "end", "hook_description"]):
                raise AIEngineError("Клип должен содержать start, end, hook_description")

        # Валидируем post_data: фильмовый формат (genre/synopsis/hook_fact)
        # или стримовый (title/context/hashtags)
        post_data = data["post_data"]
        movie_keys = {"genre", "synopsis", "hook_fact"}
        stream_keys = {"title", "context", "hashtags"}
        keys = set(post_data.keys())

        if not (movie_keys <= keys or stream_keys <= keys):
            raise AIEngineError(
                "post_data должно содержать genre, synopsis, hook_fact "
                "(фильм) или title, context, hashtags (стрим)"
            )

        return data

    def time_to_seconds(self, time_str: str) -> float:
        """
        Конвертация времени в секунды.

        Args:
            time_str: Время в формате HH:MM:SS

        Returns:
            Время в секундах
        """
        parts = time_str.split(":")
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2]) if len(parts) > 2 else 0

        return hours * 3600 + minutes * 60 + seconds

    def seconds_to_time(self, seconds: float) -> str:
        """
        Конвертация секунд в формат времени.

        Args:
            seconds: Время в секундах

        Returns:
            Строка в формате HH:MM:SS
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def get_clips_with_seconds(
        self,
        data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Получение клипов с конвертацией времени в секунды.

        Args:
            data: Данные от AI Engine

        Returns:
            Список клипов с временами в секундах
        """
        clips = []
        for clip in data.get("clips", []):
            clips.append({
                "start": clip["start"],
                "end": clip["end"],
                "start_seconds": self.time_to_seconds(clip["start"]),
                "end_seconds": self.time_to_seconds(clip["end"]),
                "hook_description": clip["hook_description"]
            })
        return clips
