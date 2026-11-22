"""
Общее ядро Movie Shorts Pipeline.

Здесь собраны шаги пайплайна, не зависящие от интерфейса: их используют
и CLI (cli.py), и графический интерфейс (gui.py).
"""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from tqdm import tqdm

from config import Config
from database import Database
from subtitles import Subtitles, SubtitleError
from ai_engine import AIEngine, AIEngineError
from video_editor import VideoEditor, VideoEditorError
from publisher import Publisher, PublisherError
from youtube_publisher import YouTubePublisher, YouTubeError, SETUP_INSTRUCTION
from twitch_downloader import TwitchDownloader, TwitchDownloadError, is_clip_url


logger = logging.getLogger(__name__)

# Режимы обработки: фильм или запись стрима
SOURCE_MOVIE = "movie"
SOURCE_STREAM = "stream"

# Функции обратного вызова для интерфейсов: лог и прогресс
LogFn = Callable[[str], None]
ProgressFn = Callable[[int, int], None]

# Год в имени файла релиза: 19xx / 20xx, отдельно стоящий
YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")

# Токены имени файла, которые относятся к релизу, а не к названию фильма
RELEASE_JUNK = {
    "web", "dl", "rip", "dlrip", "webrip", "web-dl", "webdl", "webrip",
    "bluray", "bdremux", "bdmux", "bdrip", "brrip", "dvdrip", "dvdr", "hdtv",
    "hdcam", "cam", "ts", "scr", "screener", "workprint",
    "aac", "ac3", "ac3+", "eac3", "dts", "dtshd", "dts-hd", "flac", "lossless",
    "hd", "shd", "sd", "hdr", "hdr10", "sdr", "dolby", "atmos", "upmix",
    "x264", "x265", "h264", "h265", "hevc", "avc", "divx", "xvid", "vc1", "av1",
    "2160p", "1440p", "1080p", "720p", "480p", "360p", "4k", "uhd",
    "amzn", "nf", "netflix", "hulu", "disney", "hbo", "max", "prime", "kp", "kinopoisk",
    "imdb", "tt", "ru", "en", "eng", "rus", "russian", "english", "multi",
    "sub", "subs", "subbed", "forced", "multipart", "remux",
    "proper", "repack", "internal", "unrated", "extended", "directors",
    "remastered", "hm10", "128", "256", "320", "mkv", "mp4", "m4v", "avi", "mov",
}


class PipelineError(Exception):
    """Ошибка пайплайна — текст показывается пользователю как есть."""
    pass


@dataclass
class MovieResult:
    """Итог обработки одного фильма: всё, что нужно для показа и публикации."""

    code: str
    title: str
    year: Optional[int]
    movie_dir: Path
    clip_paths: list = field(default_factory=list)
    poster_path: Optional[str] = None
    description: str = ""
    description_path: Optional[Path] = None
    post_data: dict = field(default_factory=dict)
    message_id: Optional[int] = None
    config: Optional[Config] = field(default=None, repr=False)
    source: str = SOURCE_MOVIE       # movie (фильм) или stream (стрим)
    caption: Optional[str] = None    # готовая HTML-подпись Telegram (для стримов)


def detect_year(video_path: str) -> Optional[int]:
    """
    Автоматическое определение года выпуска по имени файла.

    Args:
        video_path: Путь к видеофайлу

    Returns:
        Год или None, если в имени файла года нет
    """
    stem = Path(video_path).stem
    years = [int(y) for y in YEAR_RE.findall(stem)]

    if not years:
        return None

    # Не берём год из далёкого будущего: релиз 20xx возможен, но не на 50 лет вперёд
    current = datetime.now().year
    plausible = [y for y in years if 1900 <= y <= current + 1]

    return max(plausible) if plausible else min(years)


def _is_release_junk(token: str) -> bool:
    """Токен имени файла — часть названия или техническая метка релиза."""
    low = token.lower()

    if low in RELEASE_JUNK:
        return True

    parts = [p for p in re.split(r"[-+]", low) if p]
    return bool(parts) and all(p in RELEASE_JUNK for p in parts)


def guess_title(video_path: str) -> str:
    """
    Предположить название фильма по имени файла.

    Убираем год и технические метки релиза (WEB-DLRip, 1080p, AMZN и т.п.).
    Результат только предложение — в интерфейсе его можно поправить.

    Args:
        video_path: Путь к видеофайлу

    Returns:
        Предложенное название
    """
    stem = re.sub(r"[\[\]\(\)]", " ", Path(video_path).stem)
    tokens = [t for t in re.split(r"[._]+", stem) if t]

    kept = []
    for token in tokens:
        if YEAR_RE.fullmatch(token):
            continue
        if token.isdigit():
            continue
        if _is_release_junk(token):
            continue
        kept.append(token)

    return " ".join(kept).strip() or stem.strip()


# Кириллица в названии — значит, оно уже на русском и переводить его не нужно
CYRILLIC_RE = re.compile(r"[а-яё]", re.IGNORECASE)


def resolve_russian_title(
    title: str,
    year: Optional[int],
    config: Config,
    log: Optional[LogFn] = None
) -> str:
    """
    Официальное русское название фильма через Gemini.

    Название на латинице (из имени файла релиза) отправляем в Gemini,
    русское возвращаем как есть. При любой ошибке остаётся исходное
    название — обработка не должна падать из-за перевода.

    Args:
        title: Название из имени файла или от пользователя
        year: Год выпуска, если известен
        config: Конфигурация проекта
        log: Куда писать сообщения этапами

    Returns:
        Русское название либо исходное, если определить не удалось
    """
    if CYRILLIC_RE.search(title):
        return title

    emit = _make_emitter(log)
    emit(f"Определяем русское название для «{title}»...")

    try:
        resolved = AIEngine(config).resolve_title(title, year)
    except Exception as e:  # noqa: BLE001 — перевод не критичен для пайплайна
        logger.warning(f"Не удалось определить русское название: {e}")
        emit(f"  Не удалось определить русское название ({e}); оставляем «{title}»")
        return title

    emit(f"  Название определено: {resolved}")
    return resolved


def build_hashtags(code: str, genre: str, year: Optional[int]) -> list:
    """
    Формирование списка хэштегов для продвижения публикации.

    Args:
        code: Код фильма
        genre: Жанр через слэш
        year: Год выпуска

    Returns:
        Список хэштегов (со знаком #)
    """
    tags = [f"#код{code}"]

    # Жанровые теги — по каждому направлению через слэш
    for part in genre.split("/"):
        word = "".join(c for c in part.strip().lower() if c.isalnum() or c == "_")
        if word:
            tags.append(f"#{word}")

    # Год и универсальные продвиженческие теги
    if year:
        tags.append(f"#{year}")

    tags += ["#кино", "#фильмы", "#shorts", "#reels", "#tiktok", "#чтопосмотреть"]

    # Убираем дубликаты, сохраняя порядок
    seen = set()
    unique = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            unique.append(tag)

    return unique


def build_description(
    code: str,
    year: Optional[int],
    post_data: dict,
) -> str:
    """
    Текст описания для копирования в поле публикации любой площадки.
    Без HTML-разметки и без спойлеров.

    Первая строка — «Код: N. Название в шапке профиля.», дальше только
    год, жанр, синопсис, факт и хэштеги. Само название в описании не нужно:
    зритель ищет фильм по коду.

    Args:
        code: Код фильма
        year: Год выпуска
        post_data: Данные поста (genre, synopsis, hook_fact)

    Returns:
        Готовый текст описания
    """
    genre = post_data.get("genre", "").strip() or "Не указан"
    synopsis = post_data.get("synopsis", "").strip()
    hook_fact = post_data.get("hook_fact", "").strip()
    hashtags = " ".join(build_hashtags(code, genre, year))

    lines = [f"Код: {code}. Название в шапке профиля."]

    if year:
        lines.append(f"Год: {year}")

    lines += [
        f"Жанр: {genre}",
        "",
        synopsis,
        "",
    ]

    if hook_fact:
        lines += [f"Интересный факт: {hook_fact}", ""]

    lines.append(hashtags)

    return "\n".join(lines).strip() + "\n"


def post_link(result: MovieResult) -> Optional[str]:
    """Ссылка на опубликованный пост, если известен ID сообщения."""
    if result.message_id is None or result.config is None:
        return None

    channel = str(result.config.telegram_channel_id).lstrip("@-")
    return f"https://t.me/{channel}/{result.message_id}"


def _make_emitter(log: Optional[LogFn]) -> LogFn:
    """Лог в интерфейс вызывающего кода, с падением обратно в logging."""
    if log is not None:
        return log
    return lambda message: logger.info(message)


def process_video(
    input_path: str,
    title: str,
    clips_count: int = 3,
    year: Optional[int] = None,
    env_path: str = None,
    log: Optional[LogFn] = None,
    progress: Optional[ProgressFn] = None,
) -> MovieResult:
    """
    Полный цикл обработки фильма без публикации: субтитры → AI анализ →
    рендер клипов → описание.

    Args:
        input_path: Путь к видеофайлу
        title: Название фильма
        clips_count: Сколько клипов нарезать
        year: Год выпуска (по умолчанию определяется по имени файла)
        env_path: Путь к .env файлу
        log: Куда писать сообщения этапами (по умолчанию в logging)
        progress: Колбэк (готово, всего) для индикатора прогресса

    Returns:
        MovieResult с путями, описанием и данными поста

    Raises:
        PipelineError: на любой ошибке одного из этапов
    """
    emit = _make_emitter(log)

    path = Path(input_path)
    if not path.exists():
        raise PipelineError(f"Видеофайл не найден: {path}")

    if year is None:
        year = detect_year(str(path))

    try:
        config = Config(env_path=env_path)
    except Exception as e:
        raise PipelineError(f"Ошибка конфигурации: {e}")

    emit(f"Файл: {path}")

    # Название на латинице переводим в официальное русское через Gemini
    title = resolve_russian_title(title, year, config, log=emit)

    emit(f"Название: {title}, год: {year or 'не определён'}, клипов: {clips_count}")

    db = Database()
    subtitles = Subtitles(config)
    ai_engine = AIEngine(config)
    video_editor = VideoEditor(config)

    # Уникальный код и отдельная папка фильма
    code = db.get_unique_code()
    emit(f"Сгенерирован код фильма: {code}")

    movie_dir = config.set_movie_context(code, title, year or 0)
    emit(f"Папка фильма: {movie_dir}")

    # Шаг 1: субтитры
    emit("Шаг 1: извлечение субтитров...")
    try:
        subs_path = subtitles.extract_or_transcribe(str(path))
        emit(f"Субтитры сохранены: {subs_path}")
    except SubtitleError as e:
        raise PipelineError(f"Ошибка работы с субтитрами: {e}")

    # Шаг 2: AI анализ
    emit("Шаг 2: AI анализ субтитров (поиск вирусных сцен)...")
    try:
        ai_result = ai_engine.analyze_subtitles(clips_count=clips_count)
        clips = ai_engine.get_clips_with_seconds(ai_result)
        post_data = ai_result["post_data"]
    except AIEngineError as e:
        raise PipelineError(f"Ошибка AI анализа: {e}")

    emit(f"Найдено сцен: {len(clips)}")
    for i, clip in enumerate(clips, 1):
        emit(f"  Сцена {i}: {clip['start']} - {clip['end']} — {clip['hook_description']}")

    emit(f"Жанр: {post_data.get('genre', 'Не указан')}")

    # Шаг 3: рендер клипов — параллельно (NVENC держит несколько сессий,
    # а CPU-фильтры у каждого процесса свои)
    emit("Шаг 3: рендер вертикальных клипов (параллельно)...")
    total = len(clips)
    renders: list = [None] * total

    def render_one(args):
        i, clip = args
        try:
            render = video_editor.render_short(
                video_path=str(path),
                start=clip["start"],
                end=clip["end"],
                code=code,
                clip_index=i,
                title_text=title
            )
            return i, render, None
        except VideoEditorError as e:
            return i, None, str(e)

    done = 0
    max_workers = min(total, 3)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for i, render, error in pool.map(render_one, enumerate(clips, 1)):
            done += 1
            if error is not None:
                emit(f"  Ошибка рендера клипа {i}: {error}")
                logger.error(f"Ошибка рендера клипа {i}: {error}")
            else:
                renders[i - 1] = render
                emit(f"  Клип {i} готов: {render['clip_path']} ({render['duration']}s)")
            if progress is not None:
                progress(done, total)

    clip_paths = [r["clip_path"] for r in renders if r is not None]
    poster_path = next((r["poster_path"] for r in renders if r is not None), None)

    if not clip_paths:
        raise PipelineError("Не удалось создать ни одного клипа")

    # Сохраняем в базу и пишем описание
    db.save_movie(
        code=code,
        title=title,
        year=year or 0,
        genre=post_data.get("genre", ""),
        clip_paths=json.dumps(clip_paths),
        poster_path=poster_path,
        video_path=str(path)
    )
    emit("Информация сохранена в базу данных")

    description = build_description(code, year, post_data)
    description_path = config.get_description_path(code)
    description_path.write_text(description, encoding="utf-8")
    emit(f"Описание сохранено: {description_path}")

    return MovieResult(
        code=code,
        title=title,
        year=year,
        movie_dir=movie_dir,
        clip_paths=clip_paths,
        poster_path=poster_path,
        description=description,
        description_path=description_path,
        post_data=post_data,
        config=config,
    )


# ---------- Режим стрима ----------

def _time_string_to_seconds(text: str) -> float:
    """ЧЧ:ММ:СС | ММ:СС | СС → секунды."""
    parts = text.strip().split(":")
    if len(parts) > 3 or not all(p.isdigit() for p in parts):
        raise PipelineError(f"Непонятное время: {text!r} (ожидается ЧЧ:ММ:СС)")

    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + int(part)

    return seconds


def _seconds_to_hms(seconds: float) -> str:
    """Секунды → строка HH:MM:SS."""
    total = max(int(seconds), 0)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_moments(moments_text: str) -> list:
    """
    Разбор ручных моментов: «1:23:45-1:25:00, 00:40-00:41:30».

    Разделитель моментов — запятая или точка с запятой; время поддерживает
    форматы ЧЧ:ММ:СС, ММ:СС и просто секунды.

    Args:
        moments_text: Строка с моментами от пользователя

    Returns:
        Список клипов в том же формате, что отдаёт AIEngine

    Raises:
        PipelineError: если формат не распознан
    """
    moments = []

    for chunk in re.split(r"[,;]+", moments_text):
        chunk = chunk.strip()
        if not chunk:
            continue

        match = re.fullmatch(r"([\d:]+)\s*[-–—]\s*([\d:]+)", chunk)
        if not match:
            raise PipelineError(
                f"Не удалось разобрать момент {chunk!r}. Формат: "
                "ЧЧ:ММ:СС-ЧЧ:ММ:СС (или ММ:СС-ММ:СС), моменты через запятую"
            )

        start = _time_string_to_seconds(match.group(1))
        end = _time_string_to_seconds(match.group(2))

        if end <= start:
            raise PipelineError(f"Конец момента не раньше начала: {chunk!r}")

        moments.append({
            "start": _seconds_to_hms(start),
            "end": _seconds_to_hms(end),
            "start_seconds": start,
            "end_seconds": end,
            "hook_description": "Момент задан вручную",
        })

    if not moments:
        raise PipelineError("Не распознан ни один момент — проверьте формат")

    return moments


def build_stream_hashtags(code: str, extra_tags: str = "") -> list:
    """
    Хэштеги для нарезки стрима: код + стримовые базовые + от AI.

    Args:
        code: Код подборки
        extra_tags: Дополнительные теги от AI (слова через пробел, без #)

    Returns:
        Список хэштегов без дублей
    """
    tags = [f"#код{code}", "#стримы", "#стрим", "#twitch", "#shorts", "#reels", "#tiktok"]

    for word in (extra_tags or "").split():
        word = "".join(
            c for c in word.strip().lstrip("#").lower() if c.isalnum() or c == "_"
        )
        if word:
            tags.append(f"#{word}")

    seen = set()
    unique = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            unique.append(tag)

    return unique


def build_stream_description(code: str, title: str, post_data: dict) -> str:
    """
    Текст описания для копирования в поле публикации (Shorts/Reels/TikTok).

    Args:
        code: Код подборки
        title: Название стрима или стримера
        post_data: Данные поста (title, context, hashtags)

    Returns:
        Готовый текст описания
    """
    context = (post_data.get("context") or "").strip()
    extra_tags = (post_data.get("hashtags") or "").strip()
    hashtags = " ".join(build_stream_hashtags(code, extra_tags))

    lines = [
        f"Код: {code}. Название в шапке профиля.",
        f"Стрим: {title}",
        "",
    ]

    if context:
        lines += [context, ""]

    lines.append(hashtags)

    return "\n".join(lines).strip() + "\n"


def build_stream_caption(code: str, title: str, post_data: dict) -> str:
    """HTML-подпись поста Telegram для нарезки стрима."""
    context = (post_data.get("context") or "").strip()
    extra_tags = (post_data.get("hashtags") or "").strip()
    hashtags = " ".join(build_stream_hashtags(code, extra_tags))

    caption = f"🎮 <b>Стрим: {title}</b>\n\n"
    if context:
        caption += f"{context}\n\n"
    caption += f"🔍 <b>Код:</b> <code>{code}</code>\n\n{hashtags}"

    return caption


def build_twitch_hashtags(streamer: str, games: list) -> list:
    """
    Хэштеги для пачки клипов Twitch: стример и игра (без кода).

    Args:
        streamer: Имя стримера
        games: Названия игр (displayName с Twitch)

    Returns:
        Список хэштегов без дублей
    """
    tags = []

    for word in [streamer] + list(games):
        slug = "".join(
            c for c in (word or "").strip().lower() if c.isalnum() or c == "_"
        )
        if slug:
            tags.append(f"#{slug}")

    tags += ["#twitch", "#shorts", "#reels"]

    seen = set()
    unique = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            unique.append(tag)

    return unique


def build_twitch_description(streamer: str, games: list) -> str:
    """
    Описание публикации для пачки клипов Twitch: стример и игра в хэштегах,
    без кода — он для клипов Twitch не нужен.

    Args:
        streamer: Имя стримера
        games: Названия игр

    Returns:
        Готовый текст описания
    """
    hashtags = " ".join(build_twitch_hashtags(streamer, games))

    lines = []
    if streamer:
        lines.append(f"Стример: {streamer}")
    if games:
        lines.append("Игра: " + ", ".join(games))

    lines += ["", hashtags]

    return "\n".join(lines).strip() + "\n"


def build_twitch_caption(streamer: str, games: list) -> str:
    """HTML-подпись поста Telegram для пачки клипов Twitch (без кода)."""
    hashtags = " ".join(build_twitch_hashtags(streamer, games))

    lines = []
    if streamer:
        lines.append(f"🎮 <b>Клипы: {streamer}</b>")
    if games:
        lines.append("🕹 " + ", ".join(games))

    lines += ["", hashtags]

    return "\n".join(lines)


def download_twitch_source(
    url: str,
    env_path: str = None,
    log: Optional[LogFn] = None,
) -> tuple:
    """
    Скачивание клипа или VOD Twitch во временную папку проекта.

    Args:
        url: Ссылка на клип или VOD
        env_path: Путь к .env файлу
        log: Куда писать прогресс

    Returns:
        (путь к файлу, тип "clip"|"vod", название с Twitch или None)
    """
    emit = _make_emitter(log)
    url = url.strip()

    try:
        config = Config(env_path=env_path)
    except Exception as e:
        raise PipelineError(f"Ошибка конфигурации: {e}")

    downloader = TwitchDownloader(config.temp_dir / "twitch")
    base = f"twitch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    try:
        if is_clip_url(url):
            emit("Скачиваем клип Twitch...")
            path = downloader.download_clip(url, base)
            kind = "clip"
        else:
            emit("Скачиваем VOD Twitch — запись может быть большой, это долго...")
            path = downloader.download_vod(url, base)
            kind = "vod"
    except TwitchDownloadError as e:
        raise PipelineError(f"Ошибка скачивания Twitch: {e}")

    emit(f"Скачано: {path}")

    try:
        title = downloader.fetch_title(url)
    except TwitchDownloadError:
        title = None

    return str(path), kind, title


def process_stream(
    input_path: str,
    title: str,
    clips_count: int = 3,
    moments: Optional[str] = None,
    year: Optional[int] = None,
    env_path: str = None,
    log: Optional[LogFn] = None,
    progress: Optional[ProgressFn] = None,
) -> MovieResult:
    """
    Полный цикл нарезки стрима: субтитры → AI-моменты (или ручные таймкоды)
    → рендер вертикальных клипов → описание.

    Args:
        input_path: Путь к записи стрима (VOD, mp4/mkv)
        title: Название стрима или стримера
        clips_count: Сколько моментов искать AI (игнорируется при ручных моментах)
        moments: Моменты вручную: «ЧЧ:ММ:СС-ЧЧ:ММ:СС, ...»
        year: Год (по умолчанию текущий)
        env_path: Путь к .env файлу
        log: Куда писать сообщения этапами
        progress: Колбэк (готово, всего) для индикатора прогресса

    Returns:
        MovieResult с путями, описанием и готовой подписью для Telegram

    Raises:
        PipelineError: на любой ошибке одного из этапов
    """
    emit = _make_emitter(log)

    path = Path(input_path)
    if not path.exists():
        raise PipelineError(f"Видеофайл не найден: {path}")

    year = year or datetime.now().year

    try:
        config = Config(env_path=env_path)
    except Exception as e:
        raise PipelineError(f"Ошибка конфигурации: {e}")

    emit(f"Файл: {path}")
    emit(f"Стрим: {title}, год: {year}, клипов: {clips_count}")

    db = Database()
    subtitles = Subtitles(config)
    ai_engine = AIEngine(config)
    video_editor = VideoEditor(config)

    # Уникальный код и отдельная папка
    code = db.get_unique_code()
    emit(f"Сгенерирован код: {code}")

    movie_dir = config.set_movie_context(code, title, year)
    emit(f"Папка: {movie_dir}")

    # Моменты: вручную или через AI по субтитрам
    manual_clips = []
    if moments and moments.strip():
        manual_clips = parse_moments(moments)
        emit(f"Задано моментов вручную: {len(manual_clips)}")

    if manual_clips:
        clips = manual_clips
        post_data = {"title": title, "context": "", "hashtags": ""}
    else:
        # Шаг 1: субтитры (записи стримов обычно транскрибируются через Whisper)
        emit("Шаг 1: извлечение субтитров (для длинной записи это долго)...")
        try:
            subs_path = subtitles.extract_or_transcribe(str(path))
            emit(f"Субтитры сохранены: {subs_path}")
        except SubtitleError as e:
            raise PipelineError(f"Ошибка работы с субтитрами: {e}")

        # Длительность записи — чтобы AI не предлагал таймкоды за её пределами
        duration = None
        try:
            info = video_editor.get_video_info(str(path))
            duration = float(info.get("duration") or 0) or None
        except VideoEditorError:
            pass

        emit("Шаг 2: AI поиск ярких моментов стрима...")
        try:
            ai_result = ai_engine.analyze_subtitles_stream(clips_count=clips_count)
            clips = ai_engine.get_clips_with_seconds(ai_result)
            post_data = ai_result["post_data"]
        except AIEngineError as e:
            raise PipelineError(f"Ошибка AI анализа: {e}")

        emit(f"Найдено моментов: {len(clips)}")
        for i, clip in enumerate(clips, 1):
            emit(f"  Момент {i}: {clip['start']} - {clip['end']} — {clip['hook_description']}")

    # Шаг 3: рендер — как у фильмов, но без минимальных 40 секунд:
    # момент стрима живёт своей длительностью
    emit("Шаг 3: рендер вертикальных клипов (параллельно)...")
    total = len(clips)
    renders: list = [None] * total

    def render_one(args):
        i, clip = args
        try:
            render = video_editor.render_short(
                video_path=str(path),
                start=clip["start"],
                end=clip["end"],
                code=code,
                clip_index=i,
                title_text=title,
                min_duration=0.0,
            )
            return i, render, None
        except VideoEditorError as e:
            return i, None, str(e)

    done = 0
    max_workers = min(total, 3)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for i, render, error in pool.map(render_one, enumerate(clips, 1)):
            done += 1
            if error is not None:
                emit(f"  Ошибка рендера клипа {i}: {error}")
                logger.error(f"Ошибка рендера клипа {i}: {error}")
            else:
                renders[i - 1] = render
                emit(f"  Клип {i} готов: {render['clip_path']} ({render['duration']}s)")
            if progress is not None:
                progress(done, total)

    clip_paths = [r["clip_path"] for r in renders if r is not None]
    poster_path = next((r["poster_path"] for r in renders if r is not None), None)

    if not clip_paths:
        raise PipelineError("Не удалось создать ни одного клипа")

    # Сохраняем в базу и пишем описание
    db.save_movie(
        code=code,
        title=title,
        year=year,
        genre="",
        clip_paths=json.dumps(clip_paths),
        poster_path=poster_path,
        video_path=str(path)
    )
    emit("Информация сохранена в базу данных")

    description = build_stream_description(code, title, post_data)
    description_path = config.get_description_path(code)
    description_path.write_text(description, encoding="utf-8")
    emit(f"Описание сохранено: {description_path}")

    return MovieResult(
        code=code,
        title=title,
        year=year,
        movie_dir=movie_dir,
        clip_paths=clip_paths,
        poster_path=poster_path,
        description=description,
        description_path=description_path,
        post_data=post_data,
        config=config,
        source=SOURCE_STREAM,
        caption=build_stream_caption(code, title, post_data),
    )


def process_twitch_clips(
    urls: list,
    title: Optional[str] = None,
    year: Optional[int] = None,
    env_path: str = None,
    log: Optional[LogFn] = None,
    progress: Optional[ProgressFn] = None,
) -> MovieResult:
    """
    Пачка клипов Twitch → вертикальные шортсы под одним кодом.

    Каждый клип рендерится целиком: субтитры и AI-анализ не нужны.
    Код — только внутренний идентификатор (папка и база): на клип он
    не рисуется и в описании не пишется. В хэштегах — стример и игра,
    взятые из метаданных Twitch.

    Args:
        urls: Ссылки на клипы Twitch (clips.twitch.tv/... или twitch.tv/.../clip/...)
        title: Название подборки (по умолчанию имя стримера или заголовок клипа)
        year: Год (по умолчанию текущий)
        env_path: Путь к .env файлу
        log: Куда писать сообщения этапами
        progress: Колбэк (готово, всего) для индикатора прогресса

    Returns:
        MovieResult со всеми клипами пачки

    Raises:
        PipelineError: на ошибке скачивания, рендера или пустых ссылках
    """
    emit = _make_emitter(log)

    urls = [u.strip() for u in urls if u and u.strip()]
    if not urls:
        raise PipelineError("Не передано ни одной ссылки на клип Twitch")

    bad = [u for u in urls if not is_clip_url(u)]
    if bad:
        raise PipelineError(
            "Это ссылки не на клипы Twitch: " + ", ".join(bad) +
            "\nVOD (записи стримов) нарезайте в режиме «Стрим»"
        )

    try:
        config = Config(env_path=env_path)
    except Exception as e:
        raise PipelineError(f"Ошибка конфигурации: {e}")

    downloader = TwitchDownloader(config.temp_dir / "twitch")

    # Скачивание всех клипов
    paths = []
    for i, url in enumerate(urls, 1):
        emit(f"Скачивание клипа {i}/{len(urls)}: {url}")
        base = f"twitch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}"
        try:
            path = downloader.download_clip(url, base)
        except TwitchDownloadError as e:
            raise PipelineError(f"Ошибка скачивания: {e}")
        paths.append(str(path))
        emit(f"  Скачано: {path}")

    # Метаданные клипов: стример и игра для хэштегов, заголовок для названия
    emit("Определяем стримера и игру по метаданным Twitch...")
    streamer = ""
    games = []
    meta_title = ""
    for url in urls:
        meta = downloader.fetch_clip_metadata(url) or {}
        streamer = streamer or meta.get("streamer") or ""
        meta_title = meta_title or meta.get("title") or ""
        game = (meta.get("game") or "").strip()
        if game and game not in games:
            games.append(game)
        if meta.get("streamer"):
            break  # одного клипа достаточно для стримера, но игры собираем дальше

    if streamer:
        emit(f"Стример: {streamer}" + (f", игра: {', '.join(games)}" if games else ""))

    # Название подборки: от пользователя → имя стримера → заголовок клипа
    resolved_title = (title or "").strip() or streamer or meta_title
    if not resolved_title:
        try:
            resolved_title = downloader.fetch_title(urls[0]) or ""
        except TwitchDownloadError:
            resolved_title = ""
    resolved_title = resolved_title or "Клипы Twitch"

    year = year or datetime.now().year

    db = Database()
    video_editor = VideoEditor(config)

    code = db.get_unique_code()
    emit(f"Внутренний код подборки: {code} (на клипы не рисуется)")

    movie_dir = config.set_movie_context(code, resolved_title, year)
    emit(f"Папка: {movie_dir}")

    post_data = {
        "title": resolved_title,
        "streamer": streamer,
        "games": games,
        "context": "",
        "hashtags": "",
    }

    total = len(paths)
    renders: list = [None] * total
    done = 0

    def render_one(args):
        i, path = args
        try:
            info = video_editor.get_video_info(path)
            duration = int(float(info.get("duration") or 0))
            if duration < 1:
                raise VideoEditorError(f"Не удалось определить длительность: {path}")
            render = video_editor.render_short(
                video_path=path,
                start="00:00:00",
                end=_seconds_to_hms(duration),
                code=code,
                clip_index=i,
                title_text=resolved_title,
                min_duration=0.0,
                with_overlay=False,  # клипы Twitch — без оверлея с кодом
            )
            return i, render, None
        except VideoEditorError as e:
            return i, None, str(e)

    emit("Рендер вертикальных клипов (параллельно)...")
    with ThreadPoolExecutor(max_workers=min(total, 3)) as pool:
        for i, render, error in pool.map(render_one, enumerate(paths, 1)):
            done += 1
            if error is not None:
                emit(f"  Ошибка рендера клипа {i}: {error}")
                logger.error(f"Ошибка рендера клипа {i}: {error}")
            else:
                renders[i - 1] = render
                emit(f"  Клип {i} готов: {render['clip_path']} ({render['duration']}s)")
            if progress is not None:
                progress(done, total)

    clip_paths = [r["clip_path"] for r in renders if r is not None]
    poster_path = next((r["poster_path"] for r in renders if r is not None), None)

    if not clip_paths:
        raise PipelineError("Не удалось создать ни одного клипа")

    db.save_movie(
        code=code,
        title=resolved_title,
        year=year,
        genre="",
        clip_paths=json.dumps(clip_paths),
        poster_path=poster_path,
        video_path=paths[0],
    )
    emit("Информация сохранена в базу данных")

    description = build_twitch_description(streamer, games)
    description_path = config.get_description_path(code)
    description_path.write_text(description, encoding="utf-8")
    emit(f"Описание сохранено: {description_path}")

    return MovieResult(
        code=code,
        title=resolved_title,
        year=year,
        movie_dir=movie_dir,
        clip_paths=clip_paths,
        poster_path=poster_path,
        description=description,
        description_path=description_path,
        post_data=post_data,
        config=config,
        source=SOURCE_STREAM,
        caption=build_twitch_caption(streamer, games),
    )


def publish_result(result: MovieResult, env_path: str = None, log: Optional[LogFn] = None) -> int:
    """
    Публикация готового результата в Telegram-канал.

    Args:
        result: Итог process_video
        env_path: Путь к .env файлу (если конфиг не приехал в результате)
        log: Куда писать сообщения этапами

    Returns:
        message_id опубликованного сообщения
    """
    emit = _make_emitter(log)

    config = result.config or Config(env_path=env_path)
    result.config = config
    publisher = Publisher(config)
    db = Database()

    emit("Шаг 4: публикация в Telegram...")

    if not publisher.test_connection():
        raise PipelineError(
            "Не удалось подключиться к Telegram API. "
            "Проверьте TELEGRAM_BOT_TOKEN в .env файле"
        )

    try:
        message_id = publisher.publish_to_telegram(
            code=result.code,
            title=result.title,
            year=result.year or 0,
            post_data=result.post_data,
            poster_path=result.poster_path,
            caption=result.caption,
        )
    except PublisherError as e:
        raise PipelineError(f"Ошибка публикации: {e}")

    db.update_telegram_msg_id(result.code, message_id)
    result.message_id = message_id

    emit(f"Пост опубликован! Message ID: {message_id}")

    return message_id


def publish_youtube(result: MovieResult, log: Optional[LogFn] = None) -> list:
    """
    Загрузка всех клипов фильма в YouTube Shorts.

    Args:
        result: Итог process_video
        log: Куда писать сообщения этапами

    Returns:
        Список {"clip", "video_id", "url"} по каждому загруженному клипу

    Raises:
        PipelineError: если YouTube не настроен или загрузка не удалась
    """
    emit = _make_emitter(log)

    publisher = YouTubePublisher(result.config or Config())

    if not publisher.is_configured():
        raise PipelineError(SETUP_INSTRUCTION)

    emit("Шаг 5: загрузка клипов в YouTube Shorts...")
    emit("При первом входе откроется браузер Google для авторизации.")

    # Описание то же, что в description_<код>.txt, плюс #Shorts
    description = result.description or build_description(
        result.code, result.year, result.post_data
    )
    description += "\n#Shorts"

    # Теги YouTube — те же слова, что в хэштегах, но без #
    # Пачка клипов Twitch: без кода в заголовке и в тегах
    if result.post_data.get("streamer") is not None:
        hashtag_list = build_twitch_hashtags(
            result.post_data.get("streamer", ""),
            result.post_data.get("games", []),
        )
        code_in_title = False
    elif result.source == SOURCE_STREAM:
        hashtag_list = build_stream_hashtags(
            result.code, result.post_data.get("hashtags", "")
        )
        code_in_title = True
    else:
        hashtag_list = build_hashtags(
            result.code, result.post_data.get("genre", ""), result.year
        )
        code_in_title = True

    tags = [tag.lstrip("#") for tag in hashtag_list]

    try:
        uploaded = publisher.upload_movie(
            title=result.title,
            year=result.year,
            code=result.code,
            clip_paths=result.clip_paths,
            description=description,
            tags=tags,
            code_in_title=code_in_title,
            log=emit,
        )
    except YouTubeError as e:
        raise PipelineError(f"Ошибка YouTube: {e}")

    emit(f"В YouTube загружено клипов: {len(uploaded)}")
    return uploaded