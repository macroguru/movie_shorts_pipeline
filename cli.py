#!/usr/bin/env python3
"""
Movie Shorts Pipeline CLI
Нарезка Shorts/Reels/TikTok из фильмов и стримов с публикацией в Telegram.

Режимы:
  movie  — фильм из локального файла (субтитры → Gemini → клипы);
  stream — запись стрима: AI ищет моменты, либо моменты задаются вручную;
  twitch — скачивание клипов Twitch по ссылкам и конвертация в вертикальный формат;
  clip   — любой источник (локальный файл или ссылка на видео: Twitch, YouTube и др.)
           → вертикальный шортс.

Usage:
    # Фильм (режим по умолчанию)
    python cli.py --input movie.mkv --title "Название" --year 2024 --clips 3 --publish

    # Стрим: AI сам найдёт яркие моменты
    python cli.py --source stream --input stream.mp4 --title "Стрим" --clips 5

    # Стрим: моменты заданы вручную
    python cli.py -s stream -i stream.mp4 -t "Стрим" --moments "1:23:45-1:25:00, 2:00:10-2:01:00"

    # Клипы Twitch пачкой → вертикальные шортсы
    python cli.py --twitch https://clips.twitch.tv/Abc --twitch https://clips.twitch.tv/Def

    # VOD Twitch: скачать и нарезать
    python cli.py --twitch https://www.twitch.tv/videos/123456789 -t "Стрим" -c 5

    # Клип из любого видео по ссылке (yt-dlp скачает сам)
    python cli.py --url https://www.youtube.com/watch?v=AbCdEf -t "Клип"

    # Клип из локального файла с настройками монтажа
    python cli.py -s clip -i fragment.mp4 -t "Клип" --fps 60 --subtitle-style boxed --preset clip
"""

import argparse
import logging
import sys
from pathlib import Path

from tqdm import tqdm

from config import Config, ConfigError
from core import (
    PipelineError,
    download_twitch_source,
    parse_moments,
    post_link,
    process_clip,
    process_stream,
    process_twitch_clips,
    process_video,
    publish_result,
)
from twitch_downloader import is_clip_url


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Нарезка Shorts/Reels из фильмов и стримов с публикацией в Telegram",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python cli.py --input movie.mkv --title "Фильм" --year 2024 --clips 5 --publish
  python cli.py --source stream --input stream.mp4 -t "Стрим" --clips 5
  python cli.py -s stream -i stream.mp4 -t "Стрим" --moments "1:23:45-1:25:00, 2:00:10-2:01:00"
  python cli.py --twitch https://clips.twitch.tv/Abc --twitch https://clips.twitch.tv/Def
  python cli.py --twitch https://www.twitch.tv/videos/123456789 -t "Стрим" -c 5
  python cli.py --url https://www.youtube.com/watch?v=AbCdEf -t "Клип"
  python cli.py -s clip -i fragment.mp4 -t "Клип" --fps 60 --subtitle-style boxed --preset clip
  python cli.py -s stream -i stream.mp4 -t "Стрим" --min-duration 15 --max-duration 45
        """
    )

    parser.add_argument(
        "--input", "-i",
        default=None,
        help="Путь к видеофайлу (mkv/mp4); не нужен при --twitch"
    )

    parser.add_argument(
        "--title", "-t",
        default=None,
        help="Название фильма или стрима (для клипов Twitch определяется само)"
    )

    parser.add_argument(
        "--year", "-y",
        type=int,
        default=None,
        help="Год выпуска (опционально)"
    )

    parser.add_argument(
        "--clips", "-c",
        type=int,
        default=3,
        help="Количество клипов/моментов (по умолчанию 3)"
    )

    parser.add_argument(
        "--source", "-s",
        choices=["movie", "stream", "clip"],
        default="movie",
        help="Тип источника: movie — фильм, stream — запись стрима, "
             "clip — файл или ссылка на любое видео (по умолчанию movie)"
    )

    parser.add_argument(
        "--moments",
        default=None,
        help='Моменты стрима вручную: "ЧЧ:ММ:СС-ЧЧ:ММ:СС, ..." (без них моменты ищет AI)'
    )

    parser.add_argument(
        "--twitch",
        action="append",
        default=None,
        help="Ссылка на клип или VOD Twitch (можно указать несколько раз)"
    )

    parser.add_argument(
        "--url",
        action="append",
        default=None,
        help="Ссылка на любое видео (Twitch, YouTube и др.) — режим clip, "
             "обрабатывается через process_clip; можно указать несколько раз"
    )

    parser.add_argument(
        "--fps",
        choices=["30", "60"],
        default=None,
        help="Частота кадров выхода: 30 или 60 (по умолчанию — из .env)"
    )

    parser.add_argument(
        "--subtitle-style",
        choices=["bold", "boxed", "karaoke"],
        default=None,
        help="Стиль субтитров: bold, boxed или karaoke (по умолчанию — из .env)"
    )

    parser.add_argument(
        "--preset",
        choices=["auto", "movie", "stream", "clip", "podcast", "gaming", "off"],
        default=None,
        help="Пресет монтажа: auto, movie, stream, clip, podcast, gaming, off "
             "(по умолчанию — из .env)"
    )

    parser.add_argument(
        "--min-duration",
        type=float,
        default=None,
        help="Минимальная длительность момента в секундах для stream/clip "
             "(по умолчанию — из .env)"
    )

    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Максимальная длительность момента в секундах для stream/clip "
             "(по умолчанию — из .env)"
    )

    parser.add_argument(
        "--publish", "-p",
        action="store_true",
        help="Опубликовать посты в Telegram-канал"
    )

    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Путь к .env файлу (по умолчанию .env в директории проекта)"
    )

    return parser.parse_args()


def make_progress():
    """Прогресс-бар рендера клипов на tqdm."""
    bar = tqdm(total=1, desc="Рендер клипов", unit="клип")

    def callback(done: int, total: int):
        bar.total = max(total, 1)
        bar.n = done
        bar.refresh()

    return callback, bar


def cli_log(message: str):
    """Лог пайплайна в stdout."""
    logger.info(str(message))


def render_kwargs(args) -> dict:
    """Настройки монтажа из CLI: None означает «взять из .env»."""
    return {
        "fps": int(args.fps) if args.fps else None,
        "subtitle_style": args.subtitle_style,
        "preset": args.preset,
    }


def finish(args, result) -> int:
    """Сводка по результату и публикация в Telegram, если запрошена."""
    logger.info("=" * 50)
    logger.info("✅ Обработка завершена!")
    logger.info("=" * 50)
    logger.info(f"Код: {result.code}")
    logger.info(f"Создано клипов: {len(result.clip_paths)}")
    logger.info(f"Постер: {result.poster_path}")
    logger.info(f"Описание для публикации: {result.description_path}")
    logger.info(f"Папка: {result.movie_dir}")

    if args.publish:
        logger.info("=" * 50)
        logger.info("Публикация в Telegram...")

        message_id = publish_result(result, env_path=args.env, log=cli_log)

        logger.info(f"Пост опубликован! Message ID: {message_id}")
        link = post_link(result)
        if link:
            logger.info(f"Просмотреть: {link}")
        return 0

    logger.info("\n📝 Описание для площадок:")
    logger.info("-" * 50)
    logger.info(result.description)
    logger.info("-" * 50)
    logger.info("\nДля публикации добавьте флаг --publish")
    return 0


def run_movie(args) -> int:
    """Режим фильма: субтитры → Gemini ищет вирусные сцены → рендер."""
    if not args.title:
        logger.error("Режим movie требует --title (название фильма)")
        return 1

    logger.info(f"Обработка файла: {args.input}")
    logger.info(f"Название: {args.title}, Год: {args.year}, Клипов: {args.clips}")

    progress, bar = make_progress()
    try:
        result = process_video(
            input_path=args.input,
            title=args.title,
            clips_count=args.clips,
            year=args.year,
            env_path=args.env,
            log=cli_log,
            progress=progress,
            **render_kwargs(args),
        )
    finally:
        bar.close()

    return finish(args, result)


def run_stream(args) -> int:
    """Режим стрима: AI ищет моменты, либо моменты заданы вручную."""
    if not args.title:
        logger.error("Режим stream требует --title (название стрима)")
        return 1

    # Проверяем формат моментов до долгих этапов
    if args.moments:
        parse_moments(args.moments)

    logger.info(f"Обработка записи стрима: {args.input}")
    logger.info(f"Название: {args.title}, Клипов: {args.clips}, Моменты: {args.moments or 'ищет AI'}")

    progress, bar = make_progress()
    try:
        result = process_stream(
            input_path=args.input,
            title=args.title,
            clips_count=args.clips,
            moments=args.moments,
            year=args.year,
            env_path=args.env,
            log=cli_log,
            progress=progress,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            **render_kwargs(args),
        )
    finally:
        bar.close()

    return finish(args, result)


def run_clip(args) -> int:
    """Режим clip: локальный файл или ссылка на любое видео → вертикальный шортс."""
    sources = []
    if args.input:
        sources.append(args.input)
    if args.url:
        sources.extend(args.url)

    if not sources:
        logger.error("Режим clip требует --input (видеофайл) или --url (ссылка на видео)")
        return 1

    # Локальные пути проверяем сразу; ссылки проверит пайплайн при скачивании
    for source_item in sources:
        if not source_item.lower().startswith(("http://", "https://")) \
                and not Path(source_item).exists():
            logger.error(f"Видеофайл не найден: {source_item}")
            return 1

    exit_code = 0
    for source_item in sources:
        logger.info(f"Обработка клипа: {source_item}")
        logger.info(f"Название: {args.title or 'из заголовка видео'}, Клипов: {args.clips}")

        progress, bar = make_progress()
        try:
            result = process_clip(
                source_path_or_url=source_item,
                title=args.title,
                clips_count=args.clips,
                env_path=args.env,
                log=cli_log,
                progress=progress,
                min_duration=args.min_duration,
                max_duration=args.max_duration,
                **render_kwargs(args),
            )
        finally:
            bar.close()

        code = finish(args, result)
        if code != 0:
            exit_code = code

    return exit_code


def run_twitch(args) -> int:
    """Контент Twitch: клипы пачкой под одним кодом, VOD — как стрим."""
    exit_code = 0

    clip_urls = [u for u in args.twitch if is_clip_url(u)]
    vod_urls = [u for u in args.twitch if not is_clip_url(u)]

    # Клипы: скачать и конвертировать в вертикальные шортсы
    if clip_urls:
        logger.info(f"Клипов Twitch к обработке: {len(clip_urls)}")

        progress, bar = make_progress()
        try:
            result = process_twitch_clips(
                urls=clip_urls,
                title=args.title,
                year=args.year,
                env_path=args.env,
                log=cli_log,
                progress=progress,
            )
        finally:
            bar.close()

        code = finish(args, result)
        if code != 0:
            exit_code = code

    # VOD: скачать запись и нарезать как стрим
    for url in vod_urls:
        logger.info(f"VOD Twitch: {url}")

        path, _kind, url_title = download_twitch_source(url, env_path=args.env, log=cli_log)

        title = (args.title or url_title or "").strip()
        if not title:
            logger.error("Не удалось определить название — укажите --title")
            exit_code = 1
            continue

        progress, bar = make_progress()
        try:
            result = process_stream(
                input_path=path,
                title=title,
                clips_count=args.clips,
                moments=args.moments,
                year=args.year,
                env_path=args.env,
                log=cli_log,
                progress=progress,
                min_duration=args.min_duration,
                max_duration=args.max_duration,
                **render_kwargs(args),
            )
        finally:
            bar.close()

        code = finish(args, result)
        if code != 0:
            exit_code = code

    return exit_code


def main() -> int:
    """Основная функция CLI."""
    args = parse_args()

    # Инициализация конфигурации (ранняя валидация ключей и FFmpeg)
    try:
        config = Config(env_path=args.env)
        logger.info(f"Конфигурация загружена: {config}")
    except ConfigError as e:
        logger.error(f"Ошибка конфигурации: {e}")
        return 1

    try:
        if args.url:
            return run_clip(args)

        if args.twitch:
            return run_twitch(args)

        if not args.input:
            logger.error("Укажите --input (видеофайл), --url (ссылка на видео) "
                         "или --twitch (ссылки Twitch)")
            return 1

        if not Path(args.input).exists():
            logger.error(f"Видеофайл не найден: {args.input}")
            return 1

        if args.source == "stream":
            return run_stream(args)

        if args.source == "clip":
            return run_clip(args)

        return run_movie(args)

    except PipelineError as e:
        logger.error(f"Ошибка: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
