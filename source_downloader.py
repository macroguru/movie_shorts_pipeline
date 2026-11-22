"""
Универсальное скачивание видео по ссылке через yt-dlp.

Поддерживает всё, что умеет yt-dlp: Twitch, YouTube, VK, RuTube и т.д.
Используется режимом «Клип» и любым URL-источником.
"""

import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


class SourceDownloadError(Exception):
    """Ошибка скачивания видео по ссылке."""
    pass


def is_url(source: str) -> bool:
    """Строка выглядит как http(s)-ссылка."""
    return (source or "").strip().lower().startswith(("http://", "https://", "www."))


def download_video(
    url: str,
    output_dir: Path,
    log=None,
) -> Tuple[Path, Optional[str]]:
    """
    Скачивание видео по любой ссылке через yt-dlp.

    Args:
        url: Ссылка на видео (YouTube, Twitch, VK и другие источники yt-dlp)
        output_dir: Куда сохранять файл
        log: Колбэк для сообщений о прогрессе

    Returns:
        (путь к файлу, название видео или None)

    Raises:
        SourceDownloadError: если yt-dlp не установлен или скачивание не удалось
    """
    emit = log or (lambda message: None)
    url = url.strip()
    if url.lower().startswith("www."):
        url = "https://" + url

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = f"url_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_template = str(output_dir / f"{base}.%(ext)s")

    # yt-dlp как библиотека
    try:
        import yt_dlp  # noqa: F401
        mode = "library"
    except ImportError:
        mode = "cli"

    if mode == "library":
        path, title = _download_library(url, output_template, emit)
    else:
        exe = shutil.which("yt-dlp") or shutil.which("yt-dlp.exe")
        if not exe:
            raise SourceDownloadError("yt-dlp не установлен. Выполните: pip install yt-dlp")
        path, title = _download_cli(exe, url, output_template, emit)

    if path is None or not path.exists():
        raise SourceDownloadError(f"yt-dlp не скачал файл: {url}")

    emit(f"Скачано: {path}")
    return path, title


def _download_library(url: str, output_template: str, emit) -> Tuple[Optional[Path], Optional[str]]:
    """Скачивание через Python-API yt-dlp."""
    import yt_dlp

    opts = {
        "outtmpl": output_template,
        # Не гнаться за 4K: для вертикальных шортсов хватает 1080p,
        # меньше нагрузка и быстрее
        "format": "best[height<=1080]/best",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            requested = Path(ydl.prepare_filename(info))
            title = (info or {}).get("title")
    except Exception as e:
        raise SourceDownloadError(f"Ошибка yt-dlp: {e}")

    if requested.exists():
        return requested, title

    candidates = sorted(requested.parent.glob(requested.stem + ".*"))
    return (candidates[0], title) if candidates else (None, title)


def _download_cli(exe: str, url: str, output_template: str, emit) -> Tuple[Optional[Path], Optional[str]]:
    """Скачивание через консольный yt-dlp."""
    cmd = [
        exe,
        "--no-playlist",
        "-f", "best[height<=1080]/best",
        "-o", output_template,
        "--print", "after_move:%(title)s",
        url,
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, encoding="utf-8"
        )
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip().splitlines()
        detail = stderr[-1] if stderr else f"exit code {e.returncode}"
        raise SourceDownloadError(f"Ошибка yt-dlp: {detail}")

    title = None
    lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    if lines:
        title = lines[-1]

    base = Path(output_template.replace(".%(ext)s", ""))
    candidates = sorted(base.parent.glob(base.name + ".*"))
    return (candidates[0] if candidates else None), title
