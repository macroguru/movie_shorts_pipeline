"""
Twitch Downloader модуль: скачивание клипов и VOD со Twitch через yt-dlp.

Умеет:
- Скачивать клип по ссылке (clips.twitch.tv или twitch.tv/.../clip/...)
- Скачивать VOD (запись стрима) по ссылке или ссылке на таймкод
- Разбирать ссылку на таймкод: twitch.tv/<канал>/v/<id>?t=1h23m45s
"""

import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import parse_qs, urlparse


class TwitchDownloadError(Exception):
    """Исключение для ошибок скачивания Twitch."""
    pass


def is_clip_url(url: str) -> bool:
    """Ссылка ведёт на клип Twitch (а не на VOD или другой ресурс)."""
    low = (url or "").lower()
    return "clips.twitch.tv" in low or "/clip/" in low


def parse_clip_slug(url: str) -> Optional[str]:
    """
    Извлечение slug клипа из ссылки.

    Понимает clips.twitch.tv/<slug>, clips.twitch.tv/embed?clip=<slug>
    и twitch.tv/<канал>/clip/<slug>.

    Returns:
        Slug или None, если это не ссылка на клип
    """
    url = (url or "").strip()
    match = re.search(r"clips\.twitch\.tv/(?:embed\?clip=)?([\w-]+)", url)
    if match:
        return match.group(1)
    match = re.search(r"twitch\.tv/[\w]+/clip/([\w-]+)", url)
    if match:
        return match.group(1)
    return None


# Публичный GraphQL Twitch. Client-ID веб-клиента периодически меняется —
# сначала пробуем взять актуальный из установленного yt-dlp, иначе дефолт.
GQL_ENDPOINT = "https://gql.twitch.tv/gql"
GQL_CLIENT_ID_DEFAULT = "ue6666qo983tsx6so1t0vnawi233wa"


def _gql_client_id() -> str:
    """Актуальный Client-ID Twitch: из yt-dlp или запасное значение."""
    try:
        from yt_dlp.extractor.twitch import TwitchBaseIE

        class _StubParams:
            params = {}

        ie = TwitchBaseIE.__new__(TwitchBaseIE)
        ie._downloader = _StubParams()
        client_id = ie._CLIENT_ID
        if client_id:
            return client_id
    except Exception:
        pass

    return GQL_CLIENT_ID_DEFAULT


class TwitchDownloader:
    """Скачивание клипов и VOD Twitch через yt-dlp."""

    # Шаблоны ссылок на клипы Twitch
    CLIP_URL_RE = re.compile(
        r"(?:clips\.twitch\.tv/(?:embed\?clip=)?[\w-]+"
        r"|twitch\.tv/[\w]+/clip/[\w-]+)",
        re.IGNORECASE,
    )

    def __init__(self, output_dir: Path):
        """
        Args:
            output_dir: Куда сохранять скачанные файлы
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _check_ytdlp(self):
        """Проверка что yt-dlp доступен (библиотека или модуль)."""
        try:
            import yt_dlp  # noqa: F401
            return "library"
        except ImportError:
            if shutil.which("yt-dlp") or shutil.which("yt-dlp.exe"):
                return "cli"
        raise TwitchDownloadError(
            "yt-dlp не установлен. Выполните: pip install yt-dlp"
        )

    def download_clip(self, url: str, filename_base: str) -> Path:
        """
        Скачивание клипа Twitch.

        Args:
            url: Ссылка на клип (clips.twitch.tv/... или twitch.tv/.../clip/...)
            filename_base: База имени файла (без расширения)

        Returns:
            Путь к скачанному mp4
        """
        url = self._normalize_url(url)

        if not self.CLIP_URL_RE.search(url):
            raise TwitchDownloadError(
                f"Ссылка не похожа на клип Twitch: {url}\n"
                "Ожидается clips.twitch.tv/... или twitch.tv/<канал>/clip/..."
            )

        return self._download(url, filename_base, is_vod=False)

    def download_vod(self, url: str, filename_base: str) -> Path:
        """
        Скачивание VOD (записи стрима).

        Args:
            url: Ссылка на VOD (twitch.tv/<канал>/v/<id> или videos/<id>)
            filename_base: База имени файла (без расширения)

        Returns:
            Путь к скачанному mp4
        """
        url = self._normalize_url(url)

        if not self._is_vod_url(url):
            raise TwitchDownloadError(
                f"Ссылка не похожа на VOD Twitch: {url}\n"
                "Ожидается twitch.tv/<канал>/v/<id> или /videos/<id>"
            )

        return self._download(url, filename_base, is_vod=True)

    def parse_vod_timestamp(self, url: str) -> Optional[float]:
        """
        Извлечение таймкода из ссылки на VOD (?t=1h23m45s).

        Args:
            url: Ссылка на VOD

        Returns:
            Секунды от начала VOD или None, если таймкода нет
        """
        parsed = urlparse(url)
        t_values = parse_qs(parsed.query).get("t")
        if not t_values:
            return None

        return self._parse_duration(t_values[0])

    def fetch_clip_metadata(self, url: str) -> Optional[Dict[str, str]]:
        """
        Стример, игра и название клипа через публичный GraphQL API Twitch.

        Args:
            url: Ссылка на клип

        Returns:
            {"streamer": ..., "game": ..., "title": ...} или None при ошибке
        """
        slug = parse_clip_slug(url)
        if not slug:
            return None

        query = (
            "query ClipMeta($slug: ID!) { clip(slug: $slug) { "
            "title broadcaster { displayName } game { displayName } } }"
        )

        try:
            import httpx

            response = httpx.post(
                GQL_ENDPOINT,
                headers={"Client-Id": _gql_client_id()},
                json={"query": query, "variables": {"slug": slug}},
                timeout=15.0,
            )
            response.raise_for_status()
            clip = (response.json().get("data") or {}).get("clip") or {}
        except Exception:
            return None

        broadcaster = clip.get("broadcaster") or {}
        game = clip.get("game") or {}

        return {
            "streamer": (broadcaster.get("displayName") or "").strip(),
            "game": (game.get("displayName") or "").strip(),
            "title": (clip.get("title") or "").strip(),
        }

    def fetch_title(self, url: str) -> Optional[str]:
        """
        Название клипа/VOD с Twitch без скачивания.

        Args:
            url: Ссылка

        Returns:
            Название или None, если получить не удалось
        """
        url = self._normalize_url(url)

        try:
            mode = self._check_ytdlp()
        except TwitchDownloadError:
            return None

        if mode == "library":
            try:
                import yt_dlp

                opts = {"quiet": True, "no_warnings": True, "noplaylist": True}
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    return (info or {}).get("title")
            except Exception:
                return None

        exe = shutil.which("yt-dlp") or shutil.which("yt-dlp.exe")
        try:
            result = subprocess.run(
                [exe, "--get-title", "--no-playlist", url],
                capture_output=True,
                text=True,
                check=True,
                encoding="utf-8",
            )
            lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            return lines[-1] if lines else None
        except (subprocess.CalledProcessError, OSError):
            return None

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Добавление https:// если ссылка указана без схемы."""
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        return url

    @staticmethod
    def _is_vod_url(url: str) -> bool:
        """Проверка, что ссылка ведёт на VOD Twitch."""
        parsed = urlparse(url)
        path = parsed.path.lower()
        return ("twitch.tv" in parsed.netloc.lower()
                and ("/videos/" in path or "/v/" in path or re.search(r"/v\d+", path)))

    @staticmethod
    def _parse_duration(text: str) -> Optional[float]:
        """
        Разбор строки длительности Twitch: 1h2m3s, 45m, 30s, 90.

        Returns:
            Секунды или None
        """
        text = text.strip().lower()
        match = re.fullmatch(
            r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s?)?", text
        )
        if not match or not any(match.groups()):
            return None

        hours, minutes, seconds = (int(g) if g else 0 for g in match.groups())
        return hours * 3600 + minutes * 60 + seconds

    def _download(self, url: str, filename_base: str, is_vod: bool) -> Path:
        """
        Скачивание через yt-dlp с ограничением формата mp4.

        Args:
            url: Ссылка
            filename_base: База имени выходного файла
            is_vod: True для VOD (проверяем, что файл не гигантский)

        Returns:
            Путь к скачанному файлу
        """
        mode = self._check_ytdlp()
        safe_name = "".join(
            c if c not in '<>:"/\\|?*' else "_" for c in filename_base
        ).strip(" .") or "twitch"
        output_template = str(self.output_dir / f"{safe_name}.%(ext)s")

        if mode == "library":
            output_path = self._download_with_library(url, output_template)
        else:
            output_path = self._download_with_cli(url, output_template)

        if output_path is None or not output_path.exists():
            raise TwitchDownloadError(f"yt-dlp не скачал файл: {url}")

        if is_vod:
            size_gb = output_path.stat().st_size / 1024**3
            if size_gb > 10:
                raise TwitchDownloadError(
                    f"VOD слишком большой ({size_gb:.1f} ГБ). "
                    "Скачайте его вручную и обработайте как локальный файл."
                )

        return output_path

    def _download_with_library(self, url: str, output_template: str) -> Optional[Path]:
        """Скачивание через Python-API yt-dlp."""
        import yt_dlp

        opts = {
            "outtmpl": output_template,
            "format": "best[ext=mp4]/best",
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
        }

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                # Скачанный файл — по запрошенной базе, расширение могло отличаться
                requested = ydl.prepare_filename(info)
                path = Path(requested)
                if path.exists():
                    return path
                # yt-dlp мог подставить другой экстент
                candidates = list(path.parent.glob(path.stem + ".*"))
                return candidates[0] if candidates else None
        except Exception as e:
            raise TwitchDownloadError(f"Ошибка yt-dlp: {e}")

    def _download_with_cli(self, url: str, output_template: str) -> Optional[Path]:
        """Скачивание через консольный yt-dlp."""
        exe = shutil.which("yt-dlp") or shutil.which("yt-dlp.exe")

        cmd = [
            exe,
            "--no-playlist",
            "-f", "best[ext=mp4]/best",
            "-o", output_template,
            url,
        ]

        try:
            subprocess.run(
                cmd, capture_output=True, text=True, check=True, encoding="utf-8"
            )
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or "").strip().splitlines()
            detail = stderr[-1] if stderr else f"exit code {e.returncode}"
            raise TwitchDownloadError(f"Ошибка yt-dlp: {detail}")

        # Ищем скачанный файл по шаблону (расширение могло отличаться от mp4)
        base = Path(output_template.replace(".%(ext)s", ""))
        candidates = sorted(base.parent.glob(base.name + ".*"))
        return candidates[0] if candidates else None
