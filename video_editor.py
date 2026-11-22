"""
Video Editor модуль для обработки видео через FFmpeg.
Создает вертикальные клипы 9:16 с размытым фоном и оверлеем кода.
"""

import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional

from config import Config


class VideoEditorError(Exception):
    """Исключение для ошибок видео редактора."""
    pass


class VideoEditor:
    """Класс для обработки видео через FFmpeg."""

    # NVENC: кодирование на GPU (RTX и другие карты NVIDIA)
    ENCODER_NVENC = [
        "-c:v", "h264_nvenc",
        "-preset", "p5",
        "-rc", "vbr",
        "-cq", "23",
        "-b:v", "0",
    ]

    # CPU: libx264 как резерв, если NVENC недоступен или упал
    ENCODER_CPU = [
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
    ]

    def __init__(self, config: Config):
        """
        Инициализация видео редактора.

        Args:
            config: Конфигурация проекта
        """
        self.config = config
        self._nvenc_available = None

    def _nvenc_supported(self) -> bool:
        """Есть ли в FFmpeg кодировщик h264_nvenc (проверяется один раз)."""
        if self._nvenc_available is None:
            try:
                result = subprocess.run(
                    ["ffmpeg", "-hide_banner", "-encoders"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                self._nvenc_available = "h264_nvenc" in result.stdout
            except (subprocess.CalledProcessError, OSError):
                self._nvenc_available = False

        return self._nvenc_available

    def _encoder_args(self) -> list:
        """
        Выбор кодировщика по настройке VIDEO_ENCODER.

        auto — NVENC, если он есть в FFmpeg; nvenc — только GPU;
        cpu — принудительный libx264.
        """
        mode = getattr(self.config, "video_encoder", "auto")

        if mode in ("cpu", "x264"):
            return self.ENCODER_CPU
        if mode in ("auto", "nvenc", "gpu") and self._nvenc_supported():
            return self.ENCODER_NVENC
        return self.ENCODER_CPU

    def render_short(
        self,
        video_path: str,
        start: str,
        end: str,
        code: str,
        clip_index: int,
        title_text: str = "Название в канале",
        min_duration: float = 40.0,
        with_overlay: bool = True
    ) -> Dict[str, Any]:
        """
        Рендер вертикального короткого клипа.

        Args:
            video_path: Путь к исходному видео
            start: Время начала клипа (HH:MM:SS)
            end: Время конца клипа (HH:MM:SS)
            code: Код фильма для оверлея
            clip_index: Индекс клипа
            title_text: Текст названия для оверлея
            min_duration: Минимальная длительность в секундах
                (для фильмов 40, для стримов 0 — режем как указано)
            with_overlay: Рисовать оверлей «шапка профиля + код»
                (для клипов Twitch — False, чистая картинка)

        Returns:
            Словарь с путями к выходным файлам
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise VideoEditorError(f"Видеофайл не найден: {video_path}")

        # Пути для выхода
        output_clip = self.config.get_clip_output_path(code, clip_index)
        output_poster = self.config.get_poster_path(code)

        # Конвертируем время в секунды для FFmpeg
        start_seconds = self._time_to_seconds(start)
        end_seconds = self._time_to_seconds(end)
        duration = end_seconds - start_seconds

        # Минимальная длительность (для фильмов 40с, для стримов 0 — как указано)
        if duration < min_duration:
            duration = min_duration

        print(f"Рендер клипа {clip_index}: {start} - {end} (длительность: {duration}s)")

        # Создаем сложный фильтр для вертикального формата
        complex_filter = self._build_complex_filter(title_text, code, with_overlay=with_overlay)

        try:
            try:
                output_clip = self._run_render(
                    video_path, start_seconds, duration,
                    complex_filter, output_clip
                )
            except subprocess.CalledProcessError:
                # NVENC может упасть из-за драйвера или занятой сессии —
                # тогда один раз пробуем тем же фильтром на CPU
                if self._encoder_args() == self.ENCODER_CPU:
                    raise
                print(f"NVENC не сработал для клипа {clip_index}, пробуем CPU...")
                subprocess.run(
                    self._build_render_cmd(video_path, start_seconds, duration,
                                           complex_filter, output_clip,
                                           encoder=self.ENCODER_CPU),
                    capture_output=True,
                    text=True,
                    check=True
                )

            if not output_clip.exists():
                raise VideoEditorError("FFmpeg не создал выходной файл")

            print(f"Клип создан: {output_clip}")

            # Извлекаем постер из середины клипа
            poster_time = duration / 2
            self._extract_poster(video_path, start_seconds + poster_time, output_poster)

            return {
                "clip_path": str(output_clip),
                "poster_path": str(output_poster),
                "duration": duration
            }

        except subprocess.CalledProcessError as e:
            raise VideoEditorError(f"Ошибка FFmpeg: {e.stderr}")

    def _build_render_cmd(
        self,
        video_path: Path,
        start_seconds: float,
        duration: float,
        complex_filter: str,
        output_clip: Path,
        encoder: Optional[list] = None
    ) -> list:
        """Сборка FFmpeg-команды рендера с выбранным кодировщиком."""
        return [
            "ffmpeg",
            "-ss", str(start_seconds),
            "-t", str(duration),
            "-i", str(video_path),
            "-filter_complex", complex_filter,
            "-map", "[video]",
            "-map", "0:a:0",
            *(encoder or self._encoder_args()),
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            "-y",
            str(output_clip)
        ]

    def _run_render(self, video_path, start_seconds, duration, complex_filter, output_clip) -> Path:
        """Один прогон FFmpeg с текущим кодировщиком."""
        cmd = self._build_render_cmd(video_path, start_seconds, duration, complex_filter, output_clip)

        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        return output_clip

    def _build_complex_filter(self, title_text: str, code: str, with_overlay: bool = True) -> str:
        """
        Построение сложного фильтра FFmpeg.

        Создает:
        - Фоновый слой: масштабирование + размытие
        - Передний слой: оригинальное видео 16:9 по центру
        - Текстовый оверлей: код фильма (если with_overlay)

        Args:
            title_text: Текст названия
            code: Код фильма
            with_overlay: Рисовать ли оверлей «шапка профиля + код»

        Returns:
            Строка фильтра для FFmpeg
        """
        # Фильтр для вертикального формата 1080x1920
        # Для ultra-wide источников (например 1024x426) сначала масштабируем по высоте до 1920,
        # затем кропим центр до 1080x1920
        filter_parts = [
            # Размытый фон: масштабируем по высоте до 1920, затем кропим центр 1080x1920
            # gblur вместо boxblur: картинка та же, но считается в разы быстрее
            "[0:v]scale=-1:1920:force_original_aspect_ratio=1,crop=1080:1920,gblur=sigma=20[bg]",

            # Передний план: масштабируем по высоте до 540 (чтобы влезло в 1080x1920 с запасом)
            "[0:v]scale=-1:540:force_original_aspect_ratio=1[fg]",

            # Накладываем передний план на фон по центру
            "[bg][fg]overlay=(W-w)/2:(H-h)/2[centered]",
        ]

        if with_overlay:
            # Добавляем текстовый оверлей с кодом
            filter_parts.append(self._build_text_filter(code, title_text))
        else:
            # Без оверлея (клипы Twitch): картинка без текста
            filter_parts.append("[centered]null[video]")

        return ";".join(filter_parts)

    # Первая строка оверлея: название фильма зритель ищет в шапке профиля
    OVERLAY_HEADER = "Название в шапке профиля"

    # Полный путь к шрифту Windows (двоичное экранирование для фильтра FFmpeg)
    FONT_FILE = "C\\\\:/Windows/Fonts/arial.ttf"

    def _build_text_filter(self, code: str, title_text: str) -> str:
        """
        Построение текстового фильтра для оверлея.

        Рендерит две строки двумя отдельными drawtext (надёжнее, чем \n внутри
        одного text=, который теряется при многоуровневом экранировании фильтра):
        - "Название в шапке профиля"
        - код фильма

        Args:
            code: Код фильма
            title_text: Текст названия (в оверлее не выводится)

        Returns:
            Строка текстовых фильтров, склеенная через ;
        """
        # Экранируем специальные символы для FFmpeg drawtext
        header_escaped = self.OVERLAY_HEADER.replace("\\", "\\\\").replace(":", "\\:")
        code_escaped = str(code).replace("\\", "\\\\").replace(":", "\\:")

        common = (
            f"fontcolor=white:"
            f"borderw=2:"
            f"bordercolor=black:"
            f"box=1:"
            f"boxcolor=black@0.7:"
            f"boxborderw=14:"
            f"fontfile={self.FONT_FILE}"
        )

        return (
            f"[centered]drawtext="
            f"text='{header_escaped}':"
            f"fontsize=40:"
            f"x=(w-text_w)/2:"
            f"y=h*0.09:"
            f"{common}"
            f"[txt1];"
            f"[txt1]drawtext="
            f"text='{code_escaped}':"
            f"fontsize=96:"
            f"x=(w-text_w)/2:"
            f"y=h*0.09+90:"
            f"{common}"
            f"[video]"
        )

    def _extract_poster(
        self,
        video_path: Path,
        timestamp: float,
        output_path: Path
    ):
        """
        Извлечение кадра из видео для постера.

        Args:
            video_path: Путь к видео
            timestamp: Время кадра в секундах
            output_path: Путь для выходного изображения
        """
        try:
            cmd = [
                "ffmpeg",
                "-ss", str(timestamp),
                "-i", str(video_path),
                "-vframes", "1",
                "-q:v", "2",
                "-y",
                str(output_path)
            ]

            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            print(f"Постер создан: {output_path}")

        except subprocess.CalledProcessError as e:
            print(f"Предупреждение: не удалось создать постер: {e}")

    def _time_to_seconds(self, time_str: str) -> float:
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

    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Получение информации о видео через ffprobe.

        Args:
            video_path: Путь к видеофайлу

        Returns:
            Словарь с информацией о видео
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise VideoEditorError(f"Видеофайл не найден: {video_path}")

        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(video_path)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            data = json.loads(result.stdout)

            # Извлекаем полезную информацию
            format_info = data.get("format", {})
            video_streams = [
                s for s in data.get("streams", [])
                if s.get("codec_type") == "video"
            ]
            audio_streams = [
                s for s in data.get("streams", [])
                if s.get("codec_type") == "audio"
            ]

            return {
                "duration": float(format_info.get("duration", 0)),
                "size": int(format_info.get("size", 0)),
                "bitrate": int(format_info.get("bit_rate", 0)),
                "video_codec": video_streams[0].get("codec_name") if video_streams else None,
                "video_width": video_streams[0].get("width") if video_streams else None,
                "video_height": video_streams[0].get("height") if video_streams else None,
                "audio_codec": audio_streams[0].get("codec_name") if audio_streams else None,
                "has_subtitles": any(
                    s.get("codec_type") == "subtitle"
                    for s in data.get("streams", [])
                )
            }

        except subprocess.CalledProcessError as e:
            raise VideoEditorError(f"Ошибка ffprobe: {e.stderr}")
        except json.JSONDecodeError as e:
            raise VideoEditorError(f"Ошибка парсинга ffprobe вывода: {e}")
