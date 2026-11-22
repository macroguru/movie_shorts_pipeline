"""
Модуль извлечения и генерации субтитров.
Поддерживает извлечение встроенных субтитров и транскрибацию через Whisper.
"""

import subprocess
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from config import Config


@dataclass
class SubtitleEntry:
    """Элемент субтитров."""
    index: int
    start: str  # Формат HH:MM:SS.mmm
    end: str    # Формат HH:MM:SS.mmm
    text: str


class SubtitleError(Exception):
    """Исключение для ошибок работы с субтитрами."""
    pass


class Subtitles:
    """Класс для работы с субтитрами."""

    def __init__(self, config: Config):
        """
        Инициализация модуля субтитров.

        Args:
            config: Конфигурация проекта
        """
        self.config = config
        self.temp_dir = config.temp_dir
        self.subs_path = config.temp_subs_path

    def extract_or_transcribe(self, video_path: str) -> str:
        """
        Извлечение субтитров или транскрибация аудио.

        Сначала проверяет наличие встроенных субтитров через ffprobe.
        Если найдена русская дорожка — извлекает её.
        Иначе транскрибирует аудио через Whisper.

        Args:
            video_path: Путь к видеофайлу

        Returns:
            Путь к файлу с субтитрами (.srt)
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise SubtitleError(f"Видеофайл не найден: {video_path}")

        # Проверяем наличие встроенных субтитров
        embedded_subs = self._check_embedded_subtitles(video_path)

        if embedded_subs:
            print(f"Найдены встроенные субтитры: индекс {embedded_subs['index']}")
            return self._extract_embedded_subs(video_path, embedded_subs["index"])

        # Транскрибация через Whisper
        print("Встроенные субтитры не найдены. Запускаю транскрибацию через Whisper...")
        return self._transcribe_with_whisper(video_path)

    def _check_embedded_subtitles(self, video_path: Path) -> Optional[Dict[str, Any]]:
        """
        Проверка наличия встроенных субтитров.

        Args:
            video_path: Путь к видеофайлу

        Returns:
            Информация о субтитрах или None
        """
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "quiet",
                    "-print_format", "json",
                    "-show_streams",
                    str(video_path)
                ],
                capture_output=True,
                text=True,
                check=True
            )

            data = json.loads(result.stdout)

            for stream in data.get("streams", []):
                codec_type = stream.get("codec_type", "")
                codec_name = stream.get("codec_name", "")

                # Проверяем субтитры (subrip, ass, mov_text и т.д.)
                if codec_type == "subtitle":
                    # Проверяем язык (предпочитаем русский)
                    tags = stream.get("tags", {})
                    language = tags.get("language", tags.get("LANGUAGE", "")).lower()

                    # Возвращаем первый русский или любой найденный
                    if "ru" in language or "rus" in language:
                        return {
                            "index": stream.get("index"),
                            "codec": codec_name,
                            "language": language
                        }

                    # Запоминаем первый найденный как фоллбэк
                    if "fallback" not in locals():
                        fallback = {
                            "index": stream.get("index"),
                            "codec": codec_name,
                            "language": language
                        }

            return fallback if "fallback" in locals() else None

        except subprocess.CalledProcessError as e:
            print(f"Ошибка ffprobe: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Ошибка парсинга JSON от ffprobe: {e}")
            return None

    def _extract_embedded_subs(self, video_path: Path, stream_index: int) -> str:
        """
        Извлечение встроенных субтитров.

        Args:
            video_path: Путь к видеофайлу
            stream_index: Индекс потока субтитров

        Returns:
            Путь к файлу с субтитрами
        """
        output_path = self.subs_path

        try:
            # Пытаемся извлечь как SRT
            subprocess.run(
                [
                    "ffmpeg",
                    "-i", str(video_path),
                    "-map", f"0:{stream_index}",
                    "-c:s", "srt",
                    "-y",
                    str(output_path)
                ],
                capture_output=True,
                text=True,
                check=True
            )

            if output_path.exists() and output_path.stat().st_size > 0:
                print(f"Субтитры извлечены: {output_path}")
                return str(output_path)

        except subprocess.CalledProcessError as e:
            print(f"Ошибка извлечения субтитров: {e}")
            # Пробуем без конвертации
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-i", str(video_path),
                        "-map", f"0:{stream_index}",
                        "-y",
                        str(output_path)
                    ],
                    capture_output=True,
                    text=True,
                    check=True
                )

                if output_path.exists():
                    print(f"Субтитры извлечены (без конвертации): {output_path}")
                    return str(output_path)

            except subprocess.CalledProcessError as e2:
                print(f"Ошибка извлечения без конвертации: {e2}")

        # Фоллбэк на транскрибацию
        print("Не удалось извлечь субтитры, запускаю транскрибацию...")
        return self._transcribe_with_whisper(video_path)

    def _transcribe_with_whisper(self, video_path: Path) -> str:
        """
        Транскрибация аудио через faster-whisper.

        Args:
            video_path: Путь к видеофайлу

        Returns:
            Путь к файлу с субтитрами
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise SubtitleError(
                "faster-whisper не установлен. Выполните: pip install faster-whisper"
            )

        # Извлекаем аудио во временный файл
        audio_path = self.temp_dir / "audio_temp.mp3"

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-i", str(video_path),
                    "-vn",
                    "-acodec", "libmp3lame",
                    "-ab", "64k",
                    "-ac", "1",
                    "-y",
                    str(audio_path)
                ],
                capture_output=True,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            raise SubtitleError(f"Ошибка извлечения аудио: {e}")

        # Загружаем модель Whisper
        print(f"Загрузка модели Whisper '{self.config.whisper_model}' для устройства {self.config.whisper_device}...")

        try:
            model = WhisperModel(
                self.config.whisper_model,
                device=self.config.whisper_device,
                compute_type="int8" if self.config.whisper_device == "cpu" else "float16"
            )
        except Exception as e:
            raise SubtitleError(f"Ошибка загрузки модели Whisper: {e}")

        # Транскрибация
        print("Начинаю транскрибацию...")

        segments, info = model.transcribe(
            str(audio_path),
            language="ru",  # Явно указываем русский
            word_timestamps=False
        )

        # Конвертация в SRT формат
        srt_content = self._segments_to_srt(segments)

        # Сохраняем SRT файл
        self.subs_path.write_text(srt_content, encoding="utf-8")

        # Очищаем временные файлы
        if audio_path.exists():
            audio_path.unlink()

        print(f"Транскрибация завершена. Субтитры сохранены: {self.subs_path}")
        return str(self.subs_path)

    def _segments_to_srt(self, segments) -> str:
        """
        Конвертация сегментов Whisper в SRT формат.

        Args:
            segments: Итератор сегментов от Whisper

        Returns:
            Строка в формате SRT
        """
        srt_lines = []

        for i, segment in enumerate(segments, start=1):
            start = self._format_timestamp(segment.start)
            end = self._format_timestamp(segment.end)
            text = segment.text.strip()

            srt_lines.append(f"{i}")
            srt_lines.append(f"{start} --> {end}")
            srt_lines.append(f"{text}")
            srt_lines.append("")

        return "\n".join(srt_lines)

    def _format_timestamp(self, seconds: float) -> str:
        """
        Конвертация секунд в SRT timestamp.

        Args:
            seconds: Время в секундах

        Returns:
            Строка в формате HH:MM:SS.mmm
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def parse_srt(self, srt_path: str = None) -> List[SubtitleEntry]:
        """
        Парсинг SRT файла.

        Args:
            srt_path: Путь к SRT файлу (по умолчанию использует temp/subs.srt)

        Returns:
            Список элементов субтитров
        """
        if srt_path is None:
            srt_path = self.subs_path

        srt_path = Path(srt_path)

        if not srt_path.exists():
            raise SubtitleError(f"SRT файл не найден: {srt_path}")

        content = srt_path.read_text(encoding="utf-8")
        entries = []

        # Разделяем на блоки по пустым строкам
        blocks = content.strip().split("\n\n")

        for block in blocks:
            lines = block.strip().split("\n")

            if len(lines) < 3:
                continue

            try:
                index = int(lines[0])

                # Парсим таймкоды
                time_line = lines[1]
                if " --> " not in time_line:
                    continue

                start_str, end_str = time_line.split(" --> ")
                start = self._normalize_timestamp(start_str.strip())
                end = self._normalize_timestamp(end_str.strip())

                # Текст может быть в нескольких строках
                text = "\n".join(lines[2:])

                entries.append(SubtitleEntry(
                    index=index,
                    start=start,
                    end=end,
                    text=text
                ))

            except (ValueError, IndexError) as e:
                print(f"Пропущен блок субтитров: {e}")
                continue

        return entries

    def _normalize_timestamp(self, timestamp: str) -> str:
        """
        Нормализация timestamp в формат HH:MM:SS.mmm.

        Args:
            timestamp: Строка времени (может быть с запятой или точкой)

        Returns:
            Нормализованная строка времени
        """
        # Заменяем запятую на точку для единообразия
        timestamp = timestamp.replace(",", ".")

        # Если формат уже правильный, возвращаем как есть
        if len(timestamp) == 12 and timestamp[2] == ":" and timestamp[5] == ":":
            return timestamp.replace(".", ",")

        # Пытаемся распарсить
        try:
            parts = timestamp.replace(".", ":").split(":")
            if len(parts) >= 3:
                hours = int(parts[0])
                minutes = int(parts[1])
                # Секунды могут быть с миллисекундами
                sec_parts = parts[2].split(".")
                seconds = int(sec_parts[0])
                millis = int(sec_parts[1][:3].ljust(3, "0")) if len(sec_parts) > 1 else 0

                return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"
        except (ValueError, IndexError):
            pass

        return timestamp

    def get_subs_text(self) -> str:
        """
        Получение полного текста субтитров.

        Returns:
            Полный текст всех субтитров
        """
        entries = self.parse_srt()
        return " ".join(entry.text.replace("\n", " ") for entry in entries)
