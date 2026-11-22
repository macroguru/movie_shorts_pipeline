"""
HighlightEngine — многоэтапный поиск ярких моментов в записях стримов и клипах.

Пайплайн (без по-кадровой vision-обработки):
  A. Активность аудио: один проход ffmpeg astats → RMS по секундам.
  B. Транскрипция: faster-whisper с таймкодами слов (через subtitles.py, с кэшем).
  C. Кандидаты: окна из плотности речи + аудио-пиков (топ ~15% секунд).
  D. Текстовое рассуждение: транскрипт режется на окна ~10 минут с таймкодами,
     по окну — один вызов текстового провайдера со строгим JSON-ответом.
  E. Vision-проверка (опционально): топ 2×N кандидатов, по 3 кадра на кандидата,
     один вызов vision-провайдера на кандидата; корректировка.score ±2.
  F. Оценка/дедуп: слияние пересечений, отсечение score < 4, ранжирование,
     уточнение границ по таймкодам слов.

Чистые функции (build_candidates, merge_overlaps, rank_candidates,
refine_boundaries и др.) отделены от IO — их можно тестировать на
синтетических данных без ffmpeg и провайдеров.
"""

import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from config import Config
from providers import (
    ProviderError,
    TextProvider,
    VisionProvider,
    create_text_provider,
    create_vision_provider,
)
from usage import UsageTracker

# Доля «горячих» секунд, из которых строятся окна-кандидаты
TOP_HOT_FRACTION = 0.15
# Максимум окон-кандидатов стадии C
MAX_SEED_CANDIDATES = 24
# Длительность окна транскрипта для текстового анализа (сек)
TRANSCRIPT_WINDOW_SECONDS = 600
# Минимальный балл кандидата
MIN_CANDIDATE_SCORE = 4.0
# Максимальная корректировка score от vision-проверки
MAX_VISION_ADJUSTMENT = 2.0
# Насколько долго считается запись для стрида 2 секунды (4 часа)
LONG_VIDEO_SECONDS = 4 * 3600


class HighlightError(Exception):
    """Ошибка поиска ярких моментов — текст показывается пользователю как есть."""
    pass


@dataclass
class Highlight:
    """Найденный яркий момент (границы в секундах от начала видео)."""

    start_seconds: float
    end_seconds: float
    score: float
    reason: str
    keywords: list = field(default_factory=list)
    transcript: str = ""
    words: list = field(default_factory=list)  # [(слово, start, end), ...]


# ======================================================================
# Чистые функции (без IO) — для тестов и переиспользования
# ======================================================================

def _percentile(sorted_values: list, fraction: float) -> float:
    """Перцентиль отсортированного списка (без numpy)."""
    if not sorted_values:
        return 0.0
    index = int(fraction * (len(sorted_values) - 1))
    return sorted_values[max(0, min(index, len(sorted_values) - 1))]


def normalize_activity(rms_levels: List[float]) -> List[float]:
    """
    Нормализация уровней RMS (дБ) в 0..1 по перцентилям 10/95.

    Args:
        rms_levels: Уровень RMS в дБ на каждую секунду (или окно стрида)

    Returns:
        Список значений 0..1 той же длины
    """
    values = [max(v, -120.0) if v == v else -120.0 for v in rms_levels]
    if not values:
        return []

    ordered = sorted(values)
    low = _percentile(ordered, 0.10)
    high = _percentile(ordered, 0.95)

    if high - low < 1e-6:
        return [0.5] * len(values)

    return [min(max((v - low) / (high - low), 0.0), 1.0) for v in values]


def speech_density(segments: List[dict], total_seconds: int) -> List[float]:
    """
    Плотность речи по секундам: 1.0, если секунда покрыта сегментом, иначе 0.0.

    Args:
        segments: Сегменты транскрипта {start, end, ...}
        total_seconds: Длина видео в секундах

    Returns:
        Список 0/1 длиной total_seconds
    """
    density = [0.0] * max(total_seconds, 0)
    for segment in segments:
        start = max(int(segment.get("start", 0)), 0)
        end = min(int(segment.get("end", 0)) + 1, total_seconds)
        for second in range(start, end):
            density[second] = 1.0
    return density


def _window_score(combined: List[float], start: float, end: float) -> float:
    """Средний комбинированный балл окна (0..1)."""
    first = max(int(start), 0)
    last = min(int(end), len(combined))
    if last <= first:
        return 0.0
    return sum(combined[first:last]) / (last - first)


def build_candidates(
    activity: List[float],
    segments: List[dict],
    min_duration: float,
    max_duration: float,
    duration: Optional[float] = None,
    top_fraction: float = TOP_HOT_FRACTION,
    max_candidates: int = MAX_SEED_CANDIDATES,
) -> List[dict]:
    """
    Стадия C: окна-кандидаты из плотности речи + аудио-пиков.

    Берём топ ~15% секунд по комбинированному баллу (речь + RMS), вокруг
    них строим окна, расширяя по соседним «горячим» секундам до max_duration
    (но не короче min_duration). Окна не пересекаются между собой.

    Args:
        activity: RMS-уровни по секундам (дБ)
        segments: Сегменты транскрипта
        min_duration: Минимальная длительность окна
        max_duration: Максимальная длительность окна
        duration: Длительность видео (по умолчанию длина activity)
        top_fraction: Доля «горячих» секунд
        max_candidates: Максимум возвращаемых окон

    Returns:
        Список {"start", "end", "score"} — score 0..1, окна не пересекаются
    """
    total = int(duration) if duration and duration > 0 else len(activity)
    if total <= 0 or min_duration <= 0:
        return []

    active = normalize_activity(activity[:total])
    if len(active) < total:
        active += [0.0] * (total - len(active))

    density = speech_density(segments, total)

    # Комбинированный балл: речь важнее шума, но без речи опираемся на аудио
    if any(v > 0 for v in density):
        combined = [0.6 * density[i] + 0.4 * active[i] for i in range(total)]
    else:
        combined = active

    ordered = sorted(combined)
    threshold = _percentile(ordered, 1.0 - top_fraction)
    hot = {i for i, value in enumerate(combined) if value >= threshold and value > 0}

    limit = float(total)
    windows: List[tuple] = []

    # Самые «горячие» секунды — seeds для окон; окна не пересекаются (разнесены)
    for seed in sorted(hot, key=lambda i: -combined[i]):
        if any(start <= seed < end for start, end, _score in windows):
            continue

        start = float(seed)
        end = start + 1.0
        while end < limit and (end - start) < max_duration and int(end) in hot:
            end += 1.0

        if end - start < min_duration:
            end = min(start + min_duration, limit)

        if end - start < min_duration:
            # Даже минимума не хватает — берём всё, что есть от seed до конца
            end = limit
        if end - start < 1.0:
            continue

        windows.append((start, end, _window_score(combined, start, end)))

    # Финальный проход: по убыванию балла оставляем только непересекающиеся окна
    windows.sort(key=lambda item: -item[2])
    spaced: List[tuple] = []
    for start, end, score in windows:
        if any(start < kept_end and kept_start < end for kept_start, kept_end, _ in spaced):
            continue
        spaced.append((start, end, score))

    spaced.sort(key=lambda item: item[0])
    return [
        {"start": start, "end": end, "score": score}
        for start, end, score in spaced[:max_candidates]
    ]


def format_timecode(seconds: float) -> str:
    """
    Таймкод для промпта: MM:SS, а от часа — H:MM:SS (чтобы не было двусмысленности).
    """
    total = max(int(seconds), 0)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def timecode_to_seconds(timecode: str) -> float:
    """MM:SS | H:MM:SS | SS → секунды (при ошибке — 0.0)."""
    parts = str(timecode).strip().split(":")
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return 0.0
    seconds = 0.0
    for value in values:
        seconds = seconds * 60 + value
    return seconds


def build_transcript_windows(
    segments: List[dict],
    window_seconds: float = TRANSCRIPT_WINDOW_SECONDS,
) -> List[tuple]:
    """
    Нарезка транскрипта на окна ~10 минут с текстом вида «[MM:SS] реплика».

    Таймкоды абсолютные (от начала видео): модель возвращает start_sec/end_sec
    в тех же координатах.

    Args:
        segments: Сегменты транскрипта {start, end, text}
        window_seconds: Длительность одного окна

    Returns:
        Список (window_start, window_end, text)
    """
    if not segments:
        return []

    windows = []
    current: List[str] = []
    current_start = float(segments[0].get("start", 0))

    for segment in segments:
        start = float(segment.get("start", 0))
        if current and start - current_start >= window_seconds:
            windows.append((current_start, start, "\n".join(current)))
            current = []
            current_start = start

        timecode = format_timecode(start)
        current.append(f"[{timecode}] {segment.get('text', '').strip()}")

    if current:
        windows.append((
            current_start,
            float(segments[-1].get("end", current_start)),
            "\n".join(current),
        ))

    return windows


def extract_json(text: str) -> Optional[dict]:
    """
    Извлечение первого JSON-объекта из ответа модели (в том числе из markdown).

    Args:
        text: Сырой ответ провайдера

    Returns:
        Словарь или None, если JSON не найден
    """
    if not text:
        return None

    # Сначала ответ как есть
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # JSON в markdown-блоке
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    # Первый {...} в тексте (жадный поиск до последней })
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    return None


def parse_candidates(raw_text: str, duration: Optional[float] = None) -> List[dict]:
    """
    Разбор ответа стадии D: JSON с ключом candidates.

    Args:
        raw_text: Сырой ответ провайдера
        duration: Длительность видео — кандидаты обрезаются по ней

    Returns:
        Список {"start_sec", "end_sec", "score", "reason", "quote", "keywords"}
    """
    data = extract_json(raw_text)
    if not data:
        return []

    raw_candidates = data.get("candidates")
    if not isinstance(raw_candidates, list):
        return []

    candidates = []
    for item in raw_candidates:
        if not isinstance(item, dict):
            continue

        try:
            start = float(item.get("start_sec", 0))
            end = float(item.get("end_sec", 0))
            score = float(item.get("score", 0))
        except (TypeError, ValueError):
            continue

        if end <= start:
            continue

        score = min(max(score, 0.0), 10.0)
        if duration:
            start = max(0.0, start)
            end = min(end, duration)
            if end <= start:
                continue

        keywords = item.get("keywords") or []
        if not isinstance(keywords, list):
            keywords = []

        candidates.append({
            "start_sec": start,
            "end_sec": end,
            "score": score,
            "reason": str(item.get("reason", "")).strip(),
            "quote": str(item.get("quote", "")).strip(),
            "keywords": [str(k).strip() for k in keywords if str(k).strip()],
        })

    return candidates


def merge_overlaps(candidates: List[dict]) -> List[dict]:
    """
    Слияние пересекающихся кандидатов: границы — объединение, score — максимум.

    Причины объединяются через « | », ключевые слова — объединение без дублей.
    Слияние касающихся (стык вплотную) кандидатов не выполняется.

    Args:
        candidates: Кандидаты {start_sec, end_sec, score, reason?, keywords?}

    Returns:
        Отсортированный по start_sec список без пересечений
    """
    if not candidates:
        return []

    ordered = sorted(candidates, key=lambda c: c["start_sec"])
    merged: List[dict] = []

    for candidate in ordered:
        if merged and candidate["start_sec"] < merged[-1]["end_sec"]:
            top = merged[-1]
            top["end_sec"] = max(top["end_sec"], candidate["end_sec"])
            top["score"] = max(top["score"], candidate["score"])

            for reason in [candidate.get("reason", "")]:
                if reason and reason not in top.get("reason", ""):
                    top["reason"] = (top.get("reason", "") + " | " + reason).strip(" |")

            for keyword in candidate.get("keywords", []):
                if keyword not in top["keywords"]:
                    top["keywords"].append(keyword)

            quote = candidate.get("quote", "")
            if quote and not top.get("quote"):
                top["quote"] = quote
        else:
            merged.append({
                "start_sec": candidate["start_sec"],
                "end_sec": candidate["end_sec"],
                "score": candidate["score"],
                "reason": candidate.get("reason", ""),
                "quote": candidate.get("quote", ""),
                "keywords": list(candidate.get("keywords", [])),
            })

    return merged


def rank_candidates(candidates: List[dict], min_score: float = MIN_CANDIDATE_SCORE) -> List[dict]:
    """
    Отсечение слабых кандидатов и ранжирование по убыванию score.

    Args:
        candidates: Кандидаты с полем score
        min_score: Минимальный допустимый балл

    Returns:
        Кандидаты с score >= min_score, по убыванию score
    """
    return sorted(
        [c for c in candidates if c.get("score", 0) >= min_score],
        key=lambda c: -c.get("score", 0),
    )


def refine_boundaries(
    start_sec: float,
    end_sec: float,
    words: List[tuple],
    min_duration: float,
    max_duration: float,
    source_duration: float,
    pad: float = 1.5,
) -> tuple:
    """
    Уточнение границ клипа по таймкодам слов: старт — у первого слова
    кандидата, конец — после последнего, с подушкой до pad секунд,
    но в пределах источника и длительности [min_duration, max_duration]
    (насколько позволяет видео).

    Args:
        start_sec: Начальная граница кандидата
        end_sec: Конечная граница кандидата
        words: Слова [(word, start, end), ...] (могут быть не отсортированы)
        min_duration: Минимальная длительность клипа
        max_duration: Максимальная длительность клипа
        source_duration: Длительность видео
        pad: Максимальная подушка с каждой стороны (сек)

    Returns:
        Кортеж (start, end) в секундах
    """
    # Подушка: чуть раньше начала и позже конца, но в пределах видео
    start = max(0.0, start_sec - pad)
    end = end_sec + pad
    if source_duration > 0:
        end = min(end, source_duration)

    ordered = sorted(words, key=lambda w: w[1])

    # Слова внутри кандидата (допускаем слово, начавшееся чуть раньше границы)
    inside = [w for w in ordered if w[1] >= start_sec - 0.5 and w[1] < end_sec]

    if inside:
        # Старт — у первого слова (небольшой запас перед ним)
        start = min(max(start, inside[0][1] - 0.15), inside[0][1])
        # Конец — после последнего слова (небольшой запас после него)
        last_end = max(w[2] for w in inside)
        end = min(end, last_end + 0.35)

    # Длительность в пределах [min, max], если источник позволяет
    if end - start > max_duration:
        end = start + max_duration

    if source_duration > 0:
        if end - start < min_duration:
            end = min(source_duration, start + min_duration)
            if end - start < min_duration:
                # Уперлись в конец видео — расширяем назад
                start = max(0.0, end - min_duration)

    if end <= start:
        return (max(0.0, start_sec), max(start_sec, end_sec))

    return (start, end)


# ======================================================================
# HighlightEngine — оркестрация стадий
# ======================================================================

CANDIDATE_SYSTEM_PROMPT = """Ты редактор вирусных нарезок для TikTok, YouTube Shorts и Reels. Тебе дают фрагмент таймкодов записи стрима: летсплей, подкаст, реакции, IRL, Just Chatting, общение с чатом.

КРИТЕРИИ ВЫБОРА МОМЕНТОВ:
- Момент понятен без знания контекста всего стрима (самодостаточен)
- Цепляет зрителя в первые 3 секунды (крючок: фраза, событие, реакция)
- Имеет завершённый мини-сюжет: завязка, развитие, развязка (панчлайн, итог, реакция)
- Приоритет: юмор, яркие реакции, фейлы и победы, эмоциональные пики, драма, игровые события (клатчи, неожиданные повороты)

ВЕРНИ СТРОГИЙ JSON:
{"candidates": [{"start_sec": <число секунд от НАЧАЛА ВИДЕО>, "end_sec": <число>, "score": <0-10>, "reason": "почему зайдёт (1-2 предложения)", "quote": "самая цепляющая фраза момента", "keywords": ["тег", "тег"]}]}
- start_sec/end_sec — абсолютные секунды видео (согласованы с таймкодами [MM:SS])
- score — насколько момент вирусный, 0-10
- Верни до {clips_count} кандидатов, без пояснений вне JSON, валидный JSON"""

VISION_SYSTEM_PROMPT = """Ты проверяешь кандидат в вертикальные шортсы по кадрам из видео. Оцени: визуально ли момент цепляет (экшн, эмоции, мимика, события на экране, динамика) и самодостаточен ли он без контекста.

ВЕРНИ СТРОГИЙ JSON:
{"engaging": true|false, "adjustment": <-2..2>, "justification": "коротко почему"}
- adjustment — сколько прибавить или отнять от score кандидата (от -2 до 2)
- Без пояснений вне JSON"""


class HighlightEngine:
    """
    Поиск ярких моментов: аудио-активность → транскрипт → кандидаты →
    AI-рассуждение → vision-проверка → оценка и уточнение границ.
    """

    def __init__(self, config: Config, log: Optional[Callable[[str], None]] = None):
        """
        Args:
            config: Конфигурация проекта
            log: Куда писать прогресс этапов (по умолчанию — print)
        """
        self.config = config
        self._log_fn = log
        self.usage = UsageTracker(config.temp_dir / "usage.jsonl")

        # Границы длительности момента: ядро может переопределить после конструктора
        self.min_duration = float(config.highlight_min_duration)
        self.max_duration = float(config.highlight_max_duration)

    def _log(self, message: str) -> None:
        """Лог этапа в колбэк вызывающего кода или в print."""
        if self._log_fn is not None:
            self._log_fn(message)
        else:
            print(message)

    # ---------- Стадия A: активность аудио ----------

    def _audio_activity(self, video_path: str, duration: float) -> List[float]:
        """
        Один проход ffmpeg: RMS-уровень по каждой секунде (astats + ametadata).

        Args:
            video_path: Путь к видео
            duration: Длительность видео

        Returns:
            Список RMS (дБ) по секундам; значения -inf заменяются на -120

        Raises:
            HighlightError: если ffmpeg не выполнился
        """
        # asetnsamples выравнивает кадры по 1 секунде: один RMS-уровень на секунду
        audio_filter = (
            "aresample=48000,asetnsamples=n=48000:p=0,"
            "astats=metadata=1:reset=1,"
            "ametadata=mode=print:file=-:key=lavfi.astats.Overall.RMS_level"
        )

        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-nostats",
                    "-i", video_path,
                    "-vn",
                    "-af", audio_filter,
                    "-f", "null", "-",
                ],
                capture_output=True,
                text=True,
                check=True,
                errors="replace",
            )
        except subprocess.CalledProcessError as e:
            raise HighlightError(f"Ошибка анализа аудио (ffmpeg): {e.stderr[-500:] if e.stderr else e}")

        levels: List[float] = []
        for line in (result.stdout or "").splitlines():
            line = line.strip()
            if not line.startswith("lavfi.astats.Overall.RMS_level="):
                continue

            value = line.split("=", 1)[1].strip()
            try:
                level = float(value)
            except ValueError:
                level = -120.0  # -inf и прочее
            levels.append(min(level, 0.0) if level == level else -120.0)

        if not levels:
            raise HighlightError("ffmpeg не вернул RMS-метрики аудио")

        # Стрид 2 секунды для очень длинных записей
        stride = 2 if duration > LONG_VIDEO_SECONDS else 1
        if stride > 1:
            levels = [
                max(levels[i:i + stride]) for i in range(0, len(levels), stride)
            ]

        self._log(f"  Проанализировано {len(levels)} сек аудио (стрид {stride}с)")
        return levels

    # ---------- Стадия D: текстовое рассуждение ----------

    def _text_candidates(
        self,
        segments: List[dict],
        text_provider: TextProvider,
        clips_count: int,
        min_duration: float,
        max_duration: float,
        duration: float,
        source: str,
    ) -> List[dict]:
        """
        По одному вызову провайдера на ~10-минутное окно транскрипта (параллельно).

        Args:
            segments: Сегменты транскрипта
            text_provider: Текстовый провайдер
            clips_count: Сколько кандидатов просить из каждого окна
            min_duration: Минимальная длительность момента
            max_duration: Максимальная длительность момента
            duration: Длительность видео
            source: "stream" | "clip"

        Returns:
            Валидные кандидаты стадии D
        """
        windows = build_transcript_windows(segments)
        if not windows:
            return []

        self._log(f"  Транскрипт разбит на {len(windows)} окон по ~10 минут")

        mode_note = (
            "Найди ОДИН самый лучший момент (режим клипа)."
            if source == "clip"
            else f"Найди до {clips_count} лучших моментов."
        )

        def ask_window(window: tuple) -> List[dict]:
            window_start, window_end, text = window
            header = (
                f"ФРАГМЕНТ ЗАПИСИ: с {format_timecode(window_start)} "
                f"до {format_timecode(window_end)}. Таймкоды [MM:SS] — от начала видео.\n"
                f"Длительность момента: {int(min_duration)}-{int(max_duration)} секунд. {mode_note}\n\n"
                f"{text[:50000]}\n\n"
                "Верни ТОЛЬКО валидный JSON."
            )
            try:
                raw = text_provider.complete(header, system=CANDIDATE_SYSTEM_PROMPT)
            except ProviderError as e:
                self._log(f"  Окно {format_timecode(window_start)}: ошибка провайдера — {e}")
                return []

            return parse_candidates(raw, duration=duration)

        workers = max(1, min(self.config.highlight_workers, len(windows)))
        results: List[dict] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for candidates in pool.map(ask_window, windows):
                results.extend(candidates)

        self._log(f"  AI предложил кандидатов: {len(results)}")
        return results

    # ---------- Стадия E: vision-проверка ----------

    def _extract_frames(self, video_path: str, times: List[float], out_dir: Path) -> List[str]:
        """
        Извлечение кадров (по одному ffmpeg-вызову на кадр, масштаб 512px).

        Args:
            video_path: Путь к видео
            times: Моменты кадров в секундах
            out_dir: Папка для кадров

        Returns:
            Пути к успешно извлечённым кадрам
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []

        for i, moment in enumerate(times):
            frame_path = out_dir / f"frame_{i:02d}.jpg"
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-hide_banner", "-nostats", "-loglevel", "error",
                        "-ss", f"{max(moment, 0):.2f}",
                        "-i", video_path,
                        "-frames:v", "1",
                        "-vf", "scale=512:-2",
                        "-q:v", "3",
                        "-y", str(frame_path),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                if frame_path.exists() and frame_path.stat().st_size > 0:
                    paths.append(str(frame_path))
            except subprocess.CalledProcessError:
                continue  # кадр не критичен

        return paths

    def _vision_adjust(
        self,
        video_path: str,
        candidates: List[dict],
        vision_provider: VisionProvider,
        duration: float,
    ) -> List[dict]:
        """
        Vision-проверка кандидатов: 3 кадра на кандидата, один вызов на кандидата,
        корректировка score в пределах ±2.

        Args:
            video_path: Путь к видео
            candidates: Кандидаты (модифицируются score)
            vision_provider: Vision-провайдер
            duration: Длительность видео

        Returns:
            Те же кандидаты со скорректированными score
        """
        workers = max(1, min(self.config.highlight_workers, len(candidates)))

        def check(candidate: dict) -> None:
            start = candidate["start_sec"]
            end = candidate["end_sec"]
            length = max(end - start, 1.0)
            times = [
                start + length * 0.15,
                start + length * 0.5,
                min(start + length * 0.85, max(duration - 0.5, start)),
            ]

            frames_dir = Path(self.config.temp_dir) / "vision_frames"
            frame_paths = self._extract_frames(video_path, times, frames_dir)
            if not frame_paths:
                self._log(f"  Vision: не удалось извлечь кадры для {format_timecode(start)}")
                return

            quote = candidate.get("quote") or candidate.get("reason") or "без реплики"
            prompt = (
                f"Кандидат: {format_timecode(start)}-{format_timecode(end)}.\n"
                f"Реплика/суть: {quote[:300]}\n\n"
                "Оцени по критериям из инструкции."
            )

            try:
                raw = vision_provider.analyze_images(prompt, frame_paths)
            except ProviderError as e:
                self._log(f"  Vision: ошибка провайдера — {e}")
                return

            data = extract_json(raw) or {}
            try:
                adjustment = float(data.get("adjustment", 0))
            except (TypeError, ValueError):
                adjustment = 0.0
            adjustment = min(max(adjustment, -MAX_VISION_ADJUSTMENT), MAX_VISION_ADJUSTMENT)

            candidate["score"] = min(max(candidate["score"] + adjustment, 0.0), 10.0)
            verdict = "цепляет" if data.get("engaging") else "не цепляет"
            justification = str(data.get("justification", "")).strip()
            self._log(
                f"  Vision {format_timecode(start)}-{format_timecode(end)}: "
                f"{verdict}, {adjustment:+.0f} → score {candidate['score']:.1f}"
                + (f" ({justification})" if justification else "")
            )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(check, candidates))

        return candidates

    # ---------- Главный метод ----------

    def find_highlights(
        self,
        video_path: str,
        clips_count: int = 5,
        source: str = "stream",
    ) -> List[Highlight]:
        """
        Полный пайплайн поиска ярких моментов.

        Args:
            video_path: Путь к видео
            clips_count: Сколько моментов вернуть (в режиме clip — 1)
            source: "stream" (запись стрима) или "clip" (один лучший момент
                из короткого видео; min/max длительности обрезаются по видео)

        Returns:
            Список Highlight, отсортированный по start_seconds

        Raises:
            HighlightError: файл не найден, ошибка ffmpeg или все AI-вызовы упали
        """
        if source not in ("stream", "clip"):
            raise HighlightError(f"Неизвестный режим источника: {source!r}")

        path = Path(video_path)
        if not path.exists():
            raise HighlightError(f"Видеофайл не найден: {path}")

        duration = self._probe_duration(str(path))
        if not duration or duration <= 0:
            raise HighlightError(f"Не удалось определить длительность видео: {path}")

        self._log(f"Поиск ярких моментов: {path.name} ({format_timecode(duration)}, режим {source})")

        # Границы длительности: атрибуты движка (ядро может задать свои),
        # в режиме clip — обрезаем по длительности видео
        min_duration = self.min_duration
        max_duration = self.max_duration
        if source == "clip":
            clips_count = 1
            max_duration = min(max_duration, duration)
            min_duration = min(min_duration, max_duration)
            self._log(f"  Режим клипа: окно {min_duration:.0f}-{max_duration:.0f}с, один момент")

        # Провайдеры (лениво, общий учёт использования)
        try:
            text_provider = create_text_provider(self.config, self.usage)
        except ProviderError as e:
            raise HighlightError(f"Текстовый провайдер недоступен: {e}")

        vision_provider = None
        if self.config.highlight_vision_check == "auto":
            try:
                vision_provider = create_vision_provider(self.config, self.usage)
            except ProviderError as e:
                self._log(f"Vision-провайдер недоступен, проверка пропущена: {e}")
        else:
            self._log("Визуальная проверка отключена (highlight_vision_check=off)")

        # Стадия A: активность аудио
        self._log("Этап A: анализ активности аудио...")
        activity = self._audio_activity(str(path), duration)

        # Стадия B: транскрипт с словами (кэшируется в subtitles)
        self._log("Этап B: транскрибация с таймкодами слов...")
        from subtitles import Subtitles, SubtitleError

        try:
            segments = Subtitles(self.config).transcribe_with_words(str(path), log=self._log)
        except SubtitleError as e:
            raise HighlightError(f"Ошибка транскрибации: {e}")

        # Стадия C: окна-кандидаты из речи и аудио
        self._log("Этап C: построение окон-кандидатов...")
        seed_windows = build_candidates(
            activity, segments, min_duration, max_duration,
            duration=duration,
        )
        self._log(f"  Окна-кандидаты: {len(seed_windows)}")

        # Стадия D: текстовое рассуждение
        candidates: List[dict] = []
        if segments:
            self._log("Этап D: AI-анализ транскрипта...")
            candidates = self._text_candidates(
                segments, text_provider, clips_count,
                min_duration, max_duration, duration, source,
            )
        else:
            self._log("Этап D: речь в видео не найдена — пропускаем AI-анализ")

        # Если AI ничего не дал — опора на окна стадии C (score из аудио/речи)
        if not candidates:
            if not seed_windows:
                raise HighlightError(
                    "Не найдено ни одного кандидата: нет речи и нет аудио-активности"
                )
            self._log("AI-кандидатов нет — использую окна по речевой активности")
            candidates = [
                {
                    "start_sec": window["start"],
                    "end_sec": window["end"],
                    "score": min(window["score"] * 10.0, 10.0),
                    "reason": "Высокая речевая и аудио-активность (без AI-проверки)",
                    "quote": "",
                    "keywords": [],
                }
                for window in seed_windows
            ]

        # Стадия F (часть 1): дедуп и ранжирование
        candidates = merge_overlaps(candidates)
        candidates = rank_candidates(candidates)

        if not candidates:
            raise HighlightError(
                f"Все кандидаты слабее {MIN_CANDIDATE_SCORE:.0f} баллов — "
                "ярких моментов не найдено"
            )

        # Стадия E: vision-проверка топ 2×N кандидатов
        if vision_provider is not None:
            self._log(f"Этап E: vision-проверка топ {min(len(candidates), clips_count * 2)} кандидатов...")
            candidates = self._vision_adjust(
                str(path), candidates[:clips_count * 2], vision_provider, duration
            )
            candidates = rank_candidates(candidates)

        # Стадия F (часть 2): границы по словам, сбор Highlight
        all_words = [
            (word, start, end)
            for segment in segments
            for (word, start, end) in segment.get("words", [])
        ]

        highlights: List[Highlight] = []
        for candidate in candidates[:clips_count]:
            start, end = refine_boundaries(
                candidate["start_sec"], candidate["end_sec"], all_words,
                min_duration, max_duration, duration,
            )

            clip_words = [(w, s, e) for (w, s, e) in all_words if start <= s < end]
            transcript = " ".join(w for w, _s, _e in clip_words)

            highlights.append(Highlight(
                start_seconds=round(start, 2),
                end_seconds=round(end, 2),
                score=round(candidate["score"], 1),
                reason=candidate.get("reason", ""),
                keywords=candidate.get("keywords", []),
                transcript=transcript,
                words=clip_words,
            ))

        highlights.sort(key=lambda h: h.start_seconds)

        self._log(f"Итог: {len(highlights)} моментов")
        for i, highlight in enumerate(highlights, 1):
            self._log(
                f"  Момент {i}: {format_timecode(highlight.start_seconds)}-"
                f"{format_timecode(highlight.end_seconds)} "
                f"(score {highlight.score}) — {highlight.reason[:100]}"
            )

        return highlights

    def _probe_duration(self, video_path: str) -> Optional[float]:
        """
        Длительность видео через ffprobe.

        Args:
            video_path: Путь к видео

        Returns:
            Длительность в секундах или None
        """
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "quiet",
                    "-print_format", "json",
                    "-show_format", video_path,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            data = json.loads(result.stdout)
            return float(data["format"]["duration"])
        except (subprocess.CalledProcessError, json.JSONDecodeError,
                KeyError, ValueError, OSError) as e:
            self._log(f"ffprobe не смог определить длительность: {e}")
            return None
