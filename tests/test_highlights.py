"""
Тесты движка хайлайтов: чистые функции на синтетике и end-to-end
с фейковым провайдером и фейковой транскрипцией — без сети и whisper.
"""

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("highlights")

import highlights  # noqa: E402
from highlights import (  # noqa: E402
    Highlight,
    HighlightEngine,
    HighlightError,
    build_candidates,
    build_transcript_windows,
    extract_json,
    format_timecode,
    merge_overlaps,
    parse_candidates,
    rank_candidates,
    refine_boundaries,
    timecode_to_seconds,
)


# ---------- Чистые функции ----------

class TestTimecodes:
    def test_format_and_back(self):
        for seconds in (0, 5, 65, 3661, 7325):
            assert timecode_to_seconds(format_timecode(seconds)) == seconds

    def test_format_styles(self):
        assert format_timecode(65) == "01:05"
        assert format_timecode(3661) == "1:01:01"

    def test_bad_timecode_zero(self):
        assert timecode_to_seconds("не время") == 0.0


class TestExtractJson:
    def test_plain_json(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_markdown_fenced(self):
        text = "Модель ответила:\n```json\n{\"a\": 2}\n```"
        assert extract_json(text) == {"a": 2}

    def test_json_in_prose(self):
        assert extract_json("Ответ: {\"b\": [1, 2]} конец") == {"b": [1, 2]}

    def test_garbage_none(self):
        assert extract_json("совсем не JSON") is None
        assert extract_json("") is None


class TestParseCandidates:
    def test_valid_candidates(self):
        raw = json.dumps({"candidates": [
            {"start_sec": 10, "end_sec": 40, "score": 8.5,
             "reason": "панчлайн", "quote": "вот это да",
             "keywords": ["панч", "юмор"]},
        ]})
        items = parse_candidates(raw, duration=100)

        assert len(items) == 1
        assert items[0]["start_sec"] == 10
        assert items[0]["end_sec"] == 40
        assert items[0]["score"] == 8.5
        assert items[0]["keywords"] == ["панч", "юмор"]

    def test_clamped_to_duration(self):
        raw = json.dumps({"candidates": [
            {"start_sec": 90, "end_sec": 150, "score": 7},
        ]})
        items = parse_candidates(raw, duration=100)
        assert items[0]["start_sec"] == 90
        assert items[0]["end_sec"] == 100

    def test_score_clamped(self):
        raw = json.dumps({"candidates": [
            {"start_sec": 0, "end_sec": 10, "score": 42},
            {"start_sec": 20, "end_sec": 30, "score": -5},
        ]})
        items = parse_candidates(raw)
        assert items[0]["score"] == 10.0
        assert items[1]["score"] == 0.0

    def test_end_before_start_dropped(self):
        raw = json.dumps({"candidates": [
            {"start_sec": 30, "end_sec": 10, "score": 9},
        ]})
        assert parse_candidates(raw) == []

    def test_garbage_empty(self):
        assert parse_candidates("не JSON") == []
        assert parse_candidates('{"candidates": "строка вместо списка"}') == []


class TestBuildCandidates:
    def _synthetic(self):
        """300с записи: речь+громкость в 50-80с и 150-180с, тишина вокруг."""
        duration = 300
        activity = [-40.0] * duration
        for i in list(range(50, 80)) + list(range(150, 180)):
            activity[i] = -5.0
        segments = [
            {"start": 50, "end": 79, "text": "речь", "words": []},
            {"start": 150, "end": 179, "text": "речь", "words": []},
        ]
        return activity, segments, duration

    def test_windows_respect_bounds(self):
        activity, segments, duration = self._synthetic()
        windows = build_candidates(activity, segments, 20, 30, duration=duration)

        assert 1 <= len(windows) <= 24
        for w in windows:
            assert 20.0 <= w["end"] - w["start"] <= 30.0
            assert 0 <= w["start"] < w["end"] <= duration

    def test_windows_non_overlapping(self):
        activity, segments, duration = self._synthetic()
        windows = build_candidates(activity, segments, 20, 30, duration=duration)

        ordered = sorted(windows, key=lambda w: w["start"])
        for a, b in zip(ordered, ordered[1:]):
            assert a["end"] <= b["start"]

    def test_hot_zones_selected(self):
        activity, segments, duration = self._synthetic()
        windows = build_candidates(activity, segments, 20, 30, duration=duration)
        starts = [w["start"] for w in windows]

        # Окно внутри первого или второго «горячего» участка
        assert any(45 <= s <= 80 for s in starts)

    def test_scores_normalized(self):
        activity, segments, duration = self._synthetic()
        windows = build_candidates(activity, segments, 20, 30, duration=duration)
        for w in windows:
            assert 0.0 <= w["score"] <= 1.0

    def test_empty_input(self):
        assert build_candidates([], [], 20, 30) == []


class TestMergeOverlaps:
    def test_overlaps_merged(self):
        merged = merge_overlaps([
            {"start_sec": 0, "end_sec": 10, "score": 6,
             "reason": "первый", "quote": "", "keywords": ["a"]},
            {"start_sec": 5, "end_sec": 15, "score": 9,
             "reason": "второй", "quote": "цитата", "keywords": ["b"]},
            {"start_sec": 20, "end_sec": 30, "score": 7,
             "reason": "третий", "quote": "", "keywords": []},
        ])

        assert len(merged) == 2
        assert merged[0]["start_sec"] == 0
        assert merged[0]["end_sec"] == 15
        assert merged[0]["score"] == 9  # максимум
        assert set(merged[0]["keywords"]) == {"a", "b"}
        assert "первый" in merged[0]["reason"]
        assert "второй" in merged[0]["reason"]
        assert merged[1] == {
            "start_sec": 20, "end_sec": 30, "score": 7,
            "reason": "третий", "quote": "", "keywords": [],
        }

    def test_touching_not_merged(self):
        merged = merge_overlaps([
            {"start_sec": 0, "end_sec": 10, "score": 5, "keywords": []},
            {"start_sec": 10, "end_sec": 20, "score": 5, "keywords": []},
        ])
        assert len(merged) == 2

    def test_sorted_by_start(self):
        merged = merge_overlaps([
            {"start_sec": 50, "end_sec": 60, "score": 5, "keywords": []},
            {"start_sec": 10, "end_sec": 20, "score": 5, "keywords": []},
        ])
        assert [m["start_sec"] for m in merged] == [10, 50]

    def test_empty(self):
        assert merge_overlaps([]) == []


class TestRankCandidates:
    def test_min_score_cut_and_desc_order(self):
        ranked = rank_candidates([
            {"start_sec": 0, "end_sec": 5, "score": 3.9},
            {"start_sec": 10, "end_sec": 15, "score": 9},
            {"start_sec": 20, "end_sec": 25, "score": 4.0},
        ])
        assert [r["score"] for r in ranked] == [9, 4.0]

    def test_empty(self):
        assert rank_candidates([]) == []


class TestRefineBoundaries:
    def test_snap_to_first_word(self):
        # Старт прилипает к первому слову внутри кандидата
        words = [("начало", 9.8, 10.2), ("середина", 15.0, 15.4)]
        start, end = refine_boundaries(
            10, 20, words, min_duration=5, max_duration=15,
            source_duration=100,
        )

        assert 9.6 <= start <= 10.01  # слово на 9.8 минус запас 0.15
        assert end <= 15.75           # конец за последним словом внутри (15.4+0.35)

    def test_end_hugs_last_word_inside(self):
        words = [("начало", 10.0, 12.0), ("конец", 18.0, 19.5)]
        start, end = refine_boundaries(
            10, 20, words, min_duration=5, max_duration=15,
            source_duration=100,
        )
        # Конец поджимается к концу последнего слова внутри кандидата
        assert 19.5 <= end <= 20.0
        assert start <= 10.01

    def test_word_just_before_candidate_counts(self):
        # Слово, начавшееся на 0.5с раньше кандидата, тоже попадает в подгонку
        words = [("раннее", 9.6, 10.4)]
        start, end = refine_boundaries(
            10, 20, words, min_duration=5, max_duration=15,
            source_duration=100,
        )
        assert 9.4 <= start <= 9.7
        # Конец был поджат к слову (10.4+0.35=10.75), затем добит до min_duration
        assert 14.3 <= end <= 14.6

    def test_max_duration_respected(self):
        start, end = refine_boundaries(
            0, 50, [], min_duration=10, max_duration=30, source_duration=100,
        )
        assert end - start <= 30

    def test_min_duration_respected(self):
        # Кандидат у самого конца видео — минимум набирается расширением назад
        start, end = refine_boundaries(
            95, 98, [], min_duration=10, max_duration=60, source_duration=100,
        )
        assert end <= 100
        assert end - start >= 9.9  # допускаем погрешность округления

    def test_within_source_duration(self):
        start, end = refine_boundaries(
            10, 20, [], min_duration=5, max_duration=15, source_duration=30,
        )
        assert 0 <= start and end <= 30
        assert start < end

    def test_degenerate_fallback(self):
        # Плохие входные данные не приводят к start >= end
        start, end = refine_boundaries(
            10, 20, [], min_duration=100, max_duration=200, source_duration=12,
        )
        assert start < end


# ---------- HighlightEngine end-to-end на фейках ----------

DURATION = 300.0


def make_engine_config(tmp_path) -> SimpleNamespace:
    return SimpleNamespace(
        temp_dir=tmp_path / "temp",
        temp_subs_path=tmp_path / "temp" / "subs.srt",  # нужен Subtitles(config)
        highlight_min_duration=15.0,
        highlight_max_duration=30.0,
        highlight_workers=2,
        highlight_vision_check="off",  # vision не должен вызываться
    )


def make_segments():
    """Транскрипт со словами в двух «горячих» зонах."""
    def zone(base):
        return {
            "start": base,
            "end": base + 30,
            "text": "вот это поворот отлично",
            "words": [
                ("вот", base + 0.0, base + 0.4),
                ("это", base + 0.5, base + 0.9),
                ("поворот", base + 1.0, base + 1.8),
                ("отлично", base + 2.0, base + 2.9),
            ],
        }

    return [zone(40), zone(100), zone(200)]


CANDIDATES_JSON = json.dumps({
    "candidates": [
        {"start_sec": 40, "end_sec": 60, "score": 9.0,
         "reason": "Яркий поворот", "quote": "вот это поворот",
         "keywords": ["поворот"]},
        {"start_sec": 100, "end_sec": 115, "score": 5.5,
         "reason": "Средний момент", "quote": "отлично",
         "keywords": []},
        {"start_sec": 200, "end_sec": 260, "score": 8.0,
         "reason": "Длинный экшн", "quote": "поворот",
         "keywords": ["экшн"]},
    ]
}, ensure_ascii=False)


@pytest.fixture
def engine(tmp_path, monkeypatch):
    """HighlightEngine с фейковыми ffmpeg/ffprobe/транскриптом/провайдером."""
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake video bytes")

    config = make_engine_config(tmp_path)
    engine = HighlightEngine(config, log=lambda _msg: None)
    engine.min_duration = 15.0
    engine.max_duration = 30.0

    # Провайдер возвращает готовых кандидатов
    class _FakeProvider:
        def __init__(self):
            self.calls = []

        def complete(self, prompt, system=None):
            self.calls.append((prompt, system))
            return CANDIDATES_JSON

    fake_provider = _FakeProvider()
    monkeypatch.setattr(
        highlights, "create_text_provider",
        lambda config, tracker=None: fake_provider,
    )

    def fake_probe(_path):
        return DURATION

    def fake_activity(_path, _duration):
        activity = [-40.0] * int(DURATION)
        for i in list(range(40, 70)) + list(range(200, 260)):
            activity[i] = -5.0
        return activity

    monkeypatch.setattr(
        HighlightEngine, "_probe_duration", staticmethod(fake_probe)
    )
    monkeypatch.setattr(
        HighlightEngine, "_audio_activity", staticmethod(fake_activity)
    )
    monkeypatch.setattr(
        "subtitles.Subtitles.transcribe_with_words",
        lambda self, path, log=None: make_segments(),
    )

    engine._fake_provider = fake_provider
    engine._video_path = str(video)
    return engine


class TestFindHighlightsEndToEnd:
    def test_returns_sorted_highlights(self, engine):
        result = engine.find_highlights(engine._video_path, clips_count=2, source="stream")

        assert all(isinstance(h, Highlight) for h in result)
        assert 1 <= len(result) <= 2  # не больше clips_count
        assert [h.start_seconds for h in result] == sorted(
            h.start_seconds for h in result
        )

        # Топ по score — 9.0 и 8.0 (5.5 отсеялся как третий)
        assert any(h.score == 9.0 for h in result)

    def test_bounds_and_durations(self, engine):
        result = engine.find_highlights(engine._video_path, clips_count=2, source="stream")

        for h in result:
            assert 0 <= h.start_seconds < h.end_seconds <= DURATION
            assert h.end_seconds - h.start_seconds <= 30.0 + 1e-6
            assert h.reason
            assert h.words  # слова внутри момента найдены
            assert h.transcript  # собран из слов

    def test_first_highlight_snapped_to_words(self, engine):
        result = engine.find_highlights(engine._video_path, clips_count=2, source="stream")
        first = result[0]

        # Кандидат 40-60 → старт прилип к слову на 40.0, конец добит
        # до min_duration (слова внутри заканчиваются на 42.9)
        assert 39.5 <= first.start_seconds <= 40.05
        assert 14.9 <= first.end_seconds - first.start_seconds <= 15.1
        assert first.end_seconds <= 56

    def test_long_candidate_trimmed_and_bumped(self, engine):
        result = engine.find_highlights(engine._video_path, clips_count=3, source="stream")
        long_ones = [h for h in result if abs(h.start_seconds - 199.85) < 2]

        # Кандидат 200-260: слова внутри короткие → длительность добита
        # до min_duration, а не оставлена 60 секундами
        for h in long_ones:
            assert 14.9 <= h.end_seconds - h.start_seconds <= 30.0 + 1e-6

    def test_provider_called_with_window(self, engine):
        engine.find_highlights(engine._video_path, clips_count=2, source="stream")

        assert len(engine._fake_provider.calls) == 1
        prompt, system = engine._fake_provider.calls[0]
        assert "[00:40]" in prompt  # таймкоды транскрипта в промпте
        assert system is not None and "редактор" in system

    def test_min_max_override(self, engine):
        engine.min_duration = 5.0
        engine.max_duration = 12.0
        result = engine.find_highlights(engine._video_path, clips_count=3, source="stream")

        for h in result:
            assert h.end_seconds - h.start_seconds <= 12.0 + 1e-6


class TestFindHighlightsFallback:
    def test_provider_error_falls_back_to_activity_windows(self, tmp_path, monkeypatch):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")

        config = make_engine_config(tmp_path)
        engine = HighlightEngine(config, log=lambda _msg: None)
        engine.min_duration = 15.0
        engine.max_duration = 30.0

        from providers import ProviderError

        class _BrokenProvider:
            def complete(self, prompt, system=None):
                raise ProviderError("сеть недоступна")

        monkeypatch.setattr(
            highlights, "create_text_provider",
            lambda config, tracker=None: _BrokenProvider(),
        )
        monkeypatch.setattr(
            HighlightEngine, "_probe_duration", staticmethod(lambda _p: DURATION)
        )
        monkeypatch.setattr(
            HighlightEngine, "_audio_activity",
            staticmethod(lambda _p, _d: [-40.0] * 300),
        )
        monkeypatch.setattr(
            "subtitles.Subtitles.transcribe_with_words",
            lambda self, path, log=None: make_segments(),
        )

        result = engine.find_highlights(str(video), clips_count=2, source="stream")

        # Ошибка провайдера не уронила пайплайн: окна по активности отданы
        assert len(result) >= 1
        assert all(isinstance(h, Highlight) for h in result)
        assert any("Высокая" in h.reason for h in result)

    def test_missing_video_raises(self, tmp_path):
        config = make_engine_config(tmp_path)
        engine = HighlightEngine(config, log=lambda _msg: None)

        with pytest.raises(HighlightError, match="не найден"):
            engine.find_highlights(str(tmp_path / "missing.mp4"))

    def test_bad_source_raises(self, tmp_path):
        config = make_engine_config(tmp_path)
        engine = HighlightEngine(config, log=lambda _msg: None)
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")

        with pytest.raises(HighlightError, match="режим"):
            engine.find_highlights(str(video), source="podcast")
