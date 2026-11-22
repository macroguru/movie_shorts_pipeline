"""
Регрессионные тесты AI-движка: разбор ответов модели без сети.

AIEngine вызывается с фейковым конфигом — реальный Gemini не трогается.
"""

import json

import pytest

from ai_engine import AIEngine, AIEngineError


@pytest.fixture
def engine():
    """AIEngine без сети: клиент создаётся лениво, parse-функции его не трогают."""
    class _FakeConfig:
        gemini_model = "fake-model"
        gemini_api_key = "fake-key"
        temp_subs_path = "unused.srt"

    return AIEngine(_FakeConfig())


MOVIE_RESPONSE = {
    "clips": [
        {"start": "00:01:00", "end": "00:01:50", "hook_description": "Погоня"},
        {"start": "01:10:00", "end": "01:11:00", "hook_description": "Твист"},
    ],
    "post_data": {
        "genre": "Триллер / Детектив",
        "synopsis": "Детектив ищет серийного убийцу, но подозреваемый — он сам.",
        "hook_fact": "Снят за 30 дней",
    },
}

STREAM_RESPONSE = {
    "clips": [
        {"start": "00:05:00", "end": "00:05:45", "hook_description": "Клатч"},
    ],
    "post_data": {
        "title": "Финальный клатч",
        "context": "Игрок вытаскивает раунд в одиночку.",
        "hashtags": "летсплей клатч реакции",
    },
}


class TestParseResponseMovie:
    def test_valid_movie_json(self, engine):
        data = engine._parse_response(json.dumps(MOVIE_RESPONSE))

        assert len(data["clips"]) == 2
        assert data["clips"][0]["start"] == "00:01:00"
        post = data["post_data"]
        assert post["genre"] == "Триллер / Детектив"
        assert "серийного" in post["synopsis"]
        assert post["hook_fact"]

    def test_markdown_fenced_json(self, engine):
        raw = "Вот результат:\n```json\n" + json.dumps(MOVIE_RESPONSE) + "\n```\nГотово."
        data = engine._parse_response(raw)
        assert data["post_data"]["genre"] == MOVIE_RESPONSE["post_data"]["genre"]

    def test_json_embedded_in_prose(self, engine):
        raw = "Ответ модели: " + json.dumps(MOVIE_RESPONSE, ensure_ascii=False) + " — конец."
        data = engine._parse_response(raw)
        assert data["clips"][1]["end"] == "01:11:00"


class TestParseResponseStream:
    def test_valid_stream_json(self, engine):
        data = engine._parse_response(json.dumps(STREAM_RESPONSE))
        assert data["post_data"]["title"] == "Финальный клатч"
        assert data["post_data"]["hashtags"] == "летсплей клатч реакции"

    def test_stream_markdown_fenced(self, engine):
        raw = "```json\n" + json.dumps(STREAM_RESPONSE) + "\n```"
        data = engine._parse_response(raw)
        assert data["post_data"]["context"].startswith("Игрок")


class TestParseResponseErrors:
    def test_missing_clips(self, engine):
        bad = {"post_data": MOVIE_RESPONSE["post_data"]}
        with pytest.raises(AIEngineError, match="clips"):
            engine._parse_response(json.dumps(bad))

    def test_missing_post_data(self, engine):
        bad = {"clips": MOVIE_RESPONSE["clips"]}
        with pytest.raises(AIEngineError, match="post_data"):
            engine._parse_response(json.dumps(bad))

    def test_clip_missing_keys(self, engine):
        bad = {
            "clips": [{"start": "00:01:00", "hook_description": "нет конца"}],
            "post_data": MOVIE_RESPONSE["post_data"],
        }
        with pytest.raises(AIEngineError, match="start, end"):
            engine._parse_response(json.dumps(bad))

    def test_post_data_missing_keys(self, engine):
        bad = {
            "clips": MOVIE_RESPONSE["clips"],
            "post_data": {"genre": "Только жанр"},
        }
        with pytest.raises(AIEngineError, match="post_data"):
            engine._parse_response(json.dumps(bad))

    def test_no_json_at_all(self, engine):
        with pytest.raises(AIEngineError):
            engine._parse_response("Привет, я модель, JSON не вышел, извини")

    def test_broken_json(self, engine):
        with pytest.raises(AIEngineError):
            engine._parse_response('{"clips": [oops]}')


class TestPrepareSubtitles:
    SMALL_SRT = """1
00:00:01,000 --> 00:00:03,500
Привет мир

2
00:00:04,200 --> 00:00:06,000
Вторая реплика
в две строки

3
00:01:00,000 --> 00:01:02,000
Финал
"""

    def test_small_srt(self, engine):
        text = engine._prepare_subtitles_for_analysis(self.SMALL_SRT)
        lines = text.split("\n")

        assert len(lines) == 3
        assert lines[0].startswith("[00:00:01]")
        assert lines[0].endswith("Привет мир")

    def test_multiline_text_joined(self, engine):
        text = engine._prepare_subtitles_for_analysis(self.SMALL_SRT)
        assert "Вторая реплика в две строки" in text

    def test_milliseconds_stripped(self, engine):
        text = engine._prepare_subtitles_for_analysis(self.SMALL_SRT)
        # 00:00:04,200 → [00:00:04]
        assert "[00:00:04]" in text
        assert ",200]" not in text and ".200]" not in text

    def test_empty_input(self, engine):
        assert engine._prepare_subtitles_for_analysis("") == ""


class TestTimeToSeconds:
    def test_basic(self, engine):
        assert engine.time_to_seconds("00:01:30") == 90.0
        assert engine.time_to_seconds("01:00:00") == 3600.0
        assert engine.time_to_seconds("02:30:05.5") == 2 * 3600 + 30 * 60 + 5.5

    def test_get_clips_with_seconds(self, engine):
        data = {
            "clips": [
                {"start": "00:00:10", "end": "00:01:00", "hook_description": "x"},
            ]
        }
        clips = engine.get_clips_with_seconds(data)
        assert clips[0]["start_seconds"] == 10.0
        assert clips[0]["end_seconds"] == 60.0
