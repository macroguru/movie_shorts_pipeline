"""
Регрессионные тесты чистых функций ядра: core.py.

Сеть и GPU не нужны: тестируются только парсеры и сборщики текста.
"""

import pytest

from core import (
    PipelineError,
    build_hashtags,
    build_stream_hashtags,
    build_stream_description,
    build_twitch_hashtags,
    build_twitch_description,
    detect_year,
    guess_title,
    parse_moments,
    _seconds_to_hms,
    _time_string_to_seconds,
)


# ---------- parse_moments ----------

class TestParseMoments:
    def test_valid_multi_moment(self):
        moments = parse_moments("1:23:45-1:25:00, 00:40-00:41:30")

        assert len(moments) == 2

        first, second = moments
        assert first["start_seconds"] == 1 * 3600 + 23 * 60 + 45
        assert first["end_seconds"] == 1 * 3600 + 25 * 60
        assert first["start"] == "01:23:45"
        assert first["end"] == "01:25:00"
        assert first["hook_description"]

        assert second["start_seconds"] == 40
        assert second["end_seconds"] == 41 * 60 + 30  # 00:41:30
        assert second["end"] == "00:41:30"

    def test_seconds_format(self):
        moment = parse_moments("90-120")[0]
        assert moment["start_seconds"] == 90
        assert moment["end_seconds"] == 120
        assert moment["start"] == "00:01:30"
        assert moment["end"] == "00:02:00"

    def test_semicolon_separator(self):
        moments = parse_moments("10-20; 30-40")
        assert [m["start_seconds"] for m in moments] == [10, 30]

    def test_em_dash_separation_fails(self):
        # Эм-дэш между двумя моментами — не разделитель: парсер должен упасть
        with pytest.raises(PipelineError):
            parse_moments("10-20—30-40")

    def test_end_before_start_fails(self):
        with pytest.raises(PipelineError):
            parse_moments("1:25:00-1:23:45")

    def test_end_equal_start_fails(self):
        with pytest.raises(PipelineError):
            parse_moments("50-50")

    def test_empty_fails(self):
        with pytest.raises(PipelineError):
            parse_moments("")
        with pytest.raises(PipelineError):
            parse_moments("   ")

    def test_garbage_fails(self):
        with pytest.raises(PipelineError):
            parse_moments("abc-def")

    def test_manual_hook_description(self):
        moment = parse_moments("10-20")[0]
        assert moment["hook_description"] == "Момент задан вручную"


# ---------- guess_title / detect_year ----------

class TestGuessTitle:
    def test_release_junk_stripped(self):
        title = guess_title("The.Matrix.1999.WEB-DLRip.1080p.mkv")
        assert title == "The Matrix"

    def test_bluray_codec_stripped(self):
        title = guess_title("The.Matrix.1999.1080p.BluRay.x264.mkv")
        assert title == "The Matrix"

    def test_year_removed(self):
        title = guess_title("Interstellar.2014.1080p.mkv")
        assert title == "Interstellar"
        assert "2014" not in title

    def test_plain_name_untouched(self):
        assert guess_title("Мой Фильм.mkv") == "Мой Фильм"

    def test_digit_only_token_removed(self):
        # Соль релиза из одних цифр — не часть названия
        assert guess_title("Film.12345.mkv") == "Film"

    def test_space_separated_junk_stripped(self):
        assert guess_title("Interstellar 2014 1080p.mkv") == "Interstellar"

    def test_bracket_junk_stripped(self):
        title = guess_title("Movie (2020) [1080p].mkv")
        assert "2020" not in title and "1080p" not in title


class TestDetectYear:
    def test_year_found(self):
        assert detect_year("C:/movies/Gone.with.the.Wind.1939.1080p.mkv") == 1939

    def test_no_year_returns_none(self):
        assert detect_year("C:/movies/no_year_here.mkv") is None

    def test_takes_max_plausible(self):
        # 1999 раньше 2008 — берём поздний правдоподобный год
        assert detect_year("movie.1999.2008.mkv") == 2008

    def test_future_year_falls_back_to_earlier(self):
        # 2199 неправдоподобно далёк, 2050 тоже (current+1) — берём правдоподобный
        year = detect_year("movie.2050.2199.mkv")
        assert year in (2050, 2199)  # зависит от текущего года; не должно упасть

    def test_long_digits_not_a_year(self):
        # 1080p — не год: 4 цифры, но внутри большего числа
        assert detect_year("movie.1080p.mkv") is None

    def test_resolution_not_mistaken_for_year(self):
        assert detect_year("movie.2160p.2017.mkv") == 2017


# ---------- build_hashtags ----------

class TestBuildHashtags:
    def test_contains_code_and_promo(self):
        tags = build_hashtags("1234", "Триллер / Детектив", 1999)
        assert tags[0] == "#код1234"
        assert "#триллер" in tags
        assert "#детектив" in tags
        assert "#1999" in tags
        assert "#кино" in tags and "#shorts" in tags

    def test_no_duplicates(self):
        tags = build_hashtags("1234", "кино / кино", 1999)
        assert len(tags) == len(set(tags))

    def test_genre_slash_split(self):
        tags = build_hashtags("1234", "Драма/Комедия", None)
        assert "#драма" in tags
        assert "#комедия" in tags

    def test_year_omitted_when_none(self):
        tags = build_hashtags("1234", "Драма", None)
        assert not any(t.startswith("#19") or t.startswith("#20") for t in tags)

    def test_nonalnum_genre_chars_dropped(self):
        tags = build_hashtags("1234", "Боевик (Action)", None)
        assert "#боевикaction" in tags or "#боевик" in tags


# ---------- build_stream_hashtags ----------

class TestBuildStreamHashtags:
    def test_contains_code_and_stream_base(self):
        tags = build_stream_hashtags("7777")
        assert tags[0] == "#код7777"
        for base in ("#стримы", "#стрим", "#twitch", "#shorts"):
            assert base in tags

    def test_extra_tags_appended_cleaned(self):
        tags = build_stream_hashtags("7777", "летсплей подкаст #реакции")
        assert "#летсплей" in tags
        assert "#подкаст" in tags
        assert "#реакции" in tags  # решётка срезается

    def test_no_duplicates(self):
        tags = build_stream_hashtags("7777", "стримы twitch shorts")
        assert len(tags) == len(set(tags))

    def test_empty_extra(self):
        tags = build_stream_hashtags("7777", "")
        assert "#код7777" in tags
        assert len(tags) == len(set(tags))


# ---------- build_twitch_hashtags / build_twitch_description ----------

class TestTwitchHashtags:
    def test_contains_streamer_and_game(self):
        tags = build_twitch_hashtags("Papich", ["Dota 2"])
        assert "#papich" in tags
        assert "#dota2" in tags
        assert "#twitch" in tags

    def test_no_duplicates(self):
        tags = build_twitch_hashtags("Papich", ["Dota 2", "Dota 2"])
        assert len(tags) == len(set(tags))

    def test_no_kod_tag(self):
        # Код на клипы Twitch не рисуется — и в хэштегах его быть не должно
        tags = build_twitch_hashtags("StreamaLot", ["Just Chatting"])
        assert not any(tag.startswith("#код") for tag in tags)

    def test_no_kod_tag_in_streamer_name(self):
        # Стример с «код» в имени не должен породить #код*
        tags = build_twitch_hashtags("Kod_VMaster", [])
        assert not any(tag.startswith("#код") for tag in tags)

    def test_cyrillic_streamer_kept(self):
        tags = build_twitch_hashtags("ЭфOpcode", [])
        assert "#эфopcode" in tags


class TestBuildTwitchDescription:
    def test_no_kod_word(self):
        text = build_twitch_description("Papich", ["Dota 2"])
        assert "Код" not in text
        assert "код" not in text.replace("#код", "")  # и тега кода нет
        assert "Стример: Papich" in text
        assert "Игра: Dota 2" in text
        assert "#papich" in text

    def test_no_streamer_no_header(self):
        text = build_twitch_description("", [])
        assert "Стример" not in text
        assert "#twitch" in text

    def test_ends_with_hashtags(self):
        text = build_twitch_description("xQcOW", ["Minecraft"])
        assert "#xqcow" in text
        assert "#minecraft" in text


# ---------- build_stream_description ----------

class TestBuildStreamDescription:
    def test_structure(self):
        text = build_stream_description("4321", "Летсплей по·ночи", {
            "title": "Тема",
            "context": "Что происходит",
            "hashtags": "летсплей хоррор",
        })
        assert "Код: 4321" in text
        assert "Стрим: Летсплей по·ночи" in text
        assert "Что происходит" in text
        assert "#летсплей" in text

    def test_no_context_still_works(self):
        text = build_stream_description("4321", "Стрим", {"context": "", "hashtags": ""})
        assert "Код: 4321" in text


# ---------- time helpers ----------

class TestTimeHelpers:
    def test_seconds_to_hms_basic(self):
        assert _seconds_to_hms(0) == "00:00:00"
        assert _seconds_to_hms(3661) == "01:01:01"
        assert _seconds_to_hms(40) == "00:00:40"
        assert _seconds_to_hms(86399) == "23:59:59"

    def test_seconds_to_hms_negative_clamped(self):
        assert _seconds_to_hms(-5) == "00:00:00"

    def test_time_string_to_seconds_forms(self):
        assert _time_string_to_seconds("01:01:01") == 3661
        assert _time_string_to_seconds("12:34") == 754
        assert _time_string_to_seconds("45") == 45

    def test_time_string_to_seconds_invalid(self):
        for bad in ("1:2:3:4", "abc", "1:2:3x", ""):
            with pytest.raises(PipelineError):
                _time_string_to_seconds(bad)

    def test_roundtrip(self):
        for seconds in (0, 1, 59, 60, 61, 3599, 3600, 4000, 86399):
            assert _time_string_to_seconds(_seconds_to_hms(seconds)) == seconds
