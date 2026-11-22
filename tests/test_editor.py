"""
Тесты модуля монтажа: пресеты EditPreset и генератор ASS-субтитров.
"""

import pytest

pytest.importorskip("editor")

from types import SimpleNamespace  # noqa: E402

from editor import (  # noqa: E402
    ACCENT_COLOR,
    EditPreset,
    build_ass,
    get_preset,
)


def make_editor_config(**overrides):
    """Конфиг для get_preset: те поля, что читает editor.py."""
    base = {
        "editing_preset": "auto",
        "output_fps": 30,
        "subtitle_style": "bold",
        "audio_normalize": True,
        "noise_reduction": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class TestGetPreset:
    def test_movie_preset(self):
        preset = get_preset("movie", make_editor_config())

        assert isinstance(preset, EditPreset)
        assert preset.name == "movie"
        assert preset.layout == "blur"
        assert preset.overlay_code is True
        assert preset.min_duration == 40.0
        assert preset.zoom == 1.0

    def test_clip_preset(self):
        preset = get_preset("clip", make_editor_config())

        assert preset.name == "clip"
        assert preset.layout == "crop"
        assert preset.overlay_code is False
        assert preset.min_duration == 0.0

    def test_stream_preset(self):
        preset = get_preset("stream", make_editor_config())
        assert preset.layout == "blur"
        assert preset.overlay_code is False
        assert preset.zoom == 1.05

    def test_off_preset_minimal(self):
        preset = get_preset("off", make_editor_config())

        assert preset.name == "off"
        assert preset.overlay_code is False
        assert preset.audio_normalize is False
        assert preset.noise_reduction is False
        assert preset.zoom == 1.0

    def test_unknown_source_falls_back_to_movie(self):
        preset = get_preset("совсем-не-пресет", make_editor_config())
        assert preset.name == "movie"

    def test_source_overrides_config_preset(self):
        # Явный source сильнее EDITING_PRESET в конфиге
        preset = get_preset("clip", make_editor_config(editing_preset="gaming"))
        assert preset.name == "clip"

    def test_config_preset_used_for_auto_source(self):
        # «auto» в source → берём пресет из конфига
        preset = get_preset("auto", make_editor_config(editing_preset="gaming"))
        assert preset.name == "gaming"

    def test_config_preset_used_for_none_source(self):
        preset = get_preset(None, make_editor_config(editing_preset="podcast"))
        assert preset.name == "podcast"

    def test_fps_kwarg_honored(self):
        # Явный fps сильнее конфига
        preset = get_preset("movie", make_editor_config(output_fps=30), fps=60)
        assert preset.fps == 60

    def test_fps_invalid_kwarg_falls_back_to_30(self):
        assert get_preset("movie", make_editor_config(), fps=25).fps == 30
        assert get_preset("movie", make_editor_config(), fps=1000).fps == 30

    def test_subtitle_style_kwarg_honored(self):
        preset = get_preset(
            "movie", make_editor_config(subtitle_style="bold"), subtitle_style="boxed"
        )
        assert preset.subtitle_style == "boxed"

    def test_fps_valid_values_honored(self):
        assert get_preset("movie", make_editor_config(output_fps=60)).fps == 60
        assert get_preset("movie", make_editor_config(output_fps=30)).fps == 30

    def test_fps_invalid_falls_back_to_30(self):
        # 25 и 1000 не из {30, 60} → откат на 30
        assert get_preset("movie", make_editor_config(output_fps=25)).fps == 30
        assert get_preset("movie", make_editor_config(output_fps=1000)).fps == 30

    def test_subtitle_style_from_config(self):
        preset = get_preset("movie", make_editor_config(subtitle_style="karaoke"))
        assert preset.subtitle_style == "karaoke"

    def test_subtitle_style_invalid_falls_back(self):
        preset = get_preset("movie", make_editor_config(subtitle_style="underline"))
        assert preset.subtitle_style == "bold"

    def test_audio_flags_from_config(self):
        preset = get_preset(
            "movie", make_editor_config(audio_normalize=False, noise_reduction=True)
        )
        assert preset.audio_normalize is False
        assert preset.noise_reduction is True


class TestBuildAss:
    def test_valid_ass_structure(self, tmp_path):
        out = tmp_path / "subs.ass"
        segments = [
            {"start": 1.5, "end": 3.2, "text": "Привет мир", "words": []},
            {"start": 5.0, "end": 6.5, "text": "Вторая реплика", "words": []},
        ]

        build_ass(segments, "bold", out)
        content = out.read_text(encoding="utf-8")

        assert "[Script Info]" in content
        assert "PlayResX: 1080" in content
        assert "PlayResY: 1920" in content
        assert "[V4+ Styles]" in content
        assert "[Events]" in content
        assert "Style: Default,Arial" in content

        dialogues = [ln for ln in content.split("\n") if ln.startswith("Dialogue:")]
        assert len(dialogues) == 2
        assert "Привет мир" in dialogues[0]

    def test_timecodes_formatted(self, tmp_path):
        out = tmp_path / "t.ass"
        build_ass([{"start": 3661.5, "end": 3663.0, "text": "Час"}], "bold", out)
        line = next(
            ln for ln in out.read_text(encoding="utf-8").split("\n")
            if ln.startswith("Dialogue:")
        )
        assert "1:01:01.50" in line
        assert "1:01:03.00" in line

    def test_style_name_in_style_line(self, tmp_path):
        out = tmp_path / "boxed.ass"
        build_ass([{"start": 0, "end": 1, "text": "Плашка"}], "boxed", out)
        content = out.read_text(encoding="utf-8")
        # boxed стиль: BorderStyle 3 (плашка), полупрозрачный фон
        assert "Style: Default,Arial,64," in content
        assert "&H80000000" in content

    def test_karaoke_words_get_k_tags(self, tmp_path):
        out = tmp_path / "k.ass"
        segments = [{
            "start": 10.0,
            "end": 12.0,
            "text": "вот это поворот",
            "words": [("вот", 10.0, 10.4), ("это", 10.5, 10.9),
                      ("поворот", 11.0, 12.0)],
        }]
        build_ass(segments, "karaoke", out)
        content = out.read_text(encoding="utf-8")

        dialogue = next(
            ln for ln in content.split("\n") if ln.startswith("Dialogue:")
        )
        assert "\\k" in dialogue       # тайминг по словам
        assert ACCENT_COLOR in dialogue  # ключевое слово акцентным цветом

    def test_karaoke_without_words_falls_back_to_plain(self, tmp_path):
        out = tmp_path / "kw.ass"
        build_ass([{"start": 0, "end": 1, "text": "Слово текст"}], "karaoke", out)
        content = out.read_text(encoding="utf-8")
        dialogue = next(
            ln for ln in content.split("\n") if ln.startswith("Dialogue:")
        )
        assert "\\k" not in dialogue
        assert ACCENT_COLOR in dialogue  # подсветка самых ёмких слов

    def test_text_escaped(self, tmp_path):
        out = tmp_path / "esc.ass"
        build_ass([{"start": 0, "end": 1, "text": "{странный\\текст}"}], "bold", out)
        dialogue = next(
            ln for ln in out.read_text(encoding="utf-8").split("\n")
            if ln.startswith("Dialogue:")
        )
        # Фигурные скобки вырезаются (служебные символы ASS), бэкслеш удваивается
        assert "{" not in dialogue and "}" not in dialogue
        assert "странный" in dialogue and "текст" in dialogue
        assert "\\\\" in dialogue

    def test_empty_text_skipped(self, tmp_path):
        out = tmp_path / "empty.ass"
        build_ass([{"start": 0, "end": 1, "text": "   "}], "bold", out)
        assert not any(
            ln.startswith("Dialogue:")
            for ln in out.read_text(encoding="utf-8").split("\n")
        )

    def test_end_before_start_fixed(self, tmp_path):
        out = tmp_path / "fix.ass"
        build_ass([{"start": 5, "end": 3, "text": "Наоборот"}], "bold", out)
        dialogue = next(
            ln for ln in out.read_text(encoding="utf-8").split("\n")
            if ln.startswith("Dialogue:")
        )
        assert "0:00:05.00,0:00:05.50" in dialogue  # end = start + 0.5

    def test_unknown_style_falls_back_to_bold(self, tmp_path):
        out = tmp_path / "u.ass"
        build_ass([{"start": 0, "end": 1, "text": "Х"}], "несуществующий", out)
        content = out.read_text(encoding="utf-8")
        assert "Style: Default,Arial,72," in content  # bold fontsize

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "deep" / "nested" / "subs.ass"
        build_ass([{"start": 0, "end": 1, "text": "Х"}], "bold", out)
        assert out.exists()
