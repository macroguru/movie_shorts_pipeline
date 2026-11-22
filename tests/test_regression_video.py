"""
Регрессионные тесты VideoEditor: построение ffmpeg-фильтров без рендера
и один интеграционный smoke-тест с реальным ffmpeg (помечен slow).
"""

import shutil
import subprocess

import pytest

from video_editor import VideoEditor, VideoEditorError


class _FakeConfig:
    video_encoder = "cpu"  # смоук-тест без NVENC: детерминированный libx264


@pytest.fixture
def editor():
    return VideoEditor(_FakeConfig())


class TestBuildComplexFilter:
    def test_contains_scale_crop_overlay(self, editor):
        f = editor._build_complex_filter("Название", "1234", with_overlay=True)

        assert "scale=" in f
        assert "crop=1080:1920" in f
        assert "overlay=" in f
        assert "gblur" in f  # фон размывается

    def test_with_overlay_draws_code(self, editor):
        f = editor._build_complex_filter("Название", "1234", with_overlay=True)

        assert "drawtext" in f
        assert "1234" in f

    def test_without_overlay_no_drawtext(self, editor):
        f = editor._build_complex_filter("Название", "1234", with_overlay=False)

        assert "[centered]null[video]" in f
        assert "drawtext" not in f

    def test_layers_chained(self, editor):
        f = editor._build_complex_filter("Название", "1234", with_overlay=True)
        # Слои: bg → fg → centered → video
        assert "[bg][fg]overlay=" in f
        assert "[centered]" in f
        assert f.rstrip().endswith("[video]") or "[video]" in f


class TestTimeToSeconds:
    def test_hms(self, editor):
        assert editor._time_to_seconds("01:23:45") == 1 * 3600 + 23 * 60 + 45

    def test_two_part_interpreted_as_hh_mm(self, editor):
        # Текущая семантика: «02:30» = 2ч30м = 9000 (HH:MM), а не 150 (MM:СС).
        # core._time_string_to_seconds для той же строки даёт 150 — расхождение
        # задокументировано в отчёте QA; в пайплайне всегда HH:MM:SS.
        assert editor._time_to_seconds("02:30") == 9000

    def test_fractional_seconds(self, editor):
        assert editor._time_to_seconds("00:00:10.5") == 10.5


# ---------- Интеграционный smoke: реальный ffmpeg ----------

@pytest.mark.slow
@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe не установлены",
)
class TestFfmpegSmoke:
    def test_render_two_second_test_video(self, tmp_path):
        """2с lavfi testsrc2 → mp4 через минимальный фильтр, проверка ffprobe."""
        out = tmp_path / "smoke.mp4"

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25:duration=2",
            "-vf", "scale=160:120",  # минимальный фильтр
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-y", str(out),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        assert out.exists() and out.stat().st_size > 0

        probe = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", str(out),
            ],
            check=True, capture_output=True, text=True,
        )
        import json

        duration = float(json.loads(probe.stdout)["format"]["duration"])
        assert duration == pytest.approx(2.0, abs=0.3)

    def test_missing_file_raises(self, tmp_path):
        """Несуществующий вход → VideoEditorError."""
        editor = VideoEditor(_FakeConfig())
        missing = tmp_path / "no_such_video.mp4"

        with pytest.raises(VideoEditorError, match="не найден"):
            editor.render_short(
                video_path=str(missing),
                start="00:00:00",
                end="00:00:02",
                code="9999",
                clip_index=1,
                min_duration=0.0,
            )
