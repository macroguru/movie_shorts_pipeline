"""
Регрессионные тесты Twitch-хелперов: разбор ссылок и таймкодов без сети.
"""

import pytest

from twitch_downloader import (
    TwitchDownloader,
    is_clip_url,
    parse_clip_slug,
)


class TestIsClipUrl:
    def test_clips_domain_true(self):
        assert is_clip_url("https://clips.twitch.tv/AbCdEf") is True

    def test_channel_clip_true(self):
        assert is_clip_url("https://www.twitch.tv/papich/clip/AbCdEf?filter=clips") is True

    def test_case_insensitive(self):
        assert is_clip_url("HTTPS://CLIPS.TWITCH.TV/XyZ") is True
        assert is_clip_url("https://www.twitch.tv/user/CLIP/slug") is True

    def test_vod_false(self):
        assert is_clip_url("https://www.twitch.tv/videos/123456789") is False

    def test_other_site_false(self):
        assert is_clip_url("https://youtube.com/watch?v=abc") is False

    def test_empty_and_none_false(self):
        assert is_clip_url("") is False
        assert is_clip_url(None) is False


class TestParseClipSlug:
    def test_slug_case_preserved(self):
        # Slug чувствителен к регистру — должен остаться как есть
        assert parse_clip_slug("https://clips.twitch.tv/AbCdEf") == "AbCdEf"

    def test_embed_clip_param(self):
        slug = parse_clip_slug("https://clips.twitch.tv/embed?clip=HappyTiger42")
        assert slug == "HappyTiger42"

    def test_channel_clip_path(self):
        slug = parse_clip_slug("https://www.twitch.tv/Papich/clip/ConcernedPlayfulOx99")
        assert slug == "ConcernedPlayfulOx99"

    def test_query_suffix_ignored(self):
        slug = parse_clip_slug("https://clips.twitch.tv/PlainSlug?filter=clips&tt_medium=red")
        assert slug == "PlainSlug"

    def test_vod_returns_none(self):
        assert parse_clip_slug("https://www.twitch.tv/videos/123456") is None

    def test_garbage_returns_none(self):
        assert parse_clip_slug("не ссылка вовсе") is None
        assert parse_clip_slug("") is None
        assert parse_clip_slug(None) is None


class TestParseDuration:
    @pytest.fixture
    def downloader(self, tmp_path):
        return TwitchDownloader(tmp_path / "twitch_out")

    def test_hours_minutes_seconds(self, downloader):
        assert downloader._parse_duration("1h23m45s") == 1 * 3600 + 23 * 60 + 45

    def test_minutes_only(self, downloader):
        assert downloader._parse_duration("5m") == 300

    def test_seconds_only(self, downloader):
        assert downloader._parse_duration("30s") == 30

    def test_bare_number_is_seconds(self, downloader):
        assert downloader._parse_duration("90") == 90

    def test_case_insensitive(self, downloader):
        assert downloader._parse_duration("2H3M4S") == 2 * 3600 + 3 * 60 + 4

    def test_garbage_returns_none(self, downloader):
        assert downloader._parse_duration("abc") is None
        assert downloader._parse_duration("") is None

    def test_vod_timestamp(self, downloader):
        url = "https://www.twitch.tv/videos/123456?t=1h23m45s"
        assert downloader.parse_vod_timestamp(url) == 5025.0

    def test_vod_timestamp_minutes(self, downloader):
        url = "https://www.twitch.tv/videos/123456?t=5m30s"
        assert downloader.parse_vod_timestamp(url) == 330.0

    def test_vod_without_timestamp(self, downloader):
        url = "https://www.twitch.tv/videos/123456"
        assert downloader.parse_vod_timestamp(url) is None

    def test_downloader_creates_output_dir(self, tmp_path):
        out = tmp_path / "nested" / "twitch"
        TwitchDownloader(out)
        assert out.is_dir()
