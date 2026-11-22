"""
Тесты базы данных на временной SQLite-базе (tmp_path) — реальная база
проекта не трогается.
"""

import json

import pytest

from database import Database


@pytest.fixture
def db(tmp_path):
    return Database(db_path=str(tmp_path / "test_movies.db"))


class TestUniqueCode:
    def test_code_format(self, db):
        code = db.get_unique_code()
        assert isinstance(code, str)
        assert code.isdigit()
        assert 1000 <= int(code) <= 9999

    def test_codes_unique_with_retry(self, db, monkeypatch):
        # Детерминированный «генератор»: первый код занят → должен взять следующий
        sequence = iter(["1234", "1234", "5678"])
        import database as database_module

        monkeypatch.setattr(
            database_module.random,
            "randint",
            lambda a, b: int(next(sequence)),
        )

        db.save_movie(code="1234", title="Занятый код", year=2020, video_path="v.mp4")

        code = db.get_unique_code()
        assert code == "5678"

    def test_many_codes_distinct(self, db):
        codes = {db.get_unique_code() for _ in range(20)}
        # 20 бросков из 9000 значений — коллизии крайне маловероятны
        assert len(codes) >= 19


class TestSaveAndGet:
    def test_save_and_get(self, db):
        clip_paths = json.dumps(["c:/out/1234_1.mp4", "c:/out/1234_2.mp4"])
        record_id = db.save_movie(
            code="1234",
            title="Матрица",
            year=1999,
            genre="Триллер / Детектив",
            clip_paths=clip_paths,
            poster_path="c:/out/1234_poster.jpg",
            video_path="c:/in/movie.mkv",
        )

        assert isinstance(record_id, int) and record_id > 0

        row = db.get_by_code("1234")
        assert row is not None
        assert row["title"] == "Матрица"
        assert row["year"] == 1999
        assert row["genre"] == "Триллер / Детектив"
        assert json.loads(row["clip_paths"]) == ["c:/out/1234_1.mp4", "c:/out/1234_2.mp4"]
        assert row["poster_path"] == "c:/out/1234_poster.jpg"
        assert row["video_path"] == "c:/in/movie.mkv"
        assert row["tg_msg_id"] is None
        assert row["created_at"]

    def test_get_missing_code_returns_none(self, db):
        assert db.get_by_code("9999") is None

    def test_duplicate_code_rejected(self, db):
        db.save_movie(code="1234", title="Первый", year=2020, video_path="a.mp4")
        with pytest.raises(Exception):
            db.save_movie(code="1234", title="Дубль", year=2021, video_path="b.mp4")


class TestUpdateTelegramMsgId:
    def test_update_msg_id(self, db):
        db.save_movie(code="2222", title="Фильм", year=2020, video_path="v.mp4")

        db.update_telegram_msg_id("2222", 42)

        row = db.get_by_code("2222")
        assert row["tg_msg_id"] == 42

    def test_stats_counts_published(self, db):
        db.save_movie(code="1001", title="A", year=2020, video_path="a.mp4")
        db.save_movie(code="1002", title="B", year=2021, video_path="b.mp4")
        db.save_movie(code="1003", title="C", year=2022, video_path="c.mp4")
        db.update_telegram_msg_id("1002", 777)

        stats = db.get_stats()
        assert stats["total_movies"] == 3
        assert stats["published"] == 1
        assert stats["not_published"] == 2


class TestDelete:
    def test_delete_by_code(self, db):
        db.save_movie(code="3333", title="На удаление", year=2019, video_path="v.mp4")

        assert db.delete_by_code("3333") is True
        assert db.get_by_code("3333") is None

    def test_delete_missing_returns_false(self, db):
        assert db.delete_by_code("8888") is False

    def test_delete_does_not_touch_others(self, db):
        db.save_movie(code="4444", title="Остаюсь", year=2020, video_path="keep.mp4")
        db.save_movie(code="4445", title="Удаляюсь", year=2020, video_path="gone.mp4")

        db.delete_by_code("4445")

        assert db.get_by_code("4444") is not None
        assert db.get_stats()["total_movies"] == 1


class TestIsCodeExists:
    def test_exists_after_save(self, db):
        db.save_movie(code="5555", title="X", year=2020, video_path="v.mp4")
        assert db.is_code_exists("5555") is True
        assert db.is_code_exists("5556") is False
