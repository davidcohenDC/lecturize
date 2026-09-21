from pathlib import Path

from lecturize.checkpoint import Checkpoint, JobKey
from lecturize.model import Utterance

KEY = JobKey("abc", {"model": "tiny"})
OTHER = JobKey("abc", {"model": "small"})


def test_append_and_last_end(tmp_path):
    with Checkpoint(tmp_path / "ck.db") as ck:
        ck.open_job(KEY, Path("talk.mp3"), "tiny")
        assert ck.last_end(KEY) == 0.0
        ck.append(KEY, [Utterance(0, 2.5, "a"), Utterance(2.5, 4.0, "b")])
        ck.append(KEY, [Utterance(4.0, 6.0, "c")])
        assert ck.last_end(KEY) == 6.0
        assert [u.text for u in ck.utterances(KEY)] == ["a", "b", "c"]
        assert ck.utterances(OTHER) == []  # a different model is a different job


def test_pending_done_forget(tmp_path):
    with Checkpoint(tmp_path / "ck.db") as ck:
        ck.open_job(KEY, Path("talk.mp3"), "tiny")
        ck.append(KEY, [Utterance(0, 3, "a")])
        assert ck.pending() == [("talk.mp3", "tiny", KEY.value, 3.0)]
        ck.set_info(KEY, "en", 120.0)
        assert ck.info(KEY) == ("en", 120.0, False)
        ck.mark_done(KEY)
        assert ck.pending() == []
        assert ck.info(KEY)[2] is True
        ck.forget(KEY)
        assert ck.info(KEY) == (None, None, False)
        assert ck.utterances(KEY) == []


def test_clear_pending_keeps_done_jobs(tmp_path):
    with Checkpoint(tmp_path / "ck.db") as ck:
        ck.open_job(KEY, Path("a.mp3"), "tiny")
        ck.open_job(OTHER, Path("b.mp3"), "small")
        ck.mark_done(OTHER)
        assert ck.clear_pending() == 1
        assert ck.info(KEY) == (None, None, False)
        assert ck.info(OTHER)[2] is True


def test_open_job_updates_moved_source(tmp_path):
    with Checkpoint(tmp_path / "ck.db") as ck:
        ck.open_job(KEY, Path("/old/talk.mp3"), "tiny")
        ck.append(KEY, [Utterance(0, 1, "x")])
        ck.open_job(KEY, Path("/new/talk.mp3"), "tiny")
        assert ck.pending()[0][0].endswith("talk.mp3")
        assert "new" in ck.pending()[0][0]
        assert ck.utterances(KEY) == [Utterance(0, 1, "x")]


def test_reopen_keeps_data(tmp_path):
    db = tmp_path / "ck.db"
    with Checkpoint(db) as ck:
        ck.open_job(KEY, Path("talk.mp3"))
        ck.append(KEY, [Utterance(0, 1, "hello")])
    with Checkpoint(db) as ck:
        assert ck.utterances(KEY) == [Utterance(0, 1, "hello")]


def test_migrates_a_database_from_1_0_0(tmp_path):
    import sqlite3

    db = tmp_path / "old.db"
    with sqlite3.connect(db) as c:
        c.executescript(
            "CREATE TABLE jobs (key TEXT PRIMARY KEY, source TEXT NOT NULL, settings TEXT NOT NULL,"
            " language TEXT, duration REAL, done INTEGER NOT NULL DEFAULT 0);"
            "INSERT INTO jobs (key, source, settings) VALUES ('k', 'talk.mp3', '{}');"
        )
    with Checkpoint(db) as ck:
        assert ck.pending() == [("talk.mp3", "", "k", 0.0)]
        assert ck.conn.execute("PRAGMA user_version").fetchone()[0] == 1
