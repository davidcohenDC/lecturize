from pathlib import Path

from lecturize.checkpoint import Checkpoint, JobKey
from lecturize.model import Utterance

KEY = JobKey("abc", {"model": "tiny"})
OTHER = JobKey("abc", {"model": "small"})


def test_append_and_resume_point(tmp_path):
    with Checkpoint(tmp_path / "ck.db") as ck:
        ck.open_job(KEY, Path("talk.mp3"))
        assert ck.last_end(KEY) == 0.0
        ck.append(KEY, [Utterance(0, 2.5, "a"), Utterance(2.5, 4.0, "b")])
        ck.append(KEY, [Utterance(4.0, 6.0, "c")])
        assert ck.last_end(KEY) == 6.0
        assert [u.text for u in ck.utterances(KEY)] == ["a", "b", "c"]
        assert ck.utterances(OTHER) == []  # a different model is a different job


def test_pending_done_forget(tmp_path):
    with Checkpoint(tmp_path / "ck.db") as ck:
        ck.open_job(KEY, Path("talk.mp3"))
        ck.append(KEY, [Utterance(0, 3, "a")])
        assert ck.pending() == [("talk.mp3", KEY.value, 3.0)]
        ck.set_info(KEY, "en", 120.0)
        assert ck.info(KEY) == ("en", 120.0, False)
        ck.mark_done(KEY)
        assert ck.pending() == []
        assert ck.info(KEY)[2] is True
        ck.forget(KEY)
        assert ck.info(KEY) == (None, None, False)
        assert ck.utterances(KEY) == []


def test_reopen_keeps_data(tmp_path):
    db = tmp_path / "ck.db"
    with Checkpoint(db) as ck:
        ck.open_job(KEY, Path("talk.mp3"))
        ck.append(KEY, [Utterance(0, 1, "hello")])
    with Checkpoint(db) as ck:
        assert ck.utterances(KEY) == [Utterance(0, 1, "hello")]
