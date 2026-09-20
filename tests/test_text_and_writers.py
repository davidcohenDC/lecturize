import pytest

from lecturize.model import Transcript, Utterance
from lecturize.text import chapters, hms
from lecturize.writers import FORMATS, write

U = [
    Utterance(0.0, 2.0, "First sentence here."),
    Utterance(2.0, 4.0, "Second one follows."),
    Utterance(400.0, 402.0, "After a long pause, a new topic."),
    Utterance(402.0, 405.5, "Still the new topic."),
]


def test_chapters_split_on_long_pause_after_enough_time():
    ch = chapters(U, "en", every=300, gap=4)
    assert [c.start for c in ch] == [0.0, 400.0]
    assert ch[0].paragraphs == ["First sentence here. Second one follows."]


def test_chapters_do_not_split_before_every():
    ch = chapters(U[:2] + [Utterance(100, 101, "Soon.")], "en", every=300, gap=4)
    assert len(ch) == 1


def test_hms():
    assert hms(0) == "00:00:00" and hms(3725.9) == "01:02:05"


def test_every_writer_produces_a_file(tmp_path):
    t = Transcript(U, language="en", duration=405.5)
    for fmt in FORMATS:
        out = write(t, tmp_path / f"talk.{fmt}", fmt, title="Talk")
        assert out.exists() and out.stat().st_size > 0


def test_srt_times_and_md_chapters(tmp_path):
    t = Transcript(U, language="en", duration=405.5)
    srt = write(t, tmp_path / "t.srt", "srt").read_text(encoding="utf-8")
    assert "00:06:40,000 --> 00:06:42,000" in srt
    assert srt.count("-->") == 4
    md = write(t, tmp_path / "t.md", "md", title="Talk").read_text(encoding="utf-8")
    assert md.startswith("# Talk") and "## 00:06:40" in md


def test_unknown_format(tmp_path):
    with pytest.raises(ValueError):
        write(Transcript([], "en", 0), tmp_path / "x.pdf", "pdf")
