"""The reason this tool exists: an interrupted run continues where it stopped and ends
with the same output as a run that was never interrupted."""

from pathlib import Path

import pytest

from lecturize.device import Device
from lecturize.engine import AudioInfo
from lecturize.model import Utterance
from lecturize.pipeline import Events, Job, run

SPEECH = [Utterance(float(i * 3), float(i * 3 + 2.5), f"sentence {i}.") for i in range(50)]
CPU = Device("cpu", "int8")


class FakeEngine:
    """Yields the scripted utterances after `resume_from`; dies after `crash_after` of them."""

    def __init__(self, crash_after: int | None = None):
        self.crash_after = crash_after
        self.calls: list[float] = []

    def transcribe(
        self, audio, *, language=None, task="transcribe", vad=True, resume_from=0.0, on_info=None
    ):
        self.calls.append(resume_from)
        if on_info:
            on_info(AudioInfo("en", 0.99, 150.0))
        produced = 0
        for u in SPEECH:
            if u.start < resume_from:
                continue
            if self.crash_after is not None and produced >= self.crash_after:
                raise ConnectionError("power cut")
            produced += 1
            yield u


def _job(recording, out, **kw):
    return Job(recording, model="tiny", formats=("srt", "md"), out_dir=out, **kw)


def test_resume_matches_clean_run(recording, tmp_path):
    job = _job(recording, tmp_path / "out")

    with pytest.raises(ConnectionError):
        run(job, FakeEngine(crash_after=27), CPU)

    messages: list[str] = []
    resumed = FakeEngine()
    outputs = run(job, resumed, CPU, Events(message=messages.append))
    assert resumed.calls == [pytest.approx(SPEECH[19].end)]  # 20 utterances were flushed
    assert any("resuming from" in m for m in messages)

    clean = run(_job(recording, tmp_path / "clean", fresh=True), FakeEngine(), CPU)
    for a, b in zip(outputs, clean, strict=True):
        assert a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8")
    assert outputs[0].read_text(encoding="utf-8").count("-->") == 50


def test_finished_job_is_not_transcribed_again(recording, tmp_path):
    job = Job(recording, model="tiny", formats=("txt",), out_dir=tmp_path)
    run(job, FakeEngine(), CPU)
    again = FakeEngine()
    messages: list[str] = []
    run(job, again, CPU, Events(message=messages.append))
    assert again.calls == []
    assert any("reusing" in m for m in messages)


def test_fresh_ignores_checkpoint(recording, tmp_path):
    run(Job(recording, model="tiny", formats=("txt",), out_dir=tmp_path), FakeEngine(), CPU)
    again = FakeEngine()
    run(Job(recording, model="tiny", formats=("txt",), out_dir=tmp_path, fresh=True), again, CPU)
    assert again.calls == [0.0]


def test_translation_to_english_uses_whisper(recording):
    assert Job(recording, target="en").task == "translate"
    assert Job(recording, target="it").task == "transcribe"
    assert Job(recording, target="en").settings() != Job(recording).settings()


def test_outputs_written_where_asked(recording, tmp_path):
    outs = run(
        Job(recording, formats=("md", "docx"), out_dir=tmp_path / "notes"), FakeEngine(), CPU
    )
    assert [p.name for p in outs] == ["talk.md", "talk.docx"]
    assert all(p.parent == Path(tmp_path / "notes") for p in outs)
