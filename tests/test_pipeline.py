"""The resume logic with a scripted engine: an interrupted run continues where it stopped and
ends with the same output as a run that was never interrupted."""

from pathlib import Path

import pytest

from lecturize.device import Device
from lecturize.engine import AudioInfo, LanguageGuess
from lecturize.model import Utterance
from lecturize.pipeline import Events, Interrupted, Job, run

SPEECH = [Utterance(float(i * 3), float(i * 3 + 2.5), f"sentence {i}.") for i in range(50)]


class FakeEngine:
    """Yields the scripted utterances after `resume_from`; dies after `crash_after` of them."""

    device = Device("cpu", "int8")

    def __init__(self, crash_after: int | None = None, interrupt_after: int | None = None):
        self.crash_after = crash_after
        self.interrupt_after = interrupt_after
        self.calls: list[tuple[float, str | None]] = []

    def transcribe(
        self,
        audio,
        *,
        language=None,
        task="transcribe",
        vad=True,
        resume_from=0.0,
        on_info=None,
        on_language=None,
    ):
        self.calls.append((resume_from, language))
        if language is None and on_language:
            on_language(LanguageGuess("en", "la", 0.6, {"la": -0.4, "en": -0.2}, "sentence 0."))
        if on_info:
            on_info(AudioInfo("en", 0.99, 150.0))
        produced = 0
        for u in SPEECH:
            if u.start < resume_from:
                continue
            if self.crash_after is not None and produced >= self.crash_after:
                raise ConnectionError("power cut")
            if self.interrupt_after is not None and produced >= self.interrupt_after:
                raise KeyboardInterrupt
            produced += 1
            yield u


class Recorder(Events):
    def __init__(self):
        self.messages: list[str] = []
        self.guesses: list[LanguageGuess] = []

    def message(self, text):
        self.messages.append(text)

    def language(self, guess):
        self.guesses.append(guess)


def _job(recording, out, **kw):
    kw.setdefault("formats", ("srt", "md"))
    return Job(recording, model="tiny", out_dir=out, **kw)


def test_resume_matches_clean_run(recording, tmp_path):
    job = _job(recording, tmp_path / "out")
    with pytest.raises(ConnectionError):
        run(job, FakeEngine(crash_after=27))

    rec = Recorder()
    resumed = FakeEngine()
    outputs = run(job, resumed, rec)
    assert resumed.calls == [(pytest.approx(SPEECH[26].end), "en")]  # every utterance flushed
    assert any("resuming from" in m for m in rec.messages)

    clean = run(_job(recording, tmp_path / "clean", fresh=True), FakeEngine())
    for a, b in zip(outputs, clean, strict=True):
        assert a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8")
    assert outputs[0].read_text(encoding="utf-8").count("-->") == 50


def test_resume_uses_the_saved_language(recording, tmp_path):
    job = _job(recording, tmp_path)
    with pytest.raises(ConnectionError):
        run(job, FakeEngine(crash_after=3))
    again = FakeEngine()
    run(job, again)
    assert again.calls[0][1] == "en"  # not detected a second time


def test_ctrl_c_keeps_progress(recording, tmp_path):
    job = _job(recording, tmp_path)
    with pytest.raises(Interrupted) as e:
        run(job, FakeEngine(interrupt_after=10))
    assert e.value.seconds == pytest.approx(SPEECH[9].end)
    again = FakeEngine()
    run(job, again)
    assert again.calls[0][0] == pytest.approx(SPEECH[9].end)


def test_done_job_skips_the_engine(recording, tmp_path):
    job = _job(recording, tmp_path, formats=("txt",))
    run(job, FakeEngine())
    again, rec = FakeEngine(), Recorder()
    run(job, again, rec)
    assert again.calls == []
    assert any("reusing" in m for m in rec.messages)


def test_fresh_ignores_checkpoint(recording, tmp_path):
    run(_job(recording, tmp_path, formats=("txt",)), FakeEngine())
    again = FakeEngine()
    run(_job(recording, tmp_path, formats=("txt",), fresh=True, overwrite=True), again)
    assert again.calls[0][0] == 0.0


def test_language_guess_is_reported_once(recording, tmp_path):
    rec = Recorder()
    run(_job(recording, tmp_path), FakeEngine(), rec)
    assert [g.language for g in rec.guesses] == ["en"]
    assert rec.guesses[0].overruled


def test_existing_outputs_are_kept_unless_overwrite(recording, tmp_path):
    job = _job(recording, tmp_path, formats=("md",))
    (tmp_path / "talk.md").write_text("my edits", encoding="utf-8")
    rec = Recorder()
    assert run(job, FakeEngine(), rec) == []
    assert (tmp_path / "talk.md").read_text(encoding="utf-8") == "my edits"
    assert any("kept talk.md" in m for m in rec.messages)
    assert run(_job(recording, tmp_path, formats=("md",), overwrite=True), FakeEngine()) == [
        tmp_path / "talk.md"
    ]


def test_to_english_uses_whisper_and_names_the_file(recording, tmp_path):
    job = _job(recording, tmp_path, to="en", formats=("txt",))
    assert job.task == "translate"
    assert job.settings() != _job(recording, tmp_path).settings()
    outs = run(job, FakeEngine())
    assert [p.name for p in outs] == ["talk.en.txt"]


def test_outputs_go_where_asked(recording, tmp_path):
    outs = run(Job(recording, formats=("md", "docx"), out_dir=tmp_path / "notes"), FakeEngine())
    assert [p.name for p in outs] == ["talk.md", "talk.docx"]
    assert all(p.parent == Path(tmp_path / "notes") for p in outs)


def test_empty_transcript_writes_nothing(recording, tmp_path):
    class Silent(FakeEngine):
        def transcribe(self, audio, **kw):
            if kw.get("on_info"):
                kw["on_info"](AudioInfo("en", 0.9, 10.0))
            return iter(())

    rec = Recorder()
    assert run(_job(recording, tmp_path, language="en"), Silent(), rec) == []
    assert any("no speech" in m for m in rec.messages)
