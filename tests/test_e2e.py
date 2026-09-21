"""Real engine on a 37 second synthetic lecture. Downloads the tiny model (75 MB)."""

import wave

import pytest

from lecturize.checkpoint import Checkpoint, JobKey
from lecturize.device import Device
from lecturize.engine import Engine
from lecturize.fingerprint import fingerprint
from lecturize.paths import checkpoint_db
from lecturize.pipeline import Job, run

pytestmark = pytest.mark.slow
CPU = Device("cpu", "int8")


def _job(sample_wav, out):
    return Job(sample_wav, model="tiny", language="en", formats=("txt", "srt"), out_dir=out)


def test_tiny_transcribes_the_sample(sample_wav, tmp_path):
    outs = run(_job(sample_wav, tmp_path), Engine("tiny", CPU))
    text = outs[0].read_text(encoding="utf-8").lower()
    assert "1986" in text and "controller" in text and "architecture" in text
    assert outs[1].read_text(encoding="utf-8").count("-->") >= 5


def test_resume_with_real_engine(sample_wav, tmp_path):
    """Keep what ended before 20 s, forget the rest, run again: the timeline lines up."""
    engine = Engine("tiny", CPU)
    job = _job(sample_wav, tmp_path / "a")
    clean = run(job, engine)[1].read_text(encoding="utf-8")

    key = JobKey(fingerprint(sample_wav), job.settings())
    with Checkpoint(checkpoint_db()) as ck:
        kept = [u for u in ck.utterances(key) if u.end <= 20]
        ck.forget(key)
        ck.open_job(key, sample_wav, "tiny")
        ck.append(key, kept)
    resumed = run(_job(sample_wav, tmp_path / "b"), engine)[1].read_text(encoding="utf-8")

    def cues(s: str) -> list[str]:
        return [line for line in s.splitlines() if "-->" in line]

    with wave.open(str(sample_wav)) as w:
        duration = w.getnframes() / w.getframerate()
    last_end = cues(resumed)[-1].split(" --> ")[1]  # "00:00:36,400"
    h, m, s_ms = last_end.split(":")
    seconds = int(h) * 3600 + int(m) * 60 + float(s_ms.replace(",", "."))
    assert cues(resumed)[: len(kept)] == cues(clean)[: len(kept)]
    assert seconds >= duration - 3


def test_language_check_overrules_the_detector(sample_wav, tmp_path):
    """The detector calls this synthetic English voice Latin; decoding says English."""
    from lecturize.pipeline import Events

    class Rec(Events):
        guess = None

        def language(self, g):
            self.guess = g

    rec = Rec()
    job = Job(sample_wav, model="tiny", formats=("txt",), out_dir=tmp_path)
    run(job, Engine("tiny", CPU), rec)
    assert rec.guess is not None
    assert rec.guess.language == "en"
