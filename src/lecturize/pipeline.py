"""One recording in, notes out."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .checkpoint import Checkpoint, JobKey
from .engine import AudioInfo, Engine, LanguageGuess
from .fingerprint import fingerprint
from .model import Transcript, Utterance
from .paths import checkpoint_db
from .translate import NoRouteError, Translator
from .writers import write

DEFAULT_FORMATS = ("md", "srt")


@dataclass
class Job:
    source: Path
    model: str = "small"
    language: str | None = None  # None = detect
    to: str | None = None  # translate into this language, None = keep
    formats: tuple[str, ...] = DEFAULT_FORMATS
    out_dir: Path | None = None
    vad: bool = True
    fresh: bool = False  # ignore a previous checkpoint
    overwrite: bool = False  # replace output files that already exist

    @property
    def task(self) -> str:
        return "translate" if self.to == "en" else "transcribe"

    def settings(self) -> dict:
        return {
            "engine": "faster-whisper",
            "model": self.model,
            "language": self.language,
            "task": self.task,
            "vad": self.vad,
        }


class Events:
    """Progress hooks. The CLI overrides what it wants to show; defaults do nothing."""

    def language(self, guess: LanguageGuess) -> None: ...
    def info(self, info: AudioInfo, resumed_from: float) -> None: ...
    def progress(self, seconds_done: float) -> None: ...
    def download(self, label: str, done: int, total: int) -> None: ...
    def message(self, text: str) -> None: ...


class Interrupted(Exception):
    """Ctrl+C during transcription; what was done is in the checkpoint."""

    def __init__(self, seconds: float):
        super().__init__(f"stopped at {seconds:.0f}s")
        self.seconds = seconds


def run(job: Job, engine: Engine, events: Events | None = None) -> list[Path]:
    ev = events or Events()
    transcript = transcribe(job, engine, ev)
    outputs = _write_all(transcript, job, ev, suffix=".en" if job.task == "translate" else "")
    if job.to and job.to != transcript.language:
        outputs += _write_all(translate(transcript, job, engine, ev), job, ev, suffix=f".{job.to}")
    return outputs


def transcribe(job: Job, engine: Engine, ev: Events) -> Transcript:
    key = JobKey(fingerprint(job.source), job.settings())
    with Checkpoint(checkpoint_db()) as ck:
        if job.fresh:
            ck.forget(key)
        ck.open_job(key, job.source.resolve(), job.model)
        language, duration, done = ck.info(key)
        utterances = ck.utterances(key)

        if done:
            ev.message("already transcribed, reusing the checkpoint")
        else:
            resume_from = ck.last_end(key)
            if resume_from:
                ev.message(f"resuming from {resume_from:.0f}s ({len(utterances)} utterances kept)")

            def on_language(guess: LanguageGuess) -> None:
                nonlocal language
                language = guess.language
                ck.set_info(key, language, None)
                ev.language(guess)

            def on_info(info: AudioInfo) -> None:
                nonlocal language, duration
                language, duration = language or info.language, info.duration
                ck.set_info(key, language, duration)
                ev.info(info, resume_from)

            last_end = resume_from
            try:
                for u in engine.transcribe(
                    job.source,
                    language=job.language or language,
                    task=job.task,
                    vad=job.vad,
                    resume_from=resume_from,
                    on_info=on_info,
                    on_language=on_language,
                ):
                    ck.append(key, [u])  # one row per utterance: a crash loses nothing
                    utterances.append(u)
                    last_end = u.end
                    ev.progress(u.end)
            except KeyboardInterrupt:
                raise Interrupted(last_end) from None
            ck.mark_done(key)

    spoken = "en" if job.task == "translate" else (language or job.language or "en")
    return Transcript(
        utterances,
        language=spoken,
        duration=duration or (utterances[-1].end if utterances else 0.0),
    )


def translate(transcript: Transcript, job: Job, engine: Engine, ev: Events) -> Transcript:
    assert job.to is not None
    try:
        translator = Translator(
            transcript.language, job.to, device=engine.device.name, progress=ev.download
        )
    except NoRouteError as e:
        hint = "" if job.language else " If the detected language is wrong, pass --language."
        raise RuntimeError(f"{e}.{hint}") from None
    ev.message(f"translating {translator.describe()}")
    texts = translator.batch([u.text for u in transcript.utterances])
    return Transcript(
        [Utterance(u.start, u.end, t) for u, t in zip(transcript.utterances, texts, strict=True)],
        language=job.to,
        duration=transcript.duration,
    )


def _write_all(transcript: Transcript, job: Job, ev: Events, *, suffix: str) -> list[Path]:
    if not transcript.utterances:
        ev.message("no speech found, nothing written")
        return []
    out_dir = job.out_dir or job.source.parent
    stem = job.source.stem
    written: list[Path] = []
    for fmt in job.formats:
        target = out_dir / f"{stem}{suffix}.{fmt}"
        if target.exists() and not job.overwrite:
            ev.message(f"kept {target.name} (exists, use --overwrite to replace it)")
            continue
        written.append(write(transcript, target, fmt, title=stem))
    return written
