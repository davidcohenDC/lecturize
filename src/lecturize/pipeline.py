"""One recording in, notes out. Resumable, translation optional."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__
from .checkpoint import Checkpoint, JobKey
from .device import Device
from .engine import AudioInfo, Engine
from .fingerprint import fingerprint
from .model import Transcript, Utterance
from .paths import checkpoint_db
from .translate import Translator
from .writers import write

FLUSH_EVERY = 20  # utterances written to the checkpoint at a time


@dataclass
class Job:
    source: Path
    model: str = "small"
    language: str | None = None  # None = detect
    target: str | None = None  # translate into this language, None = keep
    formats: tuple[str, ...] = ("md", "srt")
    out_dir: Path | None = None
    vad: bool = True
    fresh: bool = False  # ignore a previous checkpoint

    @property
    def task(self) -> str:
        return "translate" if self.target == "en" else "transcribe"

    def settings(self) -> dict:
        return {
            "engine": "faster-whisper",
            "model": self.model,
            "language": self.language,
            "task": self.task,
            "vad": self.vad,
            "lecturize": __version__.split("+")[0],
        }


@dataclass
class Events:
    """Hooks the CLI uses to draw progress; every one is optional."""

    info: Callable[[AudioInfo, float], None] | None = None  # (info, resumed_from)
    progress: Callable[[float], None] | None = None  # seconds of audio done
    download: Callable[[str, int, int], None] | None = None
    message: Callable[[str], None] | None = None
    written: Callable[[Path], None] | None = None
    outputs: list[Path] = field(default_factory=list)

    def say(self, msg: str) -> None:
        if self.message:
            self.message(msg)


def run(job: Job, engine: Engine, device: Device, events: Events | None = None) -> list[Path]:
    ev = events or Events()
    key = JobKey(fingerprint(job.source), job.settings())
    with Checkpoint(checkpoint_db()) as ck:
        if job.fresh:
            ck.forget(key)
        ck.open_job(key, job.source)
        language, duration, done = ck.info(key)
        utterances = ck.utterances(key)

        if not done:
            resume_from = ck.last_end(key)
            if resume_from:
                ev.say(f"resuming from {resume_from:.0f}s ({len(utterances)} utterances kept)")

            def on_info(info: AudioInfo) -> None:
                nonlocal language, duration
                language, duration = language or info.language, info.duration
                ck.set_info(key, language, duration)
                if ev.info:
                    ev.info(info, resume_from)

            pending: list[Utterance] = []
            for u in engine.transcribe(
                job.source,
                language=job.language,
                task=job.task,
                vad=job.vad,
                resume_from=resume_from,
                on_info=on_info,
            ):
                pending.append(u)
                if ev.progress:
                    ev.progress(u.end)
                if len(pending) >= FLUSH_EVERY:
                    ck.append(key, pending)
                    utterances.extend(pending)
                    pending.clear()
            if pending:
                ck.append(key, pending)
                utterances.extend(pending)
            ck.mark_done(key)
        else:
            ev.say("already transcribed, reusing the checkpoint")

    spoken = language or job.language or "en"
    transcript = Transcript(
        utterances,
        language="en" if job.task == "translate" else spoken,
        duration=duration or (utterances[-1].end if utterances else 0.0),
    )

    out_dir = job.out_dir or job.source.parent
    stem = job.source.stem
    outputs: list[Path] = []
    for fmt in job.formats:
        outputs.append(write(transcript, out_dir / f"{stem}.{fmt}", fmt, title=stem))

    if job.target and job.target != transcript.language:
        try:
            translator = Translator(
                transcript.language, job.target, device=device.name, progress=ev.download
            )
        except LookupError as e:
            hint = "" if job.language else " If the detected language is wrong, pass --language."
            raise RuntimeError(f"{e}.{hint}") from None
        ev.say(f"translating {translator.describe()}")
        texts = translator.batch([u.text for u in transcript.utterances])
        translated = Transcript(
            [
                Utterance(u.start, u.end, t)
                for u, t in zip(transcript.utterances, texts, strict=False)
            ],
            language=job.target,
            duration=transcript.duration,
        )
        for fmt in job.formats:
            outputs.append(
                write(translated, out_dir / f"{stem}.{job.target}.{fmt}", fmt, title=stem)
            )

    for p in outputs:
        if ev.written:
            ev.written(p)
    ev.outputs = outputs
    return outputs
