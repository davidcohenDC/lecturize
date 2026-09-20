"""Speech to text with faster-whisper, streaming utterances as they are produced."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

from .device import Device  # noqa: E402
from .model import Utterance  # noqa: E402
from .paths import cache_dir  # noqa: E402

SAMPLE_RATE = 16000

MODELS = [
    "tiny",
    "base",
    "small",
    "medium",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
    "distil-large-v3",
]


@dataclass
class AudioInfo:
    language: str
    language_probability: float
    duration: float


class Engine:
    def __init__(self, model: str, device: Device, cpu_threads: int = 0):
        from faster_whisper import WhisperModel

        self.model_name = model
        self.device = device
        self._model = WhisperModel(
            model,
            device=device.name,
            compute_type=device.compute_type,
            cpu_threads=cpu_threads,
            download_root=str(cache_dir() / "whisper"),
        )

    def transcribe(
        self,
        audio: Path,
        *,
        language: str | None = None,
        task: str = "transcribe",
        vad: bool = True,
        resume_from: float = 0.0,
        on_info: Callable[[AudioInfo], None] | None = None,
    ) -> Iterator[Utterance]:
        """Yield utterances with timestamps relative to the start of the file.

        `resume_from` skips everything before that second: the audio is decoded once,
        sliced, and the offset is added back to every timestamp, so a resumed run and a
        clean run produce the same timeline.
        """
        from faster_whisper import decode_audio

        samples = decode_audio(str(audio), sampling_rate=SAMPLE_RATE)
        total = len(samples) / SAMPLE_RATE
        offset = max(0.0, min(resume_from, total))
        if offset:
            samples = samples[int(offset * SAMPLE_RATE) :]

        segments, info = self._model.transcribe(
            samples,
            language=language,
            task=task,
            vad_filter=vad,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,
            beam_size=5,
            language_detection_segments=4,
            language_detection_threshold=0.5,
        )
        if on_info:
            on_info(AudioInfo(info.language, info.language_probability, total))
        for seg in segments:
            text = seg.text.strip()
            if text:
                yield Utterance(start=seg.start + offset, end=seg.end + offset, text=text)
