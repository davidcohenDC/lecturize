"""Speech to text with faster-whisper, streaming utterances as they are produced."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE, blocks, duration
from .device import Device
from .model import Utterance
from .paths import cache_dir

PROBE_SECONDS = 60  # how much audio the language check decodes
CANDIDATES = 3  # languages tried by the check
TIE = 0.1  # log prob gap below which English wins the tie
DEAD = {"la"}  # the detector offers Latin for noise and synthetic voices; nobody lectures in it

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


@dataclass
class LanguageGuess:
    """What the language check decided and why."""

    language: str
    detected: str  # what Whisper's detector said
    probability: float  # its confidence
    scores: dict[str, float] = field(default_factory=dict)  # avg log prob per candidate
    sample: str = ""  # first words decoded in the chosen language

    @property
    def overruled(self) -> bool:
        return self.language != self.detected

    @property
    def uncertain(self) -> bool:
        if len(self.scores) < 2:
            return self.probability < 0.8
        best, second = sorted(self.scores.values(), reverse=True)[:2]
        return best - second < 0.15


class Engine:
    def __init__(self, model: str, device: Device, cpu_threads: int = 0):
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
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

    # -- decoding options shared by every call ---------------------------------

    _options: dict = {
        "beam_size": 5,
        "condition_on_previous_text": False,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
        "word_timestamps": True,
    }

    def guess_language(self, samples: np.ndarray) -> LanguageGuess:
        """Whisper's detector is confident and wrong often enough on accented speech that
        it cannot be trusted alone. Decode the first minute in the top candidates and keep
        the one the model finds easiest to explain (highest average log probability)."""
        probe = samples[: PROBE_SECONDS * SAMPLE_RATE]
        detected, prob, all_probs = self._model.detect_language(
            audio=probe, language_detection_segments=2
        )
        ranked = sorted(all_probs or [(detected, prob)], key=lambda p: p[1], reverse=True)
        # rank, not threshold: on accented speech the right language often sits second with 3%.
        # English is always tried: most lectures are in it and the detector misses it most.
        candidates = [lang for lang, p in ranked[:CANDIDATES] if p >= 0.01 and lang not in DEAD]
        if "en" not in candidates:
            candidates.append("en")
        scores: dict[str, float] = {}
        samples_text: dict[str, str] = {}
        for lang in candidates:
            segments, _ = self._model.transcribe(
                probe, language=lang, vad_filter=True, **self._options
            )
            weighted, total = 0.0, 0.0
            words: list[str] = []
            for seg in segments:
                length = max(seg.end - seg.start, 0.1)
                weighted += seg.avg_logprob * length
                total += length
                if len(words) < 12:
                    words += seg.text.split()
            scores[lang] = weighted / total if total else float("-inf")
            samples_text[lang] = " ".join(words[:12])
        best = max(scores, key=lambda k: scores[k])
        if best != "en" and scores[best] - scores["en"] < TIE:
            best = "en"  # within noise of each other: the safer bet for a lecture
        return LanguageGuess(best, detected, prob, scores, samples_text.get(best, ""))

    def transcribe(
        self,
        audio: Path | np.ndarray,
        *,
        language: str | None = None,
        task: str = "transcribe",
        vad: bool = True,
        resume_from: float = 0.0,
        on_info: Callable[[AudioInfo], None] | None = None,
        on_language: Callable[[LanguageGuess], None] | None = None,
    ) -> Iterator[Utterance]:
        """Yield utterances with timestamps relative to the start of the file.

        A file is decoded in blocks of about ten minutes (see audio.py), so memory stays
        flat however long the lecture is. `resume_from` skips everything before that
        second: decoding starts there and every timestamp gets the block start added
        back, so a resumed run and a clean run produce the same timeline.
        """
        if isinstance(audio, np.ndarray):
            total = len(samples := audio) / SAMPLE_RATE
            offset = max(0.0, min(resume_from, total))
            chunks: Iterable[tuple[float, np.ndarray]] = (
                [(offset, samples[int(offset * SAMPLE_RATE) :])] if offset < total else []
            )
            probe = samples
        else:
            total = duration(audio)
            offset = max(0.0, min(resume_from, total))
            chunks = blocks(audio, start=offset) if offset < total else []
            probe = next(blocks(audio, block=PROBE_SECONDS, search=0.0), (0.0, np.zeros(0)))[1]

        if language is None:
            guess = self.guess_language(probe)
            language = guess.language
            if on_language:
                on_language(guess)

        told = False
        for start, samples in chunks:
            segments, info = self._model.transcribe(
                samples,
                language=language,
                task=task,
                vad_filter=vad,
                vad_parameters={"min_silence_duration_ms": 500},
                **self._options,
            )
            if not told and on_info:
                on_info(AudioInfo(info.language, info.language_probability, total))
                told = True
            for seg in segments:
                text = seg.text.strip()
                if not text:
                    continue
                # With VAD the segment end can be stretched across a long silence; the last
                # word knows where the speech really stopped.
                end = seg.words[-1].end if seg.words else seg.end
                first = seg.words[0].start if seg.words else seg.start
                yield Utterance(start=first + start, end=end + start, text=text)
        if not told and on_info:
            on_info(AudioInfo(language, 1.0, total))
