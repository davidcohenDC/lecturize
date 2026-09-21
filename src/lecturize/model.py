"""The data every module exchanges: utterances with absolute timestamps."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Utterance:
    """One stretch of speech, in seconds from the start of the recording."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcript:
    utterances: list[Utterance]
    language: str  # ISO 639-1 code detected or given
    duration: float  # seconds
