"""The data every module exchanges: utterances with absolute timestamps."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Utterance:
    """One stretch of speech, in seconds from the start of the recording."""

    start: float
    end: float
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Utterance:
        return cls(start=float(d["start"]), end=float(d["end"]), text=str(d["text"]))


@dataclass(frozen=True)
class Transcript:
    utterances: list[Utterance]
    language: str  # ISO 639-1 code detected or given
    duration: float  # seconds

    @property
    def text(self) -> str:
        return " ".join(u.text.strip() for u in self.utterances if u.text.strip())
