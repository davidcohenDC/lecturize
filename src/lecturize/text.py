"""Turning utterances into readable text: sentences, paragraphs, chapters."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pysbd

from .model import Utterance


@dataclass(frozen=True)
class Chapter:
    start: float
    end: float
    paragraphs: list[str]


@lru_cache(maxsize=8)
def _segmenter(language: str) -> pysbd.Segmenter:
    try:
        return pysbd.Segmenter(language=language, clean=False)
    except ValueError:  # pysbd knows 22 languages; English rules are a fair fallback
        return pysbd.Segmenter(language="en", clean=False)


def sentences(text: str, language: str) -> list[str]:
    return [s.strip() for s in _segmenter(language).segment(text) if s.strip()]


def chapters(
    utterances: list[Utterance],
    language: str,
    *,
    every: float = 300.0,
    gap: float = 4.0,
    sentences_per_paragraph: int = 4,
) -> list[Chapter]:
    """Cut the talk into chapters at long pauses, at most `every` seconds apart,
    then group sentences into short paragraphs inside each chapter."""
    if not utterances:
        return []
    blocks: list[list[Utterance]] = [[utterances[0]]]
    for prev, cur in zip(utterances, utterances[1:], strict=False):
        block = blocks[-1]
        long_pause = cur.start - prev.end >= gap
        too_long = cur.start - block[0].start >= every
        if too_long and (long_pause or cur.start - block[0].start >= every * 1.5):
            blocks.append([cur])
        else:
            block.append(cur)
    out = []
    for block in blocks:
        text = " ".join(u.text.strip() for u in block)
        sents = sentences(text, language)
        paras = [
            " ".join(sents[i : i + sentences_per_paragraph])
            for i in range(0, len(sents), sentences_per_paragraph)
        ]
        out.append(Chapter(start=block[0].start, end=block[-1].end, paragraphs=paras))
    return out


def hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"
