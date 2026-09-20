"""Turning utterances into readable text: sentences, paragraphs, chapters."""

from __future__ import annotations

from dataclasses import dataclass

import pysbd

from .model import Utterance


@dataclass(frozen=True)
class Chapter:
    start: float
    end: float
    paragraphs: list[str]


def sentences(text: str, language: str) -> list[str]:
    """Split with pysbd when it knows the language, otherwise on sentence punctuation."""
    try:
        seg = pysbd.Segmenter(language=language, clean=False)
    except ValueError:
        seg = pysbd.Segmenter(language="en", clean=False)
    return [s.strip() for s in seg.segment(text) if s.strip()]


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
