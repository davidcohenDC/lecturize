"""Decode a recording in blocks of a few minutes, cut where the speaker is quiet.

Whole-file decoding costs about 1.7 GB of RAM per hour once Whisper's feature extractor
has run over it; a three hour lecture does not fit an 8 GB laptop. Blocks keep memory
flat, and cutting them at the quietest half second near the target length keeps words
whole.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
BLOCK = 600.0  # target block length, seconds
SEARCH = 30.0  # look this far around the target for a quiet spot
QUIET = 0.5  # length of the quiet window we look for


def duration(path: Path) -> float:
    import av

    with av.open(str(path)) as c:
        stream = c.streams.audio[0]
        if stream.duration and stream.time_base:
            return float(stream.duration * stream.time_base)
        return float(c.duration / 1_000_000) if c.duration else 0.0


def split_point(samples: np.ndarray, target: int, search: int, quiet: int) -> int:
    """Index in `samples` of the quietest `quiet`-long window within `search` of `target`."""
    lo, hi = max(0, target - search), min(len(samples) - quiet, target + search)
    if hi <= lo:
        return min(target, len(samples))
    window = samples[lo : hi + quiet].astype(np.float32)
    energy = np.convolve(window * window, np.ones(quiet), mode="valid")
    return lo + int(np.argmin(energy))


def blocks(
    path: Path, *, start: float = 0.0, block: float = BLOCK, search: float = SEARCH
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (start_seconds, float32 mono 16 kHz samples) from `start` to the end."""
    import av

    target = int(block * SAMPLE_RATE)
    margin = int(search * SAMPLE_RATE)
    quiet = int(QUIET * SAMPLE_RATE)
    position = start
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        if start > 0:
            container.seek(int(start * 1_000_000), backward=True, any_frame=False)
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        pending: list[np.ndarray] = []
        held = 0
        skip = None  # samples to drop after a seek, computed from the first frame's time
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunk = out.to_ndarray().reshape(-1)
                if skip is None:
                    at = start
                    if out.pts is not None and out.time_base is not None:
                        at = float(out.pts * out.time_base)
                    skip = max(0, int((start - at) * SAMPLE_RATE)) if start > 0 else 0
                if skip:
                    drop = min(skip, len(chunk))
                    chunk, skip = chunk[drop:], skip - drop
                    if not len(chunk):
                        continue
                pending.append(chunk)
                held += len(chunk)
                if held >= target + margin:
                    buf = np.concatenate(pending)
                    cut = split_point(buf, target, margin, quiet)
                    yield position, buf[:cut].astype(np.float32) / 32768.0
                    position += cut / SAMPLE_RATE
                    pending, held = [buf[cut:]], len(buf) - cut
        if held:
            buf = np.concatenate(pending)
            yield position, buf.astype(np.float32) / 32768.0
