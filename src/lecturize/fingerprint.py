"""A cheap, stable identity for a recording: its size plus a few samples of its bytes.

Hashing a 2 GB video on every start would take longer than loading the model, and the
path alone breaks as soon as the file is moved. Size plus 256 KB read at eight points
spread over the file is enough to tell recordings apart and survives renames.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

CHUNK = 256 * 1024
POINTS = 8


def fingerprint(path: Path) -> str:
    size = path.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with path.open("rb") as f:
        if size <= CHUNK * POINTS:
            h.update(f.read())
        else:
            for i in range(POINTS):
                # the last block ends exactly at the end of the file
                offset = (size - CHUNK) * i // (POINTS - 1)
                f.seek(offset)
                h.update(f.read(CHUNK))
    return h.hexdigest()[:32]
