"""A cheap, stable identity for a recording: size plus a hash of its head and tail.

Hashing a 2 GB video on every start would take longer than loading the model, and the
path alone breaks as soon as the file is moved. Size + 1 MB from each end is enough to
tell recordings apart and survives renames.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

CHUNK = 1024 * 1024


def fingerprint(path: Path) -> str:
    size = path.stat().st_size
    h = hashlib.sha256()
    h.update(str(size).encode())
    with path.open("rb") as f:
        h.update(f.read(CHUNK))
        if size > 2 * CHUNK:
            f.seek(size - CHUNK)
            h.update(f.read(CHUNK))
    return h.hexdigest()[:32]
