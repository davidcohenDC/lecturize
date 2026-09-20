"""Resumable jobs.

A job is identified by the recording (fingerprint) and by everything that changes the
output: model, language, task, VAD. Utterances are appended as the engine produces
them, so an interrupted run restarts from the last one written, not from zero.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .model import Utterance

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    key TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    settings TEXT NOT NULL,
    language TEXT,
    duration REAL,
    done INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS utterances (
    key TEXT NOT NULL,
    seq INTEGER NOT NULL,
    start REAL NOT NULL,
    "end" REAL NOT NULL,
    text TEXT NOT NULL,
    PRIMARY KEY (key, seq)
);
"""


@dataclass(frozen=True)
class JobKey:
    fingerprint: str
    settings: dict

    @property
    def value(self) -> str:
        return self.fingerprint + ":" + json.dumps(self.settings, sort_keys=True)


class Checkpoint:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Checkpoint:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- jobs -----------------------------------------------------------------

    def open_job(self, key: JobKey, source: Path) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO jobs (key, source, settings) VALUES (?, ?, ?)",
            (key.value, str(source), json.dumps(key.settings, sort_keys=True)),
        )
        self.conn.commit()

    def set_info(self, key: JobKey, language: str | None, duration: float | None) -> None:
        self.conn.execute(
            "UPDATE jobs SET language = COALESCE(?, language), duration = COALESCE(?, duration)"
            " WHERE key = ?",
            (language, duration, key.value),
        )
        self.conn.commit()

    def info(self, key: JobKey) -> tuple[str | None, float | None, bool]:
        row = self.conn.execute(
            "SELECT language, duration, done FROM jobs WHERE key = ?", (key.value,)
        ).fetchone()
        if row is None:
            return None, None, False
        return row[0], row[1], bool(row[2])

    def mark_done(self, key: JobKey) -> None:
        self.conn.execute("UPDATE jobs SET done = 1 WHERE key = ?", (key.value,))
        self.conn.commit()

    def forget(self, key: JobKey) -> None:
        self.conn.execute("DELETE FROM utterances WHERE key = ?", (key.value,))
        self.conn.execute("DELETE FROM jobs WHERE key = ?", (key.value,))
        self.conn.commit()

    # -- utterances ------------------------------------------------------------

    def utterances(self, key: JobKey) -> list[Utterance]:
        rows = self.conn.execute(
            'SELECT start, "end", text FROM utterances WHERE key = ? ORDER BY seq', (key.value,)
        ).fetchall()
        return [Utterance(start=r[0], end=r[1], text=r[2]) for r in rows]

    def last_end(self, key: JobKey) -> float:
        row = self.conn.execute(
            'SELECT MAX("end") FROM utterances WHERE key = ?', (key.value,)
        ).fetchone()
        return float(row[0]) if row and row[0] is not None else 0.0

    def append(self, key: JobKey, utterances: Iterable[Utterance]) -> None:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(seq), -1) FROM utterances WHERE key = ?", (key.value,)
        ).fetchone()
        seq = int(row[0]) + 1
        self.conn.executemany(
            'INSERT INTO utterances (key, seq, start, "end", text) VALUES (?, ?, ?, ?, ?)',
            [(key.value, seq + i, u.start, u.end, u.text) for i, u in enumerate(utterances)],
        )
        self.conn.commit()

    def pending(self) -> list[tuple[str, str, float]]:
        """Jobs started and not finished: (source, key, last_end)."""
        rows = self.conn.execute(
            'SELECT j.source, j.key, COALESCE(MAX(u."end"), 0) FROM jobs j'
            " LEFT JOIN utterances u ON u.key = j.key WHERE j.done = 0 GROUP BY j.key"
        ).fetchall()
        return [(r[0], r[1], float(r[2])) for r in rows]
