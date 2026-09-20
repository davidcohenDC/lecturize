"""Where lecturize keeps its own files. Nothing is written next to the recordings."""

from __future__ import annotations

import os
from pathlib import Path

from platformdirs import user_cache_dir, user_data_dir, user_log_dir

APP = "lecturize"


def data_dir() -> Path:
    return _ensure(Path(os.environ.get("LECTURIZE_HOME") or user_data_dir(APP, appauthor=False)))


def cache_dir() -> Path:
    return _ensure(Path(os.environ.get("LECTURIZE_CACHE") or user_cache_dir(APP, appauthor=False)))


def log_dir() -> Path:
    return _ensure(Path(user_log_dir(APP, appauthor=False)))


def checkpoint_db() -> Path:
    return data_dir() / "checkpoints.sqlite3"


def translation_models_dir() -> Path:
    return _ensure(cache_dir() / "translation")


def _ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
