import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Every test gets its own data and cache folders; nothing touches the user's."""
    monkeypatch.setenv("LECTURIZE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LECTURIZE_CACHE", str(tmp_path / "cache"))
    yield tmp_path


@pytest.fixture
def sample_wav() -> Path:
    return Path(__file__).parent / "data" / "lecture.wav"


@pytest.fixture
def recording(tmp_path) -> Path:
    p = tmp_path / "talk.mp3"
    p.write_bytes(os.urandom(3 * 1024 * 1024))
    return p
