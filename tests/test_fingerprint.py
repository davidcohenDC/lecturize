import os

from lecturize.fingerprint import fingerprint


def test_moved_file_keeps_id(recording, tmp_path):
    before = fingerprint(recording)
    moved = tmp_path / "renamed.mp3"
    recording.rename(moved)
    assert fingerprint(moved) == before


def test_changed_tail_changes_id(recording):
    before = fingerprint(recording)
    with recording.open("r+b") as f:
        f.seek(-10, 2)
        f.write(b"x" * 10)
    assert fingerprint(recording) != before


def test_changed_middle_changes_id(tmp_path):
    p = tmp_path / "long.mp3"
    p.write_bytes(os.urandom(12 * 1024 * 1024))
    before = fingerprint(p)
    with p.open("r+b") as f:  # eight blocks are sampled, so a 2 MB change cannot slip between them
        f.seek(5 * 1024 * 1024)
        f.write(b"y" * 2 * 1024 * 1024)
    assert fingerprint(p) != before


def test_tail_counts_for_files_between_one_and_two_megabytes(tmp_path):
    p = tmp_path / "mid.mp3"
    p.write_bytes(os.urandom(1536 * 1024))
    before = fingerprint(p)
    with p.open("r+b") as f:
        f.seek(-10, 2)
        f.write(b"z" * 10)
    assert fingerprint(p) != before


def test_tiny_file(tmp_path):
    p = tmp_path / "tiny.wav"
    p.write_bytes(b"abc")
    assert len(fingerprint(p)) == 32
