from lecturize.fingerprint import fingerprint


def test_same_content_same_id_even_when_moved(recording, tmp_path):
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


def test_small_file(tmp_path):
    p = tmp_path / "tiny.wav"
    p.write_bytes(b"abc")
    assert len(fingerprint(p)) == 32
