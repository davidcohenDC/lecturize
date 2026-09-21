from typer.testing import CliRunner

from lecturize.cli import app

runner = CliRunner()


def test_version():
    r = runner.invoke(app, ["--version"])
    assert r.exit_code == 0
    assert r.output.startswith("lecturize ")


def test_unknown_format_fails_before_loading_anything(sample_wav):
    r = runner.invoke(app, ["run", str(sample_wav), "-f", "pdf"])
    assert r.exit_code == 2
    assert "unknown format pdf" in r.output


def test_bad_language_code(sample_wav):
    r = runner.invoke(app, ["run", str(sample_wav), "-l", "italian"])
    assert r.exit_code == 2
    assert "not a language code" in r.output


def test_bad_device_value(sample_wav):
    r = runner.invoke(app, ["run", str(sample_wav), "--device", "gpu"])
    assert r.exit_code == 2


def test_file_without_audio_track(tmp_path):
    fake = tmp_path / "notes.mp3"
    fake.write_text("this is text, not audio", encoding="utf-8")
    r = runner.invoke(app, ["run", str(fake)])
    assert r.exit_code == 2
    assert "not readable" in r.output


def test_jobs_empty_and_clear():
    assert "no interrupted runs" in runner.invoke(app, ["jobs"]).output
    assert "forgot 0" in runner.invoke(app, ["jobs", "--clear"]).output


def test_paths():
    r = runner.invoke(app, ["paths"])
    assert r.exit_code == 0
    assert "checkpoint" in r.output
