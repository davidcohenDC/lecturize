"""Command line: `lecturize recording.mp4 --to it`."""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from . import __version__
from .engine import MODELS, AudioInfo, Engine
from .writers import FORMATS

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich")
console = Console(stderr=True)


def _version(value: bool) -> None:
    if value:
        typer.echo(f"lecturize {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(False, "--version", callback=_version, is_eager=True),
) -> None:
    """Turn lecture recordings into notes, on your machine."""


@app.command()
def run(
    files: list[Path] = typer.Argument(
        ..., exists=True, dir_okay=False, help="Audio or video files."
    ),
    model: str = typer.Option(
        "small", "--model", "-m", help=f"Whisper model: {', '.join(MODELS)}."
    ),
    language: str | None = typer.Option(
        None, "--language", "-l", help="Spoken language (ISO 639-1); detected if omitted."
    ),
    to: str | None = typer.Option(
        None, "--to", help="Also translate into this language (ISO 639-1)."
    ),
    formats: str = typer.Option(
        "md,srt", "--format", "-f", help=f"Comma-separated: {', '.join(FORMATS)}."
    ),
    out: Path | None = typer.Option(
        None, "--out", "-o", help="Output folder; default next to each file."
    ),
    device: str = typer.Option("auto", "--device", help="auto, cpu or cuda."),
    compute_type: str = typer.Option(
        "auto", "--compute-type", help="CTranslate2 type, e.g. int8, float16."
    ),
    vad: bool = typer.Option(
        True, "--vad/--no-vad", help="Skip silence with voice activity detection."
    ),
    fresh: bool = typer.Option(
        False, "--fresh", help="Ignore a previous checkpoint and start over."
    ),
) -> None:
    """Transcribe FILES, resuming interrupted runs, and write notes and subtitles."""
    from . import device as dev
    from .pipeline import Events, Job
    from .pipeline import run as run_job

    fmts = tuple(f.strip() for f in formats.split(",") if f.strip())
    bad = [f for f in fmts if f not in FORMATS]
    if bad:
        _fail(f"unknown format {', '.join(bad)}; choose from {', '.join(FORMATS)}")
    if model not in MODELS and not Path(model).exists():
        _fail(f"unknown model {model!r}; choose from {', '.join(MODELS)} or give a folder")
    if shutil.which("ffmpeg") is None:
        console.print(
            "[yellow]ffmpeg not found on PATH: most formats still decode through PyAV,"
            " install ffmpeg if a file is refused.[/yellow]"
        )

    try:
        chosen = dev.pick(device, compute_type)
    except RuntimeError as e:
        _fail(str(e))
    console.print(f"[dim]lecturize {__version__} · model {model} · {chosen}[/dim]")
    engine = Engine(model, chosen)

    failures = 0
    for f in files:
        job = Job(
            f,
            model=model,
            language=language,
            target=to,
            formats=fmts,
            out_dir=out,
            vad=vad,
            fresh=fresh,
        )
        try:
            _run_one(job, engine, chosen, run_job, Events)
        except Exception as e:  # keep going with the other files
            failures += 1
            console.print(f"[red]{f.name}: {e}[/red]")
    raise typer.Exit(code=1 if failures else 0)


def _run_one(job, engine, chosen, run_job, Events) -> None:  # noqa: ANN001
    console.rule(job.source.name)
    started = time.monotonic()
    duration = 0.0
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as bar:
        task = bar.add_task("loading audio", total=None)
        dl: dict[str, TaskID] = {}

        def info(i: AudioInfo, resumed: float) -> None:
            nonlocal duration
            duration = i.duration
            bar.update(
                task,
                description=f"transcribing ({i.language})",
                total=i.duration,
                completed=resumed,
            )

        def progress(done: float) -> None:
            bar.update(task, completed=done)

        def download(label: str, done: int, total: int) -> None:
            if label not in dl:
                dl[label] = bar.add_task(f"downloading {label} model", total=total or None)
            bar.update(dl[label], completed=done, total=total or None)

        events = Events(
            info=info,
            progress=progress,
            download=download,
            message=lambda m: console.print(f"[dim]{m}[/dim]"),
        )
        outputs = run_job(job, engine, chosen, events)
    for p in outputs:
        console.print(f"  [green]written[/green] {p}")
    took = time.monotonic() - started
    if duration:
        console.print(f"  [dim]{duration / 60:.1f} min of audio in {took:.1f} s[/dim]")


@app.command()
def jobs() -> None:
    """List interrupted runs that `lecturize run` would resume."""
    from .checkpoint import Checkpoint
    from .paths import checkpoint_db

    with Checkpoint(checkpoint_db()) as ck:
        rows = ck.pending()
    if not rows:
        console.print("no interrupted runs")
        return
    table = Table("file", "transcribed up to")
    for source, _key, last in rows:
        table.add_row(source, f"{last:.0f}s")
    console.print(table)


@app.command()
def languages() -> None:
    """Languages a local translation model exists for."""
    from .translate import Catalog

    console.print(" ".join(Catalog().languages()))


@app.command()
def paths() -> None:
    """Where models, checkpoints and logs live."""
    from . import paths as p

    console.print(
        f"data       {p.data_dir()}\ncache      {p.cache_dir()}\ncheckpoint {p.checkpoint_db()}"
    )


def _fail(msg: str) -> None:
    console.print(f"[red]{msg}[/red]")
    sys.exit(2)


if __name__ == "__main__":
    app()
