"""Command line: `lecturize run recording.mp4 --to it`."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import NoReturn

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
from . import device as devices
from .engine import MODELS, AudioInfo, Engine, LanguageGuess
from .pipeline import DEFAULT_FORMATS, Events, Interrupted, Job
from .pipeline import run as run_job
from .writers import FORMATS

for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252 when redirected
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich")
out = Console()  # results: file names, tables
err = Console(stderr=True)  # progress, warnings, errors


def _version(value: bool) -> None:
    if value:
        out.print(f"lecturize {__version__}")
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
        "small", "--model", "-m", help=f"Whisper model: {', '.join(MODELS)}, or a folder/HF id."
    ),
    language: str | None = typer.Option(
        None,
        "--language",
        "-l",
        help="Spoken language (ISO 639-1); checked on the first minute if omitted.",
    ),
    to: str | None = typer.Option(
        None, "--to", help="Also translate into this language (ISO 639-1)."
    ),
    formats: str = typer.Option(
        ",".join(DEFAULT_FORMATS), "--format", "-f", help=f"Comma-separated: {', '.join(FORMATS)}."
    ),
    out_dir: Path | None = typer.Option(
        None, "--out", "-o", help="Output folder; default next to each file."
    ),
    device: devices.DeviceChoice = typer.Option(
        devices.DeviceChoice.auto, "--device", help="auto, cpu or cuda."
    ),
    compute_type: str = typer.Option(
        "auto", "--compute-type", help="CTranslate2 type, e.g. int8, float16."
    ),
    vad: bool = typer.Option(
        True, "--vad/--no-vad", help="Skip silence with voice activity detection."
    ),
    fresh: bool = typer.Option(
        False, "--fresh", help="Ignore a previous checkpoint and start over."
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Replace output files that already exist."
    ),
) -> None:
    """Transcribe FILES, resuming interrupted runs, and write notes and subtitles."""
    fmts = _check_formats(formats)
    _check_language(language, "--language")
    if to:
        _check_language(to, "--to")
        _check_translation_target(to)
    for f in files:
        _check_readable(f)

    try:
        chosen = devices.pick(device, compute_type)
    except RuntimeError as e:
        _fail(str(e))
    _check_compute_type(chosen)
    err.print(f"[dim]lecturize {__version__} · model {model} · {chosen}[/dim]")
    if chosen.note:
        err.print(f"[yellow]{chosen.note}[/yellow]")

    try:
        engine = Engine(model, chosen)
    except Exception as e:  # noqa: BLE001
        _fail(_explain_engine_error(model, e))

    failures = 0
    for f in files:
        job = Job(
            f,
            model=model,
            language=language,
            to=to,
            formats=fmts,
            out_dir=out_dir,
            vad=vad,
            fresh=fresh,
            overwrite=overwrite,
        )
        try:
            _run_one(job, engine)
        except Interrupted as e:
            err.print(f"\n[yellow]{e}. Run the same command again to resume.[/yellow]")
            raise typer.Exit(code=130) from None
        except Exception as e:  # noqa: BLE001
            failures += 1
            err.print(f"[red]{f.name}: {e}[/red]")
            if "CUDA" in str(e) or "cublas" in str(e).lower():
                err.print("[red]stopping: the GPU is not usable, try --device cpu[/red]")
                raise typer.Exit(code=1) from None
    if failures:
        raise typer.Exit(code=1)


class _Cli(Events):
    def __init__(self, bar: Progress, task: TaskID):
        self.bar, self.task = bar, task
        self.downloads: dict[str, TaskID] = {}
        self.duration = 0.0

    def language(self, g: LanguageGuess) -> None:
        how = f"detector said {g.detected} {g.probability:.0%}"
        if g.overruled:
            how += f", decoding the first minute says {g.language}"
        line = f"language: [bold]{g.language}[/bold] ({how})"
        if g.uncertain:
            line += " [yellow]uncertain, pass --language if this is wrong[/yellow]"
        err.print(line)
        if g.sample:
            err.print(f'[dim]  "{g.sample} ..."[/dim]')

    def info(self, i: AudioInfo, resumed_from: float) -> None:
        self.duration = i.duration
        self.bar.update(
            self.task, description="transcribing", total=i.duration, completed=resumed_from
        )

    def progress(self, done: float) -> None:
        self.bar.update(self.task, completed=done)

    def download(self, label: str, done: int, total: int) -> None:
        if label not in self.downloads:
            self.downloads[label] = self.bar.add_task(
                f"downloading {label} model", total=total or None
            )
        self.bar.update(self.downloads[label], completed=done, total=total or None)

    def message(self, text: str) -> None:
        err.print(f"[dim]{text}[/dim]")


def _run_one(job: Job, engine: Engine) -> None:
    err.rule(job.source.name)
    started = time.monotonic()
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=err,
        transient=True,
    ) as bar:
        events = _Cli(bar, bar.add_task("loading audio", total=None))
        outputs = run_job(job, engine, events)
    for p in outputs:
        out.print(str(p))
    if events.duration:
        took = time.monotonic() - started
        err.print(f"[dim]{events.duration / 60:.1f} min of audio in {took:.1f} s[/dim]")


@app.command()
def jobs(
    clear: bool = typer.Option(False, "--clear", help="Forget every interrupted run."),
) -> None:
    """List interrupted runs that `lecturize run` would resume."""
    from .checkpoint import Checkpoint
    from .paths import checkpoint_db

    with Checkpoint(checkpoint_db()) as ck:
        if clear:
            out.print(f"forgot {ck.clear_pending()} interrupted run(s)")
            return
        rows = ck.pending()
    if not rows:
        out.print("no interrupted runs")
        return
    table = Table("file", "model", "transcribed up to")
    for source, model, _key, last in rows:
        table.add_row(source, model, f"{last:.0f}s")
    out.print(table)


@app.command()
def languages() -> None:
    """Languages a local translation model exists for."""
    from .translate import Catalog, NoRouteError

    try:
        out.print(" ".join(Catalog().languages()))
    except NoRouteError as e:
        _fail(str(e))


@app.command()
def paths() -> None:
    """Where models and checkpoints live."""
    from . import paths as p

    out.print(
        f"data       {p.data_dir()}\ncache      {p.cache_dir()}\ncheckpoint {p.checkpoint_db()}"
    )


# -- checks that run before the model is loaded ---------------------------------


def _check_formats(formats: str) -> tuple[str, ...]:
    fmts = tuple(f.strip() for f in formats.split(",") if f.strip())
    bad = [f for f in fmts if f not in FORMATS]
    if bad:
        _fail(f"unknown format {', '.join(bad)}; choose from {', '.join(FORMATS)}")
    return fmts


def _check_language(code: str | None, option: str) -> None:
    if code is None:
        return
    from faster_whisper.tokenizer import _LANGUAGE_CODES

    if code not in _LANGUAGE_CODES:
        _fail(
            f"{option} {code!r} is not a language code Whisper knows (ISO 639-1: en, it, de, ...)"
        )


def _check_translation_target(to: str) -> None:
    from .translate import Catalog, NoRouteError

    try:
        available = Catalog().languages()
    except NoRouteError as e:
        _fail(str(e))
    if to not in available:
        _fail(f"no local translation model for --to {to!r}; available: {' '.join(available)}")


def _check_readable(path: Path) -> None:
    import av

    try:
        with av.open(str(path)) as container:
            has_audio = bool(container.streams.audio)
    except Exception:  # noqa: BLE001
        _fail(f"{path.name}: not readable as an audio or video file")
    if not has_audio:
        _fail(f"{path.name}: the file has no audio track")


def _check_compute_type(chosen: devices.Device) -> None:
    import ctranslate2

    supported = ctranslate2.get_supported_compute_types(chosen.name)
    if chosen.compute_type not in supported:
        _fail(
            f"--compute-type {chosen.compute_type} is not supported on {chosen.name} here;"
            f" choose from {', '.join(sorted(supported))}"
        )


def _explain_engine_error(model: str, e: Exception) -> str:
    text = str(e)
    if (
        "LocalEntryNotFound" in type(e).__name__
        or "offline" in text.lower()
        or "connection" in text.lower()
    ):
        return f"model {model!r} is not downloaded yet and the network is not reachable"
    if "not found" in text.lower() or "does not exist" in text.lower():
        return f"model {model!r} not found; choose from {', '.join(MODELS)} or give a folder/HF id"
    return f"could not load model {model!r}: {text}"


def _fail(msg: str) -> NoReturn:
    err.print(f"[red]{msg}[/red]")
    raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
