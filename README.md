# lecturize

[![Build](https://github.com/davidcohenDC/lecturize/actions/workflows/build.yml/badge.svg)](https://github.com/davidcohenDC/lecturize/actions/workflows/build.yml)
[![PyPI](https://img.shields.io/pypi/v/lecturize)](https://pypi.org/project/lecturize/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

I record my lectures and I want them back as notes I can read, not as a two hour
audio file. `lecturize` runs on my machine, transcribes with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper), picks up where it
left off if the run is interrupted, and can translate the result with local
models. Nothing leaves the computer except the model downloads.

```sh
pipx install lecturize
lecturize run lecture.mp4 --to it
```

That writes `lecture.md`, `lecture.srt`, `lecture.it.md` and `lecture.it.srt`
next to the recording.

![lecturize transcribing and translating a recording](docs/demo.svg)

## What it does

- transcribes audio and video files (anything ffmpeg reads) with Whisper models,
  on CPU or on an NVIDIA GPU;
- skips silence with voice activity detection, which is what stops Whisper from
  inventing text during pauses;
- writes markdown with a chapter every few minutes, plain text, SRT subtitles and
  Word documents, all from the same timeline;
- resumes an interrupted run. Utterances are saved as they are produced; the
  next `lecturize run` on the same file starts from the last one and ends with
  the same output a clean run would give. `lecturize jobs` lists what is pending;
- translates into about fifty languages with the OPUS-MT models packaged by the
  [Argos Translate](https://github.com/argosopentech/argos-translate) project,
  run through CTranslate2. Pairs that have no direct model go through English.
  Translation into English uses Whisper itself, in one pass.

## Install

You need Python 3.10 or newer. [pipx](https://pipx.pypa.io/) keeps it out of
your other environments:

```sh
pipx install lecturize
```

On an NVIDIA GPU add the CUDA libraries, and the runs get several times faster:

```sh
pipx install "lecturize[cuda]"
```

Models are downloaded on first use into the user cache (`lecturize paths` shows
where): `small` is 460 MB, `large-v3` about 3 GB, a translation package 90 MB.
ffmpeg is not required for the common formats, since decoding goes through PyAV,
but having it installed does not hurt.

## Use

```sh
lecturize run talk.mp3                       # small model, language detected, md + srt
lecturize run talk.mp3 -m large-v3 -l it     # a bigger model, Italian spoken
lecturize run talk.mp3 --to en               # English notes, translated by Whisper
lecturize run talk.mp3 --to it -f md,docx    # Italian notes as markdown and Word
lecturize run *.mp4 -o notes/                # many files, outputs in one folder
lecturize run talk.mp3 --fresh               # ignore the checkpoint, start over
lecturize jobs                               # interrupted runs waiting to resume
lecturize languages                          # translation targets available
```

Options: `-m/--model` (tiny, base, small, medium, large-v2, large-v3,
large-v3-turbo, distil-large-v3, or a folder with a CTranslate2 model),
`-l/--language`, `--to`, `-f/--format`, `-o/--out`, `--device auto|cpu|cuda`,
`--compute-type`, `--no-vad`, `--fresh`.

On my machine (a laptop CPU and an RTX 4070 Ti SUPER) `small` runs at about
three times real time on the CPU in int8 and ten times on the GPU in float16;
`large-v3` on the GPU stays above real time. A two hour lecture translated into
Italian adds about a minute.

The library is usable from Python as well:

```python
from pathlib import Path
from lecturize import device
from lecturize.engine import Engine
from lecturize.pipeline import Job, run

dev = device.pick("auto")
outputs = run(Job(Path("talk.mp3"), model="small", target="it"), Engine("small", dev), dev)
```

## How it is organised

```
src/lecturize/
  cli.py          the commands (typer)
  pipeline.py     one recording in, files out; resume logic
  engine.py       faster-whisper, streaming utterances with absolute timestamps
  checkpoint.py   SQLite store of utterances per (file fingerprint, settings)
  translate.py    Argos model packages through ctranslate2 + sentencepiece, pivot via English
  text.py         sentences, paragraphs, chapters
  writers.py      txt, md, srt, docx
  device.py       CPU or CUDA, and the DLL paths the [cuda] extra needs on Windows
tests/            unit tests, a resume test with a scripted engine, two slow tests with the real one
```

`pytest` runs the fast tests; `pytest -m slow` downloads the tiny model and runs
the real engine on the 37 second synthetic lecture in `tests/data`.

## Licenses

lecturize is MIT. It downloads and runs third party models: Whisper weights
converted by SYSTRAN (MIT), and translation packages from Argos Translate, built
on OPUS-MT models, which carry a CC-BY 4.0 attribution to their authors. See
[THIRD_PARTY.md](THIRD_PARTY.md).

[MIT](LICENSE) © 2026 David Cohen
