# Contributing

Issues and pull requests are welcome. A few things that make them quick to merge:

- run `ruff check src tests`, `ruff format src tests`, `mypy src` and `pytest`
  before opening the PR; `pytest -m slow` if you touched the engine or the resume
  logic;
- one change per PR, with a conventional commit title (`fix: ...`, `feat: ...`),
  since the version number and the changelog come from the commit messages;
- if you add a dependency, say why: the package stays torch-free on purpose so
  that `pipx install lecturize` is a 200 MB download and not a 3 GB one;
- for a bug in the transcription itself, include the model, the device
  (`lecturize run` prints both on the first line) and a short clip if you can.

To work on it:

```sh
git clone https://github.com/davidcohenDC/lecturize.git
cd lecturize
python -m venv .venv && . .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -e ".[dev]"
pytest
```
