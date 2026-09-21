"""Local translation with the Argos Translate model packages (OPUS-MT converted to
CTranslate2), used directly through ctranslate2 and sentencepiece.

The Argos library itself is not a dependency: it would bring in torch through stanza.
Every package translates from or to English, so any other pair goes through English.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .paths import translation_models_dir
from .text import sentences

INDEX_URL = "https://raw.githubusercontent.com/argosopentech/argospm-index/main/index.json"
INDEX_TTL_DAYS = 30
PIVOT = "en"
USER_AGENT = f"lecturize/{__version__} (+https://github.com/davidcohenDC/lecturize)"

Progress = Callable[[str, int, int], None]  # (label, done_bytes, total_bytes)


class NoRouteError(Exception):
    """No local model, directly or through English, for this language pair."""


@dataclass(frozen=True)
class Package:
    from_code: str
    to_code: str
    version: str
    url: str

    @property
    def dir_name(self) -> str:
        return f"{self.from_code}_{self.to_code}"


class Catalog:
    """The list of available packages, cached on disk."""

    def __init__(self, models_dir: Path | None = None):
        self.models_dir = models_dir or translation_models_dir()

    def packages(self, refresh: bool = False) -> list[Package]:
        cache = self.models_dir / "index.json"
        if refresh or not cache.exists() or _older_than(cache, INDEX_TTL_DAYS):
            try:
                with urllib.request.urlopen(_request(INDEX_URL), timeout=30) as r:
                    data = r.read()
                json.loads(data)  # never cache a truncated download
                tmp = cache.with_suffix(".tmp")
                tmp.write_bytes(data)
                tmp.replace(cache)
            except (OSError, ValueError):
                if not cache.exists():
                    raise NoRouteError(
                        "the list of translation models is not available offline yet;"
                        " connect once so it can be downloaded"
                    ) from None
        try:
            raw = json.loads(cache.read_text(encoding="utf-8"))
        except ValueError:
            cache.unlink(missing_ok=True)
            return self.packages(refresh=True)
        return [
            Package(p["from_code"], p["to_code"], str(p.get("package_version", "")), p["links"][0])
            for p in raw
            if p.get("links")
        ]

    def find(self, src: str, dst: str) -> Package | None:
        return next((p for p in self.packages() if p.from_code == src and p.to_code == dst), None)

    def languages(self) -> list[str]:
        return sorted({p.to_code for p in self.packages()} | {p.from_code for p in self.packages()})

    def route(self, src: str, dst: str) -> list[Package]:
        """Direct package if there is one, otherwise src -> en -> dst."""
        if src == dst:
            return []
        direct = self.find(src, dst)
        if direct:
            return [direct]
        first, second = self.find(src, PIVOT), self.find(PIVOT, dst)
        if first and second:
            return [first, second]
        raise NoRouteError(
            f"no local translation route from {src!r} to {dst!r};"
            f" languages with a model: {', '.join(self.languages())}"
        )

    def installed(self, pkg: Package) -> Path | None:
        d = self.models_dir / pkg.dir_name
        return d if (d / "model" / "model.bin").exists() else None

    def install(self, pkg: Package, progress: Progress | None = None) -> Path:
        target = self.models_dir / pkg.dir_name
        if self.installed(pkg):
            return target
        label = f"{pkg.from_code}-{pkg.to_code}"
        with tempfile.TemporaryDirectory(dir=self.models_dir) as tmp:
            archive = Path(tmp) / "package.zip"
            _download(pkg.url, archive, lambda d, t: progress(label, d, t) if progress else None)
            with zipfile.ZipFile(archive) as z:
                names = z.namelist()
                if any(n.startswith(("/", "..")) or ".." in n.split("/") for n in names):
                    raise ValueError(f"refusing to unpack {pkg.url}: suspicious paths")
                root = names[0].split("/")[0]
                z.extractall(tmp)
            shutil.rmtree(target, ignore_errors=True)
            shutil.move(str(Path(tmp) / root), str(target))
        return target


class Step:
    def __init__(self, model_dir: Path, device: str = "cpu"):
        import ctranslate2
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor(model_file=str(model_dir / "sentencepiece.model"))
        self.translator = ctranslate2.Translator(
            str(model_dir / "model"),
            device=device,
            compute_type="int8" if device == "cpu" else "auto",
        )

    def __call__(self, texts: list[str]) -> list[str]:
        out = list(texts)
        todo = [i for i, t in enumerate(texts) if t.strip()]
        if not todo:
            return out
        tokens = [self.sp.encode(texts[i], out_type=str) for i in todo]
        results = self.translator.translate_batch(
            tokens, beam_size=4, max_batch_size=16, max_decoding_length=512
        )
        for i, r in zip(todo, results, strict=True):
            out[i] = self.sp.decode(r.hypotheses[0])
        return out


class Translator:
    """Translate paragraphs from `src` to `dst`, sentence by sentence, in batches."""

    def __init__(
        self,
        src: str,
        dst: str,
        *,
        catalog: Catalog | None = None,
        device: str = "cpu",
        progress: Progress | None = None,
    ):
        self.src, self.dst = src, dst
        cat = catalog or Catalog()
        self.route = cat.route(src, dst)
        self.steps = [Step(cat.install(p, progress), device) for p in self.route]

    def describe(self) -> str:
        return " -> ".join([self.src] + [p.to_code for p in self.route])

    def batch(self, texts: list[str]) -> list[str]:
        """Translate many texts in one pass; each text comes back as one string."""
        if not self.steps:
            return list(texts)
        pieces: list[str] = []
        owner: list[int] = []
        for i, text in enumerate(texts):
            for s in sentences(text, self.src) or [text]:
                pieces.append(s)
                owner.append(i)
        for step in self.steps:
            pieces = step(pieces)
        out = [""] * len(texts)
        for i, piece in zip(owner, pieces, strict=True):
            out[i] = (out[i] + " " + piece.strip()).strip()
        return out


def _older_than(path: Path, days: int) -> bool:
    import time

    return time.time() - path.stat().st_mtime > days * 86400


def _request(url: str) -> urllib.request.Request:
    # The Argos mirror answers 403 to the default urllib user agent.
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


def _download(url: str, dest: Path, progress: Callable[[int, int], None]) -> None:
    with urllib.request.urlopen(_request(url), timeout=60) as r, dest.open("wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            progress(done, total)
