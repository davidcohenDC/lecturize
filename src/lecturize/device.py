"""Pick CPU or CUDA, and make the CUDA libraries installed with `lecturize[cuda]` visible."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Device:
    name: str  # "cpu" or "cuda"
    compute_type: str  # what CTranslate2 will run in

    def __str__(self) -> str:
        return f"{self.name} ({self.compute_type})"


def _add_nvidia_dll_dirs() -> None:
    """On Windows the cuBLAS and cuDNN wheels drop their DLLs in site-packages/nvidia/*/bin,
    which is not on the loader path. CTranslate2 needs them before it touches the GPU."""
    if sys.platform != "win32":
        return
    try:
        import nvidia
    except ImportError:
        return
    for base in map(Path, nvidia.__path__):
        for bin_dir in base.glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def cuda_available() -> bool:
    _add_nvidia_dll_dirs()
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def pick(requested: str = "auto", compute_type: str = "auto") -> Device:
    if requested == "auto":
        name = "cuda" if cuda_available() else "cpu"
    elif requested == "cuda":
        if not cuda_available():
            raise RuntimeError(
                "CUDA requested but not usable. Install the GPU libraries with"
                " `pip install lecturize[cuda]` and check that the NVIDIA driver is recent."
            )
        name = "cuda"
    else:
        name = "cpu"
    if compute_type == "auto":
        compute_type = "float16" if name == "cuda" else "int8"
    return Device(name=name, compute_type=compute_type)
