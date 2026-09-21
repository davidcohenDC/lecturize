"""Pick CPU or CUDA, and make the CUDA libraries installed with `lecturize[cuda]` visible."""

from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class DeviceChoice(str, Enum):
    auto = "auto"
    cpu = "cpu"
    cuda = "cuda"


@dataclass(frozen=True)
class Device:
    name: str  # "cpu" or "cuda"
    compute_type: str  # what CTranslate2 will run in
    note: str = ""  # why auto ended up here, when it is worth telling the user

    def __str__(self) -> str:
        return f"{self.name} ({self.compute_type})"


CUDA_HINT = "install the GPU libraries with `pip install lecturize[cuda]`"

_dll_dirs_added = False


def _add_nvidia_dll_dirs() -> None:
    """On Windows the cuBLAS and cuDNN wheels drop their DLLs in site-packages/nvidia/*/bin,
    which is not on the loader path. CTranslate2 needs them before it touches the GPU."""
    global _dll_dirs_added
    if _dll_dirs_added or sys.platform != "win32":
        return
    _dll_dirs_added = True
    try:
        import nvidia
    except ImportError:
        return
    for base in map(Path, nvidia.__path__):
        for bin_dir in base.glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def _cublas_loads() -> bool:
    """The driver alone is not enough: CTranslate2 dlopens cuBLAS at the first encode, and
    when it is missing the process hangs instead of failing. So try to load it here."""
    if sys.platform == "win32":
        cuda_bin = Path(os.environ.get("CUDA_PATH", "")) / "bin"
        if cuda_bin.is_dir():
            os.add_dll_directory(str(cuda_bin))
        names = ["cublas64_12.dll"]
        kwargs: dict = {"winmode": 0}  # legacy search order, PATH included, as CTranslate2 does
    else:
        names = ["libcublas.so.12", "libcublas.so"]
        kwargs = {}
    for name in names:
        try:
            ctypes.CDLL(name, **kwargs)
            return True
        except OSError:
            continue
    return False


def cuda_status() -> tuple[bool, str]:
    """(usable, reason). Usable means a device is visible and cuBLAS can be loaded."""
    _add_nvidia_dll_dirs()
    try:
        import ctranslate2

        count = ctranslate2.get_cuda_device_count()
    except Exception as e:  # noqa: BLE001
        return False, f"ctranslate2 could not query CUDA ({e})"
    if count == 0:
        return False, "no CUDA device visible"
    if not _cublas_loads():
        return False, f"a CUDA device is present but cuBLAS is not installed; {CUDA_HINT}"
    return True, ""


def cuda_available() -> bool:
    return cuda_status()[0]


def pick(requested: DeviceChoice | str = DeviceChoice.auto, compute_type: str = "auto") -> Device:
    requested = DeviceChoice(requested)
    note = ""
    if requested is DeviceChoice.cuda:
        ok, reason = cuda_status()
        if not ok:
            raise RuntimeError(f"CUDA requested but not usable: {reason}")
        name = "cuda"
    elif requested is DeviceChoice.auto:
        ok, reason = cuda_status()
        name = "cuda" if ok else "cpu"
        if not ok and "cuBLAS" in reason:
            note = reason  # a GPU is there and unused: say so
    else:
        name = "cpu"
    if compute_type == "auto":
        compute_type = "float16" if name == "cuda" else "int8"
    return Device(name=name, compute_type=compute_type, note=note)
