"""Lecture recordings to notes, locally."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lecturize")
except PackageNotFoundError:  # running from a checkout without an install
    __version__ = "0.0.0"
