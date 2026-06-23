"""Căi aplicație — development vs exe PyInstaller."""

from __future__ import annotations

import sys
from pathlib import Path


def is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def app_dir() -> Path:
    """Folder de lucru: lângă exe (config, loguri, debug)."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def resource_dir() -> Path:
    """Resurse împachetate (_internal / _MEIPASS)."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", app_dir()))
    return Path(__file__).resolve().parent


def resource_path(*parts: str) -> Path:
    return resource_dir().joinpath(*parts)
