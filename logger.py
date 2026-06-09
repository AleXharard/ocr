"""Logging centralizat — fișier + panou UI."""

from __future__ import annotations

import logging
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / "logs"
DEBUG_DIR = LOG_DIR / "debug"

_ui_sink: "LogPanel | None" = None
_file_logger: logging.Logger | None = None
_lock = threading.Lock()
_session_file: Path | None = None


def init() -> Path:
    global _file_logger, _session_file
    LOG_DIR.mkdir(exist_ok=True)
    DEBUG_DIR.mkdir(exist_ok=True)

    _session_file = LOG_DIR / f"keyauto_{datetime.now():%Y%m%d_%H%M%S}.log"
    _file_logger = logging.getLogger("keyauto")
    _file_logger.setLevel(logging.DEBUG)
    _file_logger.handlers.clear()

    fh = logging.FileHandler(_session_file, encoding="utf-8")
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    _file_logger.addHandler(fh)

    info(f"Sesiune nouă · log: {_session_file.name}")
    return _session_file


def attach_ui(panel: "LogPanel") -> None:
    global _ui_sink
    _ui_sink = panel


def session_file() -> Path | None:
    return _session_file


def _emit(level: str, msg: str, exc: BaseException | None = None) -> None:
    full = msg
    if exc is not None:
        full = f"{msg}\n{traceback.format_exc()}"

    with _lock:
        if _file_logger:
            getattr(_file_logger, level.lower())(full)

        if _ui_sink is not None:
            try:
                _ui_sink.append(level, msg if exc is None else full)
            except Exception:
                pass

    if exc is not None and level == "ERROR":
        print(full, file=sys.stderr)


def debug(msg: str) -> None:
    _emit("DEBUG", msg)


def info(msg: str) -> None:
    _emit("INFO", msg)


def warn(msg: str) -> None:
    _emit("WARNING", msg)


def error(msg: str, exc: BaseException | None = None) -> None:
    _emit("ERROR", msg, exc)


def save_debug_image(img, name: str) -> Path | None:
    """Salvează frame pentru debug (numpy BGR)."""
    try:
        import cv2

        path = DEBUG_DIR / f"{name}_{datetime.now():%H%M%S_%f}.png"
        cv2.imwrite(str(path), img)
        debug(f"Screenshot debug salvat: {path.name}")
        return path
    except Exception as e:
        warn(f"Nu am putut salva screenshot debug: {e}")
        return None


class LogPanel:
    """Panou scrollabil în UI."""

    COLORS = {
        "DEBUG": "#666666",
        "INFO": "#aaaaaa",
        "WARNING": "#d4a017",
        "ERROR": "#e05555",
    }

    def __init__(self, parent, height: int = 140):
        import customtkinter as ctk

        self._frame = ctk.CTkFrame(parent, fg_color="#0d0d0d")
        self._frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        header = ctk.CTkFrame(self._frame, fg_color="transparent")
        header.pack(fill="x", padx=6, pady=(6, 2))

        ctk.CTkLabel(header, text="Log", font=ctk.CTkFont(size=11, weight="bold"), text_color="#666").pack(
            side="left"
        )

        ctk.CTkButton(
            header,
            text="Clear",
            width=50,
            height=22,
            font=ctk.CTkFont(size=10),
            fg_color="#222",
            hover_color="#333",
            command=self.clear,
        ).pack(side="right", padx=(4, 0))

        ctk.CTkButton(
            header,
            text="Folder",
            width=50,
            height=22,
            font=ctk.CTkFont(size=10),
            fg_color="#222",
            hover_color="#333",
            command=self._open_folder,
        ).pack(side="right")

        self._text = ctk.CTkTextbox(
            self._frame,
            height=height,
            font=ctk.CTkFont(family="Consolas", size=10),
            fg_color="#111111",
            text_color="#aaaaaa",
            activate_scrollbars=True,
        )
        self._text.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self._text.configure(state="disabled")

        self._lines = 0
        self._max_lines = 300

    def append(self, level: str, msg: str) -> None:
        color = self.COLORS.get(level, "#aaaaaa")
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"

        def _write():
            self._text.configure(state="normal")
            self._text.insert("end", line)
            self._text.configure(state="disabled")
            self._text.see("end")
            self._lines += 1
            if self._lines > self._max_lines:
                self._text.configure(state="normal")
                self._text.delete("1.0", "2.0")
                self._text.configure(state="disabled")
                self._lines -= 1

        # thread-safe UI update
        try:
            self._text.after(0, _write)
        except Exception:
            pass

    def clear(self) -> None:
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.configure(state="disabled")
        self._lines = 0
        info("Log UI golit")

    def _open_folder(self) -> None:
        import os
        import subprocess

        LOG_DIR.mkdir(exist_ok=True)
        os.startfile(str(LOG_DIR))
