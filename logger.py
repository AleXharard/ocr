"""Logging centralizat — fișier + panou UI."""

from __future__ import annotations

import logging
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

from app_paths import app_dir

LOG_DIR = app_dir() / "logs"
_MAX_LOG_FILES = 10  # păstrăm doar ultimele N loguri de sesiune; restul se șterg

_ui_sink: "LogPanel | None" = None
_file_logger: logging.Logger | None = None
_lock = threading.Lock()
_session_file: Path | None = None


def _prune_old_logs(keep: int = _MAX_LOG_FILES) -> None:
    """Șterge logurile vechi, păstrând doar cele mai recente `keep` fișiere."""
    try:
        logs = sorted(LOG_DIR.glob("keyauto_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in logs[keep:]:
            old.unlink(missing_ok=True)
    except Exception:
        pass


def init() -> Path:
    global _file_logger, _session_file
    LOG_DIR.mkdir(exist_ok=True)

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

    _prune_old_logs()
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


class LogPanel:
    """Panou de log scrollabil, cu culori pe nivel și filtru debug."""

    # nivel -> (badge, culoare badge, culoare mesaj)
    LEVELS = {
        "DEBUG": ("DBG", "#5b6370", "#7a828f"),
        "INFO": ("INF", "#5aa9e6", "#cdd3de"),
        "WARNING": ("WRN", "#e5b567", "#e5b567"),
        "ERROR": ("ERR", "#e06c6c", "#f08a8a"),
    }
    TS_COLOR = "#474d5a"
    MAX_LINES = 400

    def __init__(self, parent, height: int = 180):
        import customtkinter as ctk

        self._buffer: list[tuple[str, str, str]] = []  # (level, ts, msg)
        self._show_debug = ctk.BooleanVar(value=False)
        self._collapsed = False
        self._height = height

        self._frame = ctk.CTkFrame(parent, fg_color="#101014", corner_radius=12)
        self._frame.pack(fill="both", expand=True, padx=16, pady=(4, 6))

        header = ctk.CTkFrame(self._frame, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(10, 4))

        self._collapse_btn = ctk.CTkButton(
            header, text="▾ LOG", width=58, height=24, font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="transparent", hover_color="#222630", text_color="#6b7280",
            command=self.toggle_collapse,
        )
        self._collapse_btn.pack(side="left")

        self._debug_switch = ctk.CTkSwitch(
            header, text="Debug", variable=self._show_debug, command=self._rerender,
            font=ctk.CTkFont(size=11), switch_width=34, switch_height=16, text_color="#6b7280",
        )
        self._debug_switch.pack(side="left", padx=(12, 0))

        btn_kw = dict(width=52, height=24, font=ctk.CTkFont(size=11),
                      fg_color="#222630", hover_color="#2e333f")
        ctk.CTkButton(header, text="Clear", command=self.clear, **btn_kw).pack(side="right", padx=(6, 0))
        ctk.CTkButton(header, text="Folder", command=self._open_folder, **btn_kw).pack(side="right", padx=(6, 0))
        ctk.CTkButton(header, text="Copy", command=self.copy_log, **btn_kw).pack(side="right")

        self._text = ctk.CTkTextbox(
            self._frame,
            height=height,
            font=ctk.CTkFont(family="Cascadia Mono", size=11),
            fg_color="#0a0a0d",
            activate_scrollbars=True,
            wrap="word",
        )
        self._text.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self._text.tag_config("ts", foreground=self.TS_COLOR)
        for level, (_badge, badge_color, msg_color) in self.LEVELS.items():
            self._text.tag_config(f"badge_{level}", foreground=badge_color)
            self._text.tag_config(f"msg_{level}", foreground=msg_color)

        self._text.configure(state="disabled")

    def _insert(self, level: str, ts: str, msg: str) -> None:
        badge, _, _ = self.LEVELS.get(level, ("·", "#888", "#aaa"))
        self._text.insert("end", f"{ts}  ", "ts")
        self._text.insert("end", f"{badge}  ", f"badge_{level}")
        self._text.insert("end", f"{msg}\n", f"msg_{level}")

    def append(self, level: str, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._buffer.append((level, ts, msg))
        if len(self._buffer) > self.MAX_LINES:
            self._buffer.pop(0)

        if level == "DEBUG" and not self._show_debug.get():
            return

        def _write():
            self._text.configure(state="normal")
            self._insert(level, ts, msg)
            # păstrăm panoul mărginit ca să rămână fluid
            line_count = int(self._text.index("end-1c").split(".")[0])
            if line_count > self.MAX_LINES:
                self._text.delete("1.0", "2.0")
            self._text.configure(state="disabled")
            self._text.see("end")

        try:
            self._text.after(0, _write)
        except Exception:
            pass

    def _rerender(self) -> None:
        show_debug = self._show_debug.get()
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        for level, ts, msg in self._buffer:
            if level == "DEBUG" and not show_debug:
                continue
            self._insert(level, ts, msg)
        self._text.configure(state="disabled")
        self._text.see("end")

    def clear(self) -> None:
        self._buffer.clear()
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.configure(state="disabled")
        info("Log golit")

    def copy_log(self) -> None:
        """Copiază în clipboard liniile vizibile (respectând filtrul Debug)."""
        show_debug = self._show_debug.get()
        lines = [
            f"{ts} {self.LEVELS.get(level, ('·',))[0]} {msg}"
            for level, ts, msg in self._buffer
            if show_debug or level != "DEBUG"
        ]
        try:
            self._frame.clipboard_clear()
            self._frame.clipboard_append("\n".join(lines))
            info(f"Log copiat ({len(lines)} linii)")
        except Exception as e:
            warn(f"Nu pot copia logul: {e}")

    def toggle_collapse(self) -> None:
        """Ascunde/afișează panoul de text al logului (păstrând antetul)."""
        if self._collapsed:
            self._text.pack(fill="both", expand=True, padx=12, pady=(0, 12))
            self._frame.pack_configure(expand=True, fill="both")
            self._collapse_btn.configure(text="▾ LOG")
        else:
            self._text.pack_forget()
            self._frame.pack_configure(expand=False, fill="x")
            self._collapse_btn.configure(text="▸ LOG")
        self._collapsed = not self._collapsed

    def _open_folder(self) -> None:
        import os

        LOG_DIR.mkdir(exist_ok=True)
        os.startfile(str(LOG_DIR))
