"""
Key Auto — detectează casetele albe și apasă tastele în ordine.
"""

from __future__ import annotations

import json
import threading
import time
import tkinter as tk
import unicodedata
import os
from pathlib import Path

import cv2
import customtkinter as ctk
import mss
import numpy as np
from PIL import Image, ImageTk

import busteni
import game_input
import logger as log
import mina
import vision as vis
from app_paths import app_dir

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# ── Configurare ────────────────────────────────────────────────────────────
MIN_BOXES = 8
WHITE_THRESH = 210
MIN_AREA = 400
MAX_AREA = 8000
ASPECT_MIN, ASPECT_MAX = 0.6, 1.6
KEY_DELAY_MS = 25
KEY_DELAY_MIN = 0
KEY_DELAY_MAX = 200
KEY_DELAY_MAX_ENTRY = 500
KEY_HOLD_MS = 20
PRE_PRESS_MS = 35
SCAN_INTERVAL_SEC = 0.06
BUSTENI_SCAN_SEC = 0.012  # buclă rapidă pentru sincronizarea Busteni (~80fps)
MINA_SCAN_SEC = 0.002       # pauză minimă când nu e piatră / așteptăm re-arm
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

SELECTED_REGION: dict | None = None
_APP_DIR = app_dir()
PID_FILE = _APP_DIR / ".keyauto.pid"
DEBUG_DIR = _APP_DIR / "debug_busteni"
MINA_DEBUG_DIR = _APP_DIR / "debug_mina"
REGION_FILE = _APP_DIR / "regions.json"

# Zone salvate, una per minijoc (Chei / Busteni au regiuni diferite pe ecran).
# Persistăm și ultimul mod folosit, ca să pornim direct pe tab-ul potrivit.
_REGIONS: dict[str, dict] = {}


def _load_config() -> tuple[dict[str, dict], str | None, int]:
    """Întoarce (zone_per_mod, ultimul_mod, delay_taste_ms)."""
    try:
        if REGION_FILE.exists():
            data = json.loads(REGION_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                regions = {k: v for k, v in data.items() if isinstance(v, dict)}
                mode = data.get("__mode__")
                raw_delay = data.get("__key_delay_ms__", KEY_DELAY_MS)
                delay = int(raw_delay) if isinstance(raw_delay, (int, float)) else KEY_DELAY_MS
                delay = max(KEY_DELAY_MIN, min(KEY_DELAY_MAX_ENTRY, delay))
                return regions, (mode if mode in ("Chei", "Busteni", "Mina") else None), delay
    except Exception as e:
        log.warn(f"Nu pot citi zonele salvate: {e}")
    return {}, None, KEY_DELAY_MS


def _save_config(mode: str, *, key_delay_ms: int | None = None) -> None:
    try:
        data: dict = dict(_REGIONS)
        data["__mode__"] = mode
        if key_delay_ms is not None:
            data["__key_delay_ms__"] = key_delay_ms
        elif REGION_FILE.exists():
            prev = json.loads(REGION_FILE.read_text(encoding="utf-8"))
            if isinstance(prev, dict) and "__key_delay_ms__" in prev:
                data["__key_delay_ms__"] = prev["__key_delay_ms__"]
        REGION_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        log.warn(f"Nu pot salva configul: {e}")

_sct: "mss.base.MSSBase | None" = None
_dxcam = None
_ocr = None


def _extract_crops(frame: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> list[np.ndarray]:
    crops: list[np.ndarray] = []
    for x, y, w, h in boxes:
        pad = max(2, int(min(w, h) * 0.08))
        crop = frame[max(0, y + pad) : y + h - pad, max(0, x + pad) : x + w - pad]
        if crop.size:
            crops.append(crop)
    return crops


def normalize_char(ch: str) -> str:
    """Convertește fullwidth Unicode și filtrează strict A-Z0-9."""
    if not ch:
        return ""
    ch = unicodedata.normalize("NFKC", ch).upper()
    return ch if ch in vis.ALLOWED else ""


def normalize_sequence(text: str) -> str:
    return "".join(normalize_char(c) for c in text)


def _save_pid() -> None:
    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")


def _clear_pid() -> None:
    try:
        if PID_FILE.exists() and PID_FILE.read_text(encoding="utf-8").strip() == str(os.getpid()):
            PID_FILE.unlink()
    except Exception:
        pass


# ── Captură ecran (dxcam → fallback mss) ───────────────────────────────────
CAPTURE_RETRIES = 6
CAPTURE_RETRY_DELAY_SEC = 0.045
HOTKEY_CAPTURE_SETTLE_MS = 40  # minigame-ul are nevoie de un cadru după F6
def _init_capture():
    global _dxcam
    try:
        import dxcam

        _dxcam = dxcam.create(output_color="BGR")
        log.info("Captură: DXGI (dxcam) activ")
    except Exception as e:
        _dxcam = None
        log.warn(f"DXGI indisponibil, folosesc MSS — {e}")


def _get_sct() -> "mss.base.MSSBase":
    global _sct
    if _sct is None:
        _sct = mss.mss()
    return _sct


def _reset_sct() -> None:
    global _sct
    if _sct is not None:
        try:
            _sct.close()
        except Exception:
            pass
    _sct = None


def _is_capture_error(exc: BaseException) -> bool:
    if type(exc).__name__ == "ScreenShotError":
        return True
    msg = str(exc).lower()
    return "bitblt" in msg or "screenshot" in msg or "capture" in msg


def grab_screen() -> tuple[np.ndarray, str]:
    if not SELECTED_REGION:
        raise RuntimeError("Nicio zonă selectată")

    r = SELECTED_REGION
    left, top = r["left"], r["top"]
    right, bottom = left + r["width"], top + r["height"]
    region_dxcam = (left, top, right, bottom)
    last_err: Exception | None = None

    for attempt in range(CAPTURE_RETRIES):
        t0 = time.perf_counter()
        try:
            if _dxcam is not None:
                frame = _dxcam.grab(region=region_dxcam)
                if frame is not None:
                    ms = (time.perf_counter() - t0) * 1000
                    tag = f" (încercarea {attempt + 1})" if attempt else ""
                    log.debug(f"Captură DXGI {frame.shape[1]}x{frame.shape[0]} · {ms:.1f}ms{tag}")
                    return frame, "DXGI"

            shot = _get_sct().grab(r)
            frame = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)
            ms = (time.perf_counter() - t0) * 1000
            tag = f" (încercarea {attempt + 1})" if attempt else ""
            log.debug(f"Captură MSS {frame.shape[1]}x{frame.shape[0]} · {ms:.1f}ms{tag}")
            return frame, "MSS"
        except Exception as e:
            last_err = e
            _reset_sct()
            if attempt < CAPTURE_RETRIES - 1:
                log.debug(f"Captură eșuată ({attempt + 1}/{CAPTURE_RETRIES}): {e}")
                time.sleep(CAPTURE_RETRY_DELAY_SEC)

    raise last_err if last_err else RuntimeError("Captură eșuată")


def capture_monitor() -> tuple[Image.Image, dict]:
    sct = _get_sct()
    mon = sct.monitors[1]
    shot = sct.grab(mon)
    img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
    log.debug(f"Screenshot monitor: {mon['width']}x{mon['height']}")
    return img, mon


# ── OCR (RapidOCR → fallback template matching) ───────────────────────────
def _init_ocr():
    global _ocr
    for factory, label in (
        ("rapidocr", "RapidOCR v3"),
        ("rapidocr_onnxruntime", "RapidOCR"),
    ):
        try:
            if factory == "rapidocr":
                from rapidocr import RapidOCR

                _ocr = RapidOCR(params={"Global.use_cls": False})
            else:
                from rapidocr_onnxruntime import RapidOCR

                _ocr = RapidOCR()
            log.info(f"OCR: {label} încărcat")
            vis.warmup_ocr(_ocr)
            return
        except Exception as e:
            log.debug(f"{label} indisponibil: {e}")
    _ocr = None
    log.warn("OCR indisponibil — doar template/shape")


def _ocr_label() -> str:
    if _ocr is None:
        return "Shape"
    try:
        from rapidocr import RapidOCR as R3

        if isinstance(_ocr, R3):
            return "RapidOCR v3"
    except ImportError:
        pass
    return "RapidOCR"


def read_sequence(frame: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> tuple[list[str], str]:
    crops = _extract_crops(frame, boxes)
    if not crops:
        return [], "none"

    t0 = time.perf_counter()
    chars, mode = vis.read_sequence_chars(crops, _ocr)
    ms = (time.perf_counter() - t0) * 1000
    valid = sum(1 for c in chars if c and c != "?")

    if valid >= MIN_BOXES:
        log.info(f"OCR {mode} · {ms:.0f}ms: {' '.join(chars)}")
        return chars, mode

    log.warn(f"OCR incomplet ({valid}/{len(boxes)}) · {ms:.0f}ms")
    return chars, mode


# ── Detectare casete ───────────────────────────────────────────────────────
def detect_white_boxes(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, WHITE_THRESH, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1
    )

    raw_count = 0
    boxes: list[tuple[int, int, int, int]] = []
    for cnt in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        raw_count += 1
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < MIN_AREA or area > MAX_AREA:
            continue
        if not (ASPECT_MIN <= w / max(h, 1) <= ASPECT_MAX):
            continue
        if gray[y : y + h, x : x + w].mean() < 200:
            continue
        boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[0])
    filtered = _filter_row(boxes)
    log.debug(f"Contururi: {raw_count} · candidati: {len(boxes)} · rand final: {len(filtered)}")
    if filtered:
        pos = ", ".join(f"({x},{y})" for x, y, _, _ in filtered[:6])
        suffix = "..." if len(filtered) > 6 else ""
        log.debug(f"Pozitii casete: {pos}{suffix}")
    return filtered


def _filter_row(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    if len(boxes) < MIN_BOXES:
        return []

    med_h = float(np.median([b[3] for b in boxes]))
    y_tol = med_h * 0.6
    rows: dict[int, list] = {}
    for box in boxes:
        rows.setdefault(int(box[1] / y_tol) if y_tol else box[1], []).append(box)

    best = max(rows.values(), key=len)
    if len(best) < MIN_BOXES:
        return []

    med_y = float(np.median([b[1] for b in best]))
    return sorted([b for b in best if abs(b[1] - med_y) <= y_tol], key=lambda b: b[0])


def _press_key(ch: str) -> bool:
    return game_input.press_key(ch)


def scan_and_press(
    status_cb=None,
    key_delay_ms: int | None = None,
    prepare_cb=None,
    *,
    settle_ms: int = 0,
) -> tuple[bool, str]:
    t_total = time.perf_counter()
    log.info("── Scan start ──")

    if settle_ms > 0:
        time.sleep(settle_ms / 1000)

    frame, capture_mode = grab_screen()
    boxes = detect_white_boxes(frame)

    if len(boxes) < MIN_BOXES:
        log.warn(f"Prea puține casete: {len(boxes)} (min {MIN_BOXES})")
        return False, f"Casete: {len(boxes)} (min {MIN_BOXES})"

    focus_ok = [False]

    def _focus():
        focus_ok[0] = game_input.focus_game(fast=True, minimize_cb=prepare_cb)

    focus_thread = threading.Thread(target=_focus, daemon=True)
    focus_thread.start()

    chars, ocr_mode = read_sequence(frame, boxes)
    focus_thread.join(timeout=0.4)

    if len(chars) < MIN_BOXES:
        partial = " ".join(chars) if chars else "(niciuna)"
        log.warn(f"OCR incomplet: {len(chars)}/{len(boxes)} · mod {ocr_mode}")
        log.warn(f"Detectat parțial: {partial}")
        return False, f"Citite: {len(chars)}/{len(boxes)}"

    sequence = normalize_sequence("".join(chars))
    if len(sequence) != len(chars):
        log.warn(f"Caractere ignorate după normalizare: {''.join(chars)} -> {sequence}")
    chars = list(sequence)
    spaced = " ".join(chars)
    log.info(f"Detectat [{ocr_mode} · {capture_mode}]: {spaced}")
    log.info(f"Secvență: {sequence} ({len(chars)} taste)")

    if status_cb:
        status_cb(f"Apas: {sequence}")

    delay = key_delay_ms if key_delay_ms is not None else KEY_DELAY_MS
    game_input.set_hold_ms(KEY_HOLD_MS)
    log.info(f"── Apăs taste (delay {delay}ms) ──")

    if not focus_ok[0]:
        log.warn("Focus incert — folosește F6 din joc + Administrator")

    time.sleep(PRE_PRESS_MS / 1000)

    for i, ch in enumerate(chars):
        ok = _press_key(ch)
        log.info(f"Apăsat [{i + 1}/{len(chars)}]: {ch}" + ("" if ok else " (eșuat)"))
        if i < len(chars) - 1:
            time.sleep(delay / 1000)

    ms = (time.perf_counter() - t_total) * 1000
    log.info(f"── Gata · {len(chars)} taste apăsate · {ms:.0f}ms ──")
    return True, sequence


# ── Selector zonă ──────────────────────────────────────────────────────────
class RegionSelector(tk.Toplevel):
    MIN_SIZE = 40

    def __init__(self, master: tk.Misc, on_done, screenshot: Image.Image, monitor: dict):
        super().__init__(master)
        self.on_done = on_done
        self.region: dict | None = None
        self._mon = monitor
        self._start_x = self._start_y = 0
        self._rect_id: int | None = None
        self._size_id: int | None = None
        self._shade_ids: list[int] = []

        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.grab_set()
        self.geometry(f"{monitor['width']}x{monitor['height']}+{monitor['left']}+{monitor['top']}")

        self._canvas = tk.Canvas(self, cursor="crosshair", highlightthickness=0)
        self._canvas.pack(fill=tk.BOTH, expand=True)

        self._photo = ImageTk.PhotoImage(screenshot)
        self._canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)
        self._hint_id = self._canvas.create_text(
            monitor["width"] // 2,
            36,
            text="Trage peste casetele albe  ·  Enter confirmă  ·  Esc anulează",
            fill="#ffffff",
            font=("Segoe UI", 11, "bold"),
        )

        self._canvas.bind("<ButtonPress-1>", self._on_press)
        self._canvas.bind("<B1-Motion>", self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Return>", lambda _: self._confirm())
        self.bind("<Escape>", lambda _: self._cancel())
        self.focus_force()
        log.info("Selector zonă deschis")

    def _update_shade(self, x1: int, y1: int, x2: int, y2: int):
        for sid in self._shade_ids:
            self._canvas.delete(sid)
        self._shade_ids.clear()

        w, h = self._mon["width"], self._mon["height"]
        for rx1, ry1, rx2, ry2 in [(0, 0, w, y1), (0, y2, w, h), (0, y1, x1, y2), (x2, y1, w, y2)]:
            if rx2 > rx1 and ry2 > ry1:
                self._shade_ids.append(
                    self._canvas.create_rectangle(
                        rx1, ry1, rx2, ry2, fill="#000000", stipple="gray50", outline=""
                    )
                )
        self._canvas.tag_raise(self._hint_id)

    def _on_press(self, event):
        self._start_x, self._start_y = event.x, event.y
        if self._rect_id:
            self._canvas.delete(self._rect_id)
        if self._size_id:
            self._canvas.delete(self._size_id)
        self._rect_id = self._canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline="#4ade80", width=2
        )

    def _on_drag(self, event):
        if self._rect_id is None:
            return
        x1, y1 = min(self._start_x, event.x), min(self._start_y, event.y)
        x2, y2 = max(self._start_x, event.x), max(self._start_y, event.y)
        self._canvas.coords(self._rect_id, x1, y1, x2, y2)
        self._update_shade(x1, y1, x2, y2)
        if self._size_id:
            self._canvas.delete(self._size_id)
        self._size_id = self._canvas.create_text(
            (x1 + x2) // 2, y1 - 12, text=f"{x2 - x1} × {y2 - y1}", fill="#4ade80", font=("Segoe UI", 9)
        )
        self._canvas.tag_raise(self._rect_id)
        self._canvas.tag_raise(self._size_id)

    def _on_release(self, event):
        self._store_region(event.x, event.y)

    def _store_region(self, end_x: int, end_y: int):
        x1, y1 = min(self._start_x, end_x), min(self._start_y, end_y)
        w, h = abs(end_x - self._start_x), abs(end_y - self._start_y)
        if w < self.MIN_SIZE or h < self.MIN_SIZE:
            return
        self.region = {
            "left": self._mon["left"] + x1,
            "top": self._mon["top"] + y1,
            "width": w,
            "height": h,
        }

    def _confirm(self):
        if self.region is None and self._rect_id:
            c = self._canvas.coords(self._rect_id)
            if len(c) == 4:
                self._store_region(int(c[2]), int(c[3]))
        self._finish()

    def _cancel(self):
        log.warn("Selector zonă anulat (Esc)")
        self.region = None
        self._finish()

    def _finish(self):
        if self.region:
            r = self.region
            log.info(f"Zonă selectată: {r['width']}x{r['height']} @ ({r['left']}, {r['top']})")
        self.grab_release()
        self.destroy()
        self.on_done(self.region)


def pick_region(master: tk.Misc, on_done):
    screenshot, monitor = capture_monitor()
    RegionSelector(master, on_done, screenshot, monitor)


# ── UI ─────────────────────────────────────────────────────────────────────
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Key Auto")
        self.geometry("380x540")
        self.minsize(360, 500)
        self.resizable(True, True)

        self._running = False
        self._auto = False
        self._mina_active = False
        self._mode = "Chei"  # "Chei" | "Busteni" | "Mina"
        self._lock = threading.Lock()

        global _REGIONS, SELECTED_REGION
        _REGIONS, last_mode, saved_delay = _load_config()
        if last_mode:
            self._mode = last_mode  # pornim pe ultimul tab folosit
        SELECTED_REGION = _REGIONS.get(self._mode)
        self._key_delay_ms = saved_delay

        self._build_ui()
        self._set_key_delay_ms(saved_delay, save=False)
        # sincronizăm tab-ul + layout-ul cu modul salvat
        self._mode_seg.set(self._mode)
        self._apply_mode_ui(self._mode)

        log.attach_ui(self._log_panel)
        log.info(f"App pornită · {_ocr_label()} · {'DXGI' if _dxcam else 'MSS'} · delay {self._key_delay_ms}ms")
        if SELECTED_REGION:
            r = SELECTED_REGION
            log.info(f"Zonă încărcată [{self._mode}]: {r['width']}x{r['height']} @ ({r['left']}, {r['top']})")
        elif self._mode == "Mina":
            log.info("Mod Mina — ecran complet · Start sau F6")
        else:
            log.info(f"Mod {self._mode} — selectează o zonă")
        self._show_ready()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        try:
            import keyboard

            keyboard.add_hotkey("f6", self._hotkey_trigger, suppress=False)
            log.debug("Hotkey F6 înregistrat")
        except Exception as e:
            log.warn(f"Hotkey F6 indisponibil: {e}")

    def _build_ui(self):
        self.configure(fg_color="#070709")

        # ── Antet ──────────────────────────────────────────────
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=18, pady=(16, 2))

        ctk.CTkLabel(
            header, text="Key Auto", font=ctk.CTkFont(size=19, weight="bold")
        ).pack(anchor="w")

        self._status = ctk.CTkLabel(
            header, text=self._ready_text(), text_color="#6b7280", font=ctk.CTkFont(size=12)
        )
        self._status.pack(anchor="w", pady=(1, 0))

        # ── Card comenzi ──────────────────────────────────────
        card = ctk.CTkFrame(self, fg_color="#121217", corner_radius=12)
        card.pack(fill="x", padx=16, pady=(12, 4))

        self._mode_seg = ctk.CTkSegmentedButton(
            card, values=["Chei", "Busteni", "Mina"], command=self._on_mode_change,
            font=ctk.CTkFont(size=13)
        )
        self._mode_seg.set("Chei")
        self._mode_seg.pack(fill="x", padx=14, pady=(14, 10))

        self._btn = ctk.CTkButton(
            card, text="Start", height=42,
            font=ctk.CTkFont(size=15, weight="bold"), command=self._toggle
        )
        self._btn.pack(fill="x", padx=14, pady=(0, 10))

        row = ctk.CTkFrame(card, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=(0, 6))

        self._auto_var = ctk.BooleanVar(value=False)
        self._auto_switch = ctk.CTkSwitch(
            row, text="Auto-scan", variable=self._auto_var,
            command=self._on_auto_toggle, font=ctk.CTkFont(size=13)
        )
        self._auto_switch.pack(side="left")

        self._busteni_debug = ctk.BooleanVar(value=False)  # implicit OFF — fără capturi
        self._dbg_switch = ctk.CTkSwitch(
            row, text="Capturi debug", variable=self._busteni_debug,
            font=ctk.CTkFont(size=13)
        )  # afișat doar în modul Busteni (vezi _on_mode_change); pornește-l doar la nevoie

        self._mina_captures = ctk.BooleanVar(value=True)
        self._mina_cap_switch = ctk.CTkSwitch(
            row, text="Capturi detectare", variable=self._mina_captures,
            font=ctk.CTkFont(size=13)
        )

        ctk.CTkButton(
            row, text="Select zonă", width=104, height=30,
            font=ctk.CTkFont(size=12), fg_color="#24242c", hover_color="#30303a",
            command=self._pick_region,
        ).pack(side="right")

        self._delay_row = ctk.CTkFrame(card, fg_color="transparent")
        self._delay_row.pack(fill="x", padx=14, pady=(8, 14))
        ctk.CTkLabel(
            self._delay_row, text="Delay taste", text_color="#6b7280", font=ctk.CTkFont(size=12)
        ).pack(side="left")
        self._delay_entry = ctk.CTkEntry(
            self._delay_row, width=52, height=28, justify="center",
            font=ctk.CTkFont(size=12), fg_color="#141418", border_color="#30303a",
        )
        self._delay_entry.pack(side="right")
        self._delay_entry_editing: int | None = None
        self._delay_entry.bind("<FocusIn>", self._on_delay_entry_focus_in)
        self._delay_entry.bind("<Return>", self._on_delay_entry_commit)
        self._delay_entry.bind("<Escape>", self._on_delay_entry_cancel)
        self._delay_entry.bind("<FocusOut>", self._on_delay_entry_focus_out)
        self._delay_slider = ctk.CTkSlider(
            self._delay_row,
            from_=KEY_DELAY_MIN,
            to=KEY_DELAY_MAX,
            number_of_steps=KEY_DELAY_MAX - KEY_DELAY_MIN,
            command=self._on_delay_slider,
        )
        self._delay_slider.pack(side="left", fill="x", expand=True, padx=(12, 10))

        # ── Log ────────────────────────────────────────────────
        self._log_panel = log.LogPanel(self, height=190)

        self._hint = ctk.CTkLabel(
            self, text="Apasă  F6  în joc pentru a scana",
            text_color="#44464d", font=ctk.CTkFont(size=11)
        )
        self._hint.pack(pady=(0, 10))

    def _set_key_delay_ms(self, ms: int, *, save: bool = True, update_entry: bool = True) -> None:
        ms = max(KEY_DELAY_MIN, min(KEY_DELAY_MAX_ENTRY, int(ms)))
        self._key_delay_ms = ms
        self._delay_slider.set(min(KEY_DELAY_MAX, ms))
        if update_entry:
            self._delay_entry.delete(0, "end")
            self._delay_entry.insert(0, str(ms))
        if save:
            _save_config(self._mode, key_delay_ms=ms)
        log.debug(f"Delay taste: {ms}ms")

    def _on_delay_slider(self, value: float) -> None:
        self._set_key_delay_ms(int(round(value)))

    def _on_delay_entry_focus_in(self, _event=None) -> None:
        self._delay_entry_editing = self._key_delay_ms

    def _on_delay_entry_commit(self, _event=None) -> str:
        raw = self._delay_entry.get().strip().lower().replace("ms", "")
        try:
            self._set_key_delay_ms(int(raw))
        except ValueError:
            self._set_key_delay_ms(self._key_delay_ms, save=False)
        self._delay_entry_editing = None
        return "break"

    def _on_delay_entry_cancel(self, _event=None) -> str:
        if self._delay_entry_editing is not None:
            self._set_key_delay_ms(self._delay_entry_editing, save=False)
        self._delay_entry_editing = None
        self.focus()
        return "break"

    def _on_delay_entry_focus_out(self, _event=None) -> None:
        if self._delay_entry_editing is None:
            return
        raw = self._delay_entry.get().strip().lower().replace("ms", "")
        if not raw:
            self._set_key_delay_ms(self._delay_entry_editing, save=False)
        else:
            try:
                self._set_key_delay_ms(int(raw))
            except ValueError:
                self._set_key_delay_ms(self._delay_entry_editing, save=False)
        self._delay_entry_editing = None

    def _ready_text(self) -> str:
        if self._mode == "Mina":
            cap = "DXGI" if _dxcam else "MSS"
            return f"● Gata · Mina · {cap} · ecran complet"
        if not SELECTED_REGION:
            return "● Selectează o zonă pentru a începe"
        cap = "DXGI" if _dxcam else "MSS"
        return f"● Gata · {_ocr_label()} · {cap}"

    def _set_status(self, text: str, color: str = "#888888"):
        self.after(0, lambda: self._status.configure(text=text, text_color=color))

    def _show_ready(self):
        """Stare clară de disponibilitate pentru F6."""
        if self._mode == "Mina":
            if self._mina_active:
                self._set_status("● MINA activă — scanez piatra", "#5aa9e6")
            else:
                self._set_status("● GATA — Start sau F6 în joc", "#46d369")
        elif not SELECTED_REGION:
            self._set_status("● Selectează o zonă", "#e5b567")
        elif self._mode == "Busteni":
            self._set_status("● GATA — apasă F6 în joc", "#46d369")
        else:
            cap = "DXGI" if _dxcam else "MSS"
            self._set_status(f"● GATA — F6 · {_ocr_label()} · {cap}", "#46d369")

    def _pick_region(self):
        was_auto = self._auto
        self._auto = False
        self._auto_var.set(False)
        self._btn.configure(text="Start", fg_color=["#3B8ED0", "#1F6AA5"])
        self.withdraw()
        log.info("Reselectare zonă...")

        def on_done(region: dict | None):
            global SELECTED_REGION
            if region:
                SELECTED_REGION = region
                _REGIONS[self._mode] = region
                _save_config(self._mode)  # rămâne salvată între porniri, per minijoc
                log.info(f"Zonă salvată [{self._mode}]")
                self._set_status("● Zonă salvată", "#6b6")
            else:
                self._set_status("● Zonă neschimbată", "#888888")
            self.deiconify()
            self.lift()
            self.focus_force()
            if was_auto:
                self._auto_var.set(True)
                self._on_auto_toggle()

        pick_region(self, on_done)

    def _apply_mode_ui(self, mode: str):
        """Aranjează widget-urile pentru modul dat (fără efecte secundare)."""
        if mode == "Mina":
            self._auto_switch.pack_forget()
            self._dbg_switch.pack_forget()
            self._mina_cap_switch.pack(side="left", padx=(12, 0))
            self._delay_row.pack_forget()
            self._btn.configure(text="Start")
            self._hint.configure(text="Start / F6 — click automat · capturi în debug_mina/")
        elif mode == "Busteni":
            self._mina_cap_switch.pack_forget()
            self._delay_row.pack(fill="x", padx=14, pady=(8, 14))
            self._auto_switch.pack_forget()  # Busteni nu buclează — o sesiune per F6
            self._dbg_switch.pack(side="left")
            self._btn.configure(text="Arm (F6)")
            self._hint.configure(text="F6 în joc · delay = pauză după fiecare cifră apăsată")
        else:
            self._mina_cap_switch.pack_forget()
            self._delay_row.pack(fill="x", padx=14, pady=(8, 14))
            self._dbg_switch.pack_forget()
            self._auto_switch.pack(side="left")
            self._btn.configure(text="Start")
            self._hint.configure(text="Apasă  F6  în joc pentru a scana")

    def _on_mode_change(self, mode: str):
        if self._auto:  # oprim auto-scanul (specific Chei) la schimbarea modului
            self._auto_var.set(False)
            self._on_auto_toggle()
        if self._mina_active:
            self._stop_mina()
        self._mode = mode
        global SELECTED_REGION
        SELECTED_REGION = _REGIONS.get(mode)  # fiecare minijoc are zona lui salvată
        self._apply_mode_ui(mode)
        self._show_ready()
        _save_config(mode)  # reținem ultimul tab folosit
        log.info(f"Mod minijoc: {mode}")

    def _hotkey_trigger(self):
        game_input.capture_target_window()
        log.debug("F6 apăsat")
        if self._running and self._mode != "Mina":
            return
        if self._mode == "Busteni":
            self.after(0, self._run_busteni)
        elif self._mode == "Mina":
            self.after(0, self._toggle_mina)
        else:
            self.after(0, lambda: self._run_once(from_hotkey=True))

    def _toggle(self):
        if self._mode == "Mina":
            self._toggle_mina()
            return
        if self._mode == "Busteni":
            self._run_busteni()
            return
        if self._auto_var.get():
            self._on_auto_toggle()
            return
        self._run_once()

    def _on_auto_toggle(self):
        if self._auto_var.get():
            self._auto = True
            self._btn.configure(text="Stop", fg_color="#c44", hover_color="#a33")
            self._set_status("● Auto activ", "#6b6")
            log.info("Mod Auto: ON")
            threading.Thread(target=self._auto_loop, daemon=True).start()
        else:
            self._auto = False
            self._btn.configure(text="Start", fg_color=["#3B8ED0", "#1F6AA5"], hover_color=["#36719F", "#144870"])
            self._set_status(self._ready_text())
            log.info("Mod Auto: OFF")

    def _run_once(self, *, from_hotkey: bool = False):
        if not SELECTED_REGION:
            log.warn("Scan blocat: nicio zonă selectată")
            self._set_status("● Selectează o zonă", "#c88")
            return
        if self._running:
            log.debug("Scan ignorat: deja în curs")
            return
        threading.Thread(target=self._execute, args=(from_hotkey,), daemon=True).start()

    # ── Busteni: o sesiune auto-oprită per F6 ───────────────────────────────
    def _run_busteni(self):
        if not SELECTED_REGION:
            log.warn("Busteni blocat: nicio zonă selectată")
            self._set_status("● Selectează o zonă", "#c88")
            return
        if self._running:
            log.debug("Busteni ignorat: deja în curs")
            return
        threading.Thread(target=self._busteni_loop, daemon=True).start()

    def _busteni_loop(self):
        with self._lock:
            if self._running:
                return
            self._running = True

        log.info("── Busteni: sesiune armată (F6) ──")
        self._set_status("● SCANEZ (Busteni)…", "#5aa9e6")
        game_input.set_hold_ms(KEY_HOLD_MS)

        # Aducem jocul în față (asta lasă fereastra noastră în spate); NU minimizăm și
        # NU readucem fereastra — o lăsăm exact unde e, ca să nu deranjăm jocul.
        game_input.focus_game(fast=True)

        # dxcam NU e sigur în afara firului care l-a creat (crash nativ); în bucla
        # Busteni folosim o instanță mss proprie firului, creată și închisă aici.
        sct = mss.mss()
        region = SELECTED_REGION
        log.info(f"Busteni: captură MSS {region['width']}x{region['height']} @ {BUSTENI_SCAN_SEC*1000:.0f}ms")

        debug_on = self._busteni_debug.get()
        dbg_dir = saved = None
        if debug_on:
            from datetime import datetime
            DEBUG_DIR.mkdir(exist_ok=True)
            dbg_dir = DEBUG_DIR / f"sess_{datetime.now():%H%M%S}"
            dbg_dir.mkdir(exist_ok=True)
            saved = 0
            log.info(f"Busteni: capturi debug în {dbg_dir.name}")

        DBG_MAX = 120          # plafon imagini per sesiune
        DBG_EVERY = 0.25       # cadență salvare cât timp e teal pe ecran
        last_save = 0.0
        teal_announced = False

        def _save(tag, frame, st, digit=None, fired=False):
            nonlocal saved
            if not debug_on or saved is None or saved >= DBG_MAX:
                return
            # la FIRE salvăm și cadrul BRUT (neadnotat) ca să putem verifica citirea cifrei
            if fired:
                cv2.imwrite(str(dbg_dir / f"{saved:03d}_{tag}_raw.png"), frame)
            # afișăm zona CACHUITĂ (cea folosită la declanșare), nu măsurarea ocluzată
            disp = dict(st)
            if session.zone_center is not None:
                disp["zone_center"] = session.zone_center
                disp["zone_half"] = session.zone_half
            img = busteni.annotate(frame, disp, digit, fired)
            cv2.imwrite(str(dbg_dir / f"{saved:03d}_{tag}.png"), img)
            saved += 1

        session = busteni.BustenSession()
        session.start(time.perf_counter())
        reason = "cap"
        frames = 0
        try:
            while True:
                now = time.perf_counter()
                reason = session.should_stop(now)
                if reason:
                    break
                shot = sct.grab(region)
                frame = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)
                frames += 1
                action = session.process(frame, now)
                st = session.last_state

                if debug_on and frames == 1:
                    _save("first", frame, st, session.cur_digit)  # ce vede regiunea
                if st.get("has_teal") and not teal_announced:
                    teal_announced = True
                    log.info(f"Busteni: minijoc detectat (z={st['n_zone']} i={st['n_ind']}) cifră={session.cur_digit}")
                    _save("teal", frame, st, session.cur_digit)
                if debug_on and st.get("has_teal") and now - last_save >= DBG_EVERY:
                    last_save = now
                    _save("scan", frame, st, session.cur_digit)

                if action and action[0] == "press":
                    digit = action[1]
                    _save("FIRE", frame, st, digit, fired=True)
                    ok = game_input.press_key(digit)
                    import math as _m
                    zc = session.zone_center
                    ia = st.get("ind_angle")
                    ang = (f" [zonă {_m.degrees(zc):.0f}° · ind {_m.degrees(ia):.0f}°]"
                           if zc is not None and ia is not None else "")
                    log.info(f"Busteni: {digit} (runda {session.pressed_count}, locked={session._digit_locked}){ang}"
                             + ("" if ok else " — eșuat"))
                    self._set_status(f"● Busteni: {digit}", "#6b6")
                    if self._key_delay_ms > 0:
                        time.sleep(self._key_delay_ms / 1000)
                time.sleep(BUSTENI_SCAN_SEC)

            extra = f" · {saved} capturi" if debug_on else ""
            log.info(f"── Busteni stop [{reason}] · {session.pressed_count} apăsări · {frames} cadre{extra} ──")
            # NU readucem fereastra în față — rămânem în joc. Statusul se actualizează
            # oricum, iar F6 pornește o nouă sesiune fără fereastră.
            self._show_ready()  # din nou GATA — se poate apăsa F6 pentru o nouă sesiune
        except Exception as exc:
            log.error(f"Eroare Busteni: {exc}", exc)
            self._set_status(f"● Eroare: {exc}", "#c66")
        finally:
            try:
                sct.close()
            except Exception:
                pass
            self._running = False

    def _stop_mina(self, *, completed: bool = False, clicks: int = 0) -> None:
        self._mina_active = False
        self._btn.configure(
            text="Start",
            fg_color=["#3B8ED0", "#1F6AA5"],
            hover_color=["#36719F", "#144870"],
        )
        if completed:
            log.info(f"Mina: secvență completă · {clicks} click-uri")
            self._set_status("● Secvență completă — GATA", "#46d369")
        else:
            log.info("Mina: OFF")
            self._show_ready()

    def _toggle_mina(self) -> None:
        if self._mina_active:
            self._stop_mina()
            return
        self._mina_active = True
        self._btn.configure(text="Stop", fg_color="#c44", hover_color="#a33")
        self._set_status("● MINA activă — scanez piatra", "#5aa9e6")
        log.info("Mina: ON — click automat pe piatră")
        threading.Thread(target=self._mina_loop, daemon=True).start()

    def _mina_loop(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True

        sct = mss.mss()
        mon = sct.monitors[1]
        clicks = 0
        gate = mina.MinaClickGate()
        completed = False
        mina._get_scaled_templates()
        log.info(f"Mina: captură {mon['width']}x{mon['height']}")

        cap_on = self._mina_captures.get()
        dbg_dir: Path | None = None
        cap_idx = 0
        last_cap_t = 0.0
        last_cap_xy: tuple[int, int] | None = None
        MINA_CAP_EVERY = 0.18
        MINA_CAP_MOVE = 22

        if cap_on:
            from datetime import datetime

            MINA_DEBUG_DIR.mkdir(exist_ok=True)
            dbg_dir = MINA_DEBUG_DIR / f"sess_{datetime.now():%Y%m%d_%H%M%S}"
            dbg_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"Mina: capturi detectare → {dbg_dir}")

        def _save_cap(frame: np.ndarray, hit: dict | None, tag: str, caption: str = "") -> None:
            nonlocal cap_idx, last_cap_t, last_cap_xy
            if not cap_on or dbg_dir is None or cap_idx >= mina.MINA_DEBUG_MAX:
                return
            path = mina.save_detection_shot(frame, hit, dbg_dir, cap_idx, tag, caption=caption)
            if path is not None:
                cap_idx += 1
                if hit:
                    last_cap_xy = hit["center"]
                last_cap_t = time.perf_counter()
                log.debug(f"Mina cap: {path.name}")

        def _maybe_save_detect(frame: np.ndarray, hit: dict, *, force: bool = False) -> None:
            if not cap_on or dbg_dir is None:
                return
            cx, cy = hit["center"]
            now_c = time.perf_counter()
            moved = (
                last_cap_xy is None
                or (cx - last_cap_xy[0]) ** 2 + (cy - last_cap_xy[1]) ** 2 >= MINA_CAP_MOVE ** 2
            )
            if force or moved or (now_c - last_cap_t) >= MINA_CAP_EVERY:
                method = hit.get("method", "?")
                _save_cap(
                    frame, hit, "detect",
                    caption=f"{method} score={hit['score']:.2f} @ ({cx},{cy})",
                )

        try:
            while self._mina_active and not gate.sequence_complete:
                t0 = time.perf_counter()
                shot = sct.grab(mon)
                frame = cv2.cvtColor(np.asarray(shot), cv2.COLOR_BGRA2BGR)
                match = mina.find_stone(
                    frame,
                    hint=gate.search_hint(),
                    dead_zone=gate.dead_zone(),
                )

                clicked = False
                if match and gate.is_valid_target(match):
                    _maybe_save_detect(frame, match)

                now_click = time.perf_counter()
                if match and gate.should_click(match, now_click):
                    cx, cy = match["center"]
                    sx = mon["left"] + cx
                    sy = mon["top"] + cy
                    ok = game_input.click_at(sx, sy)
                    stage = gate.on_click((cx, cy), now_click)
                    clicks += 1
                    clicked = True
                    ms = (time.perf_counter() - t0) * 1000
                    stone_n = min(
                        gate.stones_done + (1 if gate.stage < mina.STAGES_PER_STONE else 0),
                        mina.STONES_PER_SEQUENCE,
                    )
                    _save_cap(
                        frame, match,
                        f"click_p{stone_n}_c{stage}",
                        caption=(
                            f"CLICK piatra {stone_n}/{mina.STONES_PER_SEQUENCE} "
                            f"· {stage}/{mina.STAGES_PER_STONE} · {match['score']:.2f}"
                        ),
                    )
                    log.info(
                        f"Mina: piatra {stone_n}/{mina.STONES_PER_SEQUENCE} "
                        f"click {stage}/{mina.STAGES_PER_STONE} @ ({sx},{sy}) "
                        f"score={match['score']:.2f} · {ms:.0f}ms"
                        + ("" if ok else " — eșuat")
                    )
                    self._set_status(
                        f"● Mina: piatra {stone_n}/{mina.STONES_PER_SEQUENCE} "
                        f"· {stage}/{mina.STAGES_PER_STONE} · {ms:.0f}ms",
                        "#6b6",
                    )
                    if gate.sequence_complete or clicks >= mina.CLICKS_PER_SEQUENCE:
                        completed = True
                        self._mina_active = False
                        break
                elif match is None:
                    gate.should_click(None, now_click)
                    time.sleep(MINA_SCAN_SEC)
                elif not clicked:
                    time.sleep(MINA_SCAN_SEC)
        except Exception as exc:
            log.error(f"Eroare Mina: {exc}", exc)
            self._set_status(f"● Eroare: {exc}", "#c66")
        finally:
            try:
                sct.close()
            except Exception:
                pass
            self._running = False
            self._mina_active = False
            done = completed or gate.sequence_complete or clicks >= mina.CLICKS_PER_SEQUENCE
            extra = f" · {cap_idx} capturi în {dbg_dir.name}" if cap_on and dbg_dir and cap_idx else ""
            self.after(0, lambda d=done, c=clicks, e=extra: self._finish_mina_loop(d, c, e))

    def _finish_mina_loop(self, completed: bool, clicks: int, extra: str = "") -> None:
        """Cleanup UI pe main thread — o singură dată la ieșirea din buclă."""
        self._stop_mina(completed=completed, clicks=clicks)
        game_input.refocus_game()
        if completed:
            log.info(f"── Mina secvență completă · {clicks} click-uri{extra} ──")
        else:
            log.info(f"── Mina stop · {clicks} click-uri{extra} ──")

    def _execute(self, from_hotkey: bool = False):
        with self._lock:
            if self._running:
                return
            self._running = True

        self._set_status("● Scanez...", "#aaaaaa")

        def _prepare():
            if from_hotkey:
                return
            evt = threading.Event()

            def _minimize():
                try:
                    if self.state() != "iconic" and self.winfo_viewable():
                        self.iconify()
                finally:
                    evt.set()

            self.after(0, _minimize)
            evt.wait(timeout=0.5)

        try:
            ok, msg = False, ""
            settle = HOTKEY_CAPTURE_SETTLE_MS if from_hotkey else 0
            for attempt in range(3):
                try:
                    ok, msg = scan_and_press(
                        status_cb=lambda m: self._set_status(f"● {m}", "#6b6"),
                        key_delay_ms=self._key_delay_ms,
                        prepare_cb=_prepare if not from_hotkey else None,
                        settle_ms=settle if attempt == 0 else 0,
                    )
                    break
                except Exception as exc:
                    if attempt < 2 and _is_capture_error(exc):
                        log.warn(f"Captură indisponibilă, reîncerc scan ({attempt + 2}/3)...")
                        time.sleep(0.08)
                        continue
                    raise
            self._set_status(f"{'● OK:' if ok else '●'} {msg}", "#6b6" if ok else "#c88")
            game_input.refocus_game()
        except Exception as exc:
            log.error(f"Eroare scan: {exc}", exc)
            self._set_status(f"● Eroare: {exc}", "#c66")
            game_input.refocus_game()
        finally:
            self._running = False

    def _auto_loop(self):
        while self._auto:
            if not self._running:
                self._execute(from_hotkey=True)
            time.sleep(SCAN_INTERVAL_SEC)

    def _on_close(self):
        self._auto = False
        self._mina_active = False
        log.info("App închisă")
        _clear_pid()
        try:
            import keyboard

            keyboard.unhook_all_hotkeys()
        except Exception:
            pass
        if _dxcam is not None:
            try:
                _dxcam.release()
            except Exception:
                pass
        if _sct is not None:
            try:
                _sct.close()
            except Exception:
                pass
        self.destroy()


if __name__ == "__main__":
    log.init()
    game_input.init()
    _init_capture()
    _init_ocr()

    _save_pid()
    try:
        App().mainloop()
    finally:
        _clear_pid()
