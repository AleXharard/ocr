"""
Key Auto — detectează casetele albe și apasă tastele în ordine.
"""

from __future__ import annotations

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

import game_input
import logger as log
import vision as vis

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# ── Configurare ────────────────────────────────────────────────────────────
MIN_BOXES = 8
WHITE_THRESH = 210
MIN_AREA = 400
MAX_AREA = 8000
ASPECT_MIN, ASPECT_MAX = 0.6, 1.6
KEY_DELAY_MS = 25
KEY_HOLD_MS = 20
PRE_PRESS_MS = 35
SCAN_INTERVAL_SEC = 0.06
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

SELECTED_REGION: dict | None = None
PID_FILE = Path(__file__).resolve().parent / ".keyauto.pid"

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


def grab_screen() -> tuple[np.ndarray, str]:
    if not SELECTED_REGION:
        raise RuntimeError("Nicio zonă selectată")

    r = SELECTED_REGION
    left, top = r["left"], r["top"]
    right, bottom = left + r["width"], top + r["height"]
    t0 = time.perf_counter()

    if _dxcam is not None:
        frame = _dxcam.grab(region=(left, top, right, bottom))
        if frame is not None:
            ms = (time.perf_counter() - t0) * 1000
            log.debug(f"Captură DXGI {frame.shape[1]}x{frame.shape[0]} · {ms:.1f}ms")
            return frame, "DXGI"

    shot = _get_sct().grab(r)
    frame = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)
    ms = (time.perf_counter() - t0) * 1000
    log.debug(f"Captură MSS {frame.shape[1]}x{frame.shape[0]} · {ms:.1f}ms")
    return frame, "MSS"


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


def scan_and_press(status_cb=None, key_delay_ms: int | None = None, prepare_cb=None) -> tuple[bool, str]:
    t_total = time.perf_counter()
    log.info("── Scan start ──")

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
        self.geometry("340x420")
        self.minsize(340, 420)
        self.resizable(True, True)

        self._running = False
        self._auto = False
        self._lock = threading.Lock()

        self._build_ui()
        log.attach_ui(self._log_panel)
        log.info(f"App pornită · {_ocr_label()} · {'DXGI' if _dxcam else 'MSS'}")

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        try:
            import keyboard

            keyboard.add_hotkey("f6", self._hotkey_trigger, suppress=False)
            log.debug("Hotkey F6 înregistrat")
        except Exception as e:
            log.warn(f"Hotkey F6 indisponibil: {e}")

    def _build_ui(self):
        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x")

        ctk.CTkLabel(top, text="Key Auto", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(14, 2))

        cap = "DXGI" if _dxcam else "MSS"
        self._status = ctk.CTkLabel(
            top, text=f"● Gata · {_ocr_label()} · {cap}", text_color="#888888", font=ctk.CTkFont(size=12)
        )
        self._status.pack(pady=(0, 10))

        row = ctk.CTkFrame(top, fg_color="transparent")
        row.pack()
        self._btn = ctk.CTkButton(row, text="Start", width=90, command=self._toggle)
        self._btn.pack(side="left", padx=4)
        ctk.CTkButton(row, text="Zonă", width=70, fg_color="#2a2a2a", hover_color="#333333", command=self._pick_region).pack(
            side="left", padx=4
        )

        self._auto_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(top, text="Auto", variable=self._auto_var, command=self._on_auto_toggle).pack(pady=(10, 0))

        delay_row = ctk.CTkFrame(top, fg_color="transparent")
        delay_row.pack(pady=(8, 0))
        ctk.CTkLabel(delay_row, text="Delay", text_color="#666666", font=ctk.CTkFont(size=11)).pack(side="left")
        self._delay_slider = ctk.CTkSlider(
            delay_row, from_=30, to=90, number_of_steps=12, width=120, command=self._on_delay_change
        )
        self._delay_slider.set(KEY_DELAY_MS)
        self._delay_slider.pack(side="left", padx=(8, 4))
        self._delay_label = ctk.CTkLabel(
            delay_row, text=f"{KEY_DELAY_MS} ms", text_color="#888888", font=ctk.CTkFont(size=11), width=48
        )
        self._delay_label.pack(side="left")
        self._key_delay_ms = KEY_DELAY_MS

        ctk.CTkLabel(top, text="F6 = scan (din joc)", text_color="#555555", font=ctk.CTkFont(size=11)).pack(pady=(6, 0))

        self._log_panel = log.LogPanel(self, height=160)

    def _on_delay_change(self, value: float):
        self._key_delay_ms = int(value)
        self._delay_label.configure(text=f"{self._key_delay_ms} ms")
        log.debug(f"Delay taste: {self._key_delay_ms}ms")

    def _ready_text(self) -> str:
        cap = "DXGI" if _dxcam else "MSS"
        return f"● Gata · {_ocr_label()} · {cap}"

    def _set_status(self, text: str, color: str = "#888888"):
        self.after(0, lambda: self._status.configure(text=text, text_color=color))

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
                self._set_status("● Zonă actualizată", "#6b6")
            else:
                self._set_status("● Zonă neschimbată", "#888888")
            self.deiconify()
            self.lift()
            self.focus_force()
            if was_auto:
                self._auto_var.set(True)
                self._on_auto_toggle()

        pick_region(self, on_done)

    def _hotkey_trigger(self):
        game_input.capture_target_window()
        log.debug("F6 apăsat")
        if not self._running:
            self.after(0, self._run_once)

    def _toggle(self):
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

    def _run_once(self):
        if not SELECTED_REGION:
            log.warn("Scan blocat: nicio zonă selectată")
            self._set_status("● Selectează o zonă", "#c88")
            return
        if self._running:
            log.debug("Scan ignorat: deja în curs")
            return
        threading.Thread(target=self._execute, daemon=True).start()

    def _execute(self):
        with self._lock:
            if self._running:
                return
            self._running = True

        self._set_status("● Scanez...", "#aaaaaa")
        def _prepare():
            evt = threading.Event()

            def _minimize():
                self.iconify()
                evt.set()

            self.after(0, _minimize)
            evt.wait(timeout=0.5)

        try:
            ok, msg = scan_and_press(
                status_cb=lambda m: self._set_status(f"● {m}", "#6b6"),
                key_delay_ms=self._key_delay_ms,
                prepare_cb=_prepare,
            )
            self.after(0, self.deiconify)
            self._set_status(f"{'● OK:' if ok else '●'} {msg}", "#6b6" if ok else "#c88")
        except Exception as exc:
            log.error(f"Eroare scan: {exc}", exc)
            self._set_status(f"● Eroare: {exc}", "#c66")
        finally:
            self._running = False

    def _auto_loop(self):
        while self._auto:
            if not self._running:
                self._execute()
            time.sleep(SCAN_INTERVAL_SEC)

    def _on_close(self):
        self._auto = False
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

    boot = tk.Tk()
    boot.withdraw()

    def on_startup(region: dict | None):
        global SELECTED_REGION
        SELECTED_REGION = region
        boot.quit()

    pick_region(boot, on_startup)
    boot.mainloop()
    boot.destroy()

    if not SELECTED_REGION:
        log.warn("Pornire anulată — nicio zonă selectată")
        raise SystemExit

    _save_pid()
    try:
        App().mainloop()
    finally:
        _clear_pid()
