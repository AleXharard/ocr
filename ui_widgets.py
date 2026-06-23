"""Widget-uri UI — design minimalist, fundal negru, text alb, bule alungite."""

from __future__ import annotations

import customtkinter as ctk

# ── Temă ─────────────────────────────────────────────────────────────────────
UI_BG = "#0a0a0a"
UI_SURFACE = "#0f1418"
UI_PILL = "#0981ad"           # albastru spălăcit — butoane / bule
UI_PILL_HOVER = "#0b6d8f"      # hover puțin mai închis
UI_PILL_ACTIVE = "#0a5f7a"
UI_ACCENT_LIGHT = "#3a9fbf"   # slider, accente subtile
UI_INNER = "#0c1a22"          # fundal interior (slider, entry)
UI_TEXT = "#ffffff"
UI_TEXT_DIM = "#c8d8e0"
UI_DOT_ON = "#ffffff"
UI_DOT_OFF = "#5a8a9a"
UI_DIVIDER = "#1a5a72"
PILL_H = 42
PILL_R = 21


def pill_font(size: int = 13, bold: bool = False) -> ctk.CTkFont:
    return ctk.CTkFont(size=size, weight="bold" if bold else "normal")


class ModePicker(ctk.CTkFrame):
    """Alegere mod: 2 opțiuni cu bulină (Chei / Busteni)."""

    def __init__(self, master, on_change, initial: str = "Chei", **kw):
        super().__init__(master, fg_color=UI_PILL, corner_radius=PILL_R, height=PILL_H, **kw)
        self.pack_propagate(False)
        self._on_change = on_change
        self._value = initial
        self._opts = [
            ("Chei", "⌨"),
            ("Busteni", "♪"),
        ]
        self._cells: dict[str, ctk.CTkButton] = {}
        inner = ctk.CTkFrame(self, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=4, pady=4)
        for i, (name, icon) in enumerate(self._opts):
            if i:
                ctk.CTkFrame(inner, width=1, fg_color=UI_DIVIDER).pack(
                    side="left", fill="y", padx=2, pady=6
                )
            btn = ctk.CTkButton(
                inner,
                text="",
                fg_color="transparent",
                hover_color=UI_PILL_HOVER,
                corner_radius=PILL_R - 4,
                height=PILL_H - 8,
                command=lambda n=name: self.set(n),
            )
            btn.pack(side="left", expand=True, fill="both")
            self._cells[name] = btn
        self.set(initial, notify=False)

    def _label(self, name: str, icon: str, selected: bool) -> str:
        dot = "●" if selected else "○"
        return f"  {dot}  {icon}  {name}  "

    def set(self, value: str, notify: bool = True) -> None:
        if value not in self._cells:
            return
        self._value = value
        icons = dict(self._opts)
        for name, btn in self._cells.items():
            sel = name == value
            btn.configure(
                text=self._label(name, icons[name], sel),
                text_color=UI_TEXT if sel else UI_TEXT_DIM,
                font=pill_font(13, bold=sel),
                fg_color=UI_PILL_HOVER if sel else "transparent",
            )
        if notify:
            self._on_change(value)

    def get(self) -> str:
        return self._value


class DelayPill(ctk.CTkFrame):
    """Slider ms + casetă editabilă, în bulă alungită."""

    def __init__(self, master, initial_ms: int, on_change, **kw):
        super().__init__(master, fg_color=UI_PILL, corner_radius=PILL_R, height=PILL_H, **kw)
        self.pack_propagate(False)
        self._on_change = on_change
        self._ms = initial_ms

        inner = ctk.CTkFrame(self, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=14, pady=8)

        ctk.CTkLabel(
            inner, text="ms", text_color=UI_TEXT_DIM, font=pill_font(12)
        ).pack(side="left", padx=(0, 8))

        self._slider = ctk.CTkSlider(
            inner, from_=5, to=150, number_of_steps=145,
            progress_color=UI_ACCENT_LIGHT,
            button_color="#ffffff",
            button_hover_color="#e0e8ec",
            fg_color=UI_INNER,
            command=self._slider_changed,
        )
        self._slider.set(initial_ms)
        self._slider.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self._entry = ctk.CTkEntry(
            inner, width=52, height=28, justify="center",
            font=pill_font(13, bold=True),
            fg_color=UI_INNER, border_color=UI_DIVIDER, text_color=UI_TEXT,
        )
        self._entry.insert(0, str(initial_ms))
        self._entry.pack(side="right")
        self._entry.bind("<Return>", self._entry_commit)
        self._entry.bind("<FocusOut>", self._entry_commit)

    def _emit(self, ms: int) -> None:
        self._ms = ms
        self._on_change(ms)

    def _slider_changed(self, value: float) -> None:
        ms = int(round(value))
        self._entry.delete(0, "end")
        self._entry.insert(0, str(ms))
        self._emit(ms)

    def _entry_commit(self, _event=None) -> None:
        raw = self._entry.get().strip().replace("ms", "")
        try:
            ms = max(5, min(500, int(raw)))
        except ValueError:
            ms = self._ms
        self._slider.set(min(150, ms))
        self._entry.delete(0, "end")
        self._entry.insert(0, str(ms))
        self._emit(ms)

    def get_ms(self) -> int:
        return self._ms


class HotkeyPill(ctk.CTkButton):
    """Buton hotkey — fundal gri, formă alungită; click pentru reînregistrare."""

    def __init__(self, master, hotkey: str, on_change, **kw):
        super().__init__(
            master,
            text=self._display(hotkey),
            height=PILL_H,
            corner_radius=PILL_R,
            fg_color=UI_PILL,
            hover_color=UI_PILL_HOVER,
            text_color=UI_TEXT,
            font=pill_font(13, bold=True),
            command=self._start_capture,
            **kw,
        )
        self._hotkey = hotkey.lower()
        self._on_change = on_change
        self._capturing = False

    @staticmethod
    def _display(key: str) -> str:
        k = key.lower()
        return f"Tastă start:  {k.upper()}"

    def _start_capture(self) -> None:
        if self._capturing:
            return
        self._capturing = True
        self.configure(text="Apasă o tastă…")
        top = self.winfo_toplevel()
        top.bind("<KeyPress>", self._on_key, add="+")
        top.focus_force()

    def _on_key(self, event) -> None:
        if not self._capturing:
            return
        sym = event.keysym
        skip = {
            "Shift_L", "Shift_R", "Control_L", "Control_R",
            "Alt_L", "Alt_R", "Meta_L", "Meta_R", "Caps_Lock",
        }
        if sym in skip:
            return
        self._capturing = False
        self.winfo_toplevel().unbind("<KeyPress>")
        key = sym.lower()
        if len(key) == 1:
            key = key
        elif key.startswith("f") and key[1:].isdigit():
            key = key  # f6
        else:
            key = key.replace("_", "").lower()
        self._hotkey = key
        self.configure(text=self._display(key))
        self._on_change(key)

    def get_hotkey(self) -> str:
        return self._hotkey

    def set_hotkey(self, key: str) -> None:
        self._hotkey = key.lower()
        self.configure(text=self._display(key))
