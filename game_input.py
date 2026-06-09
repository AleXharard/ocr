"""Trimitere taste către FiveM/GTA — focus fereastră, fără mouse."""

from __future__ import annotations

import ctypes
import sys
import time
import unicodedata
from ctypes import wintypes

import logger as log

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

INPUT_KEYBOARD = 1
KEYEVENTF_SCANCODE = 0x0008
KEYEVENTF_KEYUP = 0x0002
VK_SHIFT = 0x10
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101

ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong

_target_hwnd: int | None = None
_hold_ms = 35
_use_pydirectinput = False


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT)]

    _anonymous_ = ("_input",)
    _fields_ = [("type", wintypes.DWORD), ("_input", _INPUT)]


def init() -> None:
    global _use_pydirectinput
    try:
        import pydirectinput

        pydirectinput.PAUSE = 0
        pydirectinput.FAILSAFE = False
        _use_pydirectinput = True
        log.info("Input: pydirectinput (DirectInput)")
    except ImportError:
        _use_pydirectinput = False
        log.warn("Input: SendInput (instalează pydirectinput pentru FiveM)")

    if not is_admin():
        log.warn("Nu rulezi ca Administrator — FiveM poate bloca tastele simulate")


def is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def set_hold_ms(ms: int) -> None:
    global _hold_ms
    _hold_ms = ms


def _window_title(hwnd: int) -> str:
    length = user32.GetWindowTextLengthW(hwnd) + 1
    buf = ctypes.create_unicode_buffer(length)
    user32.GetWindowTextW(hwnd, buf, length)
    return buf.value


def _is_game_title(title: str) -> bool:
    t = title.lower()
    return any(k in t for k in ("fivem", "grand theft auto", "gta5", "cfx.re"))


def find_game_window() -> int | None:
    found: list[int] = []

    def callback(hwnd, _):
        if user32.IsWindowVisible(hwnd) and _is_game_title(_window_title(hwnd)):
            found.append(hwnd)
        return True

    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows(WNDENUMPROC(callback), 0)
    return found[0] if found else None


def get_target_hwnd() -> int | None:
    if _target_hwnd and user32.IsWindow(_target_hwnd):
        return _target_hwnd
    return find_game_window()


def capture_target_window() -> int | None:
    global _target_hwnd
    hwnd = user32.GetForegroundWindow()
    if hwnd and _is_game_title(_window_title(hwnd)):
        _target_hwnd = hwnd
        log.debug(f"Țintă salvată: {_window_title(hwnd)}")
    return _target_hwnd


def _force_foreground(hwnd: int) -> None:
    fg = user32.GetForegroundWindow()
    fg_thread = user32.GetWindowThreadProcessId(fg, None)
    our_thread = kernel32.GetCurrentThreadId()
    attached = False
    if fg_thread and fg_thread != our_thread:
        attached = bool(user32.AttachThreadInput(our_thread, fg_thread, True))
    user32.ShowWindow(hwnd, 9)
    user32.SetForegroundWindow(hwnd)
    user32.BringWindowToTop(hwnd)
    if attached:
        user32.AttachThreadInput(our_thread, fg_thread, False)


def focus_game(minimize_cb=None, fast: bool = False) -> bool:
    if minimize_cb:
        try:
            minimize_cb()
        except Exception as e:
            log.debug(f"Minimize: {e}")

    hwnd = get_target_hwnd()
    if hwnd and user32.GetForegroundWindow() == hwnd:
        log.debug(f"Focus deja activ: {_window_title(hwnd)}")
        return True

    if not fast:
        time.sleep(0.1)

    hwnd = get_target_hwnd()
    if not hwnd:
        log.warn("Nu găsesc fereastra FiveM/GTA")
        return False

    _force_foreground(hwnd)
    time.sleep(0.04 if fast else 0.15)

    if user32.GetForegroundWindow() == hwnd:
        log.info(f"Focus joc: {_window_title(hwnd)}")
        return True

    log.warn(f"Focus eșuat — activ: {_window_title(user32.GetForegroundWindow())}")
    return False


def _char_to_vk(ch: str) -> tuple[int, bool] | None:
    result = user32.VkKeyScanW(ord(ch))
    if result == -1:
        return None
    return result & 0xFF, bool((result >> 8) & 1)


def _sendinput_vk(vk: int, key_up: bool = False) -> bool:
    scan = user32.MapVirtualKeyW(vk, 0)
    flags = KEYEVENTF_KEYUP if key_up else 0
    inp = INPUT(
        type=INPUT_KEYBOARD,
        ki=KEYBDINPUT(wVk=vk, wScan=scan, dwFlags=flags, time=0, dwExtraInfo=0),
    )
    sent = user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))
    return sent == 1


def _post_key(hwnd: int, vk: int, key_up: bool = False) -> None:
    scan = user32.MapVirtualKeyW(vk, 0) & 0xFF
    if key_up:
        lparam = (1 << 31) | (1 << 30) | (scan << 16) | 1
        user32.PostMessageW(hwnd, WM_KEYUP, vk, lparam)
    else:
        lparam = (scan << 16) | 1
        user32.PostMessageW(hwnd, WM_KEYDOWN, vk, lparam)


def _press_directinput(key: str) -> bool:
    import pydirectinput

    pydirectinput.keyDown(key)
    time.sleep(_hold_ms / 1000)
    pydirectinput.keyUp(key)
    return True


def _press_sendinput(vk: int, need_shift: bool) -> bool:
    ok = True
    if need_shift:
        ok = _sendinput_vk(VK_SHIFT, False) and ok
        time.sleep(0.01)
    ok = _sendinput_vk(vk, False) and ok
    time.sleep(_hold_ms / 1000)
    ok = _sendinput_vk(vk, True) and ok
    if need_shift:
        time.sleep(0.01)
        ok = _sendinput_vk(VK_SHIFT, True) and ok
    return ok


def _press_postmessage(hwnd: int, vk: int, need_shift: bool) -> bool:
    if need_shift:
        _post_key(hwnd, VK_SHIFT, False)
        time.sleep(0.01)
    _post_key(hwnd, vk, False)
    time.sleep(_hold_ms / 1000)
    _post_key(hwnd, vk, True)
    if need_shift:
        time.sleep(0.01)
        _post_key(hwnd, VK_SHIFT, True)
    return True


def press_key(ch: str) -> bool:
    ch = unicodedata.normalize("NFKC", ch).upper()
    if not ch.isalnum():
        log.warn(f"Tastă invalidă: {ch}")
        return False

    mapped = _char_to_vk(ch.lower())
    if mapped is None:
        log.warn(f"Nu pot mapa tasta: {ch}")
        return False

    vk, need_shift = mapped
    key = ch.lower()
    hwnd = get_target_hwnd()

    # 1) pydirectinput — cel mai fiabil în GTA/FiveM
    if _use_pydirectinput:
        try:
            _press_directinput(key)
            return True
        except Exception as e:
            log.debug(f"pydirectinput eșuat: {e}")

    # 2) SendInput cu VK
    if _press_sendinput(vk, need_shift):
        return True

    log.warn(f"SendInput blocat pentru '{ch}' — rulează ca Administrator")

    # 3) PostMessage direct către fereastra jocului
    if hwnd:
        _press_postmessage(hwnd, vk, need_shift)
        log.debug(f"PostMessage fallback: {ch}")
        return True

    return False
