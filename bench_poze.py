"""Benchmark: captură + detectare + OCR + simulare apăsări (<3s țintă)."""

import time
from pathlib import Path

import cv2

import game_input
import main as app

POZE = Path(__file__).parent / "POZE"


def simulate_key_time(n_keys: int) -> float:
    gaps = max(0, n_keys - 1) * app.KEY_DELAY_MS
    holds = n_keys * app.KEY_HOLD_MS
    return (gaps + holds + app.PRE_PRESS_MS) / 1000


def bench_image(path: Path) -> None:
    exp = path.stem
    frame = cv2.imread(str(path))
    if frame is None or frame.size == 0:
        print(f"[SKIP] {path.name} | imagine invalidă")
        return
    h, w = frame.shape[:2]
    frame = frame[int(h * 0.36) : int(h * 0.54), int(w * 0.08) : int(w * 0.92)]

    t0 = time.perf_counter()
    boxes = app.detect_white_boxes(frame)
    t_det = time.perf_counter()
    chars, mode = app.read_sequence(frame, boxes)
    t_ocr = time.perf_counter()
    got = "".join(chars)
    key_sec = simulate_key_time(len(chars))
    total = (t_ocr - t0) + key_sec

    ok = got == exp
    print(
        f"[{'OK' if ok else 'FAIL'}] {path.name} | "
        f"det {(t_det-t0)*1000:.0f}ms ocr {(t_ocr-t_det)*1000:.0f}ms keys ~{key_sec*1000:.0f}ms | "
        f"total ~{total*1000:.0f}ms | {mode}"
    )
    if not ok:
        print(f"       exp: {exp}")
        print(f"       got: {got}")


def main() -> None:
    app._init_capture()
    app._init_ocr()
    game_input.init()
    print(f"Config: delay={app.KEY_DELAY_MS}ms hold={app.KEY_HOLD_MS}ms pre={app.PRE_PRESS_MS}ms")
    print("Warmup...\n")

    paths = sorted(POZE.glob("*.png"))
    if paths:
        bench_image(paths[0])

    for path in paths[1:]:
        bench_image(path)


if __name__ == "__main__":
    main()
