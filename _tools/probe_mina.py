"""Probe template matching for mina — one-off calibration."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

POZE = ROOT / "poze mina"
TEMPLATE = POZE / "piatra.png"

# zone excluse (fracții din ecran)
EXCLUDE = {
    "top_left": (0.0, 0.0, 0.32, 0.28),   # chat + fps
    "bottom_left": (0.0, 0.72, 0.22, 1.0),  # hartă + status
    "right": (0.82, 0.0, 1.0, 1.0),       # radio list
    "top_right": (0.88, 0.0, 1.0, 0.06),  # FiveM label
    "bottom_right": (0.78, 0.92, 1.0, 1.0),
}


def search_mask(h: int, w: int) -> np.ndarray:
    m = np.ones((h, w), np.uint8)
    for x1f, y1f, x2f, y2f in EXCLUDE.values():
        x1, y1 = int(x1f * w), int(y1f * h)
        x2, y2 = int(x2f * w), int(y2f * h)
        m[y1:y2, x1:x2] = 0
    return m


def _mask_result(res: np.ndarray, mask: np.ndarray, tw: int, th: int) -> None:
    """Anulează potrivirile al căror centru cade în zone excluse."""
    rh, rw = res.shape[:2]
    for y in range(rh):
        for x in range(rw):
            cx, cy = x + tw // 2, y + th // 2
            if cx >= mask.shape[1] or cy >= mask.shape[0] or not mask[cy, cx]:
                res[y, x] = -1.0


def match(frame_bgr: np.ndarray, tmpl_bgr: np.ndarray, mask: np.ndarray) -> tuple[float, tuple[int, int], int, int]:
    best_score, best_xy, best_tw, best_th = -1.0, (0, 0), 0, 0
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    tgray = cv2.cvtColor(tmpl_bgr, cv2.COLOR_BGR2GRAY)
    th0, tw0 = tgray.shape[:2]
    for scale in (0.55, 0.65, 0.75, 0.85, 0.95, 1.0, 1.1, 1.2, 1.35):
        tw, th = max(8, int(tw0 * scale)), max(8, int(th0 * scale))
        tmpl = cv2.resize(tgray, (tw, th), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        if tw >= gray.shape[1] or th >= gray.shape[0]:
            continue
        res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
        _mask_result(res, mask, tw, th)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best_score:
            best_score, best_xy = max_val, max_loc
            best_tw, best_th = tw, th
    cx = best_xy[0] + best_tw // 2
    cy = best_xy[1] + best_th // 2
    return best_score, (cx, cy), best_tw, best_th


def main() -> int:
    tmpl = cv2.imread(str(TEMPLATE))
    if tmpl is None:
        print(f"Missing template: {TEMPLATE}")
        return 1
    shots = sorted(p for p in POZE.glob("*.png") if p.name not in ("piatra.png",))
    for path in shots:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"SKIP {path.name}")
            continue
        h, w = frame.shape[:2]
        mask = search_mask(h, w)
        score, (cx, cy), tw, th = match(frame, tmpl, mask)
        has = "STONE" if score >= 0.72 else "none"
        print(f"{path.name:28s} {w}x{h} score={score:.3f} center=({cx},{cy}) size={tw}x{th} -> {has}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
