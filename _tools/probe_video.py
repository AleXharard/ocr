"""Scan rapid secvențial al videoclipului busteni (one-off)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import circle_game as cg  # noqa: E402

VIDEO = ROOT / "video" / "2026-06-12 07-36-17.mp4"
REGION = json.loads((ROOT / "busteni_region.json").read_text(encoding="utf-8"))
L, T, W, H = REGION["left"], REGION["top"], REGION["width"], REGION["height"]


def main() -> int:
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        print("Video indisponibil")
        return 1

    rounds: list[dict] = []
    cur: dict | None = None
    i = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        crop = frame[T : T + H, L : L + W]
        ui = cg.detect_active(crop)
        if ui:
            ring = cg.analyze_ring(crop, ui)
            d, c = cg._read_digit_fast(crop, ui)
            aligned = bool(ring and cg.is_aligned(ring))
            if cur is None:
                cur = {
                    "start": i,
                    "end": i,
                    "digit": d,
                    "digit_frame": i if d else -1,
                    "digit_conf": c,
                    "aligned": int(aligned),
                }
            else:
                cur["end"] = i
                if d and c > cur["digit_conf"]:
                    cur["digit"] = d
                    cur["digit_frame"] = i
                    cur["digit_conf"] = c
                cur["aligned"] += int(aligned)
        elif cur is not None:
            if cur["end"] - cur["start"] >= 15:
                rounds.append(cur)
            cur = None
        i += 1

    if cur is not None and cur["end"] - cur["start"] >= 15:
        rounds.append(cur)

    print(f"frames={i} rounds={len(rounds)}")
    for ri, r in enumerate(rounds):
        print(
            f"  R{ri}: {r['start']}-{r['end']} "
            f"digit={r['digit']}@{r['digit_frame']} conf={r['digit_conf']:.3f} "
            f"aligned={r['aligned']}"
        )

    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
