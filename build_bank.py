"""Construiește banca de glife (glyph_bank.npz) din POZE/.

Numele fișierului (fără extensie) = secvența reală de caractere din imagine.
Pentru fiecare imagine cu segmentare corectă (nr. casete == nr. caractere),
fiecare casetă a i-a (sortată stânga→dreapta) e eticheta caracterului i.

Rulează:  python build_bank.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import main as app
import vision as vis

POZE = Path(__file__).resolve().parent / "POZE"


def crop_key_band(frame):
    h, w = frame.shape[:2]
    return frame[int(h * 0.36) : int(h * 0.54), int(w * 0.08) : int(w * 0.92)]


def main() -> int:
    templates: list[np.ndarray] = []
    labels: list[str] = []
    used = skipped = 0

    for path in sorted(POZE.glob("*.png")):
        stem = path.stem.upper()
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"SKIP {path.name}: nu pot citi imaginea")
            skipped += 1
            continue
        band = crop_key_band(frame)
        boxes = app.detect_white_boxes(band)
        crops = app._extract_crops(band, boxes)
        if len(crops) != len(stem):
            print(f"SKIP {path.name}: {len(crops)} casete != {len(stem)} caractere")
            skipped += 1
            continue
        for ch, crop in zip(stem, crops):
            templates.append(vis.normalize_glyph(vis.enhance_gray(crop)))
            labels.append(ch)
        used += 1

    if not templates:
        print("Nicio glifă etichetată — banca nu a fost creată.")
        return 1

    tmpl_arr = np.stack(templates).astype(np.uint8)
    lbl_arr = np.array(labels, dtype="U1")
    np.savez_compressed(vis.BANK_PATH, templates=tmpl_arr, labels=lbl_arr)

    distinct = sorted(set(labels))
    missing = [c for c in vis.CHARS if c not in distinct]
    print(f"Images used: {used}  skipped: {skipped}")
    print(f"Bank: {len(templates)} templates, {len(distinct)}/36 chars -> {vis.BANK_PATH.name}")
    if missing:
        print(f"WARNING - characters without exemplar: {missing}")
    else:
        print("Full coverage: all 36 characters have exemplars.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
