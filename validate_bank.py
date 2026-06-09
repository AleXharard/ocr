"""Validare leave-one-image-out a băncii de glife (anti-overfit).

Demonstrează că recunoașterea generalizează: pentru fiecare imagine, banca e
construită DOAR din celelalte imagini (imaginea testată e exclusă complet), deci
nicio glifă nu se potrivește vreodată cu propriul exemplar. Folosește exact
funcțiile de producție (vis.normalize_glyph / vis.glyph_to_vec).

Rulează:  python validate_bank.py
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

import main as app
import vision as vis

POZE = Path(__file__).resolve().parent / "POZE"


def crop_key_band(frame):
    h, w = frame.shape[:2]
    return frame[int(h * 0.36) : int(h * 0.54), int(w * 0.08) : int(w * 0.92)]


def load_labeled():
    """Returnează [(stem, [vector_glifă, ...]), ...] pentru imaginile bine segmentate."""
    data = []
    for path in sorted(POZE.glob("*.png")):
        stem = path.stem.upper()
        band = crop_key_band(cv2.imread(str(path)))
        crops = app._extract_crops(band, app.detect_white_boxes(band))
        if len(crops) != len(stem):
            print(f"SKIP {path.name}: {len(crops)} boxes != {len(stem)} chars")
            continue
        vecs = [vis.glyph_to_vec(vis.normalize_glyph(vis.enhance_gray(c))) for c in crops]
        data.append((stem, vecs))
    return data


def main() -> int:
    data = load_labeled()
    total = correct = imgs_ok = 0
    min_correct = 1.0
    confusion = Counter()

    for held, vecs in data:
        # Bancă din TOATE celelalte imagini.
        bank_vecs, bank_lbls = [], []
        for other, ovecs in data:
            if other == held:
                continue
            for i, v in enumerate(ovecs):
                bank_vecs.append(v)
                bank_lbls.append(other[i])
        mat = np.stack(bank_vecs)

        ok_img = True
        for i, q in enumerate(vecs):
            expected = held[i]
            scores = mat @ q
            j = int(scores.argmax())
            pred = bank_lbls[j]
            total += 1
            if pred == expected:
                correct += 1
                min_correct = min(min_correct, float(scores[j]))
            else:
                ok_img = False
                confusion[f"{expected}->{pred}"] += 1
        imgs_ok += ok_img

    n = len(data)
    print(f"\nLeave-one-image-out: {imgs_ok}/{n} images, {correct}/{total} glyphs "
          f"({100.0*correct/max(1,total):.1f}%)")
    print(f"Min score among correct matches: {min_correct:.3f}  (threshold _MIN_SCORE={vis._MIN_SCORE})")
    print("Confusions:", dict(confusion) if confusion else "(none)")
    return 0 if correct == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
