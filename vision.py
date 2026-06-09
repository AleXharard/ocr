"""Recunoaștere caractere prin potrivire de șabloane pe glife reale.

Fontul jocului e fix, iar fiecare casetă conține exact un caracter A-Z0-9.
Potrivirea pixel-cu-pixel față de glife reale (o bancă de șabloane construită
din capturi reale) e mai exactă ȘI mai rapidă decât un OCR neural pe glife
izolate. Validat leave-one-image-out: 100% (223/223 glife, 0 confuzii).

RapidOCR rămâne doar ca plasă de siguranță: dacă multe glife au scor mic
(alt font / altă rezoluție), recurgem la recunoașterea pe bandă a RapidOCR.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import cv2
import numpy as np

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
ALLOWED = frozenset(CHARS)

# ── Parametri potrivire șabloane ─────────────────────────────────────────────
GLYPH_SIZE = 48           # latura imaginii normalizate a unei glife
_MIN_SCORE = 0.45         # sub acest scor, o glifă e considerată "nesigură"
_FALLBACK_FRAC = 0.30     # dacă >30% din glife sunt nesigure → încearcă RapidOCR
BANK_PATH = Path(__file__).resolve().parent / "glyph_bank.npz"

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
_bank: "GlyphBank | None" = None
_bank_loaded = False


def normalize_char(ch: str) -> str:
    if not ch:
        return ""
    ch = unicodedata.normalize("NFKC", ch).upper()
    return ch if ch in ALLOWED else ""


def enhance_gray(crop: np.ndarray) -> np.ndarray:
    """Gri + CLAHE + upscale, pentru contrast uniform indiferent de captură."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
    gray = _clahe.apply(gray)
    scale = max(2.0, 88 / max(gray.shape[:2]))
    if scale > 1.01:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return gray


def normalize_glyph(gray: np.ndarray) -> np.ndarray:
    """Binarizare Otsu inversată → decupare la cerneală → pătrat → GLYPH_SIZE."""
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.where(b > 0)
    if len(xs):
        b = b[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    h, w = b.shape
    if h == 0 or w == 0:
        return np.zeros((GLYPH_SIZE, GLYPH_SIZE), np.uint8)
    scale = (GLYPH_SIZE - 4) / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(b, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((GLYPH_SIZE, GLYPH_SIZE), np.uint8)
    y0, x0 = (GLYPH_SIZE - nh) // 2, (GLYPH_SIZE - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out


def glyph_to_vec(glyph: np.ndarray) -> np.ndarray:
    """Vector cu media scăzută și normat L2.

    Produsul scalar a doi astfel de vectori = coeficientul de corelație
    normalizat (identic cu cv2.TM_CCOEFF_NORMED pentru imagini de aceeași
    dimensiune), dar calculabil pentru toată banca într-o singură înmulțire.
    """
    v = glyph.astype(np.float32).ravel()
    v -= v.mean()
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 1e-6 else v


class GlyphBank:
    """Bancă de șabloane reale; clasifică o glifă prin cel mai apropiat exemplar."""

    def __init__(self, vectors: np.ndarray, labels: np.ndarray):
        self.vectors = vectors  # (N, d) float32, fiecare cu media scăzută + normat L2
        self.labels = labels    # (N,) caractere

    @classmethod
    def load(cls, path: Path = BANK_PATH) -> "GlyphBank | None":
        if not Path(path).exists():
            return None
        data = np.load(path, allow_pickle=False)
        templates = data["templates"]            # (N, S, S) uint8
        labels = data["labels"].astype("U1")
        vectors = np.stack([glyph_to_vec(t) for t in templates]).astype(np.float32)
        return cls(vectors, labels)

    def classify(self, gray: np.ndarray) -> tuple[str, float]:
        q = glyph_to_vec(normalize_glyph(gray))
        scores = self.vectors @ q
        i = int(scores.argmax())
        return str(self.labels[i]), float(scores[i])


def _get_bank() -> "GlyphBank | None":
    global _bank, _bank_loaded
    if not _bank_loaded:
        _bank = GlyphBank.load()
        _bank_loaded = True
    return _bank


# ── Plasă de siguranță: RapidOCR pe bandă ────────────────────────────────────
def _iter_text_score(out):
    """Extrage perechi (text, scor) din orice format RapidOCR, în ordine.

    Acoperă: obiect v3 (.txts/.scores), onnxruntime det+rec ([box,text,score])
    și onnxruntime rec-only ([text,score], cazul use_det=False).
    """
    if out is None:
        return
    txts = getattr(out, "txts", None)
    if txts:
        scores = getattr(out, "scores", None) or []
        for t, sc in zip(txts, scores):
            yield t, sc
        return
    result = out[0] if isinstance(out, tuple) else out
    if not result:
        return
    items = list(result)
    if len(items[0]) == 3:  # det+rec: sortează stânga→dreapta după x-ul casetei
        items = sorted(items, key=lambda it: it[0][0][0])
        for _, text, sc in items:
            yield text, sc
    else:  # rec-only: deja în ordinea de intrare
        for text, sc in items:
            yield text, sc


def _run_rec(ocr, img: np.ndarray):
    if ocr is None:
        return None
    try:
        return ocr(img, use_det=False, use_cls=False)
    except TypeError:
        return ocr(img)


def read_chars_batch(grays: list[np.ndarray], ocr) -> list[tuple[str, float]]:
    """Concatenează glifele într-o bandă și le citește dintr-un singur apel OCR."""
    if ocr is None or not grays:
        return []
    target_h = 72
    parts: list[np.ndarray] = []
    for gray in grays:
        scale = target_h / max(gray.shape[0], 1)
        parts.append(cv2.resize(gray, (max(1, int(gray.shape[1] * scale)), target_h)))
        parts.append(np.full((target_h, 4), 255, dtype=np.uint8))
    strip = cv2.cvtColor(np.hstack(parts), cv2.COLOR_GRAY2BGR)
    chars: list[tuple[str, float]] = []
    for text, sc in _iter_text_score(_run_rec(ocr, strip)):
        for c in text:
            n = normalize_char(c)
            if n:
                try:
                    chars.append((n, float(sc)))
                except (TypeError, ValueError):
                    chars.append((n, 0.7))
    return chars


# ── API public ───────────────────────────────────────────────────────────────
def read_sequence_chars(crops: list[np.ndarray], ocr) -> tuple[list[str], str]:
    """Returnează (caractere, mod). Banca de șabloane e principală; RapidOCR e plasa."""
    if not crops:
        return [], "none"

    grays = [enhance_gray(c) for c in crops]
    bank = _get_bank()
    n = len(crops)

    if bank is None:
        batch = read_chars_batch(grays, ocr)
        if len(batch) == n:
            return [c for c, _ in batch], "ocr-only"
        return ["?"] * n, "no-bank"

    results = [bank.classify(g) for g in grays]
    chars = [ch for ch, _ in results]
    weak = sum(1 for _, sc in results if sc < _MIN_SCORE)

    # Multe glife nesigure ⇒ banca probabil nu se potrivește acestui font/rezoluții.
    if ocr is not None and weak > _FALLBACK_FRAC * n:
        batch = read_chars_batch(grays, ocr)
        if len(batch) == n:
            return [c for c, _ in batch], "ocr-fallback"

    return chars, "tmpl" if weak == 0 else f"tmpl-weak{weak}"


def warmup_ocr(ocr) -> None:
    """Pre-încălzește RapidOCR (plasa de siguranță) ca să nu plătim latența la prima rulare."""
    if ocr is None:
        return
    grays = [np.full((40, 28), 255, dtype=np.uint8) for _ in range(10)]
    for i, g in enumerate(grays):
        cv2.putText(g, chr(65 + i), (2, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    read_chars_batch(grays, ocr)


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    return frame
