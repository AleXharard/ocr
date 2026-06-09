"""OCR rapid: batch rec-only + split + refine O0Q/1I + aliniere."""

from __future__ import annotations

import unicodedata

import cv2
import numpy as np

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
ALLOWED = frozenset(CHARS)
CONFUSABLE = frozenset("0OQG6C5S2Z8B1I")
_TMPL_MIN = 0.52
_O0Q_SINGLE_MIN = 0.48
_MAX_GAP_FILL = 4
_templates: dict[str, np.ndarray] | None = None
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


def normalize_char(ch: str) -> str:
    if not ch:
        return ""
    ch = unicodedata.normalize("NFKC", ch).upper()
    return ch if ch in ALLOWED else ""


def _build_templates() -> dict[str, np.ndarray]:
    templates: dict[str, np.ndarray] = {}
    size = 64
    for ch in CHARS:
        img = np.full((size, size), 255, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.5 if ch.isdigit() else 1.3
        thickness = 2
        (tw, th), _ = cv2.getTextSize(ch, font, scale, thickness)
        cv2.putText(
            img, ch, ((size - tw) // 2, (size + th) // 2), font, scale, 0, thickness, cv2.LINE_AA
        )
        _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
        templates[ch] = binary
    return templates


def _get_templates() -> dict[str, np.ndarray]:
    global _templates
    if _templates is None:
        _templates = _build_templates()
    return _templates


def enhance_gray(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
    gray = _clahe.apply(gray)
    scale = max(2.0, 88 / max(gray.shape[:2]))
    if scale > 1.01:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return gray


def ocr_canvas(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    canvas = np.full((96, 160), 255, dtype=np.uint8)
    scale = min(72 / max(h, 1), 130 / max(w, 1))
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_CUBIC)
    y0, x0 = (96 - nh) // 2, (160 - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def prep_binary_from_gray(gray: np.ndarray) -> np.ndarray:
    pad = max(6, int(min(gray.shape) * 0.15))
    padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    _, binary = cv2.threshold(padded, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def match_template(binary: np.ndarray) -> tuple[str, float]:
    if binary.size == 0 or binary.shape[0] < 8:
        return "", 0.0
    resized = cv2.resize(binary, (64, 64), interpolation=cv2.INTER_AREA)
    best_char, best_score = "", -1.0
    for ch, tmpl in _get_templates().items():
        score = float(cv2.matchTemplate(resized, tmpl, cv2.TM_CCOEFF_NORMED).max())
        if score > best_score:
            best_score, best_char = score, ch
    return (best_char, best_score) if best_score > 0.38 else ("", best_score)


def read_char_template(crop: np.ndarray, gray: np.ndarray) -> tuple[str, float]:
    ch, sc = match_template(prep_binary_from_gray(gray))
    if ch:
        return ch, sc
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    adap = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4
    )
    return match_template(adap)


def shape_features(binary: np.ndarray) -> dict:
    h, w = binary.shape
    third = max(1, h // 3)
    return {
        "holes": sum(
            1
            for c in cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0]
            if cv2.contourArea(c) > 12
        ),
        "br": float(np.sum(binary[int(h * 0.55) :, int(w * 0.45) :] > 0) / max(1, binary[int(h * 0.55) :, int(w * 0.45) :].size)),
        "tl": float(np.sum(binary[: int(h * 0.45), : int(w * 0.45)] > 0) / max(1, binary[: int(h * 0.45), : int(w * 0.45)].size)),
        "top": float(np.sum(binary[:third, :] > 0) / max(1, binary[:third, :].size)),
        "mid": float(np.sum(binary[third : 2 * third, :] > 0) / max(1, binary[third : 2 * third, :].size)),
        "bot": float(np.sum(binary[2 * third :, :] > 0) / max(1, binary[2 * third :, :].size)),
        "right": float(np.sum(binary[:, int(w * 0.65) :] > 0) / max(1, binary[:, int(w * 0.65) :].size)),
        "aspect": w / max(h, 1),
    }


def shape_disambiguate(
    crop: np.ndarray, ch: str, score: float, gray: np.ndarray | None = None
) -> tuple[str, float]:
    ch = normalize_char(ch)
    if not ch or ch not in "O0":
        return ch, score

    binary = prep_binary_from_gray(gray if gray is not None else enhance_gray(crop))
    f = shape_features(binary)
    holes, br = f["holes"], f["br"]
    candidates: list[tuple[str, float]] = [(ch, score)]

    def add(alt: str, boost: float):
        if alt != ch:
            candidates.append((alt, score + boost))

    if ch == "0":
        if holes == 1:
            return "0", score
        if holes >= 2 and br < 0.028:
            return "O", score
        return "0", score
    if ch == "O":
        if holes == 1 and br < 0.025:
            return "0", score
        if holes == 1:
            add("0", 0.18)
        elif holes >= 2 and br < 0.04:
            add("O", 0.15)

    candidates.sort(key=lambda x: x[1], reverse=True)
    best = candidates[0]
    return best if best[1] > score + 0.06 else (ch, score)


def _parse_ocr_out(out) -> tuple[str, float]:
    if out is None:
        return "", 0.0
    txts = getattr(out, "txts", None)
    scores = getattr(out, "scores", None)
    if txts:
        for t, sc in zip(txts, scores or []):
            for c in t:
                n = normalize_char(c)
                if n:
                    try:
                        return n, float(sc)
                    except (TypeError, ValueError):
                        return n, 0.7
        return "", 0.0
    if isinstance(out, tuple) and out[0]:
        result = out[0]
        result.sort(key=lambda item: item[0][0][0])
        for _, text, sc in result:
            for c in text:
                n = normalize_char(c)
                if n:
                    try:
                        return n, float(sc)
                    except (TypeError, ValueError):
                        return n, 0.7
    return "", 0.0


def _run_rec(ocr, img: np.ndarray):
    try:
        return ocr(img, use_det=False, use_cls=False)
    except TypeError:
        return ocr(img)


def _ocr_one(ocr, gray: np.ndarray) -> tuple[str, float]:
    return _parse_ocr_out(_run_rec(ocr, ocr_canvas(gray)))


def _parse_batch_out(out) -> list[tuple[str, float]]:
    chars: list[tuple[str, float]] = []
    txts = getattr(out, "txts", None)
    scores = getattr(out, "scores", None)
    if txts:
        for t, sc in zip(txts, scores or []):
            for c in t:
                n = normalize_char(c)
                if n:
                    try:
                        chars.append((n, float(sc)))
                    except (TypeError, ValueError):
                        chars.append((n, 0.7))
        return chars
    if isinstance(out, tuple) and out[0]:
        result = out[0]
        result.sort(key=lambda item: item[0][0][0])
        for _, text, sc in result:
            for c in text:
                n = normalize_char(c)
                if n:
                    try:
                        chars.append((n, float(sc)))
                    except (TypeError, ValueError):
                        chars.append((n, 0.7))
    return chars


def read_chars_batch(grays: list[np.ndarray], ocr) -> list[tuple[str, float]]:
    if ocr is None or not grays:
        return []
    target_h = 72
    parts: list[np.ndarray] = []
    for gray in grays:
        scale = target_h / max(gray.shape[0], 1)
        parts.append(cv2.resize(gray, (max(1, int(gray.shape[1] * scale)), target_h)))
        parts.append(np.full((target_h, 4), 255, dtype=np.uint8))
    strip = cv2.cvtColor(np.hstack(parts), cv2.COLOR_GRAY2BGR)
    return _parse_batch_out(_run_rec(ocr, strip))


def _contrast(gray: np.ndarray) -> np.ndarray:
    return cv2.convertScaleAbs(gray, alpha=1.35, beta=-18)


def _read_singles(grays: list[np.ndarray], ocr) -> list[tuple[str, float]]:
    return [_ocr_one(ocr, _contrast(g)) for g in grays]


def _best_batch(grays: list[np.ndarray], ocr) -> tuple[list[tuple[str, float]], str]:
    n = len(grays)
    full = read_chars_batch(grays, ocr)
    if len(full) == n:
        return full, "full"
    mid = max(1, n // 2)
    split = read_chars_batch(grays[:mid], ocr) + read_chars_batch(grays[mid:], ocr)
    if len(split) == n:
        return split, "split"
    if len(split) > len(full):
        return split, "split+"
    return full, "full+"


def _find_batch_offset(
    batch: list[tuple[str, float]], singles: list[tuple[str, float]], n: int
) -> int:
    m = len(batch)
    if m >= n or m == 0:
        return 0
    best_off, best_score = 0, -999
    for off in range(n - m + 1):
        score = 0
        if off == 0 and batch and singles and batch[0][0] == singles[0][0]:
            score += 5
        for j, (bch, _) in enumerate(batch):
            i = j + off
            sch, ssc = singles[i]
            if bch and sch == bch:
                score += 3
            elif sch and ssc >= 0.75 and bch and sch != bch:
                score -= 2
        for i in range(off):
            if singles[i][0] and singles[i][1] >= 0.55:
                score += 2
        for i in range(off + m, n):
            if singles[i][0] and singles[i][1] >= 0.55:
                score += 2
        if score > best_score:
            best_score, best_off = score, off
    return best_off


def _batch_at(batch: list[tuple[str, float]], offset: int, i: int) -> tuple[str, float]:
    j = i - offset
    if 0 <= j < len(batch):
        return batch[j]
    return "", 0.0


def _finalize_char(crop, gray, ch, sc) -> str:
    if ch in "O0":
        ch, _ = shape_disambiguate(crop, ch, sc, gray)
    return ch or "?"


def _pick_fast(crop, gray, ocr, bch: str, bsc: float) -> str:
    if bch in "O0":
        sch, ssc = _ocr_one(ocr, _contrast(gray))
        if sch in "O0" and ssc >= _O0Q_SINGLE_MIN:
            return _finalize_char(crop, gray, sch, ssc)
        if bch == "0" and sch == "O" and ssc >= 0.45:
            return _finalize_char(crop, gray, "O", ssc)
        return _finalize_char(crop, gray, bch, bsc)
    if bch in "1I":
        tch, tsc = read_char_template(crop, gray)
        if tch in "1I" and tsc > 0.65:
            return tch
    return bch or "?"


def _pick_with_single(
    crop, gray, ocr, bch: str, bsc: float, single: tuple[str, float]
) -> str:
    sch, ssc = single
    if sch and ssc >= 0.81 and sch != bch and bch not in "O0":
        return _pick_fast(crop, gray, ocr, sch, ssc)
    return _pick_fast(crop, gray, ocr, bch, bsc)


def _sandwich_dup_idx(
    batch: list[tuple[str, float]], singles: list[tuple[str, float]]
) -> int:
    chars = [c for c, _ in batch]
    for i in range(len(chars) - 2):
        if chars[i] != chars[i + 2] or chars[i] == chars[i + 1]:
            continue
        _, ssc = singles[i + 2]
        if ssc < 0.65:
            return i + 2
    return -1


def _remove_rebatch(
    crops: list[np.ndarray], grays: list[np.ndarray], ocr, remove_i: int
) -> tuple[list[np.ndarray], list[np.ndarray], list[tuple[str, float]]]:
    nc = crops[:remove_i] + crops[remove_i + 1 :]
    ng = grays[:remove_i] + grays[remove_i + 1 :]
    return nc, ng, read_chars_batch(ng, ocr)


def _might_have_sandwich(batch: list[tuple[str, float]]) -> bool:
    chars = [c for c, _ in batch]
    for i in range(len(chars) - 2):
        if chars[i] == chars[i + 2] and chars[i] != chars[i + 1]:
            return True
    return False


def _try_fix_extra_box(
    crops: list[np.ndarray], grays: list[np.ndarray], ocr, batch: list[tuple[str, float]]
) -> tuple[list[np.ndarray], list[np.ndarray], list[tuple[str, float]], bool]:
    n = len(crops)
    if len(batch) != n or n < 16 or not _might_have_sandwich(batch):
        return crops, grays, batch, False

    singles = _read_singles(grays, ocr)
    dup = _sandwich_dup_idx(batch, singles)
    if dup >= 0:
        nc, ng, nb = _remove_rebatch(crops, grays, ocr, dup)
        if len(nb) == n - 1:
            return nc, ng, nb, True
    return crops, grays, batch, False


def _pick_gap(
    crop, gray, ocr, batch_ch: str, single: tuple[str, float], tmpl: tuple[str, float]
) -> str:
    sch, ssc = single
    tch, tsc = tmpl

    if batch_ch:
        return _pick_fast(crop, gray, ocr, batch_ch, 0.65)
    if sch and ssc >= 0.72:
        return _pick_fast(crop, gray, ocr, sch, ssc)
    if sch and ssc >= 0.55:
        return _pick_fast(crop, gray, ocr, sch, ssc)
    if tch and tsc >= _TMPL_MIN:
        return tch
    return "?"


def read_sequence_chars(crops: list[np.ndarray], ocr) -> tuple[list[str], str]:
    if not crops:
        return [], "none"

    n = len(crops)
    grays = [enhance_gray(c) for c in crops]
    batch, mode = _best_batch(grays, ocr)
    crops, grays, batch, fixed = _try_fix_extra_box(crops, grays, ocr, batch)
    n = len(crops)
    if fixed:
        mode = f"{mode}-fix"

    if len(batch) == n:
        chars = [_pick_fast(crops[i], grays[i], ocr, bch, bsc) for i, (bch, bsc) in enumerate(batch)]
        return chars, mode

    singles = _read_singles(grays, ocr)
    templates = [read_char_template(c, g) for c, g in zip(crops, grays)]
    offset = _find_batch_offset(batch, singles, n)

    chars: list[str] = []
    for i in range(n):
        bch, _ = _batch_at(batch, offset, i)
        if bch and i >= offset and i < offset + len(batch):
            chars.append(_pick_with_single(crops[i], grays[i], ocr, bch, 0.65, singles[i]))
        else:
            chars.append(_pick_gap(crops[i], grays[i], ocr, bch, singles[i], templates[i]))

    filled = 0
    for i in range(n):
        if chars[i] != "?":
            continue
        if filled >= _MAX_GAP_FILL:
            break
        ch, sc = _ocr_one(ocr, _contrast(grays[i]))
        if ch:
            chars[i] = _pick_fast(crops[i], grays[i], ocr, ch, sc)
            filled += 1
        else:
            tch, tsc = templates[i]
            if tch and tsc >= _TMPL_MIN:
                chars[i] = tch
                filled += 1

    if offset:
        mode = f"{mode}@{offset}"
    if filled:
        mode = f"{mode}+{filled}"
    return chars, mode


def warmup_ocr(ocr) -> None:
    if ocr is None:
        return
    grays = [np.full((40, 28), 255, dtype=np.uint8) for _ in range(10)]
    for i, g in enumerate(grays):
        cv2.putText(g, chr(65 + i), (2, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    read_chars_batch(grays, ocr)


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    return frame
