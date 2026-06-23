"""Mina minigame — detectează piatra pe ecran și dă click.

Detecție: cerc gri (#8e9d96 predominant în disc) + template matching ca fallback.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import logger as log
from app_paths import app_dir, resource_path

TEMPLATE_PATH = resource_path("poze mina", "piatra.png")
MINA_DEBUG_DIR = app_dir() / "debug_mina"
MINA_DEBUG_MAX = 250

# #8e9d96 — culoarea predominantă în interiorul cercului pietrei (BGR).
STONE_PRIMARY_BGR = np.array([150, 157, 142], dtype=np.float32)

STONE_PALETTE_BGR = np.array(
    [
        [150, 157, 142],  # #8e9d96 — predominant
        [176, 187, 163],  # highlight
        [144, 160, 144],  # verde-gri
        [114, 122, 109],  # crateră
        [112, 112, 96],   # umbră
        [86, 88, 83],     # contur închis
    ],
    dtype=np.float32,
)

COLOR_DIST_MAX = 52
COLOR_DIST_MAX_WASHED = 95
GREY_SPREAD_MAX = 48
COLOR_MIN_SCORE = 0.42

CIRCLE_MIN_RADIUS = 11
CIRCLE_MAX_RADIUS = 44
CIRCLE_FILL_MIN = 0.30          # cât din disc e mască de culoare piatră
CIRCLE_CIRCULARITY_MIN = 0.50   # contur aproape circular
CIRCLE_PRIMARY_FRAC_MIN = 0.20    # fracție #8e9d96 în disc
CIRCLE_GREY_FRAC_MIN = 0.34       # fracție orice nuanță gri din paletă
TEXTURE_STD_MIN = 9.0             # piatra are cratere (fundalul galben e uniform)
TEXTURE_LAP_MIN = 55.0
TEXTURE_SCORE_MIN = 0.34          # scor combinat std+laplacian

MATCH_THRESHOLD = 0.75
LOCAL_MATCH_THRESHOLD = 0.75
EARLY_EXIT_SCORE = 0.82
DETECT_MAX_W = 640
DOWNSCALE_MIN_W = 850
LOCAL_RADIUS = 110
SCALES = (0.35, 0.42, 0.50, 0.58)
LOCAL_SCALES = (0.35, 0.42, 0.50)

CROP = (0.10, 0.10, 0.78, 0.78)  # gameplay; fără marginea dreaptă (radio/anunț)

# Zone ignorate la detectare (piatra poate apărea dreapta-sus lângă anunț).
DETECT_EXCLUDE_ZONES = (
    (0.00, 0.72, 0.22, 1.00),   # minimap
    (0.78, 0.00, 1.00, 1.00),   # radio / margine dreapta
    (0.78, 0.92, 1.00, 1.00),   # colț dreapta-jos
)

# Panou anunț cyan — doar validare, nu excludere totală (piatra poate fi lângă el).
ANNOUNCE_PANEL = (0.52, 0.02, 0.74, 0.22)

# Zone UI excluse (fracții ecran x1,y1,x2,y2). Anunț ~52–76% lățime, sus.
EXCLUDE_ZONES = (
    (0.00, 0.00, 0.32, 0.28),   # chat (mască teste; piatra poate apărea aici)
    *DETECT_EXCLUDE_ZONES,
    ANNOUNCE_PANEL,
)
ANNOUNCE_ZONE = ANNOUNCE_PANEL
ANNOUNCE_MIN_GREY = 0.55  # în panoul anunț acceptăm doar gri clar de piatră
MIN_STONE_GREY = 0.08    # fără #8e9d96 deloc → teren / personaj blurat

# Minigame: 3 click-uri pe aceeași piatră; secvență = 3 pietre → oprește detectia.
STAGES_PER_STONE = 3
STONES_PER_SEQUENCE = 3
CLICKS_PER_SEQUENCE = STONES_PER_SEQUENCE * STAGES_PER_STONE
CLICK_GAP_SEC = 0.02
STONE_ANCHOR_PX = 120
DEAD_ZONE_PX = 160
REARM_ABSENT_SEC = 0.02
POST_FINISH_SEC = 0.05
STRONG_COLOR_SCORE = 1.55   # skip pași lenti dacă culoarea #8e9d96 e clară
COLOR_SKIP_TEMPLATE = 0.68  # nu sărim template la gri ambiguu (piatra spălată)

_template_bgr: np.ndarray | None = None
_template_gray: np.ndarray | None = None
_template_alpha: np.ndarray | None = None
_scaled_templates: list[tuple[np.ndarray, np.ndarray, int, int, float]] | None = None


def _load_template() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    global _template_bgr, _template_gray, _template_alpha
    if _template_gray is not None and _template_alpha is not None:
        return _template_bgr, _template_gray, _template_alpha

    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Template piatră lipsă: {TEMPLATE_PATH}")

    rgba = cv2.imread(str(TEMPLATE_PATH), cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.size == 0:
        raise RuntimeError(f"Nu pot citi template-ul: {TEMPLATE_PATH}")

    if rgba.ndim == 3 and rgba.shape[2] == 4:
        _template_bgr = rgba[:, :, :3]
        _template_alpha = rgba[:, :, 3]
    else:
        _template_bgr = rgba[:, :, :3] if rgba.ndim == 3 else cv2.cvtColor(rgba, cv2.COLOR_GRAY2BGR)
        _template_alpha = np.full(_template_bgr.shape[:2], 255, np.uint8)

    _template_gray = cv2.cvtColor(_template_bgr, cv2.COLOR_BGR2GRAY)
    log.debug(f"Mina: template {_template_gray.shape[1]}x{_template_gray.shape[0]}")
    return _template_bgr, _template_gray, _template_alpha


def _get_scaled_templates() -> list[tuple[np.ndarray, np.ndarray, int, int, float]]:
    global _scaled_templates
    if _scaled_templates is not None:
        return _scaled_templates

    _, tgray, alpha = _load_template()
    th0, tw0 = tgray.shape[:2]
    out: list[tuple[np.ndarray, np.ndarray, int, int, float]] = []
    for scale in SCALES:
        tw = max(8, int(tw0 * scale))
        th = max(8, int(th0 * scale))
        tmpl = cv2.resize(
            tgray, (tw, th),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        tmpl_mask = cv2.resize(alpha, (tw, th), interpolation=cv2.INTER_NEAREST)
        tmpl_mask = (tmpl_mask > 128).astype(np.uint8)
        if int(tmpl_mask.sum()) >= 50:
            out.append((tmpl, tmpl_mask, tw, th, scale))
    _scaled_templates = out
    return out


def build_search_mask(h: int, w: int) -> np.ndarray:
    """Mască 1 = zonă validă (compatibil teste)."""
    mask = np.ones((h, w), np.uint8)
    for x1f, y1f, x2f, y2f in EXCLUDE_ZONES:
        mask[int(y1f * h): int(y2f * h), int(x1f * w): int(x2f * w)] = 0
    return mask


def _in_frac_zone(cx: int, cy: int, w: int, h: int, zone: tuple[float, float, float, float]) -> bool:
    x1f, y1f, x2f, y2f = zone
    return int(x1f * w) <= cx <= int(x2f * w) and int(y1f * h) <= cy <= int(y2f * h)


def _is_ui_excluded(cx: int, cy: int, w: int, h: int) -> bool:
    for zone in DETECT_EXCLUDE_ZONES:
        if _in_frac_zone(cx, cy, w, h, zone):
            return True
    return False


def _is_cyan_ui(roi_bgr: np.ndarray) -> bool:
    """Panou anunț — albastru/teal, nu piatra gri."""
    if roi_bgr.size == 0:
        return False
    b, g, r = cv2.split(roi_bgr)
    rf, gf, bf = r.astype(np.float32), g.astype(np.float32), b.astype(np.float32)
    spread = np.maximum(np.maximum(np.abs(rf - gf), np.abs(gf - bf)), np.abs(rf - bf))
    teal = (bf > 90) & (gf > rf + 6) & (bf > rf + 10) & (np.abs(gf - bf) < 60) & (spread > 18)
    return float(teal.mean()) > 0.10


def _reject_ui_candidate(roi_bgr: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    if _is_cyan_ui(roi_bgr):
        return True
    _, _, tw, th = box
    if tw * th > 2800 and max(tw, th) / max(min(tw, th), 1) > 1.65:
        return True
    return False


def _downscale_gray(gray: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = gray.shape[:2]
    if w <= DOWNSCALE_MIN_W:
        return gray, 1.0
    ds = DETECT_MAX_W / w
    small = cv2.resize(gray, (int(w * ds), int(h * ds)), interpolation=cv2.INTER_AREA)
    return small, ds


def _prep_search(frame_bgr: np.ndarray) -> tuple[np.ndarray, int, int, float]:
    h, w = frame_bgr.shape[:2]
    x1f, y1f, x2f, y2f = CROP
    x1, y1 = int(x1f * w), int(y1f * h)
    x2, y2 = int(x2f * w), int(y2f * h)
    crop = frame_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray, ds = _downscale_gray(gray)
    return gray, x1, y1, ds


def _color_distance_map(bgr: np.ndarray) -> np.ndarray:
    pixels = bgr.astype(np.float32)
    dists = np.linalg.norm(pixels[:, :, None, :] - STONE_PALETTE_BGR[None, None, :, :], axis=3)
    return dists.min(axis=2)


def _stone_color_mask(crop_bgr: np.ndarray) -> np.ndarray:
    """Mască binară: pixeli gri de piatră, fără fundal galben sau UI."""
    b, g, r = cv2.split(crop_bgr)
    rf, gf, bf = r.astype(np.float32), g.astype(np.float32), b.astype(np.float32)
    spread = np.maximum(np.maximum(np.abs(rf - gf), np.abs(gf - bf)), np.abs(rf - bf))
    v = (rf + gf + bf) / 3.0

    dist = _color_distance_map(crop_bgr)
    near_palette = dist < COLOR_DIST_MAX
    washed_grey = (spread < GREY_SPREAD_MAX) & (v > 85) & (v < 220) & (dist < COLOR_DIST_MAX_WASHED)

    yellow_bg = (rf > 190) & (gf > 185) & (bf > 145) & (spread < 55)
    saturated = spread > 65
    cyan_ui = (bf > 90) & (gf > rf + 6) & (bf > rf + 10) & (np.abs(gf - bf) < 60) & (spread > 18)

    stone = (near_palette | washed_grey) & ~yellow_bg & ~saturated & ~cyan_ui
    return stone.astype(np.uint8) * 255


def _roi_color_score(roi_bgr: np.ndarray) -> float:
    if roi_bgr.size == 0:
        return 0.0
    mask = _stone_color_mask(roi_bgr)
    frac = float(mask.mean()) / 255.0
    dist = _color_distance_map(roi_bgr)
    mean_dist = float(dist.mean())
    dist_score = max(0.0, 1.0 - mean_dist / COLOR_DIST_MAX_WASHED)
    return 0.65 * frac + 0.35 * dist_score


def _disk_metrics(roi_bgr: np.ndarray) -> tuple[float, float, float]:
    """Culoare + textură doar în discul central (cercul pietrei)."""
    h, w = roi_bgr.shape[:2]
    if h < 8 or w < 8:
        return 0.0, 0.0, 0.0
    cx, cy = w * 0.5, h * 0.5
    r = min(w, h) * 0.42
    disk = _disk_mask(h, w, cx, cy, r)
    if not bool(disk.any()):
        return 0.0, 0.0, 0.0

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    disk_gray = gray[disk]
    tex_std = float(disk_gray.std())
    lap = float(cv2.Laplacian(gray, cv2.CV_64F)[disk].var())

    mask = _stone_color_mask(roi_bgr)
    color_frac = float(mask[disk].mean()) / 255.0
    pixels = roi_bgr[disk].astype(np.float32)
    dist = np.linalg.norm(pixels - STONE_PRIMARY_BGR, axis=1)
    primary_frac = float((dist < COLOR_DIST_MAX).mean())
    color = 0.6 * color_frac + 0.4 * primary_frac
    tex = min(1.0, tex_std / 26.0) * 0.42 + min(1.0, lap / 700.0) * 0.58
    return color, tex, tex_std


def _stone_appearance_ok(roi_bgr: np.ndarray, *, strict_color: bool = False) -> bool:
    """Culoare #8e9d96 SAU textură de cratere în cercul central."""
    color, tex, tex_std = _disk_metrics(roi_bgr)
    if strict_color:
        return color >= COLOR_MIN_SCORE
    if color >= 0.28:
        return True
    if color >= 0.18 and tex_std >= 12.0:
        return True
    if tex_std >= 11.0:
        return True
    if tex >= TEXTURE_SCORE_MIN and tex_std >= TEXTURE_STD_MIN:
        return True
    return False


def _validate_stone_roi(
    frame_bgr: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    require_color: bool = False,
) -> bool:
    x, y, tw, th = box
    h, w = frame_bgr.shape[:2]
    if x < 0 or y < 0 or x + tw > w or y + th > h:
        return False
    cy = y + th // 2
    if cy < int(h * CROP[1]) or cy > int(h * CROP[3]):
        return False
    cx = x + tw // 2
    if cx < int(w * CROP[0]) or cx > int(w * CROP[2]):
        return False
    if _is_ui_excluded(cx, cy, w, h):
        return False
    roi = frame_bgr[y: y + th, x: x + tw]
    if roi.size == 0:
        return False
    if _reject_ui_candidate(roi, box):
        return False
    if _in_frac_zone(cx, cy, w, h, ANNOUNCE_PANEL):
        c, _, _ = _disk_metrics(roi)
        if _is_cyan_ui(roi) or c < ANNOUNCE_MIN_GREY:
            return False
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    if float((gray < 55).mean()) > 0.10:
        return False
    if float(gray.mean()) < 95:
        return False
    if require_color and not _stone_appearance_ok(roi):
        return False
    return True


def _near_xy(cx: int, cy: int, xy: tuple[int, int], radius: int) -> bool:
    ax, ay = xy
    return (cx - ax) ** 2 + (cy - ay) ** 2 <= radius ** 2


def _mask_dead_zone(gray: np.ndarray, off_x: int, off_y: int, ds: float,
                    dead_zone: tuple[int, int] | None) -> np.ndarray:
    if dead_zone is None:
        return gray
    gx = int((dead_zone[0] - off_x) * ds)
    gy = int((dead_zone[1] - off_y) * ds)
    r = max(12, int(DEAD_ZONE_PX * ds))
    out = gray.copy()
    cv2.circle(out, (gx, gy), r, 128, -1)
    return out


def _disk_mask(h: int, w: int, cx: float, cy: float, r: float) -> np.ndarray:
    ys, xs = np.ogrid[:h, :w]
    return ((xs - cx) ** 2 + (ys - cy) ** 2) <= (r * r)


def _score_circle_region(
    crop_bgr: np.ndarray,
    color_mask: np.ndarray,
    cx: float,
    cy: float,
    r: float,
    *,
    circularity: float = 1.0,
    min_r: float | None = None,
    max_r: float | None = None,
    min_disk: int = 70,
) -> tuple[float, float, float] | None:
    """Evaluează un disc: umplere, #8e9d96 predominant, formă circulară."""
    h, w = crop_bgr.shape[:2]
    r_lo = CIRCLE_MIN_RADIUS if min_r is None else min_r
    r_hi = CIRCLE_MAX_RADIUS if max_r is None else max_r
    if r < r_lo or r > r_hi:
        return None

    disk = _disk_mask(h, w, cx, cy, r)
    n_disk = int(disk.sum())
    if n_disk < min_disk:
        return None

    fill = float(color_mask[disk].mean()) / 255.0

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    bx0 = max(0, int(cx - r))
    by0 = max(0, int(cy - r))
    bx1 = min(w, int(cx + r) + 1)
    by1 = min(h, int(cy + r) + 1)
    patch = gray[by0:by1, bx0:bx1]
    disk_patch = _disk_mask(by1 - by0, bx1 - bx0, cx - bx0, cy - by0, r)
    disk_gray = patch[disk_patch] if patch.size else np.array([], dtype=np.uint8)
    tex_std = float(disk_gray.std()) if disk_gray.size else 0.0
    lap = float(cv2.Laplacian(patch, cv2.CV_64F)[disk_patch].var()) if disk_patch.any() else 0.0

    if fill < CIRCLE_FILL_MIN:
        if tex_std < TEXTURE_STD_MIN or lap < TEXTURE_LAP_MIN:
            return None
        fill = min(0.55, tex_std / 32.0)

    pixels = crop_bgr[disk].astype(np.float32)
    dist_primary = np.linalg.norm(pixels - STONE_PRIMARY_BGR, axis=1)
    primary_frac = float((dist_primary < COLOR_DIST_MAX).mean())

    dist_palette = np.linalg.norm(
        pixels[:, None, :] - STONE_PALETTE_BGR[None, :, :], axis=2,
    ).min(axis=1)
    grey_frac = float((dist_palette < COLOR_DIST_MAX).mean())

    if primary_frac < CIRCLE_PRIMARY_FRAC_MIN and grey_frac < CIRCLE_GREY_FRAC_MIN:
        if tex_std < TEXTURE_STD_MIN or lap < TEXTURE_LAP_MIN:
            return None

    color_score = 0.50 * primary_frac + 0.30 * grey_frac + 0.20 * fill
    tex_score = min(1.0, tex_std / 26.0) * 0.5 + min(1.0, lap / 700.0) * 0.5
    shape_score = min(1.0, circularity) * max(fill, tex_score * 0.5)
    total = color_score * 0.55 + tex_score * 0.30 + shape_score * 0.15
    return total, primary_frac, fill


def _hough_circles(gray_u8: np.ndarray, *, param2: int = 20) -> list[tuple[float, float, float]]:
    blurred = cv2.GaussianBlur(gray_u8, (7, 7), 1.5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.35,
        minDist=38,
        param1=55,
        param2=param2,
        minRadius=CIRCLE_MIN_RADIUS,
        maxRadius=CIRCLE_MAX_RADIUS,
    )
    if circles is None:
        return []
    out: list[tuple[float, float, float]] = []
    for c in circles[0]:
        out.append((float(c[0]), float(c[1]), float(c[2])))
    return out


def _texture_circle_map(crop_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    return np.clip(lap * 0.45, 0, 255).astype(np.uint8)


def _find_by_color(
    frame_bgr: np.ndarray,
    dead_zone: tuple[int, int] | None = None,
    hint: tuple[int, int] | None = None,
) -> dict | None:
    """Detectează piatra: cerc cu #8e9d96 predominant în interior."""
    h, w = frame_bgr.shape[:2]
    x1f, y1f, x2f, y2f = CROP
    x1, y1 = int(x1f * w), int(y1f * h)
    x2, y2 = int(x2f * w), int(y2f * h)

    off_x, off_y = x1, y1
    crop_x2, crop_y2 = x2, y2
    if hint is not None:
        hx, hy = hint
        r = LOCAL_RADIUS
        lx1 = max(x1, hx - r)
        ly1 = max(y1, hy - r)
        lx2 = min(x2, hx + r)
        ly2 = min(y2, hy + r)
        if lx2 - lx1 >= 80 and ly2 - ly1 >= 80:
            off_x, off_y, crop_x2, crop_y2 = lx1, ly1, lx2, ly2

    if (crop_x2 - off_x) < (80 if hint else 240) or (crop_y2 - off_y) < (60 if hint else 160):
        return None
    crop = frame_bgr[off_y:crop_y2, off_x:crop_x2]

    ds = 1.0
    cw = crop.shape[1]
    if cw > DOWNSCALE_MIN_W:
        ds = DETECT_MAX_W / cw
        crop = cv2.resize(
            crop,
            (int(cw * ds), int(crop.shape[0] * ds)),
            interpolation=cv2.INTER_AREA,
        )

    color_mask = _stone_color_mask(crop)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    best: tuple[float, int, int, int, int, int, int] | None = None
    seen: set[tuple[int, int, int]] = set()

    def _consider(cx: float, cy: float, r: float, circ: float) -> None:
        key = (int(cx / 8), int(cy / 8), int(r / 6))
        if key in seen:
            return
        seen.add(key)

        scored = _score_circle_region(
            crop, color_mask, cx, cy, r, circularity=circ,
            min_r=min_r, max_r=max_r, min_disk=max(45, int(70 * ds * ds)),
        )
        if scored is None:
            return

        total, primary_frac, fill = scored
        r_full = r / ds
        gcx = off_x + int(cx / ds)
        gcy = off_y + int(cy / ds)
        side = int(max(12, r_full * 2))
        fx = gcx - side // 2
        fy = gcy - side // 2
        box = (fx, fy, side, side)
        roi = frame_bgr[fy: fy + side, fx: fx + side]
        if not _stone_appearance_ok(roi):
            return
        if not _validate_stone_roi(frame_bgr, box, require_color=True):
            return

        _c, _t, _ts = _disk_metrics(roi)
        gray_mean = float(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).mean())
        if _c < MIN_STONE_GREY:
            if _ts < 18.0 or gray_mean < 90:
                return
            score = _ts * 0.045 + total * 0.18
        else:
            if _c < 0.16 and _ts < 13.0:
                return
            if _c < 0.45 and _ts > 28.0:
                return
            score = _c * 5.0 + total * 0.25 + min(1.0, _ts / 28.0) * 0.15
        if dead_zone and _near_xy(gcx, gcy, dead_zone, DEAD_ZONE_PX):
            return
        if _is_ui_excluded(gcx, gcy, w, h):
            return

        nonlocal best
        if best is None or score > best[0]:
            best = (score, gcx, gcy, fx, fy, side, side)

    min_r = max(4, int(CIRCLE_MIN_RADIUS * ds))
    max_r = max(min_r + 2, int(CIRCLE_MAX_RADIUS * ds))

    def _hough_on(mask_or_gray: np.ndarray, *, param2: int = 20) -> list[tuple[float, float, float]]:
        blurred = cv2.GaussianBlur(mask_or_gray, (5, 5), 1.2)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.35,
            minDist=max(20, int(38 * ds)),
            param1=55,
            param2=param2,
            minRadius=min_r,
            maxRadius=max_r,
        )
        if circles is None:
            return []
        return [(float(c[0]), float(c[1]), float(c[2])) for c in circles[0]]

    for hc_x, hc_y, hc_r in _hough_on(color_mask):
        _consider(hc_x, hc_y, hc_r, 0.92)

    cnts, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:8]:
        area = cv2.contourArea(c)
        if area < max(180, int(400 * ds * ds)):
            continue
        (xc, yc), radius = cv2.minEnclosingCircle(c)
        if radius < min_r or radius > max_r:
            continue
        _consider(float(xc), float(yc), float(radius), 0.95)

    strong = best is not None and best[0] >= STRONG_COLOR_SCORE
    if not strong:
        for hc_x, hc_y, hc_r in _hough_on(_texture_circle_map(crop), param2=14):
            _consider(hc_x, hc_y, hc_r, 0.88)

    if not strong:
        for c in cnts:
            area = cv2.contourArea(c)
            if area < max(140, int(280 * ds * ds)) or area > 32000:
                continue
            peri = cv2.arcLength(c, True)
            if peri < 1:
                continue
            circ = 4 * np.pi * area / (peri * peri)
            if circ < CIRCLE_CIRCULARITY_MIN:
                continue

            (xc, yc), radius = cv2.minEnclosingCircle(c)
            if radius < min_r or radius > max_r:
                continue

            enclose_fill = area / max(np.pi * radius * radius, 1.0)
            if enclose_fill < 0.38:
                continue

            _consider(xc, yc, radius, circ * min(1.0, enclose_fill / 0.55))

    if best is None:
        return None

    score, cx, cy, fx, fy, bw, bh = best
    _, tgray, _ = _load_template()
    roi = frame_bgr[fy: fy + bh, fx: fx + bw]
    display_score = max(0.76, min(0.99, 0.55 + score + _roi_color_score(roi) * 0.25))
    return {
        "center": (cx, cy),
        "score": display_score,
        "box": (fx, fy, bw, bh),
        "scale": bw / tgray.shape[1],
        "method": "circle",
    }


def _match_on_gray(
    gray: np.ndarray,
    frame_bgr: np.ndarray,
    off_x: int,
    off_y: int,
    ds: float,
    scales: tuple[float, ...] | None = None,
    threshold: float = MATCH_THRESHOLD,
    dead_zone: tuple[int, int] | None = None,
    max_peaks: int = 3,
) -> dict | None:
    gh, gw = gray.shape[:2]
    all_tmpl = _get_scaled_templates()
    if scales:
        allowed = set(scales)
        templates = [t for t in all_tmpl if t[4] in allowed]
    else:
        templates = all_tmpl

    best_hit: dict | None = None
    best_score = -1.0
    _, tgray, _ = _load_template()

    for tmpl, tmpl_mask, tw, th, _scale in templates:
        if tw >= gw or th >= gh:
            continue

        res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED, mask=tmpl_mask)
        res_work = res.copy()
        suppress_r = max(8, max(tw, th) // 2)

        for _ in range(max_peaks):
            _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(res_work)
            if not np.isfinite(max_val) or max_val < threshold:
                break

            lx, ly = max_loc
            cx = lx + tw // 2
            cy = ly + th // 2
            fx = off_x + int(cx / ds)
            fy = off_y + int(cy / ds)
            if dead_zone and _near_xy(fx, fy, dead_zone, DEAD_ZONE_PX):
                cv2.circle(res_work, (cx, cy), suppress_r, 0, -1)
                continue

            bx = off_x + int(lx / ds)
            by = off_y + int(ly / ds)
            btw, bth = max(8, int(tw / ds)), max(8, int(th / ds))
            box = (bx, by, btw, bth)
            roi = frame_bgr[by: by + bth, bx: bx + btw]

            if not _stone_appearance_ok(roi):
                cv2.circle(res_work, (cx, cy), suppress_r, 0, -1)
                continue
            c_hit, _, ts_hit = _disk_metrics(roi)
            if c_hit < MIN_STONE_GREY and ts_hit < 12.0:
                cv2.circle(res_work, (cx, cy), suppress_r, 0, -1)
                continue
            if not _validate_stone_roi(frame_bgr, box):
                cv2.circle(res_work, (cx, cy), suppress_r, 0, -1)
                continue

            if max_val > best_score:
                best_score = float(max_val)
                best_hit = {
                    "center": (fx, fy),
                    "score": float(max_val),
                    "box": box,
                    "scale": btw / tgray.shape[1],
                    "method": "template",
                }
            cv2.circle(res_work, (cx, cy), suppress_r, 0, -1)

    return best_hit


def _find_by_template(
    frame_bgr: np.ndarray,
    hint: tuple[int, int] | None = None,
    dead_zone: tuple[int, int] | None = None,
    scales: tuple[float, ...] | None = None,
) -> dict | None:
    h, w = frame_bgr.shape[:2]
    tmpl_scales = scales if scales is not None else SCALES

    if hint is not None and dead_zone is None:
        hx, hy = hint
        r = LOCAL_RADIUS
        x1, y1 = max(0, hx - r), max(0, hy - r)
        x2, y2 = min(w, hx + r), min(h, hy + r)
        if x2 - x1 >= 40 and y2 - y1 >= 40:
            crop = frame_bgr[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray, ds = _downscale_gray(gray)
            hit = _match_on_gray(
                gray, frame_bgr, x1, y1, ds, LOCAL_SCALES, LOCAL_MATCH_THRESHOLD,
                max_peaks=2,
            )
            if hit is not None:
                return hit

    gray, off_x, off_y, ds = _prep_search(frame_bgr)
    gray = _mask_dead_zone(gray, off_x, off_y, ds, dead_zone)
    return _match_on_gray(
        gray, frame_bgr, off_x, off_y, ds, tmpl_scales, dead_zone=dead_zone,
        max_peaks=2 if scales is None else 2,
    )


def _local_template_score(frame_bgr: np.ndarray, box: tuple[int, int, int, int]) -> float:
    """Cât de bine se potrivește template-ul chiar în cutia candidatului."""
    x, y, tw, th = box
    if tw < 8 or th < 8:
        return 0.0
    crop = cv2.cvtColor(frame_bgr[y: y + th, x: x + tw], cv2.COLOR_BGR2GRAY)
    if crop.size == 0:
        return 0.0
    ch, cw = crop.shape[:2]
    best = 0.0
    for tmpl, mask, ttw, tth, _sc in _get_scaled_templates():
        if ttw > cw or tth > ch:
            scale = min(cw / max(ttw, 1), ch / max(tth, 1)) * 0.95
            tw2 = max(8, int(ttw * scale))
            th2 = max(8, int(tth * scale))
            tmpl = cv2.resize(tmpl, (tw2, th2), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (tw2, th2), interpolation=cv2.INTER_NEAREST)
            ttw, tth = tw2, th2
        res = cv2.matchTemplate(crop, tmpl, cv2.TM_CCOEFF_NORMED, mask=mask)
        best = max(best, float(res.max()))
    return best


def _merge_hits(
    color_hit: dict | None,
    tmpl_hit: dict | None,
    frame_bgr: np.ndarray | None = None,
) -> dict | None:
    if color_hit and tmpl_hit:
        cx1, cy1 = color_hit["center"]
        cx2, cy2 = tmpl_hit["center"]
        dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
        if dist < 45:
            best = dict(color_hit if color_hit["score"] >= tmpl_hit["score"] * 0.92 else tmpl_hit)
            best["score"] = max(color_hit["score"], tmpl_hit["score"])
            best["method"] = "hybrid"
            return best

        c_color = 0.0
        local_t = 0.0
        if frame_bgr is not None:
            x, y, bw, bh = color_hit["box"]
            roi = frame_bgr[y: y + bh, x: x + bw]
            c_color, _, _ = _disk_metrics(roi)
            local_t = _local_template_score(frame_bgr, color_hit["box"])

        if dist > 400 and tmpl_hit["score"] >= 0.74:
            return tmpl_hit
        if dist <= 400 and c_color >= 0.65:
            return color_hit
        if local_t < 0.20 and tmpl_hit["score"] >= 0.74:
            return tmpl_hit
        if c_color < 0.68 and tmpl_hit["score"] >= 0.75:
            return tmpl_hit
        if tmpl_hit["score"] >= 0.74:
            return tmpl_hit
        return color_hit
    return color_hit or tmpl_hit


def _needs_template_verify(color_hit: dict | None, frame_bgr: np.ndarray) -> bool:
    """Template complet doar când culoarea nu e sigură (evită fals pozitive pe stâncă)."""
    if color_hit is None:
        return True
    x, y, bw, bh = color_hit["box"]
    roi = frame_bgr[y: y + bh, x: x + bw]
    if roi.size == 0:
        return True
    c, _, _ = _disk_metrics(roi)
    if c < 0.65:
        return True
    if _local_template_score(frame_bgr, color_hit["box"]) < 0.25:
        return True
    return False


def find_stone(
    frame_bgr: np.ndarray,
    hint: tuple[int, int] | None = None,
    dead_zone: tuple[int, int] | None = None,
) -> dict | None:
    """Caută piatra: culoare + template (template omis când culoarea e sigură)."""
    h, w = frame_bgr.shape[:2]
    if h < 32 or w < 32:
        return None

    if hint is not None:
        color_hit = _find_by_color(frame_bgr, dead_zone=dead_zone, hint=hint)
        if color_hit is not None:
            return color_hit
        return _find_by_template(frame_bgr, hint=hint, dead_zone=dead_zone)

    color_hit = _find_by_color(frame_bgr, dead_zone=dead_zone)
    if not _needs_template_verify(color_hit, frame_bgr):
        return color_hit
    tmpl_hit = _find_by_template(frame_bgr, dead_zone=dead_zone)
    return _merge_hits(color_hit, tmpl_hit, frame_bgr)


def annotate(frame_bgr: np.ndarray, match: dict | None) -> np.ndarray:
    img = frame_bgr.copy()
    if not match:
        cv2.putText(img, "no stone", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return img

    x, y, tw, th = match["box"]
    cx, cy = match["center"]
    score = match["score"]
    method = match.get("method", "?")
    cv2.rectangle(img, (x, y), (x + tw, y + th), (0, 255, 0), 2)
    cv2.drawMarker(img, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 12, 2)
    cv2.putText(
        img, f"stone {score:.2f} {method}", (x, max(16, y - 6)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
    )
    return img


def save_detection_shot(
    frame_bgr: np.ndarray,
    match: dict | None,
    folder: Path,
    index: int,
    tag: str,
    *,
    caption: str = "",
) -> Path | None:
    """Salvează captură anotată în folderul de debug Mina."""
    folder.mkdir(parents=True, exist_ok=True)
    img = annotate(frame_bgr, match)
    if caption:
        cv2.putText(
            img, caption, (8, img.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 220, 255), 1, cv2.LINE_AA,
        )
    path = folder / f"{index:04d}_{tag}.png"
    if not cv2.imwrite(str(path), img):
        return None
    return path


class MinaClickGate:
    """3 click-uri per piatră; secvență de 3 pietre apoi stop."""

    def __init__(self) -> None:
        self.stage = 0
        self._stones_done = 0
        self._anchor: tuple[int, int] | None = None
        self._finished_anchor: tuple[int, int] | None = None
        self._last_click_t = 0.0
        self._absent_since: float | None = None

    @property
    def stones_done(self) -> int:
        return self._stones_done

    @property
    def sequence_complete(self) -> bool:
        return self._stones_done >= STONES_PER_SEQUENCE

    def reset(self) -> None:
        self.stage = 0
        self._anchor = None
        self._absent_since = None

    def dead_zone(self) -> tuple[int, int] | None:
        return self._finished_anchor

    def search_hint(self) -> tuple[int, int] | None:
        if self._finished_anchor:
            return None
        if self._anchor and 0 < self.stage < STAGES_PER_STONE:
            return self._anchor
        return None

    def is_valid_target(self, match: dict) -> bool:
        if self.sequence_complete:
            return False
        cx, cy = match["center"]
        if self._finished_anchor and _near_xy(cx, cy, self._finished_anchor, DEAD_ZONE_PX):
            return False
        if self.stage >= STAGES_PER_STONE:
            return self._is_new_stone(cx, cy)
        return True

    def _is_new_stone(self, cx: int, cy: int) -> bool:
        if self._anchor is None:
            return True
        ax, ay = self._anchor
        return (cx - ax) ** 2 + (cy - ay) ** 2 > STONE_ANCHOR_PX ** 2

    def _maybe_reset_after_absence(self, now: float) -> None:
        if self._absent_since is None:
            return
        if now - self._absent_since < REARM_ABSENT_SEC:
            return
        self.reset()

    def should_click(self, match: dict | None, now: float) -> bool:
        if self.sequence_complete:
            return False
        if now - self._last_click_t < CLICK_GAP_SEC:
            return False

        if match is None:
            if self._absent_since is None:
                self._absent_since = now
            self._maybe_reset_after_absence(now)
            return False

        cx, cy = match["center"]

        if self.stage >= STAGES_PER_STONE:
            if self._is_new_stone(cx, cy):
                self.reset()
                return True
            if now - self._last_click_t >= POST_FINISH_SEC:
                self.reset()
                return True
            return False

        self._absent_since = None

        if self.stage > 0 and self._is_new_stone(cx, cy):
            self.reset()
            return True

        return True

    def on_click(self, center: tuple[int, int], now: float) -> int:
        if self._finished_anchor and not _near_xy(
            center[0], center[1], self._finished_anchor, DEAD_ZONE_PX
        ):
            self._finished_anchor = None

        self._last_click_t = now
        self._anchor = center
        self.stage += 1
        if self.stage >= STAGES_PER_STONE:
            self._finished_anchor = center
            if self.stage == STAGES_PER_STONE:
                self._stones_done += 1
        return self.stage
