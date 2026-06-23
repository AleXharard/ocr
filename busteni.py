"""Busteni minigame — apasă cifra când indicatorul atinge zona verde.

Joc de sincronizare, complet separat de jocul cu casete albe (vezi `vision.py`):
un inel cu o cifră 1-9 în centru, o zonă-țintă teal (`#068f6d`) apărută într-o
poziție aleatoare și un indicator teal mai deschis (`#38d5af`) care se rotește.
Apăsăm cifra exact când indicatorul atinge zona. Checkpointul verde (`#01b802`)
NU conține teal, deci e ignorat automat.

Culorile pot varia ușor cu rezoluția/gamma → potrivire cu toleranță.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

import logger as log
import vision as vis

# ── Culori de referință (RGB) ────────────────────────────────────────────────
ZONE_RGB = (6, 143, 109)        # #068f6d — zona țintă (teal mai închis)
IND_RGB = (56, 213, 175)        # #38d5af — indicatorul (teal mai deschis)
CHECKPOINT_RGB = (1, 184, 2)    # #01b802 — checkpoint verde, de IGNORAT

COLOR_TOL = 75.0                # rază de potrivire a culorii teal (toleranță pt. gamma/rezoluție)
# Inelul real are MEREU și o zonă (arc gros) ȘI un indicator. Decorul jocului poate
# avea pete teal, dar rar pe ambele simultan → cerem prag pe fiecare.
MIN_ZONE_PX = 120               # arcul zonei (real ~600+)
MIN_IND_PX = 15                 # indicatorul (real ~130, mai subțire)
NEW_ROUND_DEG = 25.0            # saltul zonei care semnalează o rundă nouă

# ── Auto-stop ─────────────────────────────────────────────────────────────
# Sesiunea rulează cât timp minijocul e detectat (oricât de lung), și se închide
# după NO_DETECT_SEC fără nicio detecție. HARD_CAP_SEC e doar o plasă de siguranță.
NO_DETECT_SEC = 5.0             # 5s fără detecție → închide bucla
HARD_CAP_SEC = 1800.0           # plafon de siguranță (30 min) — nu se atinge în joc normal

LEAD_SEC = 0.015                # latența reală de apăsare e mică → lead mic (altfel apasă prea devreme)
MAX_LEAD_RAD = math.radians(10) # plafon al predicției ca să nu „sară" peste zonă
TRIGGER_HALF = math.radians(12) # apăsăm când indicatorul e aproape de CENTRUL zonei (nu la margine)
MAX_ZONE_HALF = math.radians(55)  # o zonă reală nu depășește ~110° → restul e zgomot
DIGITS = "123456789"
MIN_DIGIT_SCORE = 0.28          # sub atât, citirea cifrei e nesigură → nu apăsăm
MIN_VOTE_SCORE = 0.30           # prag pentru un vot în consens
LOCK_SCORE = 0.38               # o singură citire foarte sigură → îngheță cifra
LOCK_VOTES = 3                  # consens din cadre curate (indicator departe de zonă)
DIGIT_CLEAR_DEG = 30.0          # indicatorul e „departe" de zonă → cifra centrală e curată


# ── Segmentare culoare ───────────────────────────────────────────────────────
def _color_masks(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Întoarce (zone_mask, ind_mask) bool, excluzând checkpointul verde."""
    img = frame_bgr.astype(np.float32)
    b, g, r = img[..., 0], img[..., 1], img[..., 2]

    def dist(rgb: tuple[int, int, int]) -> np.ndarray:
        rr, gg, bb = rgb
        return np.sqrt((r - rr) ** 2 + (g - gg) ** 2 + (b - bb) ** 2)

    d_zone = dist(ZONE_RGB)
    d_ind = dist(IND_RGB)
    d_chk = dist(CHECKPOINT_RGB)

    teal_min = np.minimum(d_zone, d_ind)
    teal = (teal_min < COLOR_TOL) & (teal_min < d_chk)
    ind_mask = teal & (d_ind <= d_zone)
    zone_mask = teal & (d_zone < d_ind)
    return zone_mask, ind_mask


def _largest_blob(mask: np.ndarray) -> np.ndarray:
    """Păstrează doar cea mai mare componentă conexă (arcul/indicatorul real).

    Marginile anti-aliased ale indicatorului pot fi clasificate ca „zonă"; izolând
    cel mai mare blob eliminăm aceste pete răzlețe care altfel deformează unghiul.
    """
    m = mask.astype(np.uint8)
    if m.sum() == 0:
        return mask
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return mask
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == k


def _circular_mean(angles: np.ndarray) -> float:
    return math.atan2(float(np.sin(angles).mean()), float(np.cos(angles).mean()))


def ang_diff(a: float, b: float) -> float:
    """Diferență unghiulară semnată minimă a-b în (-pi, pi]."""
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def _angles(mask: np.ndarray, cx: float, cy: float) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.empty(0, np.float32)
    return np.arctan2(ys - cy, xs - cx)


# ── Stare detectată dintr-un cadru ───────────────────────────────────────────
def detect_state(frame_bgr: np.ndarray) -> dict:
    """Citește starea minijocului din cadru. Centrul = centrul regiunii alese."""
    h, w = frame_bgr.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    zone_mask, ind_mask = _color_masks(frame_bgr)
    # izolăm arcul real și indicatorul real (elimină petele răzlețe)
    zone_mask = _largest_blob(zone_mask)
    ind_mask = _largest_blob(ind_mask)
    n_zone, n_ind = int(zone_mask.sum()), int(ind_mask.sum())

    # Inelul real => zonă ȘI indicator simultan; altfel e decor/zgomot.
    if n_zone < MIN_ZONE_PX or n_ind < MIN_IND_PX:
        return {"has_teal": False, "n_zone": n_zone, "n_ind": n_ind}

    za = _angles(zone_mask, cx, cy)
    zone_center = _circular_mean(za)
    dev = np.abs(np.angle(np.exp(1j * (za - zone_center))))
    zone_half = min(float(np.percentile(dev, 95)), MAX_ZONE_HALF)

    ind_angle = _circular_mean(_angles(ind_mask, cx, cy))

    return {
        "has_teal": True,
        "center": (cx, cy),
        "zone_center": zone_center,
        "zone_half": zone_half,
        "ind_angle": ind_angle,
        "zone_mask": zone_mask,
        "ind_mask": ind_mask,
        "n_zone": n_zone,
        "n_ind": n_ind,
    }


# ── Citirea cifrei centrale (1-9) ────────────────────────────────────────────
def read_digit(frame_bgr: np.ndarray, teal_mask: np.ndarray | None = None) -> tuple[str | None, float]:
    """Cifra din centrul inelului. Bancă de glife restrânsă la 1-9 (alb-pe-negru)."""
    h, w = frame_bgr.shape[:2]
    cx, cy = w // 2, h // 2
    # ROI strâns pe centru (cifra) ca să excludem inelul/indicatorul de pe margine
    rw, rh = max(8, int(w * 0.24)), max(8, int(h * 0.28))
    roi = frame_bgr[max(0, cy - rh): cy + rh, max(0, cx - rw): cx + rw]
    if roi.size == 0:
        return None, 0.0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    if teal_mask is not None:
        sub = teal_mask[max(0, cy - rh): cy + rh, max(0, cx - rw): cx + rw].astype(np.uint8)
        # dilatare mică — una prea mare poate mânca pixeli ai cifrei albe
        sub = cv2.dilate(sub, np.ones((3, 3), np.uint8))
        gray[sub > 0] = 0  # scoatem teal din ROI ca să nu fie confundat cu cerneala

    # Cifra e ALBĂ pe fundal închis — invers față de jocul cu casete (negru pe alb).
    inv = 255 - gray
    enhanced = vis.enhance_gray(inv)

    bank = vis._get_bank()
    if bank is None:
        return None, 0.0
    glyph = vis.normalize_glyph(enhanced)
    if int((glyph > 0).sum()) < 0.05 * glyph.size:
        return None, 0.0  # prea puțină cerneală → nu e o cifră reală
    q = vis.glyph_to_vec(glyph)
    scores = bank.vectors @ q
    idx = np.array([i for i, lab in enumerate(bank.labels) if str(lab) in DIGITS])
    if idx.size == 0:
        return None, 0.0
    j = int(idx[int(np.argmax(scores[idx]))])
    score = float(scores[j])
    if score < MIN_DIGIT_SCORE:
        return None, score  # ROI gol / citire nesigură → nu apăsăm o tastă greșită
    return str(bank.labels[j]), score


# ── Sesiune cu auto-stop ─────────────────────────────────────────────────────
class BustenSession:
    """Procesează cadre unul câte unul; decide când să apese și când să se oprească."""

    def __init__(self) -> None:
        self.t0 = 0.0
        self.seen = False
        self.last_teal_t = 0.0
        self.cur_digit: str | None = None
        self.round_pressed = False
        self.round_zone_angle: float | None = None  # unghiul zonei la momentul apăsării
        self.zone_center: float | None = None  # zonă cachuită din cadre curate
        self.zone_half: float | None = None
        self.ind_hist: list[tuple[float, float]] = []
        self.pressed_count = 0
        self._votes: dict[str, int] = {}  # voturi cifră în runda curentă (cifra e fixă/rundă)
        self._digit_locked = False       # odată înghețată, cifra NU se schimbă până la rundă nouă
        self.last_state: dict = {"has_teal": False}

    def start(self, now: float) -> None:
        self.t0 = now
        self.last_teal_t = now
        self.seen = False

    def should_stop(self, now: float) -> str | None:
        if now - self.t0 > HARD_CAP_SEC:
            return "cap"  # plasă de siguranță (30 min)
        # cât timp se detectează minijocul, last_teal_t se reîmprospătează → rulează oricât;
        # 5s fără nicio detecție = lotul s-a terminat (sau nu a apărut) → închide.
        if now - self.last_teal_t > NO_DETECT_SEC:
            return "done" if self.seen else "no-show"
        return None

    def _velocity(self) -> float:
        """Viteză unghiulară din eșantioanele recente (fereastră 0.12s, fără cele vechi)."""
        if len(self.ind_hist) < 2:
            return 0.0
        t_now = self.ind_hist[-1][0]
        recent = [(t, a) for (t, a) in self.ind_hist if t_now - t <= 0.12]
        if len(recent) < 2:
            return 0.0
        (t0, a0), (t1, a1) = recent[0], recent[-1]
        dt = t1 - t0
        return ang_diff(a1, a0) / dt if dt > 1e-4 else 0.0

    def _clean_for_digit(self, ia: float | None, zc: float | None, zh: float | None) -> bool:
        """Cifra centrală e citibilă când indicatorul nu e peste zona țintă."""
        if ia is None or zc is None or zh is None:
            return True
        return abs(ang_diff(ia, zc)) > zh + math.radians(DIGIT_CLEAR_DEG)

    def _read_and_lock_digit(
        self, frame_bgr: np.ndarray, teal_mask: np.ndarray, ia: float | None,
        zc: float | None, zh: float | None,
    ) -> None:
        """Citește cifra doar din cadre curate; o îngheță după consens — nu o suprascrie."""
        if self._digit_locked or self.round_pressed:
            return
        if not self._clean_for_digit(ia, zc, zh):
            return
        d_raw, d_score = read_digit(frame_bgr, teal_mask)
        if d_raw is None:
            return
        if d_score >= LOCK_SCORE:
            self.cur_digit = d_raw
            self._digit_locked = True
            return
        if d_score >= MIN_VOTE_SCORE:
            self._votes[d_raw] = self._votes.get(d_raw, 0) + 1
            best = max(self._votes, key=self._votes.get)
            if self._votes[best] >= LOCK_VOTES:
                self.cur_digit = best
                self._digit_locked = True

    def process(self, frame_bgr: np.ndarray, now: float) -> tuple[str, str] | None:
        """Întoarce ('press', cifra) când trebuie apăsat, altfel None."""
        st = detect_state(frame_bgr)
        self.last_state = st
        if not st["has_teal"]:
            return None

        self.seen = True
        self.last_teal_t = now
        zc_now, zh_now, ia = st["zone_center"], st["zone_half"], st["ind_angle"]
        teal_mask = st["zone_mask"] | st["ind_mask"]

        # Cifra e constantă pe rundă — o citim doar din cadre curate și o ÎNGHEȚĂM.
        # Citiri eronate când indicatorul trece prin zonă NU mai pot schimba alegerea.
        self._read_and_lock_digit(frame_bgr, teal_mask, ia, zc_now, zh_now)
        majority = max(self._votes, key=self._votes.get) if self._votes else None

        # Rundă nouă după o apăsare. Trei semnale (oricare e suficient):
        #  • indicatorul a PĂRĂSIT zona  → cel mai sigur; o nouă intrare = rundă nouă
        #  • cifra (votată) s-a schimbat → merge chiar dacă zona e lângă cea veche
        #  • zona a sărit                → rezervă
        if self.round_pressed:
            left_zone = (ia is not None and self.zone_center is not None and self.zone_half is not None
                         and abs(ang_diff(ia, self.zone_center)) > self.zone_half + math.radians(22))
            digit_changed = (majority is not None and majority != self.cur_digit
                             and self._votes[majority] >= 3)
            zone_jumped = (zc_now is not None and self.round_zone_angle is not None
                           and abs(ang_diff(zc_now, self.round_zone_angle)) > math.radians(NEW_ROUND_DEG))
            if left_zone or digit_changed or zone_jumped:
                self.round_pressed = False
                self.cur_digit = None
                self.ind_hist.clear()
                self.zone_center = self.zone_half = None
                self._votes = {}
                self._digit_locked = False

        # cifra curentă rămâne cea înghețată; nu o suprascriem cu voturi noi
        # Cachuim zona din cadre „curate" (indicatorul departe de ea). Când indicatorul
        # o atinge, el ocluzează arcul și deformează măsurarea → folosim valoarea cachuită.
        if zc_now is not None and zh_now is not None:
            clean = ia is None or abs(ang_diff(ia, zc_now)) > zh_now + math.radians(12)
            if clean or self.zone_center is None:
                self.zone_center, self.zone_half = zc_now, zh_now

        if ia is not None:
            self.ind_hist.append((now, ia))
            self.ind_hist = self.ind_hist[-5:]

        zc, zh = self.zone_center, self.zone_half
        if (not self.round_pressed and self._digit_locked and self.cur_digit
                and ia is not None and zc is not None and zh is not None):
            lead = self._velocity() * LEAD_SEC
            lead = max(-MAX_LEAD_RAD, min(MAX_LEAD_RAD, lead))  # nu sărim peste zonă
            pred = ia + lead
            # apăsăm aproape de CENTRUL zonei (fereastră strânsă), nu la marginea ei
            eff_half = min(zh, TRIGGER_HALF)
            if abs(ang_diff(pred, zc)) <= eff_half:
                self.round_pressed = True
                self.round_zone_angle = zc  # îngheață unghiul zonei pentru detecția saltului
                self.pressed_count += 1
                return ("press", self.cur_digit)

        return None


# ── Diagnostic: desenează ce „vede" detecția peste cadru ─────────────────────
def annotate(frame_bgr: np.ndarray, state: dict, digit: str | None = None,
             fired: bool = False) -> np.ndarray:
    """Suprapune măștile teal, centrul, arcul zonei și unghiul indicatorului."""
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    cx, cy = int((w - 1) / 2), int((h - 1) / 2)
    R = int(min(w, h) * 0.45)

    if state.get("has_teal"):
        img[state["zone_mask"]] = (0, 0, 255)    # zona  → roșu
        img[state["ind_mask"]] = (0, 255, 255)   # indicator → galben
        zc, zh, ia = state["zone_center"], state["zone_half"], state["ind_angle"]
        if zc is not None and zh is not None:
            for a in (zc - zh, zc + zh):
                cv2.line(img, (cx, cy), (int(cx + R * math.cos(a)), int(cy + R * math.sin(a))),
                         (0, 0, 255), 1)
        if ia is not None:
            cv2.line(img, (cx, cy), (int(cx + R * math.cos(ia)), int(cy + R * math.sin(ia))),
                     (0, 255, 255), 2)

    cv2.drawMarker(img, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 14, 1)

    def deg(a):
        return f"{math.degrees(a):.0f}" if a is not None else "-"

    line1 = [f"d={digit}" if digit else "d=?",
             f"z={state.get('n_zone', 0)}", f"i={state.get('n_ind', 0)}"]
    if fired:
        line1.append("FIRE")
    line2 = [f"zc={deg(state.get('zone_center'))}", f"zh={deg(state.get('zone_half'))}",
             f"ia={deg(state.get('ind_angle'))}"]
    color = (0, 255, 0) if fired else (255, 255, 255)
    for i, txt in enumerate((" ".join(line1), " ".join(line2))):
        y = 18 + i * 20
        cv2.putText(img, txt, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(img, txt, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img
