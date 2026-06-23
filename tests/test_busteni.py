"""Teste busteni — pe capturile reale + rotații sintetice."""

import math
import os
import sys

import cv2
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import busteni  # noqa: E402

SHOT_DIR = r"C:/Users/mihai/Pictures/Screenshots"
MINIGAME = os.path.join(SHOT_DIR, "Screenshot 2026-06-13 013920.png")
CHECKPOINT = os.path.join(SHOT_DIR, "Screenshot 2026-06-13 013842.png")

# Centrul inelului în captura completă (din analiza pixelilor).
RING_CENTER = (351, 388)
HALF = 78  # jumătatea laturii regiunii ce încadrează strâns inelul


def _load_bgr(path):
    return cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)


def _crop_region(bgr, center, half):
    cx, cy = center
    return bgr[cy - half: cy + half, cx - half: cx + half].copy()


@pytest.fixture(scope="module")
def minigame_region():
    if not os.path.exists(MINIGAME):
        pytest.skip("captura minigame lipsește")
    return _crop_region(_load_bgr(MINIGAME), RING_CENTER, HALF)


@pytest.fixture(scope="module")
def checkpoint_region():
    if not os.path.exists(CHECKPOINT):
        pytest.skip("captura checkpoint lipsește")
    # checkpointul verde, încadrat similar
    bgr = _load_bgr(CHECKPOINT)
    return _crop_region(bgr, (286, 217), 90)


def test_minigame_has_teal(minigame_region):
    st = busteni.detect_state(minigame_region)
    assert st["has_teal"] is True
    assert st["n_zone"] > 0 and st["n_ind"] > 0


def test_checkpoint_ignored(checkpoint_region):
    """Checkpointul verde NU are teal → has_teal False, deci e ignorat."""
    st = busteni.detect_state(checkpoint_region)
    assert st["has_teal"] is False


def test_zone_below_and_indicator_right(minigame_region):
    st = busteni.detect_state(minigame_region)
    # în captură zona e jos (≈ +90°, adică math.pi/2 cu y în jos)
    assert abs(busteni.ang_diff(st["zone_center"], math.pi / 2)) < math.radians(35)
    # indicatorul e jos-dreapta, la un unghi mai mic decât zona
    assert st["ind_angle"] is not None
    assert busteni.ang_diff(st["zone_center"], st["ind_angle"]) > 0


def test_read_digit_is_2(minigame_region):
    zone_mask, ind_mask = busteni._color_masks(minigame_region)
    digit, score = busteni.read_digit(minigame_region, zone_mask | ind_mask)
    assert digit == "2", f"citit {digit!r} (scor {score:.2f})"


# ── Rotații sintetice ────────────────────────────────────────────────────────
def _synthetic(zone_deg, ind_deg, size=160, radius=60, half_deg=18):
    """Cadru cu o zonă teal la zone_deg (±half_deg) și un indicator la ind_deg (y în jos)."""
    img = np.zeros((size, size, 3), np.uint8)
    cx = cy = (size - 1) / 2.0
    for d in np.arange(zone_deg - half_deg, zone_deg + half_deg, 1.5):
        a = math.radians(d)
        x, y = int(cx + radius * math.cos(a)), int(cy + radius * math.sin(a))
        cv2.circle(img, (x, y), 4, busteni.ZONE_RGB[::-1], -1)  # BGR
    # indicator: punct îngust
    a = math.radians(ind_deg)
    x, y = int(cx + radius * math.cos(a)), int(cy + radius * math.sin(a))
    cv2.circle(img, (x, y), 4, busteni.IND_RGB[::-1], -1)
    return img


def test_zone_center_robust_to_indicator_bleed():
    """Pete „zonă" lângă indicator (anti-aliasing) NU trebuie să mute centrul zonei."""
    frame = _synthetic(zone_deg=90, ind_deg=0)
    # adăugăm pete de culoarea zonei chiar lângă indicator (la 0°)
    cx = cy = (frame.shape[0] - 1) / 2.0
    import math as _m
    for d in (-4, 0, 4):
        a = _m.radians(d)
        x, y = int(cx + 60 * _m.cos(a)), int(cy + 60 * _m.sin(a))
        cv2.circle(frame, (x, y), 2, busteni.ZONE_RGB[::-1], -1)
    st = busteni.detect_state(frame)
    # centrul zonei rămâne jos (≈90°), nu e tras spre 0°
    assert abs(busteni.ang_diff(st["zone_center"], _m.pi / 2)) < _m.radians(30)


def test_no_misfire_when_indicator_far_despite_bleed():
    """Reproduce bug-ul: indicator la 0°, zona jos → NU declanșează."""
    s = busteni.BustenSession()
    s.start(0.0)
    s.cur_digit = "1"
    s._digit_locked = True
    # mișcare rapidă spre zonă, dar indicatorul e încă la 90° distanță
    s.process(_synthetic(90, -20), 0.00)
    action = s.process(_synthetic(90, 0), 0.02)  # viteză mare, dar departe de zonă
    assert action is None


def _sweep_until_press(s, zone_deg, start, stop, step=8, t0=0.0, dt=0.03):
    """Mătură indicatorul prin zonă (mișcare continuă realistă) și întoarce acțiunea."""
    for k, idg in enumerate(range(start, stop + 1, step)):
        a = s.process(_synthetic(zone_deg, idg), t0 + k * dt)
        if a is not None:
            return a
    return None


def test_trigger_fires_when_indicator_on_zone():
    s = busteni.BustenSession()
    s.start(0.0)
    s.cur_digit = "5"  # sărim peste OCR în acest test sintetic
    s._digit_locked = True
    s.process(_synthetic(90, 40), 0.0)  # citire curată a zonei (indicator departe)
    # indicatorul mătură prin zona de jos → trebuie să apese
    assert _sweep_until_press(s, 90, 48, 120, t0=0.04) == ("press", "5")


def test_no_trigger_when_indicator_far_from_zone():
    frame = _synthetic(zone_deg=90, ind_deg=-90)
    s = busteni.BustenSession()
    s.start(0.0)
    s.cur_digit = "5"
    s._digit_locked = True
    assert s.process(frame, 0.0) is None


def test_moving_indicator_fires_near_zone_edge():
    """Indicatorul care mătură intră în zonă → apasă pe la marginea ei (lead mic)."""
    s = busteni.BustenSession()
    s.start(0.0)
    s.cur_digit = "7"
    s._digit_locked = True
    s.process(_synthetic(90, 50, half_deg=15), 0.0)  # citire curată a zonei
    assert _sweep_until_press(s, 90, 58, 110, step=4, t0=0.04) == ("press", "7")


def test_no_fire_far_below_zone():
    """La 60° (zonă ±15°, centru 90°) indicatorul e clar în afara zonei → NU apasă."""
    s = busteni.BustenSession()
    s.start(0.0)
    s.cur_digit = "7"
    s._digit_locked = True
    assert s.process(_synthetic(90, 60, half_deg=15), 0.0) is None


def test_one_press_per_round_then_rearm_on_zone_jump():
    s = busteni.BustenSession()
    s.start(0.0)
    s.cur_digit = "3"
    s._digit_locked = True
    s.process(_synthetic(90, 40), 0.0)  # citire curată zonă jos
    assert _sweep_until_press(s, 90, 48, 120, t0=0.04) == ("press", "3")
    # același loc → nu mai apasă (o singură apăsare per rundă)
    assert s.process(_synthetic(90, 90), 1.0) is None
    # rundă nouă: zona sare la 0°, indicator departe (fără teleport de viteză)
    s.cur_digit = None
    s._digit_locked = False
    s.process(_synthetic(0, -52), 1.1)  # detectează saltul + citire curată
    s.cur_digit = "8"
    s._digit_locked = True
    assert _sweep_until_press(s, 0, -44, 44, t0=1.2) == ("press", "8")


def test_second_round_fires_when_zone_lands_near_previous():
    """Bug real: runda 2 cu zona la 9° de runda 1 (saltul NU declanșează) → tot apasă."""
    s = busteni.BustenSession()
    s.start(0.0)
    s.cur_digit = "3"
    s._digit_locked = True
    s.process(_synthetic(69, 0), 0.0)  # citire curată runda 1
    assert _sweep_until_press(s, 69, 12, 96, t0=0.04) == ("press", "3")
    # indicatorul continuă și PĂRĂSEȘTE zona → re-armare
    s.process(_synthetic(69, 170), 0.5)
    assert s.round_pressed is False
    # runda 2: zona la 60° (doar 9° față de 69) și cifră nouă
    s.cur_digit = "2"
    s._digit_locked = True
    assert _sweep_until_press(s, 60, 0, 96, t0=0.6) == ("press", "2")


def test_digit_lock_holds_against_later_reads():
    """Citiri eronate după blocare NU schimbă cifra aleasă."""
    s = busteni.BustenSession()
    s.start(0.0)
    s.cur_digit = "2"
    s._digit_locked = True
    s._votes = {"9": 20}
    assert s.cur_digit == "2"


def test_session_auto_stops():
    gap = busteni.NO_DETECT_SEC

    # niciun teal vreodată → no-show după 5s
    s = busteni.BustenSession()
    s.start(100.0)
    assert s.should_stop(100.0 + gap - 0.5) is None
    assert s.should_stop(100.0 + gap + 0.5) == "no-show"

    # teal văzut, apoi dispărut > 5s → done
    s2 = busteni.BustenSession()
    s2.start(0.0)
    s2.seen = True
    s2.last_teal_t = 50.0
    assert s2.should_stop(50.0 + gap - 0.3) is None
    assert s2.should_stop(50.0 + gap + 0.3) == "done"


def test_session_runs_long_while_detected():
    """Cât timp se detectează (last_teal_t se reîmprospătează), NU se oprește — oricât."""
    s = busteni.BustenSession()
    s.start(0.0)
    s.seen = True
    # simulăm detecție continuă timp de 60s (peste vechiul plafon de 11s)
    for t in range(0, 60):
        s.last_teal_t = float(t)
        assert s.should_stop(float(t) + 0.1) is None
