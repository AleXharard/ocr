"""Teste Mina — detectare piatră pe capturi reale din poze mina/."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mina  # noqa: E402

POZE = Path(__file__).resolve().parents[1] / "poze mina"

# Centru așteptat (x, y) pe fiecare captură 1024×576 — calibrat cu template alpha.
EXPECT_STONE: dict[str, tuple[int, int]] = {
    "ecran_01.png": (534, 265),
    "ecran_02.png": (361, 175),
    "ecran_03.png": (767, 142),
    "ecran_04.png": (767, 142),
    "ecran_05.png": (573, 358),
    "minigame_mina.png": (385, 336),
    "piatra_mai_mica.png": (361, 176),
    "piatra_mica.png": (573, 358),
    "piatra_pe_mijloc.png": (534, 264),
}

CENTER_TOL = 28


def _load(name: str):
    path = POZE / name
    if not path.exists():
        pytest.skip(f"lipsește {name}")
    img = cv2.imread(str(path))
    if img is None:
        pytest.skip(f"imagine invalidă: {name}")
    return img


@pytest.fixture(scope="module", autouse=True)
def _warm_template():
    mina._load_template()


@pytest.mark.parametrize("filename,expected", list(EXPECT_STONE.items()))
def test_finds_stone_on_screenshots(filename: str, expected: tuple[int, int]):
    frame = _load(filename)
    hit = mina.find_stone(frame)
    assert hit is not None, f"piatra nedetectată în {filename}"
    assert hit["score"] >= mina.MATCH_THRESHOLD
    cx, cy = hit["center"]
    ex, ey = expected
    assert abs(cx - ex) <= CENTER_TOL, f"{filename}: center ({cx},{cy}) vs ({ex},{ey})"
    assert abs(cy - ey) <= CENTER_TOL


def test_no_stone_in_chat_region():
    """Chat-ul din stânga-sus nu trebuie confundat cu piatra."""
    frame = _load("ecran_01.png")
    h, w = frame.shape[:2]
    chat = frame[0: int(h * 0.28), 0: int(w * 0.32)].copy()
    assert mina.find_stone(chat) is None


def test_no_stone_on_minimap():
    """Minimap-ul din stânga-jos nu trebuie confundat cu piatra."""
    frame = _load("ecran_02.png")
    h, w = frame.shape[:2]
    minimap = frame[int(h * 0.72):, 0: int(w * 0.22)].copy()
    assert mina.find_stone(minimap) is None


def test_no_stone_on_radio_list():
    """Lista radio / anunț din dreapta nu trebuie confundată cu piatra."""
    frame = _load("ecran_03.png")
    h, w = frame.shape[:2]
    radio = frame[:, int(w * 0.78):].copy()
    assert mina.find_stone(radio) is None


def test_template_file_exists():
    assert mina.TEMPLATE_PATH.exists()


def test_search_mask_covers_center():
    mask = mina.build_search_mask(576, 1024)
    assert mask[288, 512] == 1
    assert mask[50, 50] == 0      # chat
    assert mask[520, 50] == 0     # minimap
    assert mask[300, 950] == 0    # radio
