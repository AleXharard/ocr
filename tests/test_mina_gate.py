"""Teste MinaClickGate — 3 click-uri per piatră."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mina  # noqa: E402


def _match(cx: int, cy: int, score: float = 0.9) -> dict:
    return {"center": (cx, cy), "score": score}


def test_three_clicks_same_stone():
    gate = mina.MinaClickGate()
    t0 = time.perf_counter()
    stone = _match(500, 400)

    assert gate.should_click(stone, t0) is True
    assert gate.on_click((500, 400), t0) == 1

    assert gate.should_click(stone, t0 + 0.01) is False

    t2 = t0 + mina.CLICK_GAP_SEC + 0.02
    assert gate.should_click(stone, t2) is True
    assert gate.on_click((500, 400), t2) == 2

    t3 = t2 + mina.CLICK_GAP_SEC + 0.02
    assert gate.should_click(stone, t3) is True
    assert gate.on_click((500, 400), t3) == 3

    assert gate.should_click(stone, t3 + 0.03) is False


def test_reset_after_stone_gone():
    gate = mina.MinaClickGate()
    t0 = time.perf_counter()
    stone = _match(500, 400)

    for i in range(mina.STAGES_PER_STONE):
        t = t0 + i * (mina.CLICK_GAP_SEC + 0.03)
        gate.on_click((500, 400), t)

    assert gate.stage == mina.STAGES_PER_STONE
    t_done = t0 + 1.0
    gate.should_click(None, t_done)
    gate.should_click(None, t_done + mina.REARM_ABSENT_SEC + 0.05)
    assert gate.stage == 0

    assert gate.should_click(_match(700, 500), t_done + 1.5) is True


def test_no_spam_during_gap():
    gate = mina.MinaClickGate()
    t0 = time.perf_counter()
    gate.on_click((500, 400), t0)
    stone = _match(500, 400)
    for dt in (0.005, 0.01, 0.015):
        assert gate.should_click(stone, t0 + dt) is False


def test_second_stone_different_position():
    """După 3/3, piatra nouă în altă parte → click imediat."""
    gate = mina.MinaClickGate()
    t0 = time.perf_counter()
    for i in range(mina.STAGES_PER_STONE):
        gate.on_click((500, 400), t0 + i * (mina.CLICK_GAP_SEC + 0.03))

    assert gate.stage == mina.STAGES_PER_STONE
    assert gate.dead_zone() == (500, 400)
    t_last = t0 + (mina.STAGES_PER_STONE - 1) * (mina.CLICK_GAP_SEC + 0.03)
    t_new = t_last + mina.CLICK_GAP_SEC + 0.02
    new_stone = _match(800, 600)
    assert gate.should_click(new_stone, t_new) is True
    assert gate.on_click((800, 600), t_new) == 1
    assert gate.dead_zone() is None


def test_sequence_stops_after_three_stones():
    """3 pietre × 3 click-uri → secvență completă, fără click-uri suplimentare."""
    gate = mina.MinaClickGate()
    t0 = time.perf_counter()
    stones = [(500, 400), (800, 600), (300, 300)]
    t = t0
    for si, (x, y) in enumerate(stones):
        if si > 0:
            gate.reset()
        for _ in range(mina.STAGES_PER_STONE):
            t += mina.CLICK_GAP_SEC + 0.02
            stage = gate.on_click((x, y), t)
        assert gate.stones_done == si + 1
        assert stage == mina.STAGES_PER_STONE

    assert gate.sequence_complete
    assert gate.should_click(_match(100, 100), t + 1) is False


def test_hard_stop_at_nine_clicks():
    """Siguranță: după 9 click-uri total, secvența e completă."""
    assert mina.CLICKS_PER_SEQUENCE == 9
    gate = mina.MinaClickGate()
    t0 = time.perf_counter()
    t = t0
    for i in range(9):
        t += mina.CLICK_GAP_SEC + 0.02
        if i % mina.STAGES_PER_STONE == 0 and i > 0:
            gate.reset()
        gate.on_click((500 + i * 10, 400), t)
    assert gate.sequence_complete

