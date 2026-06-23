"""Analyze busteni debug session digit reads."""
import math
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import busteni  # noqa: E402


def analyze(sess: Path) -> None:
    print(f"\n=== {sess.name} ===")
    votes: dict[str, int] = {}
    scores: dict[str, list[float]] = {}
    for p in sorted(sess.glob("*.png")):
        if "_raw" in p.name:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        st = busteni.detect_state(img)
        if not st.get("has_teal"):
            continue
        mask = st["zone_mask"] | st["ind_mask"]
        d, s = busteni.read_digit(img, mask)
        ia = st.get("ind_angle")
        zc = st.get("zone_center")
        zh = st.get("zone_half")
        gap = abs(busteni.ang_diff(ia, zc)) if ia is not None and zc is not None else None
        if d:
            votes[d] = votes.get(d, 0) + 1
            scores.setdefault(d, []).append(s)
        tag = p.stem.split("_", 1)[-1]
        if tag in ("first", "teal", "FIRE") or "scan" in tag:
            gap_d = f"{math.degrees(gap):.0f}" if gap is not None else "-"
            print(f"  {p.name}: d={d} sc={s:.3f} ind-zone={gap_d}°")

    if votes:
        best = max(votes, key=votes.get)
        print(f"  votes={votes} majority={best}")
        for k, v in scores.items():
            print(f"    {k}: n={len(v)} avg={sum(v)/len(v):.3f}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "debug_busteni"
    targets = sys.argv[1:] or ["sess_151127", "sess_151145", "sess_151002"]
    for name in targets:
        analyze(root / name)
