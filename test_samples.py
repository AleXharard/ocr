"""Test OCR pe toate pozele din folderul POZE/."""

import sys
from pathlib import Path

import cv2

import main as app

POZE_DIR = Path(__file__).resolve().parent / "POZE"
ASSETS = Path(r"C:\Users\orbis\.cursor\projects\c-Users-orbis-tot-nou\assets")


def crop_key_band(frame):
    """Decupează zona centrală unde apare randul de casete (poze fullscreen)."""
    h, w = frame.shape[:2]
    return frame[int(h * 0.36) : int(h * 0.54), int(w * 0.08) : int(w * 0.92)]

# Poze vechi din assets (optional)
LEGACY = [
    ("assets-1", "UUC0IZC7VVXVH6NJ", ASSETS / "c__Users_orbis_AppData_Roaming_Cursor_User_workspaceStorage_empty-window_images_image-e0439954-455b-486b-bcbc-e68132d65858.png"),
    ("assets-2", "Y5KFWD7OT5N0JTLTZF", ASSETS / "c__Users_orbis_AppData_Roaming_Cursor_User_workspaceStorage_empty-window_images_image-91210d1d-b757-4e57-a575-c096afe7bf3c.png"),
    ("assets-3", "5FMWA5LIWW0Q3K21Y", ASSETS / "c__Users_orbis_AppData_Roaming_Cursor_User_workspaceStorage_empty-window_images_image-fadf5170-757b-429e-a586-e538412c6883.png"),
    ("assets-4", "76QC5G1HEGZ38BS5I", ASSETS / "c__Users_orbis_AppData_Roaming_Cursor_User_workspaceStorage_empty-window_images_image-112c8232-b314-42d2-8bd4-596b0555367e.png"),
]


def load_samples() -> list[tuple[str, str, Path, bool]]:
    samples: list[tuple[str, str, Path, bool]] = []

    if POZE_DIR.is_dir():
        for path in sorted(POZE_DIR.glob("*.png")):
            expected = path.stem.upper()
            samples.append((path.name, expected, path, True))

    for name, expected, path in LEGACY:
        if path.exists():
            samples.append((name, expected, path, False))

    return samples


def run_test(name: str, expected: str, path: Path, from_poze: bool = False) -> bool:
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"[SKIP] {name} — nu pot citi imaginea")
        return False

    if from_poze:
        frame = crop_key_band(frame)

    boxes = app.detect_white_boxes(frame)
    chars, mode = app.read_sequence(frame, boxes)
    got = "".join(chars)
    match = got == expected

    status = "OK" if match else "FAIL"
    print(f"[{status}] {name} · {len(boxes)} casete · {mode}")
    print(f"       asteptat: {expected}")
    print(f"       detectat: {got}")
    if not match:
        width = max(len(expected), len(got))
        got_pad = got.ljust(width)
        diff = "".join("^" if a != b else " " for a, b in zip(expected.ljust(width), got_pad))
        print(f"       diff:     {diff}")
    print()
    return match


def main() -> int:
    app._init_ocr()
    samples = load_samples()

    if not samples:
        print(f"Nicio poza gasita in {POZE_DIR}")
        return 1

    print(f"Test OCR · {len(samples)} poze\n" + "-" * 50)

    ok = sum(run_test(name, expected, path, from_poze) for name, expected, path, from_poze in samples)
    total = len(samples)
    print(f"Rezultat: {ok}/{total} corecte")
    return 0 if ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
