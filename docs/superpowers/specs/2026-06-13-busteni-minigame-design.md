# Busteni minigame — design

Date: 2026-06-13

## Problem

The app currently automates one FiveM minigame: a row of white boxes, each with a
letter/digit, pressed left-to-right (the **Chei** game — `detect_white_boxes` +
`vision.py` glyph bank + `scan_and_press`).

A second, unrelated minigame — **Busteni** (logs/timber; forest scene) — needs
automating too. It is a *timing* game, not a sequence game:

- A ring sits on screen with a single **digit 1-9** in the centre.
- A **target zone** arc (`#068f6d`, darker teal) appears at a **random position**
  on the ring each round.
- An **indicator** (`#38d5af`, brighter teal) sweeps around the ring.
- You press the centre digit **the instant the indicator crosses the zone**.
- Rounds come in **random batches**: after a correct press, a new round may pop
  (new digit, new zone position) until the batch finishes. A batch lasts ~10 s.

A bright-green spinning **checkpoint** (`#01b802`) may also be on screen and must be
**ignored entirely**. It contains zero teal — that is the discriminator.

Colours may shift slightly with resolution/gamma, so matching uses tolerance.

## Goals / non-goals

- Add Busteni **without touching** the Chei path. `vision.py`, the glyph bank,
  `detect_white_boxes`, `scan_and_press` are unchanged.
- A **Chei / Busteni segmented toggle** selects the active game; Start/F6 route to it.
- Busteni runs as a **self-stopping single session** per F6 — no perpetual loop
  (performance is critical for hitting the timing window).
- Non-goal: auto-detecting which game is on screen. The user picks via the toggle.

## Architecture

New isolated module **`busteni.py`** holding all detection + timing. `main.py` gains
a mode toggle and a busteni session loop that reuses existing capture (`grab_screen`),
input (`game_input`), and the `vision` glyph bank (read-only).

### Detection (`busteni.py`)

The user boxes the region tightly around the circle, so **ring centre = region
centre**.

1. `_color_masks(frame_bgr)` → `(zone_mask, ind_mask)`: per-pixel Euclidean distance
   in RGB to the zone and indicator references; a pixel is teal if it is within
   `COLOR_TOL` of either *and* closer to teal than to the checkpoint green (so the
   checkpoint is excluded). Each teal pixel is assigned to whichever teal it is
   nearer.
2. `detect_state(frame_bgr)` → dict: `has_teal`, `center`, `zone_center` (circular
   mean angle of zone pixels), `zone_half` (95th-percentile angular half-width),
   `ind_angle` (circular mean of indicator pixels), pixel counts. `has_teal` is False
   when total teal pixels < `MIN_TEAL_PX` (no minigame / checkpoint only) → nothing
   happens.
3. `read_digit(frame_bgr, teal_mask)` → `(digit, score)`: central ROI, teal pixels
   blanked, grayscale **inverted** (busteni digits are white-on-dark, the opposite of
   Chei's dark-on-white), then the `vision` glyph bank restricted to **digits 1-9**.
   RapidOCR is the fallback.

### Timing (`busteni.BustenSession`)

Stateful, fed one frame at a time:

- Tracks recent indicator angles → angular **velocity**.
- **Trigger**: predicted indicator angle (`ind_angle + velocity * LEAD_SEC`) falls
  within the zone arc (`|Δ| ≤ zone_half`). The lead compensates input latency so the
  press lands on-zone rather than late.
- **One press per round**, then debounced. A **new round** is detected when the zone
  jumps angularly (> ~25°), which re-arms and re-reads the digit.

### Auto-stop (session lifecycle)

F6 in-game (or Start) begins **one** session. The fast loop runs only during a
session. It ends on the first of:

- **done** — teal seen, then absent for `GONE_GRACE_SEC` (~0.5 s): batch finished.
- **no-show** — no teal within `NO_SHOW_SEC` (~3 s): nothing appeared.
- **cap** — `HARD_CAP_SEC` (~15 s) absolute ceiling.

Then idle (zero work) until the next F6.

### UI (`main.py`)

- `CTkSegmentedButton` "Chei / Busteni" at the top of the control card.
- Busteni mode: Start label becomes "Arm (F6)"; the Auto switch is hidden (continuous
  looping is the thing we are avoiding). Chei mode unchanged.
- `_hotkey_trigger` / Start route to `_run_busteni` or `_run_once` by mode.
- Busteni loop focuses the game once (`fast=True`), then grabs the region and feeds
  frames to `BustenSession`, pressing the returned digit. No per-press refocus
  (latency). App stays in the background; capture is screen-region based.

## Testing

- Unit tests in `tests/test_busteni.py` against the real screenshot
  (`013920` minigame, `013842` checkpoint): centre, digit = "2", zone arc present,
  indicator angle, `has_teal` True for the minigame and **False** for the checkpoint.
- Synthetic rotation tests: indicator placed inside vs outside the zone → trigger
  fires / does not; velocity-lead trigger fires slightly early.
- Live end-to-end timing is confirmed in-game by the user.

## Risks

- Digit font differs from the Chei glyph bank → if template scores are weak, RapidOCR
  fallback covers it; tests verify the real "2" reads correctly.
- Loose region selection shifts the assumed centre. Mitigation: centre is robust to
  small offsets because triggering uses *relative* angles; gross mis-selection is a
  user error surfaced by no/!wrong detection.
