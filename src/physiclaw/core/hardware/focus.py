"""Software focus lock — freeze autofocus once it has converged.

The rig is rigid: the camera is bolted at a fixed position and the
phone screen sits at a constant distance, so the correct lens position
is a constant of the rig — a calibration value, like the affine
transforms. `lock` runs ONCE, during the camera-mapping calibration
step (`camera_map._pin_focus`), against the dedicated ``focus`` page:
a full-screen checkerboard, the one scene where the absolute sharpness
gate is trustworthy. The other calibration pages are mostly flat and
meter far below the gate even in perfect focus on a far camera — the
corners page measured 14-34 against the 80 gate on a rig at minimum
phone coverage, while the checkerboard meters like session content
(hundreds). The verified position is read back, re-applied and
re-verified by pixels, then persisted in the calibration bundle and
seeded into every fresh `Camera`
(`rig.connect_camera`); `configure_capture` — the remembered-state
choke point — re-applies it at the first open and every reconnect.
Nothing re-locks at runtime: continuous AF
hunts (the stylus crossing the lens re-triggers one after every
gesture) can't happen under a pinned lens, and persistent blur means
the rig physically changed — recalibrate, don't adjust.

Like `exposure.converge`, `lock` is pure logic over injected
callables — `meter` (capture + score the focus page's
sharpness), `freeze`, `unfreeze` — so the whole flow is unit-testable
with a fake camera. And like exposure, it trusts only measured
pixels: focus writes can be silently ignored just as exposure writes
are, and a UVC GET_CUR readback can be junk (see uvc.py's measured
firmware quirks) — the freeze is judged by the sharpness of the frame
it produced, never by the driver's word.

A scene that never meters sharp is no reference to lock against: the
lens may be mid-hunt, or the screen may be dark/blank with nothing to
focus on — the lock fails, AF stays in charge, and no position is
persisted. A freeze whose verification meter comes back soft is
reverted (`unfreeze`): firmware AF beats a lens frozen mid-hunt.
"""

import logging
from dataclasses import dataclass
from typing import Callable

# The pin gate deliberately rides the session gate: calibration proves
# the lens clears the same number every session judges views by, and an
# optics re-tune of `[vision] blur_threshold` moves both scenes
# together. The checkerboard's wide margin (hundreds vs 80) keeps that
# coupling safe.
from physiclaw.core.vision.quality import BLUR_THRESHOLD

log = logging.getLogger(__name__)

# Frames the caller's meter must let pass before each read — the AF
# loop iterates per frame, so consecutive meters separated by a few
# frames sample distinct lens positions instead of one.
SETTLE_FRAMES = 5

# Bounded settle loop: the scene must meter sharp on two CONSECUTIVE
# reads before the freeze — one sharp read can be the instant an AF
# hunt sweeps through focus. Sharpness clears BLUR_THRESHOLD with a
# wide margin on a focused phone-screen crop (300+ vs the 80 floor),
# so two agreeing reads mean converged, not lucky.
CONFIRM_METERS = 4


@dataclass(frozen=True)
class LockResult:
    """Outcome of one lock attempt — for the log, and for tests."""

    locked: bool  # True = AF frozen and the frozen frame metered sharp
    detail: str  # human line for the lock log


def lock(
    meter: Callable[[], float | None],
    freeze: Callable[[], bool],
    unfreeze: Callable[[], None],
) -> LockResult:
    """Freeze autofocus at its converged position; verify by pixels.

    Phases:
      1. Wait for AF to settle: meter until two consecutive reads are
         sharp (>= BLUR_THRESHOLD), bounded by CONFIRM_METERS. A scene
         that never gets there — dark screen, AF still hunting — fails
         the lock; AF stays in charge.
      2. Freeze. `freeze` disables continuous AF (and pins the current
         absolute focus where the channel allows). A refused freeze
         fails open — AF stays in charge, no retry loop.
      3. Verify from the frozen frame: still sharp means the lens was
         held where AF converged; a collapse means the freeze wedged a
         mid-hunt position — `unfreeze` and leave AF in charge.

    A `meter` returning None (no frame) fails open immediately.
    """
    prev: float | None = None
    for _ in range(CONFIRM_METERS):
        cur = meter()
        if cur is None:
            return LockResult(False, "no frame — focus left on auto")
        if prev is not None and prev >= BLUR_THRESHOLD and cur >= BLUR_THRESHOLD:
            break
        prev = cur
    else:
        return LockResult(
            False,
            f"scene never metered sharp (last {cur:.0f} < "
            f"{BLUR_THRESHOLD:.0f}) — no converged focus to freeze",
        )
    if not freeze():
        return LockResult(False, "driver refused the focus lock — AF left in charge")
    after = meter()
    if after is None or after < BLUR_THRESHOLD:
        unfreeze()
        got = "no frame" if after is None else f"sharpness {after:.0f}"
        return LockResult(False, f"frozen frame soft ({got}) — reverted to autofocus")
    return LockResult(True, f"focus locked (sharpness {after:.0f})")
