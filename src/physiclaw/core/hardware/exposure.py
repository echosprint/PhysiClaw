"""Software exposure convergence — the fallback when firmware AE fails.

OpenCV cannot reliably control camera exposure cross-platform: property
`set()`/`get()` are driver lotteries, and encodings differ per backend.
So this module trusts only measured pixels. `converge` is pure logic over
three injected callables — `meter` (capture + assess the phone-screen
crop), `set_auto`, `set_manual` — so the whole search is unit-testable
with a fake camera.

The acceptance band reuses `quality.py`'s verdicts: a frame this tune
accepts (`not blown`) is by construction one the runtime QualityMonitor
won't immediately warn about. Only a "too dark" floor is added here —
quality.py has no dark verdict (a dark frame can be a sleeping phone at
runtime), but during a tune WE made it dark, so it's an overshoot.

Every failure path ends in `set_auto()` + a descriptive TuneResult
(unless the user pinned manual in config): firmware AE plus the runtime
⚠ warnings beat a bad frozen manual value.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Literal

from physiclaw.core.vision.quality import BLOWN_MEDIAN_LUMA, QualityReport

log = logging.getLogger(__name__)

# Frames the caller's meter must let pass after each property set —
# drivers apply exposure changes with latency, and metering the pre-set
# frame would misread every step.
SETTLE_FRAMES = 5

# Manual-stepping bounds, log2-seconds scale (-11 ≈ 0.5ms, -2 = 250ms).
# This is the shared contract for `set_manual` values: Windows passes it
# raw (the DirectShow CameraControl_Exposure unit); Linux converts to
# V4L2 ~100µs ticks (one stop = one halving). Values outside are dark
# noise / motion-blur territory.
MIN_EXPOSURE = -11
MAX_EXPOSURE = -2
MAX_STEPS = 6

# Below this median luma the screen content is crushed to black — the
# manual-stepping overshoot floor (see module docstring for why this
# lives here and not in quality.py).
DARK_MEDIAN_LUMA = 25.0

# A set that changes measured median luma by less than this did nothing:
# the driver is ignoring exposure writes (common — see opencv#9738), and
# further stepping is theater. One log2 stop should move luma far more.
LUMA_STALL_EPSILON = 3.0


@dataclass(frozen=True)
class TuneResult:
    """Outcome of one convergence run — for the log, and for tests."""

    mode: Literal["auto", "manual"]
    exposure: int | None  # the held manual value; None in auto mode
    ok: bool  # True = final metered frame is in band
    detail: str  # human line for the tune log


def _good(r: QualityReport) -> bool:
    return not r.blown and r.median_luma >= DARK_MEDIAN_LUMA


def converge(
    meter: Callable[[], QualityReport | None],
    set_auto: Callable[[], None],
    set_manual: Callable[[int], None],
    *,
    start: int,
    max_steps: int = MAX_STEPS,
    prefer_auto: bool = True,
) -> TuneResult:
    """Verify exposure by metering; converge manually only if AE fails.

    Phases:
      1. Meter as-is — in band means firmware AE is doing its job
         (the healthy-rig path on every OS): touch nothing.
      2. (`prefer_auto` only) re-assert AE and re-meter — recovers the
         driver-left-AE-off-after-renegotiation case without going manual.
      3. Integer-step a manual exposure from `start`: darker while
         blown/too-bright, brighter while too dark. Stops on: in band
         (success), stalled luma (driver ignores writes), two direction
         flips (band lies between two integer steps — keep the darkest
         non-blown step tried), range exhausted, or `max_steps`.

    A `meter` returning None (no frame) fails open immediately. With
    `prefer_auto=False` (user pinned manual in config) phase 2 is
    skipped and failure keeps the best manual step instead of reverting
    to auto.
    """
    r = meter()
    if r is None:
        return TuneResult("auto", None, False, "no frame — exposure left as-is")
    if _good(r):
        return TuneResult(
            "auto",
            None,
            True,
            f"in band as-is (median {r.median_luma:.0f}, clip {r.clip_pct:.0%})",
        )

    if prefer_auto:
        set_auto()
        r = meter()
        if r is None:
            return TuneResult("auto", None, False, "no frame after AE re-assert")
        if _good(r):
            return TuneResult(
                "auto",
                None,
                True,
                f"recovered by AE re-assert (median {r.median_luma:.0f})",
            )

    # Phase 3: manual stepping. On the log2 scale, lower = darker.
    exp = max(MIN_EXPOSURE, min(MAX_EXPOSURE, start))
    prev_luma = r.median_luma
    last_dir = 0
    flips = 0
    best: int | None = None  # darkest non-blown exposure tried
    reason = f"no progress within {max_steps} steps"
    for step in range(1, max_steps + 1):
        set_manual(exp)
        r = meter()
        if r is None:
            reason = "meter lost frames mid-stepping"
            break
        if abs(r.median_luma - prev_luma) < LUMA_STALL_EPSILON:
            reason = f"driver ignores exposure writes (luma stuck ~{r.median_luma:.0f})"
            break
        prev_luma = r.median_luma
        if _good(r):
            return TuneResult(
                "manual",
                exp,
                True,
                f"converged at {exp} in {step} step(s) "
                f"(median {r.median_luma:.0f}, clip {r.clip_pct:.0%})",
            )
        if not r.blown and (best is None or exp < best):
            best = exp
        direction = -1 if (r.blown or r.median_luma > BLOWN_MEDIAN_LUMA) else 1
        if last_dir and direction != last_dir:
            flips += 1
            if flips >= 2:
                if best is not None:
                    set_manual(best)
                    return TuneResult(
                        "manual",
                        best,
                        False,
                        f"band between steps — held darkest non-blown {best}",
                    )
                reason = "oscillating with no non-blown step"
                break
        last_dir = direction
        nxt = max(MIN_EXPOSURE, min(MAX_EXPOSURE, exp + direction))
        if nxt == exp:
            reason = f"exposure range exhausted at {exp}"
            break
        exp = nxt

    if not prefer_auto and best is not None:
        set_manual(best)
        return TuneResult(
            "manual",
            best,
            False,
            f"{reason} — held best manual {best} (auto disabled in config)",
        )
    set_auto()
    return TuneResult("auto", None, False, f"{reason} — reverted to auto")
