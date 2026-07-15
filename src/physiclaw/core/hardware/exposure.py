"""Software exposure convergence — the fallback when firmware AE fails.

OpenCV cannot reliably control camera exposure cross-platform: property
`set()`/`get()` are driver lotteries, and encodings differ per backend.
So this module trusts only measured pixels. `converge` is pure logic over
three injected callables — `meter` (capture + assess the phone-screen
crop), `set_auto`, `set_manual` — so the whole search is unit-testable
with a fake camera.

Acceptance targets highlight headroom, not a median band: a phone
screen's white level is constant at fixed brightness, so the one manual
exposure that renders white *just below* clipping stays correct for any
screen content — dark UIs simply have fewer lit pixels. Concretely: the
crop must have (almost) no clipped pixels and clear quality.py's dark
floor. Tuning to "median looks nice" on whatever happens to be showing
is exactly how a dark lock screen freezes a value that blows out the
home screen after unlock.

For the same reason, a pitch-dark crop is not a reference at all (phone
asleep / lock screen): the tune DEFERS — reports `deferred=True` and
leaves firmware AE in charge — instead of converging against blackness.
The caller retries when a bright view shows up.

Every other failure path ends in `set_auto()` + a descriptive TuneResult
(unless the user pinned manual in config): firmware AE plus the runtime
⚠ warnings beat a bad frozen manual value.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Literal

from physiclaw.core.vision.quality import (
    BLOWN_BLOB_COUNT,
    BLOWN_CLIP_PCT,
    QualityReport,
)

log = logging.getLogger(__name__)

# Frames the caller's meter must let pass after each property set —
# drivers apply exposure changes with latency, and metering the pre-set
# frame would misread every step.
SETTLE_FRAMES = 5

# Manual-stepping bounds, log2-seconds scale (-8 ≈ 3.9ms, -5 = 31.2ms).
# This is the shared contract for `set_manual` values: Windows passes it
# raw (the DirectShow CameraControl_Exposure unit); Linux converts to
# V4L2 ~100µs ticks (one stop = one halving).
#
# The window is sized for filming an OLED phone screen at mid brightness:
# iPhone panels PWM-dim at ~240-480Hz, and an exposure that doesn't
# average several PWM cycles records banding — below ~4ms is banding
# territory. The ceiling keeps the stream at full frame rate: a 30fps
# sensor can't expose past ~33ms without halving its rate, and
# contrast-detect autofocus iterates per frame — measured on the bench
# rig, a 62.5ms hold dropped the stream to ~16fps and every post-gesture
# AF hunt outlived the 2s view settle, blurring a whole session. The -6
# default start (15.6ms ≈ 1/60s) is the textbook screen-filming shutter.
MIN_EXPOSURE = -8
MAX_EXPOSURE = -5
MAX_STEPS = 6

# Below this median luma the screen content is crushed to black — the
# manual-stepping overshoot floor (see module docstring for why this
# lives here and not in quality.py).
DARK_MEDIAN_LUMA = 25.0

# Acceptance ceiling for clipped pixels during a tune — much tighter than
# quality.py's 12% warning threshold. Near-zero clip on the crop IS the
# highlight-headroom criterion: the screen's whites render just below
# 250, so no content can blow out later. Not exactly zero — sensor
# sparkle and the phone's specular bezel edge leak a few clipped pixels
# into any crop.
TUNE_CLIP_PCT = 0.02

# Below this median the screen itself is dark (asleep, lock screen at
# rest) — there is nothing to expose FOR, and any value converged here
# would be wildly hot for a lit screen. The tune defers instead.
DARK_REFERENCE_LUMA = 15.0

# A set that changes measured median luma by less than this did nothing:
# the driver is ignoring exposure writes (common — see opencv#9738), and
# further stepping is theater. One log2 stop should move luma far more.
LUMA_STALL_EPSILON = 3.0

# After re-asserting auto-exposure the firmware loop re-converges over
# tens of frames (Imatest measures AE settle in ~1/3s windows) — one
# post-set meter can catch it mid-swing. Meter until two consecutive
# reads agree within the tolerance, bounded by this many meters.
AE_SETTLE_METERS = 4
AE_SETTLE_TOLERANCE = 0.05


@dataclass(frozen=True)
class TuneResult:
    """Outcome of one convergence run — for the log, and for tests."""

    mode: Literal["auto", "manual"]
    exposure: int | None  # the held manual value; None in auto mode
    ok: bool  # True = final metered frame is in band
    detail: str  # human line for the tune log
    # True = the screen was dark, so there was no reference to tune
    # against — the caller should retry once a bright view arrives.
    deferred: bool = False


def _good(r: QualityReport) -> bool:
    """In band: highlight headroom (near-zero clip) + readable shadows.
    `not blown` folds in every failure signature quality.py knows
    (including the icon-grid blob axis) — the tune must never accept a
    frame the runtime QualityMonitor would flag, or each such view
    triggers a re-tune that changes nothing."""
    return (
        not r.blown
        and r.clip_pct <= TUNE_CLIP_PCT
        and r.median_luma >= DARK_MEDIAN_LUMA
    )


def _usable(r: QualityReport) -> bool:
    """Fallback-holdable: bright enough to read and under the runtime
    warning threshold — worse than `_good` (some highlights clip) but
    strictly better than a frame the QualityMonitor would flag."""
    return (
        not r.blown
        and r.clip_pct <= BLOWN_CLIP_PCT
        and r.median_luma >= DARK_MEDIAN_LUMA
    )


def is_reference(r: QualityReport) -> bool:
    """Whether the metered crop is bright enough to tune against.

    The single home for the dark-reference threshold: `converge` defers
    when this is False, and the re-tune policy (perception) retries a
    deferred tune when a view where this is True arrives — the two
    decisions are inverses and must never drift apart."""
    return r.median_luma >= DARK_REFERENCE_LUMA


def _settled_meter(
    meter: Callable[[], QualityReport | None],
) -> QualityReport | None:
    """Meter after an AE-mode change, waiting out the firmware loop's
    re-convergence: re-meter until two consecutive reads agree within
    AE_SETTLE_TOLERANCE (relative median luma), bounded by
    AE_SETTLE_METERS. Returns the last read (None if any read failed)."""
    prev = meter()
    if prev is None:
        return None
    for _ in range(AE_SETTLE_METERS - 1):
        cur = meter()
        if cur is None:
            return None
        if abs(cur.median_luma - prev.median_luma) <= AE_SETTLE_TOLERANCE * max(
            prev.median_luma, 1.0
        ):
            return cur
        prev = cur
    return prev


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
      2. (`prefer_auto` only) re-assert AE and re-meter (settled — the
         firmware loop needs frames to re-converge) — recovers the
         driver-left-AE-off-after-renegotiation case without going manual.
      3. Dark-reference gate: a pitch-dark crop even under AE means the
         screen itself is dark (asleep / resting lock screen) — there is
         no white level to expose for, and converging here is how a
         frozen value ends up blowing out the screen once it lights up.
         Defer (`deferred=True`); the caller retries on a bright view.
      4. Integer-step a manual exposure from `start`: darker while
         highlights clip, brighter while too dark. Stops on: in band
         (success), stalled luma (driver ignores writes), two direction
         flips (band lies between two integer steps — keep the darkest
         usable step tried), range exhausted, or `max_steps`.

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
        r = _settled_meter(meter)
        if r is None:
            return TuneResult("auto", None, False, "no frame after AE re-assert")
        if _good(r):
            return TuneResult(
                "auto",
                None,
                True,
                f"recovered by AE re-assert (median {r.median_luma:.0f})",
            )

    if not is_reference(r):
        return TuneResult(
            "auto" if prefer_auto else "manual",
            None,
            False,
            f"screen dark (median {r.median_luma:.0f}) — "
            "no reference to tune against; deferred",
            deferred=True,
        )

    # Phase 4: manual stepping. On the log2 scale, lower = darker.
    exp = max(MIN_EXPOSURE, min(MAX_EXPOSURE, start))
    prev_luma = r.median_luma
    last_dir = 0
    flips = 0
    best: int | None = None  # darkest usable exposure tried
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
        if _usable(r) and (best is None or exp < best):
            best = exp
        # Darker while highlights clip — globally (clip fraction) or as
        # burned icon blobs (which can ride on a low global clip);
        # brighter only when neither says overexposed.
        overexposed = r.clip_pct > TUNE_CLIP_PCT or r.white_blobs >= BLOWN_BLOB_COUNT
        direction = -1 if overexposed else 1
        if last_dir and direction != last_dir:
            flips += 1
            if flips >= 2:
                if best is not None:
                    set_manual(best)
                    return TuneResult(
                        "manual",
                        best,
                        False,
                        f"band between steps — held darkest usable {best}",
                    )
                reason = "oscillating with no usable step"
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
