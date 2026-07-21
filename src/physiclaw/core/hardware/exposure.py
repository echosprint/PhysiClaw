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
crop must have (almost) no clipped pixels and must not be underexposed
(`QualityReport.dark`, the two-axis predicate — so correctly-exposed
dark-content screens are accepted as-is).

A dark (underexposed — see `QualityReport.dark`) crop is still worth
tuning against: at night auto-brightness dims the phone to a few nits,
and a value held for a lit daytime screen crushes the lock screen (and
its passcode keypad) to black — the observed night-unlock failure
mode. The tune therefore steps BRIGHTER on a dark crop; if the screen
lights up later and the held value runs hot, the blown-view inline fix
corrects it before the view ships (the saturation clauses in quality.py
— wall-to-wall clip, and white-median + clip + featureless — exist so
that white-out is detectable at all).

EVERY failed search defers (`deferred=True`): the scene was judged
unfixable as-is — dark at the ceiling, blown/oscillating, or the meter
lost frames — and re-proving that per view is the livelock this design
exists to prevent. The hold records a baseline median metered under the
RESTORED exposure state (firmware AE under `prefer_auto`; the held
manual otherwise), and `wants_retry` releases it only when the scene
itself changes or a fully in-band view arrives. Failure still ends in
`set_auto()` + a descriptive TuneResult (unless the user pinned manual
in config): firmware AE plus the runtime ⚠ warnings beat a bad frozen
manual value.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Literal

from physiclaw.core.vision.quality import (
    BLOWN_BLOB_COUNT,
    BLOWN_CLIP_PCT,
    WASHED_HIGHLIGHT_PCT,
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
# The window is sized for filming an OLED phone screen: iPhone panels
# PWM-dim at ~240-480Hz, and an exposure that doesn't average several
# PWM cycles records banding — below ~4ms is banding territory. The
# -4 ceiling (62.5ms) drops the stream to ~16fps; affordable ONLY on a
# rig whose lens is pinned from the calibration bundle (hardware/
# focus.py) — contrast-detect AF iterates per frame, and at 16fps every
# post-gesture hunt outlived the 2s view settle (measured; it blurred a
# whole session). 16fps still serves the watchdog and view grabs, and
# the extra stop is what renders a night-dimmed screen readable. The -6
# default start (15.6ms ≈ 1/60s) is the textbook screen-filming shutter.
MIN_EXPOSURE = -8
MAX_EXPOSURE = -4
MAX_STEPS = 6

# Ceiling for rigs whose lens is NOT pinned (no calibrated focus in the
# bundle, a failed calibration-time pin, or an apply refused at open —
# every one of which leaves the lens on live AF): the old
# full-frame-rate ceiling, so live AF never has to iterate at 16fps.
# Callers pass it via `converge(..., max_exposure=UNPINNED_MAX_EXPOSURE)`.
UNPINNED_MAX_EXPOSURE = -5

# Acceptance ceiling for clipped pixels during a tune — much tighter than
# quality.py's 12% warning threshold. Near-zero clip on the crop IS the
# highlight-headroom criterion: the screen's whites render just below
# 250, so no content can blow out later. Not exactly zero — sensor
# sparkle and the phone's specular bezel edge leak a few clipped pixels
# into any crop.
TUNE_CLIP_PCT = 0.02

# Below this median the screen shows no lit content at all (asleep,
# resting lock screen) — the gate on the stall check only: a black
# screen can't move the meter regardless of the driver, so "ignored
# writes" is judgeable only above this floor.
DARK_REFERENCE_LUMA = 15.0

# The scene must move by this much (either direction) from a failed
# tune's recorded baseline before a re-tune is worth trying — the
# hold's release valve for still-mis-exposed views (an in-band view
# releases via its own branch). The baseline is metered under the same
# exposure state as the views it gates (see the failure hold in
# `converge`), so this compares scene against scene; it only needs to
# clear per-frame metering noise on a static scene (~2-3 luma), not an
# exposure-regime gap.
RETRY_LUMA_DELTA = 5.0

# A set that changes measured median luma by less than this did nothing:
# the driver is ignoring exposure writes (common — see opencv#9738), and
# further stepping is theater. One log2 stop should move luma far more.
# Judged only between consecutive MANUAL reads (from step 2 on): step 1's
# baseline is the auto-mode read, and a manual start that matches AE's
# converged brightness is normal, not an ignored write.
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
    # True on EVERY failed search (dark ending, blown/oscillating
    # ending, meter loss) — the scene was judged unfixable as-is and
    # nothing more can be learned until it changes. `wants_retry` is
    # the release valve.
    deferred: bool = False
    # The hold's baseline median, metered under the RESTORED exposure
    # state so it shares a regime with the views `wants_retry` gates.
    # None on ok results and on sentinels (crash / meter failure),
    # where the hold spans the dark band and blown stays urgent.
    median: float | None = None


def _good(r: QualityReport) -> bool:
    """In band: highlight headroom (near-zero clip) + not underexposed.
    `not blown` / `not dark` fold in every failure signature quality.py
    knows — the tune must never accept a frame the runtime
    QualityMonitor would flag (each such view triggers a re-tune that
    changes nothing), and must never REJECT a frame the monitor calls
    fine: `dark` is the two-axis predicate, so a correctly-exposed
    dark-content screen (median ~7, bright text) is accepted as-is
    instead of laddering a correct exposure hot."""
    return not r.blown and r.clip_pct <= TUNE_CLIP_PCT and not r.dark


def _usable(r: QualityReport) -> bool:
    """Fallback-holdable: readable and under the runtime warning
    threshold — worse than `_good` (some highlights clip) but strictly
    better than a frame the QualityMonitor would flag."""
    return not r.blown and r.clip_pct <= BLOWN_CLIP_PCT and not r.dark


def is_reference(r: QualityReport) -> bool:
    """Whether the metered crop shows a lit screen at all — the gate on
    the stall check (a black screen can't move the meter no matter what
    the driver does, so "ignored writes" is only judgeable on a lit
    crop)."""
    return r.median_luma >= DARK_REFERENCE_LUMA


def wants_retry(last: TuneResult | None, r: QualityReport) -> bool:
    """Should this view trigger a (re-)tune, given the last outcome?

    The single home for the ENTIRE fire/hold policy — dark and blown
    alike. `converge` decides when to give up on a scene (every failed
    tune defers with a restored-regime baseline); this decides when the
    scene has changed enough to try again. Keeping both here is what
    stops them drifting apart:

    - view is clean (neither dark nor blown) → tune only when a hold is
      outstanding AND the view is fully in band (`_good`): the
      verify-tune is then genuinely one meter (phase 1 accepts as-is)
      and clears the stale hold. A merely-usable clean view (the held
      darkest-usable glare frame) must NOT clear it — the re-run would
      repeat the search that just failed, every view;
    - mis-exposed view, no hold → tune (a stale value crushes a night
      screen / saturates a morning one);
    - mis-exposed view under a hold → only when the scene moved past
      the hold's baseline by RETRY_LUMA_DELTA in EITHER direction (a
      stuck scene re-proves what the failed tune measured — the thrash
      bound for sleeping phones and persistent glare alike);
    - baseline-less hold (crash / meter-failure sentinel): dark views
      hold (no baseline to compare), blown views fire — saturation is
      urgent and the crash may have been dark-side."""
    bad = r.dark or r.blown
    if last is None or not last.deferred:
        return bad
    if not bad:
        return _good(r)
    if last.median is None:
        return r.blown
    return abs(r.median_luma - last.median) > RETRY_LUMA_DELTA


def _restored_median(meter: Callable[[], QualityReport | None]) -> float | None:
    """One settled read under the just-restored exposure state — the
    same-regime baseline a dark deferral records for `wants_retry`.
    Metering BEFORE the restore would gate the release on the exposure
    difference (ceiling-manual vs reverted-AE), not on scene change."""
    r = _settled_meter(meter)
    return r.median_luma if r is not None else None


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
    max_exposure: int = MAX_EXPOSURE,
) -> TuneResult:
    """Verify exposure by metering; converge manually only if AE fails.

    Phases:
      1. Meter as-is — in band means firmware AE is doing its job
         (the healthy-rig path on every OS): touch nothing.
      2. (`prefer_auto` only) re-assert AE and re-meter (settled — the
         firmware loop needs frames to re-converge) — recovers the
         driver-left-AE-off-after-renegotiation case without going manual.
      3. Integer-step a manual exposure from `start`: darker while
         highlights clip, brighter while too dark (including a dark
         crop — a night-dimmed screen tunes like any other). Stops on:
         in band (success), stalled luma (driver ignores writes; judged
         only on lit crops — a black screen can't move the meter no
         matter what the driver does), two direction flips (band lies
         between two integer steps — keep the darkest usable step
         tried), range exhausted, or `max_steps`.
      4. Failure hold: every search that ends without an accepted frame
         defers (`deferred=True`; firmware AE restored under
         `prefer_auto`, pinned-manual holds the brightest probed step
         on a dark ending) and records a baseline median under the
         RESTORED state, so `wants_retry` compares scene against scene
         for dark and blown alike. The loop's diagnosis stays in the
         detail; the state note is appended to it.

    A `meter` returning None (no frame) fails open immediately — as a
    DEFERRED sentinel (no baseline), so the caller's re-tune policy
    holds instead of re-firing on every dark view of a meter-dead
    camera. With `prefer_auto=False` (user pinned manual in config)
    phase 2 is skipped and failure keeps the best manual step instead
    of reverting to auto. `max_exposure` caps the brightening ladder —
    pass UNPINNED_MAX_EXPOSURE on rigs whose lens isn't pinned (live AF
    can't afford the 16fps the -4 ceiling costs).
    """
    r = meter()
    if r is None:
        return TuneResult(
            "auto", None, False, "no frame — exposure left as-is", deferred=True
        )
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
            return TuneResult(
                "auto", None, False, "no frame after AE re-assert", deferred=True
            )
        if _good(r):
            return TuneResult(
                "auto",
                None,
                True,
                f"recovered by AE re-assert (median {r.median_luma:.0f})",
            )

    # Phase 3: manual stepping. On the log2 scale, lower = darker.
    # Clamp `start` once for the whole manual phase — config doesn't
    # range-check, and every manual value written below (steps and the
    # pinned-manual fallback) must respect set_manual's contract.
    start = max(MIN_EXPOSURE, min(max_exposure, start))
    exp = start
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
        if _good(r):
            return TuneResult(
                "manual",
                exp,
                True,
                f"converged at {exp} in {step} step(s) "
                f"(median {r.median_luma:.0f}, clip {r.clip_pct:.0%})",
            )
        # Stall check after the in-band accept (a step that lands in band
        # is a success however little the luma moved), and only from the
        # second step on: step 1's baseline is the auto-mode read, and a
        # manual start that happens to match AE's converged brightness is
        # normal — declaring it a stall would abort a search whose next
        # step could still converge. Lit crops only: on a black screen
        # the meter can't move regardless of the driver, and the asleep
        # verdict below is the right conclusion, not "ignored writes".
        if (
            step > 1
            and is_reference(r)
            and abs(r.median_luma - prev_luma) < LUMA_STALL_EPSILON
        ):
            reason = f"driver ignores exposure writes (luma stuck ~{r.median_luma:.0f})"
            break
        prev_luma = r.median_luma
        if _usable(r) and (best is None or exp < best):
            best = exp
        # Darker while highlights are too hot — globally (clip fraction),
        # as burned icon blobs (which ride on a low global clip), or as a
        # washed highlight fraction (whites at ~230-249 that don't clip at
        # 250 yet still burn `blown`); brighter only when none says so. The
        # washed term must mirror `blown`'s washed clause, or the acceptance
        # test rejects a frame the stepper then walks the wrong way.
        overexposed = (
            r.clip_pct > TUNE_CLIP_PCT
            or r.white_blobs >= BLOWN_BLOB_COUNT
            or r.highlight_pct >= WASHED_HIGHLIGHT_PCT
        )
        direction = -1 if overexposed else 1
        if last_dir and direction != last_dir:
            flips += 1
            if flips >= 2:
                if best is not None:
                    # Held-usable exit still defers with a baseline:
                    # without the hold, a scene the search already
                    # proved band-less (persistent glare) re-fires the
                    # full tune on every subsequent view.
                    set_manual(best)
                    return TuneResult(
                        "manual",
                        best,
                        False,
                        f"band between steps — held darkest usable {best}",
                        deferred=True,
                        median=_restored_median(meter),
                    )
                reason = "oscillating with no usable step"
                break
        last_dir = direction
        nxt = max(MIN_EXPOSURE, min(max_exposure, exp + direction))
        if nxt == exp:
            reason = f"exposure range exhausted at {exp}"
            break
        exp = nxt

    # Failure hold: EVERY failed search defers — the scene was judged
    # unfixable as-is, and `wants_retry` re-fires only when the scene
    # itself moves (or a clean view arrives, clearing the hold with one
    # cheap verify-tune). The baseline is metered AFTER the exposure
    # state is restored below, so release and any re-defer judge the
    # SAME regime — a stepped-manual verdict against restored-regime
    # views livelocks. The loop's own diagnosis (stall, oscillation,
    # exhaustion) stays in `reason`; the state note is appended, never
    # a replacement.
    if r is not None and (r.dark or r.blown):
        state = "dark" if r.dark else "blown"
        reason += f"; crop still {state} (median {r.median_luma:.0f})"

    if not prefer_auto:
        # Manual pinned in config: never flip to firmware AE, even when
        # no usable step was found — honoring the pin matters more than
        # a good exposure the user opted out of. On a dark ending hold
        # the LAST step tried — the brightest probe, least-bad for a
        # dark scene — rather than re-crushing the screen back to the
        # configured start.
        if best is not None:
            value, held = best, "best manual"
        elif r is not None and r.dark:
            value, held = exp, "brightest probed"
        else:
            value, held = start, "manual start"
        set_manual(value)
        return TuneResult(
            "manual",
            value,
            False,
            f"{reason} — held {held} {value} (auto disabled in config)",
            deferred=True,
            median=_restored_median(meter),
        )
    set_auto()
    return TuneResult(
        "auto",
        None,
        False,
        f"{reason} — reverted to auto",
        deferred=True,
        median=_restored_median(meter),
    )
