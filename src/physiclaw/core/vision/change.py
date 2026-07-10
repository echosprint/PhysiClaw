"""Before/after gesture frame diff — did the screen visibly change?

Feeds the `screen: changed` / `screen: no visible change` verdict that
the orchestrator appends to every gesture result (`physiclaw.verdict`).
The point is catching SILENT REFUSALS: an app that rejects a tap with a
1–2s toast looks, by the time the arm parks and the camera looks, exactly
like a tap that never landed. The agent can't see the toast — but it can
be told "nothing is different", which is the evidence that matters.

Pipeline (calibrated per the TOCTOU-defense literature: pixel noise
threshold ~20–25/255, real transitions change >0.2% of pixels while
benign flicker stays under ~0.05%):

    grayscale → Gaussian blur (kills sensor noise + sub-pixel wobble)
    → clamped brightness align → translation align
    → absdiff → threshold → morphological open
    → changed if DISTRIBUTED (ratio > RATIO_THRESHOLD)
             or LOCALIZED (a coherent blob ≥ the local floor)

Two triggers because a global ratio dilutes small changes: a cart
badge `2→3`, a one-line toast, a single quantity digit each cover a
fraction of a percent of the frame — below the distributed threshold,
but a solid connected region. Morphological opening first erases
scattered noise and thin alignment-residue edges, so what remains for
the blob test is genuinely coherent change, not camera artifact.

The status-bar strip is cropped out first — the clock ticks over
mid-gesture and would count as a change on every slow gesture.

The arm's motion makes the raw diff physically unreliable; a false
`changed` is the harmful direction (it resets the stuck guard's counters
and re-hides silent refusals). Three fixes:

  - AUTOFOCUS HUNTING (the arm crossing the lens re-triggers AF) makes
    frames blurry. Handled at ACQUISITION: `GestureObserver.grab_screen` retries a
    low-sharpness capture once so AF settles. Not handled here — a
    diff-time focus-mismatch guard cannot distinguish "AF blur" from
    "blank screen gained content", a real change it must not eat.
  - AE RE-METERING after occlusion shifts global brightness. Fix:
    mean-align the after frame, CLAMPED to ±NOISE_THRESHOLD — a modest
    mean shift is exposure, a large one is the page actually changing
    (dark → light) and must survive as change.
  - VIBRATION shifts the rig a few pixels between frames → every edge
    would diff. Fix: estimate the global shift (phase correlation) and
    compare the aligned overlap; a shift beyond MAX_ALIGN_SHIFT means
    the rig itself moved → verdict UNRELIABLE (None).

`None` = "could not judge" and rides the existing fail-open path:
`verdict.attach(None)` leaves the result unmarked and the stuck guard
skips counting.

The verdict is still evidence, not proof — a change smaller than the
camera can resolve (sub-blob) is missed, so doctrine keeps telling the
agent to READ the value it tried to change, and the stuck guard only
accumulates, never acts on a single verdict.
"""
import cv2
import numpy as np

# Fraction of frame height to drop from the top before diffing — covers
# the iOS status bar (clock, signal) on the cropped phone-screen frame.
STATUS_BAR_FRAC = 0.06

# Per-pixel gray delta below this is sensor noise, not signal. Also the
# clamp on the brightness (AE) correction — see module docstring.
NOISE_THRESHOLD = 25

# DISTRIBUTED-change trigger: fraction of pixels that must differ to
# call the screen changed regardless of layout. 0.3% ≈ a keyboard
# popping, a page turning; camera flicker on a static screen is well
# under 0.05%. Small localized changes (a badge, a toast, one digit)
# sit below this — the LOCALIZED trigger below catches those.
RATIO_THRESHOLD = 0.003

# LOCALIZED-change trigger: after morphological opening removes scattered
# noise + thin alignment-residue edges, one connected changed region this
# large = a real coherent change (badge, toast, digit), even at 0.05% of
# the frame. Resolution-relative with an absolute floor for small frames.
# A blinking text cursor (~2px wide) is erased by the opening, so it never
# reaches this.
LOCAL_BLOB_FRAC = 0.0002
LOCAL_BLOB_MIN_PX = 60

# Vibration correction bound, px. A larger confident shift = the rig
# itself moved; alignment can't be trusted, so neither can the diff.
MAX_ALIGN_SHIFT = 8

# Phase-correlation peak response below this = no coherent global
# translation (content changed, or periodic content spreading the peak
# across harmonics) — fall back to anchored template matching.
# Measured on this pipeline: true shifts ≈ 0.8–1.2, garbage ≈ 0.001–0.01.
_PHASE_RESPONSE_MIN = 0.3

# Template-match (TM_CCOEFF_NORMED) confidence below this = the frames
# share no alignable content at any small shift — a real transition;
# diff unaligned. Same-content matches score ≈ 0.9+.
_TEMPLATE_CONF_MIN = 0.5

# Below this gray variance a frame has no texture to align against
# (blank screens, synthetic frames) — skip alignment.
_MIN_TEXTURE_VAR = 1.0

_BLUR_KSIZE = 5
_OPEN_KERNEL = np.ones((3, 3), np.uint8)


def _prepare(before: np.ndarray, after: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Crop, gray, blur, brightness-align (clamped), vibration-align.
    Returns the comparable gray pair, or None when the rig moved too far
    to compare honestly."""
    if before.shape != after.shape:
        after = cv2.resize(after, (before.shape[1], before.shape[0]))
    top = int(before.shape[0] * STATUS_BAR_FRAC)
    a = cv2.GaussianBlur(cv2.cvtColor(before[top:], cv2.COLOR_BGR2GRAY), (_BLUR_KSIZE,) * 2, 0)
    b = cv2.GaussianBlur(cv2.cvtColor(after[top:], cv2.COLOR_BGR2GRAY), (_BLUR_KSIZE,) * 2, 0)

    # Brightness align, clamped: correct AE drift, keep real transitions.
    delta = round(float(a.mean()) - float(b.mean()))
    delta = max(-NOISE_THRESHOLD, min(NOISE_THRESHOLD, delta))
    if delta:
        b = np.clip(b.astype(np.int16) + delta, 0, 255).astype(np.uint8)

    # Translation align — needs texture on both sides to correlate.
    if min(float(a.var()), float(b.var())) > _MIN_TEXTURE_VAR:
        shift = _estimate_shift(a, b)
        if shift is None:
            return None
        dx, dy = shift
        if dx or dy:
            h, w = a.shape
            a = a[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
            b = b[max(0, dy):h - max(0, -dy), max(0, dx):w - max(0, -dx)]
    return a, b


def _estimate_shift(a: np.ndarray, b: np.ndarray) -> tuple[int, int] | None:
    """Global translation (dx, dy) with b[y+dy, x+dx] ≈ a[y, x] (sign
    convention verified empirically). Three outcomes:

      (dx, dy)  a confident small shift → align (vibration).
      (0, 0)    no alignable translation → diff as-is. This is the REAL
                PAGE TRANSITION case: unrelated content gives phase
                correlation a garbage peak, which must not be mistaken
                for rig motion — the change must survive to the diff.
      None      a CONFIDENT shift beyond MAX_ALIGN_SHIFT → the rig
                moved; nothing about the diff can be trusted.

    Periodic content (keyboard rows, list stripes) spreads the phase
    peak across harmonics → low response even for true small shifts; an
    anchored template match resolves those within ±MAX_ALIGN_SHIFT. A
    peak pinned to the window border means the true shift may exceed
    the window (rig move) → None, same as the confident-phase case."""
    (sx, sy), resp = cv2.phaseCorrelate(a.astype(np.float32), b.astype(np.float32))
    if resp >= _PHASE_RESPONSE_MIN:
        dx, dy = round(sx), round(sy)
        if abs(dx) > MAX_ALIGN_SHIFT or abs(dy) > MAX_ALIGN_SHIFT:
            return None
        return dx, dy
    m = MAX_ALIGN_SHIFT
    if a.shape[0] <= 4 * m or a.shape[1] <= 4 * m:
        return 0, 0  # too small to anchor a template
    tpl = b[m:-m, m:-m]
    res = cv2.matchTemplate(a, tpl, cv2.TM_CCOEFF_NORMED)
    _minv, maxv, _minloc, (mx, my) = cv2.minMaxLoc(res)
    if maxv < _TEMPLATE_CONF_MIN:
        return 0, 0
    # Per-axis border check: a peak PINNED to the search-window border
    # means the true shift may lie beyond ±m (a real rig move on content
    # that also spread the phase peak) — aligning to it, or diffing
    # unaligned, would read as a false `changed` → None. Exception: when
    # the response is FLAT along an axis (content uniform that way, e.g.
    # horizontal stripes make every x-offset match equally) the border
    # position is an artifact of tie-breaking, and the shift on that
    # axis is simply unidentifiable → 0.
    x_flat = float(res[my, :].max() - res[my, :].min()) < 1e-3
    y_flat = float(res[:, mx].max() - res[:, mx].min()) < 1e-3
    if (not x_flat and mx in (0, 2 * m)) or (not y_flat and my in (0, 2 * m)):
        return None
    return 0 if x_flat else m - mx, 0 if y_flat else m - my


def _opened_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """0/1 mask of significantly-differing pixels, morphologically opened
    to erase speckle noise and 1–2px alignment-residue edges — what
    survives is coherent change (a badge, a toast, a region)."""
    mask = (cv2.absdiff(a, b) > NOISE_THRESHOLD).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, _OPEN_KERNEL)


def change_ratio(before: np.ndarray, after: np.ndarray) -> float | None:
    """Fraction of comparable pixels that visibly differ (post-opening),
    or None when the frames can't be compared honestly (rig moved — see
    module docstring). Frames are BGR crops of the phone screen from the
    same fixed camera; a re-crop size drift of a pixel is absorbed."""
    pair = _prepare(before, after)
    if pair is None:
        return None
    mask = _opened_mask(*pair)
    return int(np.count_nonzero(mask)) / mask.size


def frames_changed(before: np.ndarray, after: np.ndarray) -> bool | None:
    """True/False iff the screen visibly changed; None when the frames
    are unreliable — callers fail open (no verdict, no guard counting).

    Changed if EITHER a distributed change crosses RATIO_THRESHOLD OR a
    single coherent region crosses the localized-blob floor — so a small
    badge/toast/digit registers even when it's a fraction of the frame."""
    pair = _prepare(before, after)
    if pair is None:
        return None
    mask = _opened_mask(*pair)
    total = mask.size
    if int(np.count_nonzero(mask)) > RATIO_THRESHOLD * total:
        return True
    n, _labels, stats, _c = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return False
    largest = int(stats[1:, cv2.CC_STAT_AREA].max())
    return largest >= max(LOCAL_BLOB_MIN_PX, LOCAL_BLOB_FRAC * total)
