"""Camera-view quality assessment — is autofocus / auto-exposure failing?

The camera's AF/AE run in firmware and the OS backend; OpenCV can neither
rely on controlling them (AVFoundation exposes no exposure props at all,
Windows MSMF/DSHOW drivers honor them inconsistently) nor detect their
failure except from the pixels. So this module judges the frames:

- **Blur** (autofocus failed, lens smudged, rig bumped): variance of the
  Laplacian, same estimator the orchestrator's grab-retry uses. Sharp
  phone-screen crops score 300+; out-of-focus drops under 80.
- **Blown highlights** (auto-exposure failed, or glare on the glass):
  fraction of pixels clipped to near-white. A plain clip fraction can't
  work alone — a well-exposed white page (chat, order form) legitimately
  clips its background — so the rule is two-factor: lots of clipped
  pixels on a screen whose MEDIAN is not white. That is the signature of
  icons/dock burned to white blobs on a dark home screen or Spotlight
  overlay, which is exactly the failure that strands the agent.

  A white-out whose median itself crosses 200 (AE caught mid-swing
  right after a dark→bright screen flip) evades the histogram rule —
  and no luma statistic can separate it from a legit white page
  (measured 2026-07: a readable chat page clips MORE than a nuked home
  screen). Two complementary defenses: STRUCTURE — the icon-grid
  variant is many icon-sized solid clipped blobs (`clipped_icon_blobs`)
  where a white page is one page-sized component, so it folds into
  `blown`; and ACQUISITION — `GestureObserver.with_view` detects the
  brightness flip from the before/after median jump and re-settles the
  grab, covering white-outs with no icon structure at all.

Thresholds were calibrated on real session frames (2026-07): a bad-rig
corpus (Windows, overexposed + glare) vs a good-rig corpus (macOS). At
`clip > 12% and median < 200` the rule flagged ~25% of bad-rig frames —
precisely its dark-screen views — and 1.5% of good-rig frames (that one
had genuine glare washout, so the warning was fair).

`QualityMonitor` composes the agent-facing ⚠ line and tracks how long
the problem has persisted: transient hits (one AF hunt, one bright
animation) warn softly; a streak means the rig itself needs attention,
so the line escalates to "report this to your user".
"""

from dataclasses import dataclass

import cv2
import numpy as np

from physiclaw.common.config import CONFIG
from physiclaw.core.vision.preprocess import grayscale, resize_to_width

# Laplacian variance is not scale-invariant: the same screen captured at
# a higher resolution scores differently, so sharpness scores (and the
# thresholds tuned against them) only transfer between rigs when measured
# at one working width. Frames wider than this are downscaled before
# scoring; narrower ones are scored as-is (upscaling would interpolation-
# smooth genuinely sharp pixels into false blur). 480 ≈ the crop width of
# the corpora the thresholds were calibrated on.
NORMALIZED_WIDTH = 480


def laplacian_variance(frame: np.ndarray) -> float:
    """Variance of Laplacian — a focus/blur estimate. Higher = sharper.

    Accepts a BGR or already-gray frame; frames wider than
    NORMALIZED_WIDTH are downscaled first so scores are comparable
    across cameras and resolutions. Sharp phone screenshots with
    text/icons typically score 300+; severe motion blur or out-of-focus
    drops it under 80. Run on the cropped phone-screen region —
    backgrounds (cutting mat, ruler) contain their own edges that would
    mask real blur on the screen.
    """
    gray = resize_to_width(grayscale(frame), NORMALIZED_WIDTH)
    return float(cv2.Laplacian(gray, cv2.CV_16S).var())


# Below this Laplacian variance a phone-screen crop is unreadably soft.
# One number, one scale: `laplacian_variance` (above) normalizes frames
# to NORMALIZED_WIDTH, and the orchestrator's PEEK/GRAB blur-retry
# thresholds are defined FROM this constant — this check runs AFTER
# those retries, so a frame still under it means the retry didn't
# recover (the pinned lens position no longer matches the scene — the
# camera or phone moved — not a transient).
# Seeded from `[vision]` config (deliberate CONFIG read at import, same
# tradeoff as preprocess.py) so a rig with different optics/lighting can
# re-tune without a source edit; the defaults are the calibrated values.
BLUR_THRESHOLD = CONFIG.vision.blur_threshold

# A pixel this bright is treated as clipped (detail destroyed).
CLIP_LUMA = 250
# Below this median luma the screen content is crushed to black — the
# view is underexposed (stale bright-scene value, or a dim night screen
# the exposure hasn't caught up with). Shared with exposure.py: its
# tune acceptance and the dark-view triggers must agree on one floor.
DARK_MEDIAN_LUMA = 25.0
# The second axis of `dark`: a low median alone can't distinguish an
# underexposed frame from CORRECTLY-EXPOSED dark content — dark-mode
# UI (black background, white text) meters median ~7 on real session
# frames while being perfectly readable. What separates them is the
# highlights: readable dark UI carries near-white text (p99 ≈ 250);
# a crushed frame has no bright pixels at all (p99 well under this).
DARK_P99_LUMA = 180.0
# Above this clip fraction the frame is saturated wall-to-wall — a
# gross overexposure (e.g. a manual value held from a far dimmer
# scene). The two-factor blown rule can't catch it (median ≥ 250
# evades the BLOWN_MEDIAN_LUMA guard, and a page-sized clipped region
# is not icon-shaped), so it gets its own clause. Legit light pages
# measured well below this (a readable chat page clips ~70%).
SATURATED_CLIP_PCT = 0.95
# Two-factor blown-highlights rule — see module docstring for calibration.
BLOWN_CLIP_PCT = CONFIG.vision.blown_clip_pct
BLOWN_MEDIAN_LUMA = CONFIG.vision.blown_median_luma

# Icon-grid white-out: many disconnected, icon-sized, solid, roughly
# square clipped blobs — burned app icons. A legit white page is ONE
# page-sized component (excluded by the area band) and its fragments
# between bubbles/cards are irregular (excluded by fill/aspect).
# Calibrated on real session frames (2026-07): blown home screens
# scored 14 qualifying blobs, every good frame (chat pages, clean
# homes, lock screen) ≤ 2 — threshold 6 sits in that gap.
BLOB_MIN_AREA_FRAC = 0.002  # of the crop, ~half an app icon
BLOB_MAX_AREA_FRAC = 0.08  # ~a 2x2 widget; a white page is far bigger
BLOB_MIN_FILL = 0.8  # solid rounded square, not a ragged fragment
BLOB_ASPECT = (0.6, 1.6)  # roughly square
BLOWN_BLOB_COUNT = CONFIG.vision.blown_blob_count

# Bad views in a row before the warning escalates from "this view is
# unreliable" to "tell your user the rig needs attention".
PERSIST_AFTER = 3

# Agent-facing fragments — pinned by tests; edits change agent behavior.
BLUR_NOTE = "image blurry — the camera or phone moved, or focus calibration is stale"
BLOWN_NOTE = "washed out to white — auto-exposure failed or glare on the screen"
DARK_NOTE = "image dark — the screen is dim/asleep or exposure is stale"
UNRELIABLE_NOTE = "labels in this view may be wrong or missing"
PERSIST_REMINDER = (
    "Bad camera views {n} in a row — the rig needs physical attention: "
    "report it to your user (camera focus, glare / lighting, screen "
    "brightness) instead of retrying blind."
)


def clipped_icon_blobs(gray: np.ndarray) -> int:
    """Count icon-like clipped blobs — the icon-grid white-out signature.

    Connected components of the >= CLIP_LUMA mask, kept when icon-sized
    (area within the BLOB_*_AREA_FRAC band), solid (bbox fill >=
    BLOB_MIN_FILL) and roughly square (BLOB_ASPECT). See the constants
    above for the calibration. Runs on the grayscale crop."""
    h, w = gray.shape[:2]
    total = h * w
    mask = (gray >= CLIP_LUMA).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    count = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        frac = area / total
        if not (BLOB_MIN_AREA_FRAC <= frac <= BLOB_MAX_AREA_FRAC):
            continue
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area / max(1, bw * bh) < BLOB_MIN_FILL:
            continue
        aspect = bw / max(1, bh)
        if BLOB_ASPECT[0] <= aspect <= BLOB_ASPECT[1]:
            count += 1
    return count


@dataclass(frozen=True)
class QualityReport:
    """Measured quality of one cropped phone-screen frame."""

    sharpness: float  # variance of Laplacian — higher = sharper
    clip_pct: float  # fraction of pixels >= CLIP_LUMA
    median_luma: float  # median gray level, 0-255
    white_blobs: int = 0  # icon-like clipped blobs (clipped_icon_blobs)
    # 99th-percentile gray level — the highlight axis that separates
    # underexposure from dark content (see DARK_P99_LUMA). Defaults to
    # 0.0 ("no highlights") so hand-built low-median reports read dark.
    p99_luma: float = 0.0

    @property
    def blurry(self) -> bool:
        return self.sharpness < BLUR_THRESHOLD

    @property
    def dark(self) -> bool:
        """Underexposed: content crushed to black — a low median AND no
        bright pixels. Correctly-exposed dark-mode content (median ~7
        with near-white text) is NOT dark: its p99 clears the highlight
        floor. Underexposed frames are always low-sharpness too —
        consumers should report dark INSTEAD of blurry, or a camera
        problem gets blamed for an exposure one."""
        return self.median_luma < DARK_MEDIAN_LUMA and self.p99_luma < DARK_P99_LUMA

    @property
    def blown(self) -> bool:
        """Three failure signatures, any one:
        - the two-factor histogram rule (clipped pixels on a non-white
          median), which a white-median frame evades by construction;
        - the icon-grid white-out (many icon-sized clipped blobs), which
          catches white-median catastrophes when the burned content is
          icon-shaped;
        - wall-to-wall saturation (clip >= SATURATED_CLIP_PCT), which
          catches the gross case both others evade — a manual value
          held from a far dimmer scene nuking the whole frame white."""
        if self.clip_pct >= SATURATED_CLIP_PCT:
            return True
        if self.clip_pct > BLOWN_CLIP_PCT and self.median_luma < BLOWN_MEDIAN_LUMA:
            return True
        return self.white_blobs >= BLOWN_BLOB_COUNT


def assess(frame: np.ndarray) -> QualityReport:
    """Measure a cropped phone-screen BGR frame.

    Run on the crop, not the raw camera frame — the background (cutting
    mat, desk) carries its own edges and shadows that would mask screen
    blur and skew the luma stats.

    Sharpness goes through `laplacian_variance`, which normalizes wide
    frames to NORMALIZED_WIDTH so the score is comparable across cameras
    and resolutions. The luma stats are distribution-shaped and scale-
    independent, so they read the native crop directly.
    """
    gray = grayscale(frame)
    return QualityReport(
        sharpness=laplacian_variance(gray),
        clip_pct=float((gray >= CLIP_LUMA).mean()),
        median_luma=float(np.median(gray)),
        white_blobs=clipped_icon_blobs(gray),
        p99_luma=float(np.percentile(gray, 99)),
    )


class QualityMonitor:
    """Turns per-view reports into the agent-facing ⚠ line.

    Stateful across views (peeks and gesture views share one monitor) so
    persistence can escalate; any good view resets the streak.
    """

    def __init__(self) -> None:
        self._streak = 0

    @property
    def streak(self) -> int:
        """Consecutive bad views so far (0 after any good view) — handed
        to the orchestration layer's quality callback as context for its
        re-tune decisions and logs."""
        return self._streak

    def observe(self, report: QualityReport) -> str | None:
        """Return the ⚠ warning line for a bad view, else None."""
        issues: list[str] = []
        if report.dark:
            # Dark takes the blur slot: a crushed-black frame always
            # meters blurry, and naming the sharpness would steer the
            # agent at the camera when the problem is light.
            issues.append(f"{DARK_NOTE} (median {report.median_luma:.0f})")
        elif report.blurry:
            issues.append(
                f"{BLUR_NOTE} (sharpness {report.sharpness:.0f} < {BLUR_THRESHOLD:.0f})"
            )
        if report.blown:
            issues.append(f"{report.clip_pct:.0%} of the view {BLOWN_NOTE}")
        if not issues:
            self._streak = 0
            return None
        self._streak += 1
        line = f"⚠ camera: {'; '.join(issues)} — {UNRELIABLE_NOTE}."
        if self._streak >= PERSIST_AFTER:
            line += " " + PERSIST_REMINDER.format(n=self._streak)
        return line
