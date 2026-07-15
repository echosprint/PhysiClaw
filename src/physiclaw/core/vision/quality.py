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

  Known blind spot, by construction: a CATASTROPHIC white-out whose
  median itself crosses 200 (AE caught mid-swing right after a
  dark→bright screen flip) reads as a legit white page — and no luma
  statistic can separate the two (measured 2026-07: a readable chat
  page clips MORE than a nuked home screen). Acquisition handles that
  case instead: `GestureObserver.with_view` detects the flip from the
  before/after median jump and re-settles the grab.

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
# recover (AF genuinely failing, not just mid-hunt).
BLUR_THRESHOLD = 80.0

# A pixel this bright is treated as clipped (detail destroyed).
CLIP_LUMA = 250
# Two-factor blown-highlights rule — see module docstring for calibration.
BLOWN_CLIP_PCT = 0.12
BLOWN_MEDIAN_LUMA = 200.0

# Bad views in a row before the warning escalates from "this view is
# unreliable" to "tell your user the rig needs attention".
PERSIST_AFTER = 3

# Agent-facing fragments — pinned by tests; edits change agent behavior.
BLUR_NOTE = "image blurry — autofocus failed or the camera moved"
BLOWN_NOTE = "washed out to white — auto-exposure failed or glare on the screen"
UNRELIABLE_NOTE = "labels in this view may be wrong or missing"
PERSIST_REMINDER = (
    "Bad camera views {n} in a row — the rig needs physical attention: "
    "report it to your user (camera focus, glare / lighting, screen "
    "brightness) instead of retrying blind."
)


@dataclass(frozen=True)
class QualityReport:
    """Measured quality of one cropped phone-screen frame."""

    sharpness: float  # variance of Laplacian — higher = sharper
    clip_pct: float  # fraction of pixels >= CLIP_LUMA
    median_luma: float  # median gray level, 0-255

    @property
    def blurry(self) -> bool:
        return self.sharpness < BLUR_THRESHOLD

    @property
    def blown(self) -> bool:
        return self.clip_pct > BLOWN_CLIP_PCT and self.median_luma < BLOWN_MEDIAN_LUMA


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
        if report.blurry:
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
