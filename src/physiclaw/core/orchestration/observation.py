"""Gesture observation — the "see" half of every mutating gesture.

`GestureObserver` owns the observation pipeline that brackets a gesture:
park (clearing the arm from the lens) → grab the cropped phone-screen
frame with a blur-retry (autofocus settles after the arm's move) →
detect UI on the after-frame → assess camera quality → attach the
screen-change verdict. Gesture bodies stay thin callables; a new
diagnostic plugs in here without touching gesture code.

The observer doesn't touch hardware directly — it takes `park`, `grab`,
and `detect` callables from the orchestrator, so its logic is testable
without an arm or camera.

Careful: the verdict marker text (`physiclaw.common.verdict`) is a shared
vocabulary with the agent engine's stuck guard — byte-stable.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

from physiclaw.common import verdict
from physiclaw.core.vision import quality
from physiclaw.core.vision.change import frames_changed
from physiclaw.core.vision.quality import laplacian_variance
from physiclaw.core.vision.util import encode_view_jpeg

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GestureResult:
    """What a screen-mutating gesture hands to the MCP tool layer: the
    action text (screen-change verdict attached) plus the fused
    post-gesture view — annotated JPEG + element listing detected on the
    same parked after-frame the verdict used. `jpeg`/`listing` are None
    when the camera or detector hiccupped; the tool layer then falls
    back to a text-only result telling the agent to `peek`."""

    text: str
    jpeg: bytes | None = None
    listing: str | None = None


class GestureObserver:
    """Before/after observation around a gesture body.

    Stateful only through its `QualityMonitor` — shared across peeks and
    gesture views so a persistent rig problem escalates to "tell your
    user".
    """

    # Post-gesture settle before the fused view is captured: added to the
    # ~1s of stylus retract + park, total ≈ 2s — enough for most page
    # transitions (Anthropic's computer-use reference uses a 2.0s delay).
    GESTURE_SETTLE_SECONDS = 1.0
    # The arm crossing the lens re-triggers autofocus; a frame captured
    # mid-hunt is blurry and poisons both the verdict and the fused view.
    # Below this Laplacian variance, wait and re-grab once. Same number
    # and scale as the quality monitor's blur verdict.
    GRAB_BLUR_THRESHOLD = quality.BLUR_THRESHOLD
    GRAB_BLUR_RETRY_SECONDS = 1.5

    # Blur-retry gate for peeks — same measurement and scale as the
    # gesture grabs above and the quality monitor's verdict
    # (laplacian_variance normalizes internally): one number, defined
    # once. Peek waits longer but re-grabs only once, without a re-check.
    PEEK_BLUR_THRESHOLD = quality.BLUR_THRESHOLD
    PEEK_BLUR_RETRY_SECONDS = 2.0

    def __init__(
        self,
        park: Callable[[], None],
        grab: Callable[[], Any],
        detect: Callable[[Any], tuple[str, Any]],
        monitor: quality.QualityMonitor | None = None,
    ):
        self._park = park
        self._grab = grab  # () -> cropped phone-screen BGR frame
        self._detect = detect  # frame -> (element listing, annotated frame)
        self._quality = monitor if monitor is not None else quality.QualityMonitor()

    def observe_quality(self, source: str, frame) -> str | None:
        """Judge a camera view for AF/AE failure; on a bad one, warn in the
        log and return the agent-facing ⚠ line for the caller to attach.

        Runs AFTER the blur-retry grabs, so a frame still failing here means
        the retry didn't recover it. Fail-open: a crash in the check never
        costs the view."""
        try:
            warning = self._quality.observe(quality.assess(frame))
        except Exception:
            log.exception("camera-quality check failed — skipping")
            return None
        if warning is not None:
            log.warning("%s: %s", source, warning)
        return warning

    def peek_frame(self):
        """Park, grab the cropped phone-screen frame, and re-grab once
        after a settle if it's too blurry — peek's acquisition path.

        Unlike `grab_screen`, failures propagate (a peek with no frame is
        a tool error, not a withheld verdict) and the retry frame is used
        as-is: peek has no verdict to protect, so a still-soft frame is
        better shown than dropped. Caller must hold the lock."""
        self._park()
        frame = self._grab()
        sharpness = laplacian_variance(frame)
        if sharpness < self.PEEK_BLUR_THRESHOLD:
            log.warning(
                "peek: blurry frame (laplacian var=%.1f < %.0f) — retrying",
                sharpness,
                self.PEEK_BLUR_THRESHOLD,
            )
            time.sleep(self.PEEK_BLUR_RETRY_SECONDS)
            frame = self._grab()
        return frame

    def grab_screen(self, settle: float = 0.0):
        """Park (clearing the arm from the lens), let the screen and
        autofocus settle for `settle`s, then capture the cropped
        phone-screen frame for the gesture diff and fused view. Parking
        first means the settle covers the AF hunt the arm's move
        triggered, so the capture is usually sharp; a still-blurry frame
        is re-grabbed once as a fallback.

        Returns `(frame, sharp)`: frame is None on any failure (the view
        is best-effort, never a reason to fail a gesture); sharp is False
        when even the retry stayed below GRAB_BLUR_THRESHOLD — such a
        frame still serves the fused view, but blur erases edges so it
        diffs as "changed everywhere": the caller must skip the verdict
        (None, the fail-open direction) rather than emit a false
        `changed` that would reset the stuck guard. Caller must hold the
        lock."""
        try:
            self._park()
            if settle:
                time.sleep(settle)
            frame = self._grab()
            if laplacian_variance(frame) < self.GRAB_BLUR_THRESHOLD:
                log.debug("gesture frame blurry — waiting for autofocus")
                time.sleep(self.GRAB_BLUR_RETRY_SECONDS)
                frame = self._grab()
                if laplacian_variance(frame) < self.GRAB_BLUR_THRESHOLD:
                    log.warning(
                        "gesture frame still blurry after retry — verdict withheld"
                    )
                    return frame, False
            return frame, True
        except Exception:
            log.warning("screen frame grab failed", exc_info=True)
            return None, False

    def with_view(self, action) -> GestureResult:
        """Run a gesture body bracketed by before/after frames; return
        the action text with the screen-change verdict attached PLUS the
        fused post-gesture view (annotated JPEG + listing) detected on
        the same after-frame. One capture serves both:

          - the verdict is the agent's only evidence for SILENT REFUSALS
            — a toast vanishes long before the arm parks, so "the screen
            looks exactly the same" distinguishes a refused tap from a
            landed one;
          - the view replaces the act-then-peek turn pair — the agent
            reads the new screen from the gesture result itself.

        Caller must hold the lock. The before-grab needs no settle (the
        arm was already parked from the previous op, screen static); the
        after-grab settles so the transition completes and AF recovers
        from the gesture's arm move before capture.
        """
        before, before_sharp = self.grab_screen()
        result = action()
        after, after_sharp = self.grab_screen(settle=self.GESTURE_SETTLE_SECONDS)
        changed = None
        jpeg = listing = None
        if after is not None:
            # Verdict only from two sharp frames — a blurry side diffs as
            # "changed everywhere" (false `changed`, the harmful
            # direction). The view below is best-effort either way.
            if before is not None and before_sharp and after_sharp:
                try:
                    changed = frames_changed(before, after)
                except Exception:
                    log.debug("screen-verdict diff failed", exc_info=True)
            try:
                listing, annotated = self._detect(after)
                jpeg = encode_view_jpeg(annotated)
            except Exception:
                log.warning("post-gesture view failed", exc_info=True)
            warning = self.observe_quality("gesture view", after)
        else:
            warning = None
        text = verdict.attach(result, changed)
        if warning is not None:
            if listing is not None:
                listing = f"{listing}\n{warning}"
            else:
                # Detection failed, so there's no listing to carry the line —
                # ride the action text: a blurry/blown frame is a plausible
                # CAUSE of the failed view, and the agent needs to hear it
                # before blindly re-peeking.
                text = f"{text}\n{warning}"
        return GestureResult(text=text, jpeg=jpeg, listing=listing)
