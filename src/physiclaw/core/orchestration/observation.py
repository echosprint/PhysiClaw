"""Gesture observation — the "see" half of every mutating gesture.

`GestureObserver` owns the observation pipeline that brackets a gesture:
park (clearing the arm from the lens) → grab the cropped phone-screen
frame with a blur/blown-retry (autofocus and auto-exposure settle after
the arm's move and screen flips), escalating a persistently blown grab —
or a deferred tune's first bright reference — to an inline exposure
re-tune so the agent gets a corrected view, not a warning → re-settle
the after-grab when the before/after median jump says the screen
flipped brightness class (AE mid-swing white-outs are invisible to luma
statistics; see FLIP_MEDIAN_DELTA) → detect UI on the after-frame →
assess camera quality → attach the screen-change verdict. Gesture
bodies stay thin callables; a new diagnostic plugs in here without
touching gesture code.

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
    # The arm crossing the lens re-triggers autofocus, and a screen-content
    # flip (dark lock screen → bright home) sends firmware auto-exposure
    # re-converging; a frame captured mid-hunt is blurry or blown and
    # poisons both the verdict and the fused view. On either verdict,
    # wait and re-grab once. Blur uses the same number and scale as the
    # quality monitor's verdict.
    GRAB_BLUR_THRESHOLD = quality.BLUR_THRESHOLD
    GRAB_RETRY_SECONDS = 1.5

    # Retry gate for peeks — same measurements as the gesture grabs above
    # and the quality monitor's verdict: one number, defined once. Peek
    # waits longer but re-grabs only once, without a re-check.
    PEEK_BLUR_THRESHOLD = quality.BLUR_THRESHOLD
    PEEK_RETRY_SECONDS = 2.0

    # A gesture that flips the screen's brightness class (lock → home,
    # dark app → white page) sends firmware AE re-converging over
    # seconds; the settled after-grab can still catch it mid-swing,
    # recording a white-out the phone never showed. Luma statistics
    # can't flag that frame — a *readable* white page meters MORE
    # clipped than a nuked one (measured 2026-07: good chat page 70%
    # clip vs nuked home screen 48%) — but the median jump between the
    # gesture's own before/after frames detects the flip itself: above
    # this delta, settle again and regrab. 100 clears normal same-page
    # deltas (<30) and moderate overlays, and trips on dark↔bright
    # flips (measured ~180).
    FLIP_MEDIAN_DELTA = 100.0
    FLIP_SETTLE_SECONDS = 1.5

    def __init__(
        self,
        park: Callable[[], None],
        grab: Callable[[], Any],
        detect: Callable[[Any], tuple[str, Any]],
        monitor: quality.QualityMonitor | None = None,
        on_quality: Callable[[quality.QualityReport, int], None] | None = None,
        fix_exposure: Callable[[], None] | None = None,
        needs_fix: Callable[[quality.QualityReport], bool] | None = None,
    ):
        self._park = park
        self._grab = grab  # () -> cropped phone-screen BGR frame
        self._detect = detect  # frame -> (element listing, annotated frame)
        self._quality = monitor if monitor is not None else quality.QualityMonitor()
        # (report, streak) after every judged view — perception hangs its
        # re-tune policy here without observation knowing about tuning.
        self._on_quality = on_quality
        # Synchronous exposure fix (perception.tune_now) and the
        # predicate deciding when a grab warrants it
        # (perception.needs_inline_fix: washed out, or a deferred tune
        # owed its bright reference). Fixing BEFORE the view ships beats
        # sending the agent a mis-exposed frame with a warning attached.
        self._fix_exposure = fix_exposure
        self._needs_fix = needs_fix if needs_fix is not None else lambda r: r.blown

    def _fix_exposure_grab(self, source: str, report: quality.QualityReport):
        """The injected predicate says this grab's exposure is wrong
        (still blown after the settle-retry, or a deferred tune just met
        its bright reference) — run the synchronous tune and grab the
        corrected frame. Returns `(frame, report, fixed)`; on any
        failure the original report is kept and `fixed` is False
        (fail-open)."""
        if self._fix_exposure is None:
            return None, report, False
        log.info(
            "%s: %s (clip %.0f%%) — tuning exposure before the view ships",
            source,
            "frame still blown after retry" if report.blown else "deferred tune",
            report.clip_pct * 100,
        )
        try:
            self._fix_exposure()
            frame = self._grab()
            return frame, quality.assess(frame), True
        except Exception:
            log.warning("inline exposure fix failed", exc_info=True)
            return None, report, False

    def observe_quality(self, source: str, report: quality.QualityReport) -> str | None:
        """Judge an already-assessed camera view for AF/AE failure; on a
        bad one, warn in the log and return the agent-facing ⚠ line for
        the caller to attach.

        Takes the `QualityReport` the acquisition path (`peek_frame` /
        `grab_screen`) already computed for its retry decision — the
        frame is never re-measured here. Runs AFTER the retry grabs, so
        a report still failing means the retry didn't recover it. Every
        judged view is also reported to `on_quality` (with the running
        bad-view streak) so the owner can react — e.g. re-tune exposure
        on a washed-out view. Fail-open: a crash in the check never costs
        the view."""
        try:
            warning = self._quality.observe(report)
        except Exception:
            log.exception("camera-quality check failed — skipping")
            return None
        if warning is not None:
            log.warning("%s: %s", source, warning)
        if self._on_quality is not None:
            try:
                self._on_quality(report, self._quality.streak)
            except Exception:
                log.exception("quality-report callback failed — ignoring")
        return warning

    def peek_frame(self):
        """Park, grab the cropped phone-screen frame, and re-grab once
        after a settle if it's too blurry or blown — peek's acquisition
        path. Blown covers the AE-lag transient: the screen just flipped
        dark→bright and firmware auto-exposure hasn't re-converged yet.

        Returns `(frame, report)` — the report always describes the
        returned frame, so the caller's quality judgment
        (`observe_quality`) reuses it instead of re-measuring.

        Unlike `grab_screen`, failures propagate (a peek with no frame is
        a tool error, not a withheld verdict). A still-blurry retry frame
        is used as-is (no verdict to protect — better shown than
        dropped); a frame the `needs_fix` predicate flags (still blown,
        or a deferred tune just met its bright reference) escalates to
        the inline exposure fix, since a mis-exposed frame is
        correctable, not just observable. Caller must hold the lock."""
        self._park()
        frame = self._grab()
        report = quality.assess(frame)
        blurry = report.sharpness < self.PEEK_BLUR_THRESHOLD
        if blurry or report.blown:
            log.warning(
                "peek: %s frame (sharpness %.1f, clip %.0f%%) — retrying",
                "blurry" if blurry else "blown",
                report.sharpness,
                report.clip_pct * 100,
            )
            time.sleep(self.PEEK_RETRY_SECONDS)
            frame = self._grab()
            report = quality.assess(frame)
        if self._needs_fix(report):
            fixed_frame, report, fixed = self._fix_exposure_grab("peek", report)
            if fixed:
                frame = fixed_frame
        return frame, report

    def grab_screen(self, settle: float = 0.0):
        """Park (clearing the arm from the lens), let the screen and
        autofocus settle for `settle`s, then capture the cropped
        phone-screen frame for the gesture diff and fused view. Parking
        first means the settle covers the AF hunt the arm's move
        triggered, so the capture is usually sharp; a still-blurry frame
        is re-grabbed once as a fallback.

        A blown frame (auto-exposure still re-converging after the screen
        content flipped dark→bright) triggers the same wait-and-regrab as
        blur; when the `needs_fix` predicate then says the exposure
        itself is wrong (still blown, or a deferred tune just met its
        bright reference), the injected `fix_exposure` tune runs and the
        corrected frame replaces the mis-exposed one — the agent gets a
        readable view instead of a warning.

        Returns `(frame, sharp, report, retuned)`: frame is None on any
        failure (the view is best-effort, never a reason to fail a
        gesture) and the report — always describing the returned frame —
        feeds the caller's `observe_quality` so nothing is re-measured.
        sharp is False when even the retry stayed below
        GRAB_BLUR_THRESHOLD — such a frame still serves the fused view,
        but blur erases edges so it diffs as "changed everywhere": the
        caller must skip the verdict (None, the fail-open direction)
        rather than emit a false `changed` that would reset the stuck
        guard. retuned is True when the exposure was changed during THIS
        grab — a frame captured under a different exposure than its
        pair-mate diffs as changed everywhere too, so the caller must
        likewise skip the verdict. Caller must hold the lock."""
        try:
            self._park()
            if settle:
                time.sleep(settle)
            frame = self._grab()
            report = quality.assess(frame)
            blurry = report.sharpness < self.GRAB_BLUR_THRESHOLD
            retuned = False
            if blurry or report.blown:
                log.debug(
                    "gesture frame %s — waiting for AF/AE",
                    "blurry" if blurry else "blown",
                )
                time.sleep(self.GRAB_RETRY_SECONDS)
                frame = self._grab()
                report = quality.assess(frame)
            if self._needs_fix(report):
                fixed_frame, report, retuned = self._fix_exposure_grab(
                    "gesture view", report
                )
                if retuned:
                    frame = fixed_frame
            if report.sharpness < self.GRAB_BLUR_THRESHOLD:
                log.warning("gesture frame blurry — verdict withheld")
                return frame, False, report, retuned
            return frame, True, report, retuned
        except Exception:
            log.warning("screen frame grab failed", exc_info=True)
            return None, False, None, False

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
        from the gesture's arm move before capture. A brightness-class
        flip between the pair (median jump > FLIP_MEDIAN_DELTA) earns
        the after-grab an extra settle + regrab — firmware AE needs
        seconds to re-converge across such a flip, and luma statistics
        cannot flag the mid-swing white-out after the fact.
        """
        before, before_sharp, before_report, _ = self.grab_screen()
        result = action()
        after, after_sharp, after_report, after_retuned = self.grab_screen(
            settle=self.GESTURE_SETTLE_SECONDS
        )
        if (
            before_report is not None
            and after_report is not None
            and abs(after_report.median_luma - before_report.median_luma)
            > self.FLIP_MEDIAN_DELTA
        ):
            # Brightness-class flip: the after frame was likely grabbed
            # mid-AE-swing (see FLIP_MEDIAN_DELTA) — settle again and
            # regrab through the full quality pipeline.
            log.info(
                "gesture view: screen flipped brightness (median %.0f → %.0f) "
                "— extra settle for AE",
                before_report.median_luma,
                after_report.median_luma,
            )
            time.sleep(self.FLIP_SETTLE_SECONDS)
            after, after_sharp, after_report, flip_retuned = self.grab_screen()
            after_retuned = after_retuned or flip_retuned
        changed = None
        jpeg = listing = None
        if after is not None:
            # Verdict only from two comparable sharp frames — a blurry
            # side diffs as "changed everywhere" (false `changed`, the
            # harmful direction), and so does an exposure change between
            # the pair (an inline re-tune during the after-grab; one
            # during the BEFORE-grab is fine — both frames then share the
            # new exposure). The view below is best-effort either way.
            if (
                before is not None
                and before_sharp
                and after_sharp
                and not after_retuned
            ):
                try:
                    changed = frames_changed(before, after)
                except Exception:
                    log.debug("screen-verdict diff failed", exc_info=True)
            try:
                listing, annotated = self._detect(after)
                jpeg = encode_view_jpeg(annotated)
            except Exception:
                log.warning("post-gesture view failed", exc_info=True)
            warning = self.observe_quality("gesture view", after_report)
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
