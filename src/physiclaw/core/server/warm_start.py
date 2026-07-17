"""Warm-start: resume from the saved Calibration bundle.

``try_resume`` (called only when ``physiclaw server --warm-start`` or
``--hot-start`` is passed) loads the bundle from disk into
``rig.calibration``, reconnects hardware, runs an end-to-end sanity tap,
and flips the ready flag only if every test passes. ``--hot-start``
(``verify=False``) is the same resume minus every step that touches the
phone: no bridge wait, no sanity tap, no home-screen swipe — the user
vouches that nothing moved. A plain ``physiclaw server`` boot ignores
the bundle entirely — see ``core/server/app.py``.

The resting-position invariant is what makes warm-start work at all. The
arm parks at the off-screen spot (``PARK_PCT``) between every operation
and on clean shutdown, so on reconnect the tip is sitting there — not at
the calibrated origin. ``arm.setup()``'s ``G92 X0 Y0`` would wrongly
declare the park spot to be ``(0, 0)`` and shift the whole frame, so
``restore_park_origin`` re-pins the origin from the park coordinate to
keep ``pct_to_grbl`` valid. The sanity tap is the only mechanism that
catches violations of this invariant (crash mid-move, power cut, arm
bumped — anything that leaves the tip somewhere other than the park
spot). Under ``--hot-start`` nothing catches them: the first symptom of
a violated invariant is a gesture landing off-target.
"""

import logging
import sys
from typing import TYPE_CHECKING

from physiclaw.common.config import CONFIG

# Re-exported from core.server.net (its true home) so existing callers —
# `warm_start.wait_for_port`, `from ...warm_start import wait_for_port` —
# keep working. The redundant alias marks it an intentional re-export.
from physiclaw.core.server.net import wait_for_port as wait_for_port

if TYPE_CHECKING:
    from physiclaw.core.bridge import CalibrationState, PageState
    from physiclaw.core.orchestration import PhysiClaw

log = logging.getLogger(__name__)


def resume_flag(verify: bool) -> str:
    """User-facing CLI flag for a resume mode — the single source of the
    verify → ``--warm-start`` / ``--hot-start`` mapping, shared with
    ``cli/server.py``'s log messages."""
    return "--warm-start" if verify else "--hot-start"


# How long to wait for the phone to (re)load /bridge, in seconds.
BRIDGE_WAIT_TIMEOUT = CONFIG.warm_start.bridge_wait_timeout_seconds
# After a /bridge load is detected, let the page finish rendering before
# we start tapping dots at it.
BRIDGE_SETTLE_SECONDS = CONFIG.warm_start.bridge_settle_seconds


def _sanity(
    physiclaw: "PhysiClaw", calib: "CalibrationState", phone: "PageState"
) -> bool:
    """Run a compact end-to-end tap verification. Returns True iff every
    tap landed within tolerance. No-touches counts as failure — warm-start
    only succeeds when we've proven the calibration still holds.

    On failure, logs the specific diagnosis (no touches = /bridge not open;
    touches but off = arm/phone/camera moved) so the caller can stay terse.
    """
    from physiclaw.core.calibration.calibrate import validate_calibration

    rig = physiclaw.rig
    cal = rig.calibration
    phone.set_mode("calibrate")
    try:
        results = validate_calibration(
            rig.arm,
            rig.cam,
            calib,
            cal.effective_rotation(),
            cal.pct_to_grbl,
            cal.pct_to_cam,
            cam_size=cal.cam_size,
            num_tests=2,
        )
    finally:
        phone.set_mode("bridge")

    total = len(results)
    received = sum(1 for r in results if r.get("error", 999) < 999)
    passed = sum(1 for r in results if r["passed"])
    if passed == total:
        log.info(f"--warm-start: sanity passed ({passed}/{total} taps)")
        return True
    if received == 0:
        log.error(
            "--warm-start: sanity — no taps registered. "
            "Is the phone's /bridge page open and foregrounded?"
        )
    else:
        log.error(
            f"--warm-start: sanity — {passed}/{total} taps within tolerance "
            f"({received}/{total} touches received). Calibration looks stale "
            f"(arm, phone, or camera likely moved since last setup)."
        )
    return False


def try_resume(
    cam_index_override: int | None,
    *,
    verify: bool = True,
    physiclaw: "PhysiClaw",
    calib: "CalibrationState",
    phone: "PageState",
) -> bool:
    """Connect hardware, run sanity, flip ready if everything holds.

    Camera index comes from ``--cam-index`` if provided, else from
    ``bundle.cam_index``, else 0.

    ``verify=False`` (``--hot-start``) trusts the resting-position
    invariant outright: the bridge wait, the sanity tap, and the final
    home-screen swipe are all skipped, so the phone is never touched and
    the server is ready as soon as hardware reconnects.

    The assembled state objects come in as parameters (the caller,
    ``cli/server.py``, passes the default assembly's) — no back-import
    into ``core.server.app``, so this runs against fakes in tests.

    Returns True on success; False (with a logged reason) if the bundle
    is incomplete, hardware reconnect fails, or sanity doesn't pass.
    The caller exits non-zero so the user can fall back to plain
    ``uv run physiclaw`` + ``setup.py``.
    """
    from physiclaw.core.calibration.state import BUNDLE_PATH, Calibration

    flag = resume_flag(verify)
    rig = physiclaw.rig
    loaded = Calibration.load()
    if loaded is None:
        if BUNDLE_PATH.exists():
            # The file is there but didn't parse (corrupt, or an
            # incompatible schema version) — a different fix than
            # "run setup for the first time".
            log.error(
                f"{flag}: calibration bundle on disk is unreadable or "
                "incompatible — recalibrate with `physiclaw setup hardware`"
            )
        else:
            log.error(f"{flag}: no calibration bundle on disk")
        return False
    log.info(
        f"{flag}: loaded bundle (rotation={loaded.cam_rotation}, "
        f"complete={loaded.complete})"
    )
    if not loaded.complete:
        # Gate BEFORE any state lands in the session: an incomplete
        # bundle must not leak a stale viewport shift / screen dimension
        # into the live calib state the interactive wizard then runs
        # against.
        log.error(f"{flag}: bundle on disk is incomplete")
        return False
    rig.calibration = loaded
    # Mirror into the bridge-side state so calibration handlers that
    # read `calib.viewport_shift` (e.g. show_assistive_touch) see it,
    # and restore the CSS-pt dimensions so warm-start's validate can run
    # without waiting for the phone's /bridge page to POST them again.
    calib.viewport_shift = loaded.viewport_shift
    rig.assistive_touch.compute_at_screen_pos(loaded.viewport_shift)
    calib.screen_dimension = loaded.screen_dimension

    cal = rig.calibration
    cam_index = (
        cam_index_override
        if cam_index_override is not None
        else (cal.cam_index if cal.cam_index is not None else 0)
    )
    # Hold the hardware lock from BEFORE the reconnect: connect_arm /
    # connect_camera flip `hardware_ready` the moment they succeed, and the
    # MCP plane is already serving — an unlocked gap between connect and
    # restore_park_origin would let a concurrent tool call pass
    # require_hardware() and drive (or auto-park) the arm in the mis-pinned
    # G92 frame arm.setup() just declared (physically displacing the tip),
    # or grab the lock first so the resume's own non-blocking acquire aborts
    # with "busy". Acquiring up front makes the invariant structural: tools
    # can never observe hardware_ready with the lock free until the origin
    # is re-pinned; racing callers fail fast with the busy error instead.
    # (rig.locked() can't be used here — its require_hardware() gate would
    # reject the not-yet-connected rig.)
    rig.acquire()
    origin_pinned = False
    try:
        try:
            rig.connect_arm()
            rig.connect_camera(cam_index)
        except Exception as e:
            log.error(f"{flag}: hardware reconnect failed: {e}")
            return False

        # The camera may have negotiated a different resolution than the
        # bundle was calibrated at (e.g. the [camera] config changed, or the
        # default moved). Same aspect → the 0-1 mapping holds, adopt the live
        # pixel size; different aspect → the mapping is broken, recalibrate.
        live = rig.cam.peek()
        if live is None:
            log.error(f"{flag}: camera produced no frame")
            return False
        live_h, live_w = live.shape[:2]
        if not cal.reconcile_cam_size((live_w, live_h)):
            log.error(
                f"{flag}: capture aspect ratio changed since calibration "
                "— run `physiclaw` setup again"
            )
            return False

        # The tip rests at the park spot, not the calibrated origin that
        # arm.setup() just assumed — re-pin the frame from it (see
        # restore_park_origin). The sanity tap below catches the cases where
        # the tip didn't hold that spot (killed mid-move, power yank, arm
        # bumped).
        rig.restore_park_origin()
        origin_pinned = True
        if verify:
            if sys.stdin.isatty():
                print()
                print("━" * 60)
                print("Warm-start")
                print(
                    "  Open or refresh /bridge on the phone (foreground, not locked)."
                )
                print(
                    f"  Server waits up to {BRIDGE_WAIT_TIMEOUT}s for steady polling."
                )
                print("━" * 60)
                if not rig.bridge.wait_for_connection(
                    BRIDGE_WAIT_TIMEOUT, BRIDGE_SETTLE_SECONDS
                ):
                    log.error(
                        f"{flag}: /bridge page not polling within "
                        f"{BRIDGE_WAIT_TIMEOUT}s — open or refresh /bridge on the phone."
                    )
                    return False
            else:
                log.info(f"{flag}: non-interactive; running sanity immediately")

            if not _sanity(physiclaw, calib, phone):
                # _sanity logged the specific diagnosis.
                return False
        else:
            log.info(
                f"{flag}: trusting the parked arm — skipping bridge wait, "
                "sanity tap, and home-screen swipe"
            )
    finally:
        # Mirror locked()'s exit: auto-park restores the resting spot on
        # every path — but only once the origin is re-pinned, because park()
        # in the mis-pinned frame would itself displace the tip (the exact
        # hazard the lock span prevents). home_screen() below runs after the
        # release: it takes locked() itself.
        if origin_pinned:
            try:
                rig.park()
            except Exception:
                log.exception(f"{flag}: auto-park after resume failed")
        rig.release()
    if verify:
        # Match setup.py's final step: send the phone home (swipe from
        # bottom), then flip ready. home_screen's locked() context auto-parks
        # the arm off-screen on exit, so nothing is hovering over the glass
        # afterward.
        physiclaw.home_screen()
    # Settle the camera, then flip ready — become_ready owns that order
    # (under --warm-start the home screen is showing — the dark scene that
    # exposes AE failure; under --hot-start whatever's on screen still
    # beats not checking at all). Synchronous is fine: try_resume already
    # runs on a background thread, and the settle half is fail-open +
    # bounded by frame-wait timeouts.
    physiclaw.become_ready()
    log.info(f"{flag}: resumed from bundle (cam={cam_index}) — MCP tools ready")
    return True
