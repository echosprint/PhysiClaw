"""Mechanical iOS passcode unlock — the one imperative gesture.

Unlike the data-recipe macros in ``gestures.py`` (home/back/force-quit),
unlocking is a timing dance: wake, swipe up, wait out the Face-ID
attempt, OCR-poll for the keypad "1", and — if that ran long enough that
the keypad has timed out — re-arm a fresh keypad and blind-tap the
already-located digit.

The passcode keypad "1" sits at a fixed screen position (screen-fraction
coordinates, stable across recalibrations), so ``PhoneUnlock`` caches it
on the first successful locate: later unlocks skip OCR entirely and just
wake, swipe, wait for the keypad, and blind-tap. Held by the orchestrator
facade (which stays thin) and run inside its lock + observation bracket
via injected collaborators: ``execute`` runs one validated gesture
(``PhysiClaw._execute``) and ``perception`` supplies the keypad OCR. The
caller holds the lock.
"""

import time
from typing import TYPE_CHECKING, Callable

from physiclaw.core.orchestration import gestures

if TYPE_CHECKING:
    from physiclaw.core.orchestration.perception import Perception

# Runs one already-validated gesture (``PhysiClaw._execute``).
Execute = Callable[[gestures.Gesture], str]

# --- Keypad timing (seconds, measured since the unlock swipe) ---------
# iOS runs a Face-ID attempt on swipe-up; the passcode numpad appears
# only after it fails. Wait this long after the swipe before the first
# OCR (or, on the cached path, before the tap) so we don't act on the
# Face-ID screen.
NUMPAD_APPEAR_SECONDS = 3.0
# OCR latency can outlast the numpad. If locating the "1" takes at least
# this long (wall-clock since the swipe), the keypad has likely timed
# out — re-arm it with a fresh swipe and blind-tap the known position
# rather than tapping a stale target (or paying another slow OCR).
STALE_SECONDS = 8.0
# Wall-clock to wait to before that re-swipe, so the stale numpad has
# fully reset first: a bottom-edge swipe on a LIVE numpad is iOS's cancel
# gesture, so it must be gone before we re-open it.
NUMPAD_RESET_SECONDS = 12.0


class PhoneUnlock:
    """Stateful passcode entry that caches the keypad "1" location.

    First unlock OCR-locates the digit and remembers its bbox; every
    later unlock skips the (slow) OCR and blind-taps the cached position.
    Instance is long-lived (held by the orchestrator facade), so the
    cache spans agent sessions within one server run.
    """

    def __init__(self) -> None:
        # Cached screen-fraction bbox of the "1" key; None until located.
        self._digit_bbox: list[float] | None = None

    def unlock_phone(self, execute: Execute, perception: "Perception") -> str:
        """Wake, open the passcode keypad, and tap the repdigit code.

        Runs inside the caller's lock + observation bracket. ``execute``
        runs one already-validated gesture; ``perception`` warms/holds the
        OCR and locates the keypad digit. Returns the action text the
        observer scans.
        """
        # Cached: we already know where "1" is — open the keypad and
        # blind-tap it, no OCR.
        if self._digit_bbox is not None:
            self._open_keypad(execute)
            self._tap_passcode(execute, self._digit_bbox)
            return "Passcode entered"

        # Cold: warm OCR (a cold RapidOCR load would outlast the keypad),
        # open the keypad, and locate the "1".
        perception.ocr_reader()
        swipe_at = self._open_keypad(execute)
        bbox = perception.wait_for_numpad_digit(gestures.UNLOCK_PASSCODE[0])
        if bbox is None:
            return "Failed to find passcode keypad"
        self._digit_bbox = bbox  # cache so later unlocks skip OCR entirely

        # If the locate outran the keypad's lifetime, the numpad is stale:
        # let it fully reset (a swipe on a LIVE one cancels), re-open a
        # fresh one, and blind-tap the located bbox — no second OCR.
        if time.monotonic() - swipe_at >= STALE_SECONDS:
            reset_in = NUMPAD_RESET_SECONDS - (time.monotonic() - swipe_at)
            if reset_in > 0:
                time.sleep(reset_in)
            execute(gestures.UNLOCK_SWIPE)
            time.sleep(NUMPAD_APPEAR_SECONDS)

        self._tap_passcode(execute, bbox)
        return "Passcode entered"

    def _open_keypad(self, execute: Execute) -> float:
        """Wake and swipe up to raise the passcode keypad, waiting out the
        Face-ID attempt. Returns the swipe's monotonic time (for the
        stale-keypad check)."""
        execute(gestures.WAKE_SCREEN)
        time.sleep(0.5)  # let the woken screen settle before the swipe
        execute(gestures.UNLOCK_SWIPE)
        swipe_at = time.monotonic()
        time.sleep(NUMPAD_APPEAR_SECONDS)
        return swipe_at

    def _tap_passcode(self, execute: Execute, bbox: list[float]) -> None:
        """Tap the repdigit passcode: the one key, once per digit."""
        for _ in range(len(gestures.UNLOCK_PASSCODE)):
            execute(gestures.Tap(bbox))
