"""Clipboard-sync state machine — miss counting, decay, and escalation.

The orchestrator's `_send_to_clipboard` queues text on the bridge,
long-presses AssistiveTouch, and waits for the phone's Shortcut to fetch
it. This module owns the policy around that wait: how long the confirm
window is, when repeated misses escalate the error text, and when stale
miss state is forgotten. Keeping it separate from the hardware call
makes the timing rules testable with an injected clock instead of
mocking `time.monotonic`.

State is guarded by the orchestrator's gesture lock — begin/confirm/
record_miss are only ever called with it held, so no locking here.
"""

import time
from typing import Callable


class ClipboardSyncError(RuntimeError):
    """AT long-press fired but the phone never fetched the queued text.

    Raised (not returned) so the `sequence` path ABORTS before any
    paste/send step runs against the phone's stale clipboard — returning
    a message here once let batches paste the previous query into an IM
    and hit send, repeatedly messaging garbage to a real person."""


class ClipboardSyncState:
    """Consecutive unconfirmed clipboard syncs, and what they imply.

    Drives the shorter retry timeout + escalating error text in the
    orchestrator's `_send_to_clipboard`; reset by any confirmed sync, or
    by `MISS_DECAY_SECONDS` elapsing since the last miss.
    """

    # A healthy clipboard Shortcut confirms in ~1s of the long-press; the
    # generous first window covers a cold Shortcuts app. After a miss the
    # Shortcut is almost certainly broken (moved AT button, permission
    # dialog, phone off the LAN) — don't burn the full window per retry.
    CONFIRM_SECONDS = 30.0
    RETRY_CONFIRM_SECONDS = 8.0
    # Miss state older than this is forgotten: the server process spans
    # sessions, and a "Miss #2 in a row" escalation (or the short retry
    # window) hours after an unrelated miss would be misleading — the
    # phone may have been fixed or rebooted in between.
    MISS_DECAY_SECONDS = 600.0

    def __init__(self, now: Callable[[], float] = time.monotonic):
        self._now = now
        self._misses = 0
        self._miss_at = 0.0  # clock reading of the last miss

    def begin(self) -> float:
        """Start a sync attempt: decay stale miss state, then return the
        confirm window to wait on the bridge."""
        if (
            self._misses
            and self._now() - self._miss_at > self.MISS_DECAY_SECONDS
        ):
            self._misses = 0
        return (
            self.CONFIRM_SECONDS
            if self._misses == 0
            else self.RETRY_CONFIRM_SECONDS
        )

    def confirm(self) -> None:
        """The phone fetched the text — any miss streak is over."""
        self._misses = 0

    def record_miss(self) -> str:
        """Count an unconfirmed sync; return the agent-facing error text,
        escalated once the misses look like a broken Shortcut rather than
        a one-off."""
        self._misses += 1
        self._miss_at = self._now()
        msg = (
            "clipboard sync FAILED — AssistiveTouch long-pressed but the "
            "phone never fetched the text; its clipboard still holds the "
            "PREVIOUS content, so do NOT paste"
        )
        if self._misses >= 2:
            msg += (
                f". Miss #{self._misses} in a row — the clipboard "
                "Shortcut is broken (moved AssistiveTouch button, a "
                "Shortcuts permission dialog, or the phone lost the LAN). "
                "STOP retrying this path: type with the keyboard instead, "
                "or report the problem to your user"
            )
        return msg
