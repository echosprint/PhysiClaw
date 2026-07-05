"""BridgeState — text-to-clipboard transfer and screenshot upload state.

Thread-safe shared state between Starlette route handlers and blocking
MCP tool threads.
"""

import logging
import threading
import time
from collections import deque

from physiclaw.core.logger import save_screenshot

log = logging.getLogger(__name__)

# Bounded rolling window of recent raw uploads the layout tool can fetch.
RECENT_SCREENSHOTS_MAX = 3


class BridgeState:
    """Shared state for the LAN bridge between server and phone browser.

    The phone opens /bridge in Safari and polls GET /api/bridge/state every
    250ms. The server returns the current state; the phone renders it.
    The phone is stateless — it always renders from the latest server response.

    Three data flows use this state:

    1. Text → clipboard via tap (page must be open):
       - Agent calls send_text("hello") → sets self.text
       - Phone polls, sees text, displays it large on screen
       - Agent's arm physically taps the phone screen
       - Phone JS copies text to clipboard on touch event
       - Phone POSTs /api/bridge/tapped → mark_clipboard_copied()
       - Agent's bridge_tap() tool was blocking on wait_clipboard(),
         now unblocks and returns success

    2. Text → clipboard via iOS Shortcut (no page needed):
       - Agent calls send_text("hello") → sets self.text
       - User long-presses AssistiveTouch (or any trigger) to run the
         "PhysiClaw Clipboard" Shortcut
       - Shortcut GETs /api/bridge/clipboard → server returns the text
         and calls mark_clipboard_copied()
       - Shortcut writes the response into the iOS clipboard directly
       - This path bypasses the bridge page and the physical tap entirely.

    3. Screenshot upload:
       - Agent taps AssistiveTouch on phone (via arm)
       - iOS Shortcut fires: takes screenshot, POSTs image bytes
         to /api/bridge/screenshot → receive_screenshot()
       - Agent's phone_screenshot() tool was blocking on
         wait_screenshot(), now unblocks and returns the image

    Thread-safe: accessed from async Starlette route handlers and
    blocking MCP tool threads concurrently.
    """

    def __init__(self):
        self.lock = (
            threading.Lock()
        )  # protects shared fields (text, screenshot) across threads
        self._text: str | None = None  # current text queued for the phone bridge page
        self.last_seen: float = (
            0  # timestamp of last phone poll, for connection detection
        )
        self._clipboard_copied = (
            threading.Event()
        )  # set when phone confirms tap-to-copy
        self._screenshot_data: bytes | None = (
            None  # PNG/JPEG bytes from iOS Shortcut upload
        )
        self._screenshot_ready = threading.Event()  # set when screenshot upload arrives
        # Recent raw uploads, oldest→newest. Independent of the consume path
        # above (wait/clear never touch it), so a reader can't disturb it.
        self._recent_screens: deque[bytes] = deque(maxlen=RECENT_SCREENSHOTS_MAX)

    @property
    def connected(self) -> bool:
        """True if phone polled within the last 0.5 seconds.

        bridge.html polls every 250ms, so 0.5s = 2× the interval — tight
        enough to detect a tab going to sleep almost immediately, loose
        enough to ride through one dropped poll.
        """
        return time.time() - self.last_seen < 0.5

    def wait_for_connection(
        self, timeout: float, settle_seconds: float = 1.0
    ) -> bool:
        """Block until the phone has been polling steadily for
        `settle_seconds` continuously, or `timeout` elapses.

        Sustained polling implies the page is foreground and actively
        repainting; `connected` alone could be a tab that briefly opened
        and got backgrounded. Any dropout resets the clock. Used before
        operations that require active canvas rendering on the phone
        (auto-pick RGBM corners, warm-start sanity tap).
        """
        deadline = time.monotonic() + timeout
        stable_since: float | None = None
        while time.monotonic() < deadline:
            if self.connected:
                if stable_since is None:
                    stable_since = time.monotonic()
                elif time.monotonic() - stable_since >= settle_seconds:
                    return True
            else:
                stable_since = None
            time.sleep(0.2)
        return False

    def send_text(self, text: str):
        """Set text for the phone to display and copy on tap."""
        with self.lock:
            self._text = text
            self._clipboard_copied.clear()

    def clear_text(self):
        """Clear any queued text so the phone bridge page goes blank."""
        with self.lock:
            self._text = None
            self._clipboard_copied.clear()

    def current_text(self) -> str | None:
        """Thread-safe read of the queued text. Returns None if empty."""
        with self.lock:
            return self._text

    def is_copied(self) -> bool:
        """True once the queued text has been copied to the phone clipboard
        — via the clipboard Shortcut fetch (:meth:`fetch_text`) or a
        confirmed tap (:meth:`mark_clipboard_copied`). Cleared whenever new
        text is queued or the text is cleared, so it always refers to the
        CURRENT text. Drives the phone's on-screen "copied — paste to
        verify" confirmation (see ``PageState.get_state``)."""
        return self._clipboard_copied.is_set()

    def fetch_text(self) -> str | None:
        """Atomically read the queued text and mark it as copied.

        Used by /api/bridge/clipboard so the iOS Shortcut can fetch the text
        in one round trip — no separate confirm-tap is needed. Returns None
        if no text is queued; in that case the clipboard event is not set.
        """
        with self.lock:
            text = self._text
        if text is not None:
            self._clipboard_copied.set()
        return text

    def mark_clipboard_copied(self):
        """Phone confirms the tap-to-copy succeeded."""
        self._clipboard_copied.set()

    def wait_clipboard(self, timeout: float = 30.0) -> bool:
        """Block until phone confirms clipboard copy, or timeout."""
        return self._clipboard_copied.wait(timeout=timeout)

    def poll(self):
        """Update last-seen timestamp (called on every phone poll)."""
        self.last_seen = time.time()

    # ─── Screenshot upload ────────────────────────────────────

    def receive_screenshot(self, data: bytes):
        """Store an uploaded screenshot and signal waiters.

        Called from the async route handler when the iOS Shortcut POSTs
        the image. Writes data under lock, then signals the event so
        ``wait_screenshot()`` sees consistent data. When
        ``PHYSICLAW_SAVE_SCREENSHOTS`` is set, also dumps the raw bytes
        to ``data/screenshots/``.
        """
        save_screenshot(data)
        with self.lock:
            self._screenshot_data = data
            self._recent_screens.append(data)
        self._screenshot_ready.set()

    def clear_screenshot(self):
        """Clear any pending screenshot so wait_screenshot blocks for a fresh one.

        Only touches the consume path — the `_recent_screens` window is
        deliberately left intact so a fetch after the MCP tool consumed a
        shot still sees it.
        """
        self._screenshot_ready.clear()
        with self.lock:
            self._screenshot_data = None

    def recent_screenshots(self, n: int = RECENT_SCREENSHOTS_MAX) -> list[bytes]:
        """Snapshot of the last `n` raw uploads, oldest→newest. Read-only
        w.r.t. the MCP screenshot pipeline."""
        with self.lock:
            shots = list(self._recent_screens)
        return shots[-n:] if n > 0 else shots

    def wait_screenshot(self, timeout: float = 10.0) -> bytes | None:
        """Block until a screenshot arrives, or timeout. Returns PNG/JPEG bytes.

        Called from the blocking MCP tool thread. Waits on the event
        without holding the lock (otherwise receive_screenshot couldn't
        acquire it to store data). Once signaled, grabs the lock briefly
        to read the data safely.
        """
        if self._screenshot_ready.wait(timeout=timeout):
            self._screenshot_ready.clear()
            with self.lock:
                return self._screenshot_data
        return None
