"""BridgeState — text-to-clipboard transfer and screenshot upload state.

Thread-safe shared state between Starlette route handlers and blocking
MCP tool threads.
"""

import logging
import threading
import time

log = logging.getLogger(__name__)


class BridgeState:
    """Shared state for the LAN bridge between server and phone browser.

    The phone opens /bridge in Safari and polls GET /api/bridge/state every
    250ms. The server returns the current state; the phone renders it.
    The phone is stateless — it always renders from the latest server response.

    Two data flows use this state:

    1. Text → clipboard:
       - Agent calls send_text("hello") → sets self.text
       - Phone polls, sees text, displays it large on screen
       - Agent's arm physically taps the phone screen
       - Phone JS copies text to clipboard on touch event
       - Phone POSTs /api/bridge/tapped → mark_clipboard_copied()
       - Agent's bridge_tap() tool was blocking on wait_clipboard(),
         now unblocks and returns success

    2. Screenshot upload:
       - Agent taps AssistiveTouch on phone (via arm)
       - iOS Shortcut fires: takes screenshot, POSTs image bytes
         to /api/bridge/screenshot → receive_screenshot()
       - Agent's phone_screenshot() tool was blocking on
         wait_screenshot(), now unblocks and returns the image

    Thread-safe: accessed from async Starlette route handlers and
    blocking MCP tool threads concurrently.
    """

    def __init__(self):
        self.lock = threading.Lock()  # protects shared fields (text, screenshot) across threads
        self.text: str | None = None  # current text displayed on phone bridge page
        self.last_seen: float = 0  # timestamp of last phone poll, for connection detection
        self._clipboard_copied = threading.Event()  # set when phone confirms tap-to-copy
        self._screenshot_data: bytes | None = None  # PNG/JPEG bytes from iOS Shortcut upload
        self._screenshot_ready = threading.Event()  # set when screenshot upload arrives

    @property
    def connected(self) -> bool:
        """True if phone polled within the last 5 seconds."""
        return time.time() - self.last_seen < 5.0

    def send_text(self, text: str):
        """Set text for the phone to display and copy on tap."""
        with self.lock:
            self.text = text
            self._clipboard_copied.clear()

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
        """Store an uploaded screenshot, save to disk, and signal waiters.

        Called from the async route handler when the iOS Shortcut POSTs
        the image. Saves to data/phone/screenshot/, writes data under lock,
        then signals the event so wait_screenshot() sees consistent data.
        """
        from datetime import datetime
        from pathlib import Path

        save_dir = Path("data/phone/screenshot")
        save_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        ext = "png" if data[:4] == b"\x89PNG" else "jpg"
        save_path = save_dir / f"{ts}.{ext}"
        save_path.write_bytes(data)
        log.info(f"Bridge: screenshot saved to {save_path} ({len(data)} bytes)")

        with self.lock:
            self._screenshot_data = data
        self._screenshot_ready.set()

    def clear_screenshot(self):
        """Clear any pending screenshot so wait_screenshot blocks for a fresh one."""
        self._screenshot_ready.clear()
        with self.lock:
            self._screenshot_data = None

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
