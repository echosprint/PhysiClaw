"""BridgeState — text-to-clipboard transfer and screenshot upload state.

Thread-safe shared state between Starlette route handlers and blocking
MCP tool threads.
"""

import logging
import threading
import time
from collections import deque

from physiclaw.common.dumps import save_screenshot

log = logging.getLogger(__name__)

# Bounded rolling window of recent raw uploads the layout tool can fetch.
RECENT_SCREENSHOTS_MAX = 3

# Upload arming: POST /api/bridge/screenshot is only accepted while a
# consumer is expecting one. `clear_screenshot()` (the tap-AT choke point)
# opens this window; `wait_screenshot()` extends it past its own timeout so
# manual flows (viewport pre-cal's "double-tap AT yourself") stay covered.
UPLOAD_WINDOW_SECONDS = 60.0
# Slack past a waiter's timeout: an upload racing the waiter's last poll
# should still land rather than 403.
UPLOAD_WINDOW_GRACE_SECONDS = 5.0
# Calibration asks the user to double-tap AssistiveTouch themselves; eager
# taps land before wait_screenshot() arms, so entering calibrate mode opens
# a wider window up front (see PageState.set_mode).
CAL_UPLOAD_WINDOW_SECONDS = 120.0
# Upload-eligible IP freshness: an IP that polled /api/bridge/state (or had
# an upload accepted) within this window may upload. A TTL'd SET, not a
# single "last poller wins" pin: pinning the most recent poller flip-flopped
# whenever a second /bridge tab (a leftover laptop tab) polled alongside the
# phone, 403ing the real phone's screenshot roughly every other upload — and
# the iOS Shortcut doesn't retry, so each rejection became a 60s screenshot
# timeout. Expiry doubles as the DHCP heal: an address that goes quiet ages
# out, and once nothing fresh remains the next armed uploader is admitted
# trust-on-first-use. While the phone is active its entry stays fresh, so a
# hijack needs the phone (and every other poller) silent this long AND an
# armed window — the off-path attacker the gate targets doesn't get both.
PHONE_IP_TTL_SECONDS = 180.0

# Callers on the host itself bypass the phone IP pin entirely.
_LOOPBACK_IPS = frozenset({"127.0.0.1", "::1"})


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
        self._screenshot_received_at: float = 0.0  # monotonic, last upload arrival
        # Recent raw uploads, oldest→newest. Independent of the consume path
        # above (wait/clear never touch it), so a reader can't disturb it.
        self._recent_screens: deque[bytes] = deque(maxlen=RECENT_SCREENSHOTS_MAX)
        # Upload gate: monotonic deadline until which an upload is expected
        # (0.0 = disarmed), plus the TTL'd set of upload-eligible IPs —
        # ip → monotonic last-seen (see PHONE_IP_TTL_SECONDS).
        self._upload_armed_until: float = 0.0
        self._phone_ips: dict[str, float] = {}

    @property
    def connected(self) -> bool:
        """True if phone polled within the last 0.5 seconds.

        bridge.html polls every 250ms, so 0.5s = 2× the interval — tight
        enough to detect a tab going to sleep almost immediately, loose
        enough to ride through one dropped poll.
        """
        return time.time() - self.last_seen < 0.5

    def wait_for_connection(self, timeout: float, settle_seconds: float = 1.0) -> bool:
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
        The copied event is set in the SAME lock hold as the read, so
        :meth:`expire_text` can atomically distinguish "already fetched"
        from "never fetched" on the sync-timeout path.
        """
        with self.lock:
            text = self._text
            if text is not None:
                self._clipboard_copied.set()
        return text

    def expire_text(self) -> bool:
        """Retire the queued text after a sync timeout. Returns True if it
        had in fact already been copied — the caller should treat the sync
        as a (late) success.

        The check-and-clear shares one lock hold with :meth:`fetch_text`'s
        read-and-mark: a late Shortcut fetching in the gap between the
        caller's ``wait_clipboard`` timing out and a plain ``clear_text()``
        would put the text on the phone clipboard while the caller reports
        it didn't — and leave the copied event set with no text queued.
        On the copied path the text stays queued (matching a confirmed
        sync); otherwise it is cleared so a later fetch can never deliver
        it after the caller reported failure.
        """
        with self.lock:
            if self._clipboard_copied.is_set():
                return True
            self._text = None
            return False

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
        ``wait_screenshot()`` sees consistent data. Disarms the upload
        window — one armed window admits one upload. When
        ``PHYSICLAW_SAVE_SCREENSHOTS`` is set, also dumps the raw bytes
        to ``data/screenshots/``.
        """
        save_screenshot(data)
        with self.lock:
            self._screenshot_data = data
            self._screenshot_received_at = time.monotonic()
            self._recent_screens.append(data)
            self._upload_armed_until = 0.0
        self._screenshot_ready.set()

    def clear_screenshot(self):
        """Clear any pending screenshot so wait_screenshot blocks for a fresh one.

        Also arms the upload window: this is called exactly when a consumer
        is about to trigger an upload (tap AT / start a verify), so it is
        the natural "expect an upload now" signal.

        Only touches the consume path — the `_recent_screens` window is
        deliberately left intact so a fetch after the MCP tool consumed a
        shot still sees it.
        """
        self._screenshot_ready.clear()
        with self.lock:
            self._screenshot_data = None
            self._arm_upload_locked(UPLOAD_WINDOW_SECONDS)

    def _arm_upload_locked(self, seconds: float):
        """Extend (never shrink) the upload window. Caller holds the lock."""
        self._upload_armed_until = max(
            self._upload_armed_until, time.monotonic() + seconds
        )

    def arm_upload(self, seconds: float = UPLOAD_WINDOW_SECONDS):
        """Open the upload window ahead of an expected manual upload,
        without touching any pending screenshot. Entering calibrate mode
        calls this so a user's eager double-tap isn't 403'd."""
        with self.lock:
            self._arm_upload_locked(seconds)

    def take_pending_screenshot(self, received_after: float = 0.0) -> bytes | None:
        """Consume the pending screenshot if it arrived after `received_after`
        (monotonic), else return None and leave state untouched.

        Lets a step use an upload the user already sent — e.g. double-tapping
        while the calibrate page shows the orange square, before pressing
        Measure — instead of making them upload again. The cutoff excludes
        shots that predate whatever the step needs on screen."""
        if not self._screenshot_ready.is_set():
            return None
        with self.lock:
            if self._screenshot_data is None:
                return None
            if self._screenshot_received_at <= received_after:
                return None
            self._screenshot_ready.clear()
            return self._screenshot_data

    def upload_armed(self) -> bool:
        """True while an upload is expected — a `clear_screenshot()` or a
        live `wait_screenshot()` opened the window and it hasn't expired
        or been consumed."""
        with self.lock:
            return time.monotonic() < self._upload_armed_until

    # ─── Phone IP gate (TTL'd recently-seen set) ──────────────

    def _prune_phone_ips_locked(self, now: float) -> None:
        """Drop IPs not seen within `PHONE_IP_TTL_SECONDS`. Caller holds
        the lock."""
        self._phone_ips = {
            ip: seen
            for ip, seen in self._phone_ips.items()
            if now - seen <= PHONE_IP_TTL_SECONDS
        }

    def upload_ip_allowed(self, ip: str | None) -> bool:
        """Gate an upload by source IP: any IP seen recently — a live
        /api/bridge/state poll or a prior accepted upload within
        `PHONE_IP_TTL_SECONDS` — is accepted. When no fresh IP is known,
        the first non-loopback uploader is admitted trust-on-first-use.
        Loopback is always allowed (the host itself is trusted). The gate
        is risk reduction, not authentication: it excludes only the
        *off-path* LAN sender (e.g. a compromised IoT device) — an
        on-path attacker on the same Wi-Fi can spoof an admitted address.
        """
        if ip is None or ip in _LOOPBACK_IPS:
            return True
        now = time.monotonic()
        with self.lock:
            self._prune_phone_ips_locked(now)
            if ip in self._phone_ips:
                self._phone_ips[ip] = now
                return True
            if self._phone_ips:
                # Fresh senders are known and this isn't one of them.
                return False
            # Nothing fresh (first upload ever, or everyone aged out —
            # e.g. the phone came back from a DHCP lease change): admit
            # and remember the sender.
            self._phone_ips[ip] = now
        log.info(f"Bridge: no fresh phone IP — admitted uploader {ip}")
        return True

    def note_phone_ip(self, ip: str | None):
        """Record a live bridge-page poller as upload-eligible. Every
        recently-seen poller stays eligible (see `PHONE_IP_TTL_SECONDS`),
        not just the latest one: a single last-poller-wins pin flip-flopped
        whenever a second /bridge tab polled alongside the phone, 403ing
        the real phone's uploads roughly every other time. A poll is also
        the freshest "this is the phone" signal, so a post-DHCP-change
        address is admitted immediately, without waiting out any TTL."""
        if ip is None or ip in _LOOPBACK_IPS:
            return
        now = time.monotonic()
        with self.lock:
            self._prune_phone_ips_locked(now)
            new = ip not in self._phone_ips
            self._phone_ips[ip] = now
        if new:
            log.info(f"Bridge: upload-eligible phone IP {ip}")

    def recent_screenshots(self, n: int = RECENT_SCREENSHOTS_MAX) -> list[bytes]:
        """Snapshot of the last `n` raw uploads, oldest→newest. Read-only
        w.r.t. the MCP screenshot pipeline."""
        with self.lock:
            shots = list(self._recent_screens)
        return shots[-n:] if n > 0 else shots

    def wait_screenshot(self, timeout: float = 10.0) -> bytes | None:
        """Block until a screenshot arrives, or timeout. Returns PNG/JPEG bytes.

        Called from the blocking MCP tool thread. Arms the upload window
        for its own duration (some flows wait without a prior
        `clear_screenshot`, e.g. viewport pre-cal's manual double-tap).
        Waits on the event without holding the lock (otherwise
        receive_screenshot couldn't acquire it to store data). Once
        signaled, grabs the lock briefly to read the data safely.
        """
        with self.lock:
            self._arm_upload_locked(timeout + UPLOAD_WINDOW_GRACE_SECONDS)
        if self._screenshot_ready.wait(timeout=timeout):
            self._screenshot_ready.clear()
            with self.lock:
                return self._screenshot_data
        return None
