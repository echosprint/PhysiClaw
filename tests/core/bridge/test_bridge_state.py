"""Tests for `physiclaw.core.bridge.state` — thread-safe LAN bridge.

The class wraps shared state behind a `threading.Lock` and uses two
`threading.Event` instances for the clipboard / screenshot signals.
Tests drive state changes via the public methods on a single thread —
race conditions aren't validated, only the post-condition state.

`time.time` and `time.monotonic` are mocked when needed; `time.sleep`
gets stubbed so waits return immediately. `save_screenshot` is patched
to a no-op so writes don't hit disk.

Accepted equivalent / unreachable mutmut survivors:

  - `module.log = logging.getLogger(__name__)` ↔ `log = None` — the
    module body never invokes the logger; the import is reserved for
    future debug use.
  - Local-variable type annotations `Type | None` ↔ `Type & None` —
    annotations are not evaluated at runtime.
  - Multi-line `_screenshot_data` declaration collapsed to one line —
    formatting only, no behavior change.
  - Inside `wait_for_connection`, `time.monotonic() - stable_since`
    ↔ `time.monotonic() + stable_since` — both expressions cross
    `settle_seconds` for any monotonically increasing time stream;
    the sign affects WHEN (not WHETHER) the helper returns True, and
    asserting iteration count via sleep-call-count would be a
    brittle implementation test.
  - Loop deadline check `time.monotonic() < deadline` ↔ `<= deadline`
    — discriminator only fires when monotonic equals deadline to the
    nanosecond, which the time-mocking harness can't reliably hit.
"""

from __future__ import annotations

import itertools
import time

import pytest

from physiclaw.core.bridge import state as state_mod
from physiclaw.core.bridge.state import BridgeState


@pytest.fixture(autouse=True)
def _no_real_screenshot_writes(mocker) -> None:
    mocker.patch.object(state_mod, "save_screenshot")


@pytest.fixture
def bs() -> BridgeState:
    return BridgeState()


# ---------- init ----------


def test_init_starts_with_no_text_and_no_screenshot(bs: BridgeState) -> None:
    assert bs.current_text() is None
    assert bs._screenshot_data is None
    assert bs.last_seen == 0


def test_init_clipboard_and_screenshot_events_start_unset(bs: BridgeState) -> None:
    assert bs._clipboard_copied.is_set() is False
    assert bs._screenshot_ready.is_set() is False


# ---------- connected property ----------


def test_connected_false_at_init() -> None:
    bs = BridgeState()  # last_seen = 0; time.time() far in the future

    assert bs.connected is False


def test_connected_true_when_polled_within_500ms(bs: BridgeState, mocker) -> None:
    fake_time = mocker.patch.object(state_mod.time, "time")
    fake_time.return_value = 1000.0
    bs.poll()  # writes last_seen = 1000.0
    fake_time.return_value = 1000.4  # 400ms later

    assert bs.connected is True


def test_connected_false_when_last_poll_was_over_500ms_ago(
    bs: BridgeState, mocker
) -> None:
    fake_time = mocker.patch.object(state_mod.time, "time")
    fake_time.return_value = 1000.0
    bs.poll()
    fake_time.return_value = 1000.6  # 600ms later — over the 500ms cutoff

    assert bs.connected is False


def test_connected_false_at_exactly_500ms_boundary(bs: BridgeState, mocker) -> None:
    # `< 0.5` (strict). Exactly 0.5s difference must NOT count as
    # connected — mutating to `<= 0.5` would flip this case.
    fake_time = mocker.patch.object(state_mod.time, "time")
    fake_time.return_value = 1000.0
    bs.poll()
    fake_time.return_value = 1000.5  # exactly 500ms

    assert bs.connected is False


# ---------- wait_for_connection ----------


def test_wait_for_connection_returns_true_after_settle_period(
    bs: BridgeState, mocker
) -> None:
    # Fake time advances 0.1s per call (an infinite stream). With
    # connected=True always and settle_seconds=0.5, the running stable
    # interval crosses the threshold within a few iterations.
    mocker.patch.object(state_mod.time, "sleep")  # no real sleep
    mocker.patch.object(BridgeState, "connected", new=True)
    counter = itertools.count(start=0.0, step=0.1)
    mocker.patch.object(state_mod.time, "monotonic", side_effect=lambda: next(counter))

    assert bs.wait_for_connection(timeout=10.0, settle_seconds=0.5) is True


def test_wait_for_connection_returns_false_when_timeout_elapses_disconnected(
    bs: BridgeState, mocker
) -> None:
    mocker.patch.object(state_mod.time, "sleep")
    mocker.patch.object(BridgeState, "connected", new=False)
    counter = itertools.count(start=0.0, step=0.3)
    mocker.patch.object(state_mod.time, "monotonic", side_effect=lambda: next(counter))

    assert bs.wait_for_connection(timeout=0.5, settle_seconds=10.0) is False


def test_wait_for_connection_resets_settle_timer_on_dropout(
    bs: BridgeState, mocker
) -> None:
    # connected sequence: True, False (drops), True, True, True...
    # With dropout, stable_since resets to None — settle timer restarts.
    mocker.patch.object(state_mod.time, "sleep")
    connected_seq = itertools.chain([True, False], itertools.repeat(True))

    class _Prop:
        def __get__(self, instance, owner=None):
            return next(connected_seq)

    mocker.patch.object(BridgeState, "connected", new=_Prop())
    counter = itertools.count(start=0.0, step=0.2)
    mocker.patch.object(state_mod.time, "monotonic", side_effect=lambda: next(counter))

    # settle_seconds=0.5, timeout=10 — eventually the long True-stretch
    # accumulates 0.5s of stable time and the helper returns True.
    assert bs.wait_for_connection(timeout=10.0, settle_seconds=0.5) is True


def test_wait_for_connection_sleeps_with_200ms_polling_interval(
    bs: BridgeState, mocker
) -> None:
    # The sleep-between-iterations interval is 0.2s; mutating it would
    # change the polling cadence noticeably.
    sleep = mocker.patch.object(state_mod.time, "sleep")
    mocker.patch.object(BridgeState, "connected", new=False)
    counter = itertools.count(start=0.0, step=0.5)
    mocker.patch.object(state_mod.time, "monotonic", side_effect=lambda: next(counter))

    bs.wait_for_connection(timeout=1.0, settle_seconds=10.0)

    sleep.assert_called_with(0.2)


def test_wait_for_connection_default_settle_seconds_is_one() -> None:
    import inspect

    sig = inspect.signature(BridgeState.wait_for_connection)
    assert sig.parameters["settle_seconds"].default == 1.0


def test_wait_for_connection_returns_true_at_exact_settle_boundary(
    bs: BridgeState, mocker
) -> None:
    # `>= settle_seconds` (inclusive). Crafted timeline:
    #   call 1: deadline   = 0.0 + 10 = 10.0
    #   iter 1 while       = 0.1  (in deadline)
    #   iter 1 stable_since= 0.2
    #   sleep ×1
    #   iter 2 while       = 0.3
    #   iter 2 elif        = 0.4   →  0.4 − 0.2 = 0.2 == settle_seconds
    # Original `>=` returns True here. Mutating to `>` requires a 3rd
    # iteration — at least one extra sleep call would land.
    sleep = mocker.patch.object(state_mod.time, "sleep")
    mocker.patch.object(BridgeState, "connected", new=True)
    times = iter([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    mocker.patch.object(state_mod.time, "monotonic", side_effect=lambda: next(times))

    bs.wait_for_connection(timeout=10.0, settle_seconds=0.2)

    # Original returns at exact-equal boundary after one sleep. A `>`
    # mutation would force at least a 2nd sleep before passing.
    assert sleep.call_count == 1


# ---------- send_text / clear_text / current_text ----------


def test_send_text_stores_value_and_clears_clipboard_event(bs: BridgeState) -> None:
    bs._clipboard_copied.set()  # simulate stale signal

    bs.send_text("hello")

    assert bs.current_text() == "hello"
    assert bs._clipboard_copied.is_set() is False


def test_clear_text_resets_text_to_none_and_clears_event(bs: BridgeState) -> None:
    bs.send_text("hello")
    bs._clipboard_copied.set()

    bs.clear_text()

    assert bs.current_text() is None
    assert bs._clipboard_copied.is_set() is False


# ---------- fetch_text ----------


def test_fetch_text_returns_text_and_marks_clipboard_copied(bs: BridgeState) -> None:
    bs.send_text("payload")

    text = bs.fetch_text()

    assert text == "payload"
    assert bs._clipboard_copied.is_set() is True


def test_fetch_text_when_no_text_returns_none_and_does_not_set_event(
    bs: BridgeState,
) -> None:
    text = bs.fetch_text()

    assert text is None
    assert bs._clipboard_copied.is_set() is False


# ---------- is_copied ----------


def test_is_copied_false_before_any_copy(bs: BridgeState) -> None:
    bs.send_text("payload")

    assert bs.is_copied() is False


def test_is_copied_true_after_fetch(bs: BridgeState) -> None:
    bs.send_text("payload")
    bs.fetch_text()

    assert bs.is_copied() is True


def test_is_copied_true_after_tap_confirmation(bs: BridgeState) -> None:
    bs.mark_clipboard_copied()

    assert bs.is_copied() is True


def test_is_copied_resets_when_new_text_queued(bs: BridgeState) -> None:
    # New text hasn't been copied yet — the flag must not linger from the
    # previous copy, or the phone would falsely show a stale confirmation.
    bs.mark_clipboard_copied()
    bs.send_text("fresh")

    assert bs.is_copied() is False


def test_is_copied_resets_when_text_cleared(bs: BridgeState) -> None:
    bs.mark_clipboard_copied()
    bs.clear_text()

    assert bs.is_copied() is False


# ---------- mark_clipboard_copied / wait_clipboard ----------


def test_mark_clipboard_copied_sets_the_event(bs: BridgeState) -> None:
    bs.mark_clipboard_copied()

    assert bs._clipboard_copied.is_set() is True


def test_wait_clipboard_returns_true_when_signal_already_set(
    bs: BridgeState,
) -> None:
    bs.mark_clipboard_copied()

    assert bs.wait_clipboard(timeout=0.01) is True


def test_wait_clipboard_returns_false_when_timeout_elapses(
    bs: BridgeState,
) -> None:
    assert bs.wait_clipboard(timeout=0.01) is False


def test_wait_clipboard_default_timeout_is_30_seconds() -> None:
    import inspect

    sig = inspect.signature(BridgeState.wait_clipboard)
    assert sig.parameters["timeout"].default == 30.0


# ---------- poll ----------


def test_poll_updates_last_seen_to_current_time(bs: BridgeState, mocker) -> None:
    mocker.patch.object(state_mod.time, "time", return_value=12345.0)

    bs.poll()

    assert bs.last_seen == 12345.0


# ---------- screenshot ----------


def test_receive_screenshot_stores_bytes_and_signals_waiters(
    bs: BridgeState,
) -> None:
    payload = b"\x89PNG fake bytes"

    bs.receive_screenshot(payload)

    assert bs._screenshot_data == payload
    assert bs._screenshot_ready.is_set() is True


def test_receive_screenshot_persists_via_save_screenshot(
    bs: BridgeState, mocker
) -> None:
    save = mocker.patch.object(state_mod, "save_screenshot")

    bs.receive_screenshot(b"x")

    save.assert_called_once_with(b"x")


def test_clear_screenshot_resets_data_and_event(bs: BridgeState) -> None:
    bs.receive_screenshot(b"first")

    bs.clear_screenshot()

    assert bs._screenshot_data is None
    assert bs._screenshot_ready.is_set() is False


# ─── recent-screenshots ring ──────────────────────────────────


def test_recent_screenshots_empty_at_init(bs: BridgeState) -> None:
    assert bs.recent_screenshots() == []


def test_recent_screenshots_returns_uploads_oldest_to_newest(
    bs: BridgeState,
) -> None:
    for p in (b"a", b"b", b"c"):
        bs.receive_screenshot(p)

    assert bs.recent_screenshots() == [b"a", b"b", b"c"]


def test_recent_screenshots_caps_at_max(bs: BridgeState) -> None:
    for i in range(state_mod.RECENT_SCREENSHOTS_MAX + 5):
        bs.receive_screenshot(bytes([i]))

    shots = bs.recent_screenshots()
    assert len(shots) == state_mod.RECENT_SCREENSHOTS_MAX
    # Oldest 5 evicted; newest retained.
    assert shots[-1] == bytes([state_mod.RECENT_SCREENSHOTS_MAX + 4])


def test_recent_screenshots_honours_n(bs: BridgeState) -> None:
    for p in (b"a", b"b", b"c", b"d"):
        bs.receive_screenshot(p)

    assert bs.recent_screenshots(2) == [b"c", b"d"]


def test_clear_screenshot_does_not_touch_ring(bs: BridgeState) -> None:
    # The MCP consume path must not disturb the ring the layout tool reads.
    bs.receive_screenshot(b"first")
    bs.clear_screenshot()
    bs.wait_screenshot(timeout=0.01)

    assert bs.recent_screenshots() == [b"first"]


def test_wait_screenshot_returns_data_when_already_received(
    bs: BridgeState,
) -> None:
    bs.receive_screenshot(b"image-bytes")

    out = bs.wait_screenshot(timeout=0.01)

    assert out == b"image-bytes"


def test_wait_screenshot_clears_event_after_returning(bs: BridgeState) -> None:
    # Once consumed, a subsequent wait should block again — the event
    # gets cleared inside wait_screenshot so the next caller doesn't
    # see a stale signal.
    bs.receive_screenshot(b"first")
    bs.wait_screenshot(timeout=0.01)

    assert bs._screenshot_ready.is_set() is False


def test_wait_screenshot_returns_none_when_timeout_elapses(
    bs: BridgeState,
) -> None:
    assert bs.wait_screenshot(timeout=0.01) is None


def test_wait_screenshot_default_timeout_is_10_seconds() -> None:
    import inspect

    sig = inspect.signature(BridgeState.wait_screenshot)
    assert sig.parameters["timeout"].default == 10.0


# ---------- upload arming + phone IP pin ----------


def test_clear_screenshot_arms_and_receive_consumes(bs: BridgeState) -> None:
    assert not bs.upload_armed()
    bs.clear_screenshot()
    assert bs.upload_armed()
    bs.receive_screenshot(b"png")
    assert not bs.upload_armed()  # one window, one upload


def test_arm_upload_opens_window_without_touching_pending(bs: BridgeState) -> None:
    # Calibrate steps arm ahead of the user's manual double-tap; a pending
    # screenshot must survive (arming is not a consume/clear).
    bs.receive_screenshot(b"pending")
    bs.arm_upload()
    assert bs.upload_armed()
    assert bs._screenshot_data == b"pending"
    assert bs._screenshot_ready.is_set()


def test_arm_upload_extends_never_shrinks(bs: BridgeState) -> None:
    bs.arm_upload(100.0)
    deadline = bs._upload_armed_until
    bs.arm_upload(1.0)
    assert bs._upload_armed_until == deadline


def test_take_pending_screenshot_returns_none_when_nothing_pending(
    bs: BridgeState,
) -> None:
    assert bs.take_pending_screenshot() is None


def test_take_pending_screenshot_consumes_recent_upload(bs: BridgeState) -> None:
    cutoff = time.monotonic()
    bs.receive_screenshot(b"early-upload")

    assert bs.take_pending_screenshot(received_after=cutoff) == b"early-upload"
    # Consumed — a second take (or a wait) must not see it again.
    assert bs.take_pending_screenshot(received_after=cutoff) is None
    assert bs._screenshot_ready.is_set() is False


def test_take_pending_screenshot_rejects_upload_before_cutoff(
    bs: BridgeState,
) -> None:
    bs.receive_screenshot(b"stale")
    cutoff = time.monotonic()

    # Received before the cutoff (e.g. before the square was shown) —
    # left pending, not consumed.
    assert bs.take_pending_screenshot(received_after=cutoff) is None
    assert bs._screenshot_ready.is_set() is True


def test_wait_screenshot_arms_for_its_own_duration(bs: BridgeState) -> None:
    # Manual flows (viewport pre-cal) wait without a prior clear — the
    # wait itself must open the window.
    bs.wait_screenshot(timeout=0.0)
    assert bs.upload_armed()


def test_upload_ip_pin_tofu_then_match(bs: BridgeState) -> None:
    assert bs.upload_ip_allowed("192.168.1.30")  # first LAN sender pins
    assert not bs.upload_ip_allowed("192.168.1.66")  # fresh pin → reject
    assert bs.upload_ip_allowed("192.168.1.30")  # pinned IP stays fine


def test_upload_ip_pin_always_allows_loopback(bs: BridgeState) -> None:
    assert bs.upload_ip_allowed("127.0.0.1")
    assert bs.upload_ip_allowed(None)


def test_upload_ip_pin_repins_when_stale(
    bs: BridgeState, monkeypatch: pytest.MonkeyPatch
) -> None:
    # DHCP heal: the pinned IP going quiet past the staleness window lets
    # the next sender re-pin instead of being locked out until restart.
    assert bs.upload_ip_allowed("192.168.1.30")
    monkeypatch.setattr(state_mod, "PIN_STALE_SECONDS", 0.0)
    assert bs.upload_ip_allowed("192.168.1.66")


def test_note_phone_ip_repins_immediately(bs: BridgeState) -> None:
    # A live bridge-page poll IS the phone — it may move the pin at once,
    # no staleness wait (the page reload after an IP change).
    assert bs.upload_ip_allowed("192.168.1.30")
    bs.note_phone_ip("192.168.1.66")
    assert bs.upload_ip_allowed("192.168.1.66")
    assert not bs.upload_ip_allowed("192.168.1.30")
