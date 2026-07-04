"""Tests for `physiclaw.agent.engine.stuck` — the same-target press guard."""
from __future__ import annotations

import pytest

from physiclaw.agent.engine import stuck
from physiclaw.agent.engine.stuck import BLOCK_AT, WARN_AT, StuckGuard


BOX = [0.908, 0.526, 0.983, 0.562]  # the log's dead + stepper
NEAR_BOX = [0.911, 0.527, 0.983, 0.562]  # re-transcription jitter, same target
FAR_BOX = [0.100, 0.100, 0.200, 0.200]


def _guard() -> StuckGuard:
    return StuckGuard(_exempt=[])


def _press(guard: StuckGuard, bbox, *, tool: str = "tap", changed: bool | None = False):
    """One dispatched press with its verdict fed back. Returns the warning."""
    return guard.record(tool, {"bbox": bbox}, changed)


# ---------- counting & tiers ----------


def test_warns_on_third_fruitless_press() -> None:
    g = _guard()
    warnings = [_press(g, BOX) for _ in range(WARN_AT)]
    assert warnings[:-1] == [None, None]
    assert "press #3" in warnings[-1]
    assert "CONVENTION § Stuck" in warnings[-1]


def test_blocks_fifth_press_on_exhausted_target() -> None:
    g = _guard()
    for _ in range(BLOCK_AT - 1):
        _press(g, BOX)
    msg = g.should_block("tap", {"bbox": BOX})
    assert msg is not None and msg.startswith("BLOCKED")
    # Not exhausted one press earlier:
    g2 = _guard()
    for _ in range(BLOCK_AT - 2):
        _press(g2, BOX)
    assert g2.should_block("tap", {"bbox": BOX}) is None


def test_jittered_bbox_is_the_same_target() -> None:
    # The log's failure mode: tap → double_tap → nudged-coordinate tap on
    # one element. All of it is one target to the guard.
    g = _guard()
    _press(g, BOX, tool="tap")
    _press(g, NEAR_BOX, tool="double_tap")
    warning = _press(g, BOX, tool="long_press")
    assert warning is not None


def test_different_target_counts_separately() -> None:
    g = _guard()
    for _ in range(WARN_AT):
        _press(g, BOX)
    assert _press(g, FAR_BOX) is None
    assert g.should_block("tap", {"bbox": FAR_BOX}) is None


# ---------- resets & fail-open ----------


def test_screen_change_resets_the_target() -> None:
    # A qty stepper that actually increments must never MISS-trip or
    # block (the warn-only orbit advisory at press 5 is the designed
    # exception — it explicitly says "carry on if progressing").
    g = _guard()
    for _ in range(20):
        w = _press(g, BOX, changed=True)
        assert w is None or "same spot" in w
    assert g.should_block("tap", {"bbox": BOX}) is None
    # ...and a change after misses wipes the accumulated count.
    _press(g, BOX)
    _press(g, BOX)
    _press(g, BOX, changed=True)
    w = _press(g, BOX)
    assert w is None or "same spot" in w  # no miss warning: count restarted


def test_no_verdict_fails_open() -> None:
    # No verdict → no miss counting, no block. (The press still executed,
    # so the warn-only orbit advisory legitimately fires once.)
    g = _guard()
    for _ in range(20):
        w = _press(g, BOX, changed=None)
        assert w is None or "same spot" in w
    assert g.should_block("tap", {"bbox": BOX}) is None


def test_step_change_resets_all_targets() -> None:
    g = _guard()
    g.observe_step("update quantity to 4")
    for _ in range(BLOCK_AT - 1):
        _press(g, BOX)
    assert g.should_block("tap", {"bbox": BOX}) is not None
    g.observe_step("pay for the order")
    assert g.should_block("tap", {"bbox": BOX}) is None


def test_same_step_observation_keeps_counters() -> None:
    g = _guard()
    g.observe_step("step A")
    for _ in range(WARN_AT - 1):
        _press(g, BOX)
    g.observe_step("step A")
    assert _press(g, BOX) is not None  # warn fires — nothing was reset


# ---------- exemptions & non-press tools ----------


def test_swipes_never_counted() -> None:
    g = _guard()
    args = {"bbox": BOX, "direction": "up"}
    for _ in range(20):
        assert g.record("swipe", args, False) is None
    assert g.should_block("swipe", args) is None


@pytest.mark.parametrize("tool", ["peek", "screenshot", "note", "wait", "go_back"])
def test_non_gesture_tools_ignored(tool: str) -> None:
    g = _guard()
    assert g.record(tool, {}, False) is None
    assert g.should_block(tool, {}) is None


def test_learned_keyboard_keys_exempt(mocker) -> None:
    backspace = [0.864, 0.804, 0.986, 0.856]
    mocker.patch.object(
        stuck.screen_layout, "repeatable_key_boxes", return_value=[backspace],
    )
    g = StuckGuard()
    for _ in range(20):
        assert g.record("tap", {"bbox": backspace}, False) is None
    assert g.should_block("tap", {"bbox": backspace}) is None


def test_malformed_bbox_ignored() -> None:
    g = _guard()
    assert g.record("tap", {"bbox": ["x", 1, 2]}, False) is None
    assert g.record("tap", {}, False) is None
    assert g.should_block("tap", {"bbox": None}) is None


# ---------- sequence steps ----------


def test_sequence_press_steps_counted() -> None:
    g = _guard()
    args = {"actions": [
        {"tool_name": "tap", "arg": BOX},
        {"tool_name": "send_to_clipboard", "arg": "hello"},
    ]}
    for _ in range(BLOCK_AT - 1):
        g.record("sequence", args, False)
    # The exhausted target blocks the whole batch that includes it...
    assert g.should_block("sequence", args) is not None
    # ...and blocks the bare press too.
    assert g.should_block("tap", {"bbox": NEAR_BOX}) is not None


def test_sequence_with_clean_targets_not_blocked() -> None:
    g = _guard()
    args = {"actions": [{"tool_name": "tap", "arg": FAR_BOX}]}
    assert g.should_block("sequence", args) is None


def test_sequence_malformed_actions_ignored() -> None:
    g = _guard()
    assert g.record("sequence", {"actions": "not-a-list"}, False) is None
    assert g.record("sequence", {}, False) is None


# ---------- ping-pong ----------


BOX_A = [0.100, 0.100, 0.200, 0.200]
BOX_B = [0.700, 0.700, 0.800, 0.800]


def _pp(guard: StuckGuard, bbox, *, tool: str = "tap") -> None:
    """One executed, screen-CHANGING press — the verdict that same-target
    counting ignores but ping-pong must still see."""
    guard.record(tool, {"bbox": bbox}, True)


def test_pingpong_warns_then_blocks() -> None:
    g = _guard()
    for _ in range(WARN_AT - 1):
        _pp(g, BOX_A)
        _pp(g, BOX_B)
    _pp(g, BOX_A)
    # The WARN_AT-th pair completes on this press:
    warning = g.record("tap", {"bbox": BOX_B}, True)
    assert warning is not None and "repeated the same action cycle" in warning

    for _ in range(BLOCK_AT - WARN_AT - 1):
        _pp(g, BOX_A)
        _pp(g, BOX_B)
    _pp(g, BOX_A)
    # Continuing the alternation would be the BLOCK_AT-th pair:
    msg = g.should_block("tap", {"bbox": BOX_B})
    assert msg is not None and msg.startswith("BLOCKED")
    assert "action cycle" in msg


def test_pingpong_fires_despite_changed_verdicts() -> None:
    # The whole point: screen-changing loops evade same-target counting.
    g = _guard()
    for _ in range(BLOCK_AT):
        _pp(g, BOX_A)
        _pp(g, BOX_B)
    assert g.should_block("tap", {"bbox": BOX_A}) is not None
    # ...but a genuinely third target breaks the pattern and is free —
    # same-target never tripped either (all verdicts were `changed`):
    third = [0.400, 0.400, 0.500, 0.500]
    assert g.should_block("tap", {"bbox": third}) is None


def test_pingpong_nav_and_press_alternation() -> None:
    # The yogurt-log shape: tap product row ↔ go_back, forever.
    g = _guard()
    for _ in range(BLOCK_AT):
        _pp(g, BOX_A)
        g.record("go_back", {}, True)
    assert g.should_block("tap", {"bbox": BOX_A}) is not None


def test_pingpong_browsing_pattern_not_flagged() -> None:
    # tap(item1) → back → tap(item2) → back → … : the press targets
    # differ each round, so no two-signature alternation forms.
    g = _guard()
    for i in range(10):
        bbox = [0.1, 0.05 * i, 0.2, 0.05 * i + 0.05]
        _pp(g, bbox)
        g.record("go_back", {}, True)
    assert g.should_block("go_back", {}) is None


def test_pingpong_same_action_repetition_not_flagged() -> None:
    # Feed scrolling: swipe up ×10 is A,A,A… — not an alternation.
    g = _guard()
    for _ in range(10):
        g.record("swipe", {"bbox": BOX_A, "direction": "up"}, True)
    assert g.should_block("swipe", {"bbox": BOX_A, "direction": "up"}) is None


def test_pingpong_swipe_direction_hunting_flagged() -> None:
    # up/down/up/down scroll hunting is the swipe-shaped loop.
    g = _guard()
    for _ in range(BLOCK_AT):
        g.record("swipe", {"bbox": BOX_A, "direction": "up"}, True)
        g.record("swipe", {"bbox": BOX_A, "direction": "down"}, True)
    assert g.should_block("swipe", {"bbox": BOX_A, "direction": "up"}) is not None


def test_pingpong_broken_by_a_third_action() -> None:
    g = _guard()
    for _ in range(BLOCK_AT):
        _pp(g, BOX_A)
        _pp(g, BOX_B)
    _pp(g, FAR_BOX)  # C breaks the A,B tail
    assert g.should_block("tap", {"bbox": BOX_A}) is None


def test_pingpong_step_change_resets_history() -> None:
    g = _guard()
    g.observe_step("step A")
    for _ in range(BLOCK_AT):
        _pp(g, BOX_A)
        _pp(g, BOX_B)
    g.observe_step("step B")
    assert g.should_block("tap", {"bbox": BOX_A}) is None


def test_pingpong_exempt_keyboard_keys_never_tracked(mocker) -> None:
    # Typing "ababab…" on two keyboard keys must not read as ping-pong.
    key_a = [0.100, 0.800, 0.160, 0.860]
    key_b = [0.300, 0.800, 0.360, 0.860]
    mocker.patch.object(
        stuck.screen_layout, "repeatable_key_boxes", return_value=[key_a, key_b],
    )
    g = StuckGuard()
    for _ in range(10):
        g.record("tap", {"bbox": key_a}, True)
        g.record("tap", {"bbox": key_b}, True)
    assert g.should_block("tap", {"bbox": key_a}) is None


def test_pingpong_views_do_not_break_pattern() -> None:
    # A peek between the alternating actions has no signature — the
    # pattern continues through it.
    g = _guard()
    for _ in range(BLOCK_AT):
        _pp(g, BOX_A)
        g.record("peek", {}, None)
        _pp(g, BOX_B)
    assert g.should_block("tap", {"bbox": BOX_A}) is not None


def test_pingpong_history_bounded() -> None:
    g = _guard()
    for i in range(100):
        bbox = [0.01 * (i % 40), 0.5, 0.01 * (i % 40) + 0.05, 0.6]
        _pp(g, bbox)
    assert len(g._history) <= stuck._HISTORY_MAX


# ---------- period-3 cycles & error loops ----------


def test_period3_cycle_flagged() -> None:
    # cart → product row → checkout → cart → … (the log's nav loop).
    g = _guard()
    box_c = [0.400, 0.400, 0.500, 0.500]
    for _ in range(BLOCK_AT):
        _pp(g, BOX_A)
        _pp(g, BOX_B)
        _pp(g, box_c)
    assert g.should_block("tap", {"bbox": BOX_A}) is not None


def test_period3_distinct_content_not_flagged() -> None:
    # search → add → back with DIFFERENT items each round is a legit
    # workflow — differing tap targets break the cycle identity.
    g = _guard()
    for i in range(8):
        item = [0.1, 0.06 * i, 0.2, 0.06 * i + 0.05]
        _pp(g, item)
        _pp(g, BOX_B)          # "add to cart" button, same every round
        g.record("go_back", {}, True)
    assert g.should_block("go_back", {}) is None


def test_error_repeat_warns_then_blocks() -> None:
    # The AssistiveTouch-overlap trap: identical call raises every time.
    g = _guard()
    warnings = [g.record_error("tap", {"bbox": BOX}) for _ in range(WARN_AT)]
    assert warnings[:-1] == [None, None]
    assert "failed" in warnings[-1]

    for _ in range(BLOCK_AT - 1 - WARN_AT):
        g.record_error("tap", {"bbox": BOX})
    msg = g.should_block("tap", {"bbox": BOX})
    assert msg is not None and "failed" in msg


def test_error_count_cleared_by_success() -> None:
    g = _guard()
    for _ in range(BLOCK_AT - 1):
        g.record_error("tap", {"bbox": BOX})
    _press(g, BOX, changed=True)  # the call finally worked
    assert g.should_block("tap", {"bbox": BOX}) is None


def test_error_repeat_different_args_counted_separately() -> None:
    g = _guard()
    for _ in range(10):
        g.record_error("tap", {"bbox": BOX})
    assert g.should_block("tap", {"bbox": FAR_BOX}) is None


def test_error_repeat_step_change_resets() -> None:
    g = _guard()
    g.observe_step("step A")
    for _ in range(BLOCK_AT):
        g.record_error("tap", {"bbox": BOX})
    g.observe_step("step B")
    assert g.should_block("tap", {"bbox": BOX}) is None


# ---------- position orbit (detector 4, warn-only) ----------


def test_orbit_warns_on_fifth_same_pos_press_mixed_everything() -> None:
    # 5 of the last 10 presses on one spot — across gesture types,
    # verdicts, errors, and interleaved other targets — must warn.
    g = _guard()
    g.record("tap", {"bbox": BOX}, True)            # 1: changed
    g.record("tap", {"bbox": FAR_BOX}, True)        # other spot
    g.record("double_tap", {"bbox": BOX}, False)    # 2: no change
    g.record_error("long_press", {"bbox": BOX})     # 3: raised
    g.record("tap", {"bbox": FAR_BOX}, True)        # other spot
    g.record("tap", {"bbox": BOX}, True)            # 4: changed
    warning = g.record("tap", {"bbox": NEAR_BOX}, True)  # 5: jittered

    assert warning is not None and "same spot" in warning
    # Advisory only — nothing blocks:
    assert g.should_block("tap", {"bbox": BOX}) is None


def test_orbit_warns_once_per_position() -> None:
    g = _guard()
    for _ in range(4):
        g.record("tap", {"bbox": BOX}, True)
    first = g.record("tap", {"bbox": BOX}, True)
    later = [g.record("tap", {"bbox": BOX}, True) for _ in range(4)]

    assert first is not None and "same spot" in first
    assert all(w is None or "same spot" not in w for w in later)


def test_orbit_silent_below_threshold() -> None:
    g = _guard()
    for _ in range(4):
        assert g.record("tap", {"bbox": BOX}, True) is None


def test_orbit_window_slides() -> None:
    # 4 presses on BOX, then 10 elsewhere pushes them out of the window —
    # the next BOX press is 1-of-10, not 5-of-anything.
    g = _guard()
    for _ in range(4):
        g.record("tap", {"bbox": BOX}, True)
    for i in range(10):
        other = [0.05, 0.05 * i + 0.01, 0.10, 0.05 * i + 0.05]
        g.record("tap", {"bbox": other}, True)
    assert g.record("tap", {"bbox": BOX}, True) is None


def test_orbit_step_change_resets() -> None:
    g = _guard()
    g.observe_step("a")
    for _ in range(4):
        g.record("tap", {"bbox": BOX}, True)
    g.observe_step("b")
    assert g.record("tap", {"bbox": BOX}, True) is None


def test_orbit_exempt_keys_never_tracked(mocker) -> None:
    # Typing runs — backspace, space, return — must never enter the
    # orbit window, whatever the mix of keys and verdicts.
    backspace = [0.864, 0.804, 0.986, 0.856]
    ret = [0.749, 0.864, 0.987, 0.916]
    space = [0.256, 0.865, 0.742, 0.916]
    mocker.patch.object(
        stuck.screen_layout, "repeatable_key_boxes",
        return_value=[backspace, ret, space],
    )
    g = StuckGuard()
    for _ in range(10):
        assert g.record("tap", {"bbox": backspace}, True) is None
        assert g.record("tap", {"bbox": space}, False) is None
        assert g.record_error("tap", {"bbox": ret}) is None
    assert g._presses == []  # nothing entered the window at all
