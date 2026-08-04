"""Tests for `physiclaw.agent.macros.runner` — replaying a spec over a
fake MCP caller: step order, args substitution, per-step guards (free
check, step-1 peek, forbid tripwire), the `wait` step, abort semantics, the
single-verdict result composition, and the run+record fold."""

from __future__ import annotations

import pytest

from physiclaw.agent.macros import runner as runner_mod
from physiclaw.agent.macros import stats as macro_stats
from physiclaw.agent.macros.model import (
    REASON_BAD_INPUT,
    REASON_EXPECT_FAILED,
    REASON_GUARD_FAILED,
    REASON_TIMEOUT,
    REASON_TOOL_ERROR,
    MacroError,
    Screen,
)
from physiclaw.agent.macros.parse import parse_macro
from physiclaw.agent.macros.runner import run, run_and_record
from physiclaw.common import verdict
from physiclaw.common.listing import format_row

IMAGE = {"type": "image", "mime_type": "image/jpeg", "data": "aGk="}


class FakeCaller:
    """Scripted MCP caller: pops one reply (or exception) per call and
    records `(name, args)` pairs."""

    def __init__(self, replies: list):
        self.replies = list(replies)
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, args: dict | None = None) -> list[dict]:
        self.calls.append((name, args or {}))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _gesture(
    text: str, *, changed: bool | None = True, listing: str = ""
) -> list[dict]:
    blocks = [{"type": "text", "text": verdict.attach(text, changed)}, IMAGE]
    if listing:
        blocks.append({"type": "text", "text": listing})
    return blocks


def _spec(text: str, name: str = "demo"):
    return parse_macro(text, name)


TWO_STEPS = """
name: demo
description: d
inputs:
  msg:
    description: text
steps:
  - name: home-screen-3
    tool: home_screen
  - name: stage
    tool: send_to_clipboard
    with:
      text: "{msg}"
"""


def _guarded(guard_body: str):
    """The guard tests' shared scaffold: home_screen, then a guarded tap."""
    indented = "".join(f"      {line}\n" for line in guard_body.splitlines())
    return _spec(
        "name: demo\ndescription: d\nsteps:\n"
        "  - name: home-screen-1\n    tool: home_screen\n"
        "  - name: open-app\n"
        "    tool: tap\n"
        "    with:\n"
        "      bbox: [0.1, 0.2, 0.3, 0.4]\n"
        f"    guard:\n{indented}"
    )


GUARDED = 'require: "WeChat"\nhint: "open it manually"'
GUARD_FORBID = 'forbid: "Upgrade now"'

GUARD_FIRST = """
name: demo
description: d
steps:
  - name: entry
    tool: tap
    with:
      bbox: [0.1, 0.2, 0.3, 0.4]
    guard:
      require: "Home"
"""

WAIT_STEPS = """
name: demo
description: d
steps:
  - name: go-home
    tool: home_screen
  - name: settle
    tool: wait
    with:
      seconds: 3
"""

# The canonical replacement for the old `wait_seconds` guard: settle, then
# check. The wait invalidates the held listing, so the guard re-reads.
WAIT_THEN_GUARD = """
name: demo
description: d
steps:
  - name: go-home
    tool: home_screen
  - name: settle
    tool: wait
    with:
      seconds: 2
  - name: open-app
    tool: tap
    with:
      bbox: [0.1, 0.2, 0.3, 0.4]
    guard:
      require: "WeChat"
"""


# ---------- happy path ----------


async def test_run_calls_each_step_with_substituted_args() -> None:
    mcp = FakeCaller([_gesture("went home"), _gesture("copied")])

    await run(_spec(TWO_STEPS), {"msg": "hello"}, mcp)

    assert mcp.calls == [
        ("home_screen", {}),
        ("send_to_clipboard", {"text": "hello"}),
    ]


async def test_run_success_result_reports_all_steps() -> None:
    mcp = FakeCaller([_gesture("went home"), _gesture("copied")])

    result = await run(_spec(TWO_STEPS), {"msg": "hello"}, mcp)

    assert result.ok is True
    assert result.aborted_step is None
    log = result.blocks[0]["text"]
    assert "all 2 steps completed" in log
    assert "✓ 1. home_screen" in log
    assert '✓ 2. send_to_clipboard "stage"' in log


async def test_run_keeps_only_last_step_view() -> None:
    first = _gesture("went home", listing="home listing")
    second = _gesture("copied", listing="final listing")
    mcp = FakeCaller([first, second])

    result = await run(_spec(TWO_STEPS), {"msg": "hello"}, mcp)

    texts = [b["text"] for b in result.blocks if b["type"] == "text"]
    assert not any("home listing" in t for t in texts)
    assert any("final listing" in t for t in texts)
    assert sum(1 for b in result.blocks if b["type"] == "image") == 1


async def test_run_attaches_last_verdict_to_first_block_only() -> None:
    mcp = FakeCaller(
        [_gesture("went home", changed=True), _gesture("copied", changed=False)]
    )

    result = await run(_spec(TWO_STEPS), {"msg": "hello"}, mcp)

    assert verdict.parse(result.blocks[0]["text"]) is False
    later = "\n".join(b["text"] for b in result.blocks[1:] if b["type"] == "text")
    assert verdict.parse(later) is None  # retained view is defanged


async def test_run_step_log_marks_changed_and_no_change() -> None:
    mcp = FakeCaller(
        [_gesture("went home", changed=True), _gesture("copied", changed=False)]
    )

    result = await run(_spec(TWO_STEPS), {"msg": "hello"}, mcp)

    log = result.blocks[0]["text"]
    assert "✓ 1. home_screen" in log and "(changed)" in log
    assert "(no change)" in log


# ---------- guards: free check against the previous view ----------


async def test_run_guard_passes_when_text_in_previous_result() -> None:
    mcp = FakeCaller([_gesture("went home", listing="WeChat 微信"), _gesture("tapped")])

    result = await run(_guarded(GUARDED), {}, mcp)

    assert result.ok is True
    assert len(mcp.calls) == 2  # no peek needed — free check passed


async def test_run_guard_failure_aborts_before_the_call() -> None:
    mcp = FakeCaller([_gesture("went home", listing="Settings only")])

    result = await run(_guarded(GUARDED), {}, mcp)

    assert result.ok is False
    assert result.aborted_step == 2
    assert result.reason == REASON_GUARD_FAILED
    assert "'WeChat'" in result.detail
    assert ("tap", {"bbox": [0.1, 0.2, 0.3, 0.4]}) not in mcp.calls


async def test_run_guard_abort_carries_hint_for_the_agent() -> None:
    mcp = FakeCaller([_gesture("went home", listing="Settings only")])

    result = await run(_guarded(GUARDED), {}, mcp)

    assert "(hint: open it manually)" in result.blocks[0]["text"]
    assert "(hint: open it manually)" in result.detail


async def test_run_guard_failure_result_carries_previous_view() -> None:
    mcp = FakeCaller([_gesture("went home", listing="Settings only")])

    result = await run(_guarded(GUARDED), {}, mcp)

    log = result.blocks[0]["text"]
    assert "ABORTED at step 2/2" in log
    assert '✗ 2. tap "open-app"' in log
    assert any("Settings only" in b.get("text", "") for b in result.blocks[1:])


GUARD_ANY = 'require: {or: ["微信", "WeChat"]}'


@pytest.mark.parametrize("listing", ["chats 微信 here", "chats WeChat here"])
async def test_run_require_any_of_passes_on_either_alternative(listing: str) -> None:
    mcp = FakeCaller([_gesture("went home", listing=listing), _gesture("tapped")])

    result = await run(_guarded(GUARD_ANY), {}, mcp)

    assert result.ok is True


async def test_run_require_any_of_miss_names_all_alternatives() -> None:
    mcp = FakeCaller([_gesture("went home", listing="Settings only")])

    result = await run(_guarded(GUARD_ANY), {}, mcp)

    assert result.ok is False
    assert "require ('微信' or 'WeChat') not on screen" in result.detail


GUARD_REGION = 'require: {text: "微信", within: [0.2, 0.2, 0.8, 0.3]}'


async def test_run_region_guard_passes_when_element_center_inside() -> None:
    row = format_row(3, "text", "微信", [0.4, 0.22, 0.6, 0.28], 0.93)
    mcp = FakeCaller([_gesture("went home", listing=row), _gesture("tapped")])

    result = await run(_guarded(GUARD_REGION), {}, mcp)

    assert result.ok is True


GUARD_LETTER_KEYS = (
    'require: {or: [{text: "q", within: [0.0, 0.65, 1.0, 0.87]},'
    '               {text: "a", within: [0.0, 0.65, 1.0, 0.87]}]}'
)


async def test_run_single_char_matches_whole_key_label_only() -> None:
    # A standalone "q" key element matches; "quit" containing q does not —
    # exact-label semantics are what make letter-key anchors usable.
    key_row = format_row(4, "text", "q", [0.01, 0.69, 0.09, 0.73], 0.9)
    mcp = FakeCaller([_gesture("went home", listing=key_row), _gesture("tapped")])

    result = await run(_guarded(GUARD_LETTER_KEYS), {}, mcp)

    assert result.ok is True


async def test_run_single_char_does_not_match_inside_longer_labels() -> None:
    chat_row = format_row(4, "text", "quit at dawn", [0.1, 0.7, 0.9, 0.75], 0.9)
    mcp = FakeCaller([_gesture("went home", listing=chat_row)])

    result = await run(_guarded(GUARD_LETTER_KEYS), {}, mcp)

    assert result.ok is False  # chat text can't fake a keyboard key


async def test_run_region_guard_rejects_same_text_elsewhere_on_screen() -> None:
    # 微信 IS on screen, but its element centers outside the region — the
    # exact false-positive whole-screen matching cannot catch.
    row = format_row(3, "text", "微信", [0.4, 0.80, 0.6, 0.90], 0.93)
    mcp = FakeCaller([_gesture("went home", listing=row)])

    result = await run(_guarded(GUARD_REGION), {}, mcp)

    assert result.ok is False
    assert "'微信' within [0.2,0.2,0.8,0.3] not on screen" in result.detail


async def test_run_forbid_text_aborts_when_present() -> None:
    # The popup tripwire: a forbidden text on screen kills the replay.
    mcp = FakeCaller([_gesture("went home", listing="Upgrade now — 50% off")])

    result = await run(_guarded(GUARD_FORBID), {}, mcp)

    assert result.ok is False
    assert result.reason == REASON_GUARD_FAILED
    assert "forbid 'Upgrade now' on screen" in result.detail


async def test_run_forbid_passes_when_absent() -> None:
    mcp = FakeCaller(
        [_gesture("went home", listing="clean screen"), _gesture("tapped")]
    )

    result = await run(_guarded(GUARD_FORBID), {}, mcp)

    assert result.ok is True


# ---------- guards: step-1 peek and wait polling ----------


async def test_run_step_one_guard_peeks_before_acting() -> None:
    # No previous view exists — a step-1 guard anchors the start state
    # with one peek before any gesture fires.
    entry_view = _gesture("current", changed=None, listing="Home screen")
    mcp = FakeCaller([entry_view, _gesture("tapped")])

    result = await run(_spec(GUARD_FIRST), {}, mcp)

    assert result.ok is True
    assert mcp.calls[0] == ("peek", {})
    assert mcp.calls[1][0] == "tap"


async def test_run_step_one_guard_failure_aborts_with_no_gesture() -> None:
    mcp = FakeCaller([_gesture("current", changed=None, listing="Lock screen")])

    result = await run(_spec(GUARD_FIRST), {}, mcp)

    assert result.ok is False
    assert result.aborted_step == 1
    assert all(name == "peek" for name, _ in mcp.calls)


async def test_run_guard_checks_once_and_does_not_poll() -> None:
    # A guard is a predicate now. The free check misses and that is the
    # answer — no second look, no camera cycle spent hoping.
    mcp = FakeCaller([_gesture("went home", listing="loading…")])

    result = await run(_guarded(GUARDED), {}, mcp)

    assert result.ok is False
    assert result.reason == REASON_GUARD_FAILED
    # Zero peeks: the free check decided, and the previous step's view is
    # still current, so even the abort report costs nothing extra.
    assert [name for name, _ in mcp.calls] == ["home_screen"]


async def test_wait_step_sleeps_and_calls_no_tool(mocker) -> None:
    sleep = mocker.patch("physiclaw.agent.macros.steps.asyncio.sleep")
    # The trailing peek is the end-of-run recovery read, not the wait: a
    # wait leaves the held frame `seconds` out of date, so it cannot be
    # shipped as "the current screen".
    mcp = FakeCaller([_gesture("went home"), _gesture("peeked")])

    result = await run(_spec(WAIT_STEPS), {}, mcp)

    assert result.ok is True
    sleep.assert_awaited_once_with(3)
    assert [name for name, _ in mcp.calls] == ["home_screen", "peek"]
    assert 'wait "settle"' in result.blocks[0]["text"]
    assert "waited 3s" in result.blocks[0]["text"]


async def test_a_trailing_wait_does_not_ship_the_pre_wait_frame_as_current(
    mocker,
) -> None:
    # `wait` sleeps BECAUSE the screen is expected to change, so the frame
    # held afterwards is stale by construction. Shipping it under "the view
    # below is the current screen" would hand the agent — and, through
    # dispatch's screen supersede, its whole notion of "now" — a frame that
    # is `seconds` old.
    mocker.patch("physiclaw.agent.macros.steps.asyncio.sleep")
    mcp = FakeCaller(
        [
            _gesture("went home", listing="BEFORE the wait"),
            _gesture("peeked", listing="AFTER the wait"),
        ]
    )

    result = await run(_spec(WAIT_STEPS), {}, mcp)

    assert result.ok is True
    assert "fresh `peek` of the current screen" in result.blocks[0]["text"]
    shipped = "\n".join(b.get("text", "") for b in result.blocks[1:])
    assert "AFTER the wait" in shipped
    assert "BEFORE the wait" not in shipped


async def test_a_wait_whose_expect_peeked_needs_no_recovery_read(mocker) -> None:
    # The expect already paid for a fresh frame — don't buy a second one.
    mocker.patch("physiclaw.agent.macros.steps.asyncio.sleep")
    row = format_row(0, "text", "WeChat", [0.3, 0.03, 0.6, 0.09], 0.95)
    mcp = FakeCaller([_gesture("went home"), _gesture("peeked", listing=row)])
    spec = _spec(
        "name: demo\ndescription: d\nsteps:\n"
        "  - name: go-home\n    tool: home_screen\n"
        "  - name: settle\n    tool: wait\n    with: {seconds: 1}\n"
        '    expect: {text: "WeChat", within: [0.15, 0.0, 0.85, 0.15]}\n'
    )

    result = await run(spec, {}, mcp)

    assert result.ok is True
    assert [name for name, _ in mcp.calls].count("peek") == 1


async def test_wait_step_drops_held_screen_text_so_the_next_guard_re_reads(
    mocker,
) -> None:
    # The whole point of waiting is that the screen CHANGES during it.
    # Reusing the pre-wait listing would hand the next guard exactly the
    # screen the wait existed to move past.
    mocker.patch("physiclaw.agent.macros.steps.asyncio.sleep")
    mcp = FakeCaller(
        [
            _gesture("went home", listing="loading…"),
            _gesture("current", changed=None, listing="WeChat 微信"),
            _gesture("tapped"),
        ]
    )

    result = await run(_spec(WAIT_THEN_GUARD), {}, mcp)

    assert result.ok is True
    # The peek between the wait and the tap is the guard re-reading.
    assert [name for name, _ in mcp.calls] == ["home_screen", "peek", "tap"]


async def test_wait_step_does_not_clobber_the_abort_view(mocker) -> None:
    # `wait` returns no blocks; `last_view` must survive it, or an abort on
    # the next step would have nothing to show.
    mocker.patch("physiclaw.agent.macros.steps.asyncio.sleep")
    mcp = FakeCaller(
        [
            _gesture("went home", listing="loading…"),
            RuntimeError("camera hiccup"),
        ]
    )

    result = await run(_spec(WAIT_THEN_GUARD), {}, mcp)

    assert result.ok is False
    assert result.reason == REASON_GUARD_FAILED


async def test_run_guard_peek_failure_is_retried_once_then_aborts() -> None:
    # A failed READ is not a failed guard, and the retry is a property of
    # the camera — not of any wait the author did or didn't write.
    # Guard on step 1, so nothing is held and the guard must peek.
    mcp = FakeCaller(
        [
            RuntimeError("camera hiccup"),
            RuntimeError("camera hiccup"),
            RuntimeError("camera hiccup"),
        ]
    )

    result = await run(_spec(GUARD_FIRST), {}, mcp)

    assert result.ok is False
    assert result.reason == REASON_GUARD_FAILED
    assert "could not read the screen" in result.detail
    # Two reads: the first attempt plus exactly one retry.
    assert [name for name, _ in mcp.calls][:2] == ["peek", "peek"]


# ---------- skip_when (idempotence) ----------

SKIP_STEPS = """
name: demo
description: d
inputs:
  msg:
    description: text
steps:
  - name: home-screen-4
    tool: home_screen
  - name: focus-input-box
    tool: tap
    with:
      bbox: [0.1, 0.915, 0.69, 0.955]
    skip_when: {or: ["空格", "space"]}
  - name: stage
    tool: send_to_clipboard
    with:
      text: "{msg}"
"""


async def test_run_skips_step_whose_postcondition_holds() -> None:
    # Keyboard already up ("空格" visible) — the focus tap must not fire.
    mcp = FakeCaller([_gesture("went home", listing="空格 键盘"), _gesture("copied")])

    result = await run(_spec(SKIP_STEPS), {"msg": "hi"}, mcp)

    assert result.ok is True
    assert [name for name, _ in mcp.calls] == ["home_screen", "send_to_clipboard"]
    assert (
        '↷ 2. tap "focus-input-box" — skipped (already satisfied)'
        in (result.blocks[0]["text"])
    )


async def test_run_executes_step_when_postcondition_not_met() -> None:
    mcp = FakeCaller(
        [
            _gesture("went home", listing="no keyboard"),
            _gesture("tapped"),
            _gesture("copied"),
        ]
    )

    result = await run(_spec(SKIP_STEPS), {"msg": "hi"}, mcp)

    assert result.ok is True
    assert [name for name, _ in mcp.calls] == [
        "home_screen",
        "tap",
        "send_to_clipboard",
    ]


async def test_run_skip_when_on_first_step_peeks_once() -> None:
    spec = _spec(
        "name: demo\ndescription: d\nsteps:\n"
        "  - name: entry\n    tool: tap\n    with:\n"
        "      bbox: [0.1, 0.2, 0.3, 0.4]\n"
        '    skip_when: "already there"\n'
    )
    already = _gesture("current", changed=None, listing="already there")
    mcp = FakeCaller([already])

    result = await run(spec, {}, mcp)

    assert result.ok is True
    assert [name for name, _ in mcp.calls] == ["peek"]  # skipped the tap


async def test_run_skipped_step_bypasses_its_guard() -> None:
    # A skipped step's guard must not abort the run — the step isn't needed.
    spec = _spec(
        "name: demo\ndescription: d\nsteps:\n"
        "  - name: home-screen-2\n    tool: home_screen\n"
        "  - name: focus\n    tool: tap\n    with:\n"
        "      bbox: [0.1, 0.9, 0.7, 0.96]\n"
        '    skip_when: "空格"\n'
        '    guard:\n      require: "never-on-screen"\n'
    )
    mcp = FakeCaller([_gesture("went home", listing="空格")])

    result = await run(spec, {}, mcp)

    assert result.ok is True


# ---------- tool error ----------


async def test_run_tool_error_aborts_with_detail() -> None:
    mcp = FakeCaller([_gesture("went home"), RuntimeError("arm busy")])

    result = await run(_spec(TWO_STEPS), {"msg": "x"}, mcp)

    assert result.ok is False
    assert result.aborted_step == 2
    assert result.reason == REASON_TOOL_ERROR
    assert result.detail == "arm busy"
    assert "arm busy" in result.blocks[0]["text"]


async def test_run_tool_error_on_first_step_returns_log_only() -> None:
    mcp = FakeCaller([RuntimeError("no server")])

    result = await run(_spec(TWO_STEPS), {"msg": "x"}, mcp)

    assert result.ok is False
    assert result.aborted_step == 1
    assert [b["type"] for b in result.blocks] == ["text"]


async def test_run_first_step_abort_header_admits_no_view() -> None:
    # Step 1 failed AND the recovery peek failed — the header must not
    # claim a view, and must hand `peek` back to the agent.
    mcp = FakeCaller([RuntimeError("no server"), RuntimeError("still down")])

    result = await run(_spec(TWO_STEPS), {"msg": "x"}, mcp)

    log = result.blocks[0]["text"]
    assert "no steps executed by this run" in log
    assert "no view available (recovery `peek` failed)" in log
    assert "view below" not in log


async def test_run_abort_header_steers_against_rerun() -> None:
    # Completed steps already moved the phone — the result must explicitly
    # forbid a whole-macro re-run and hand off to manual recovery.
    mcp = FakeCaller([_gesture("went home", listing="Settings only")])

    result = await run(_guarded(GUARDED), {}, mcp)

    log = result.blocks[0]["text"]
    assert "steps 1–1 already executed" in log
    assert "Do NOT re-run the macro" in log
    assert "the view below is the current screen" in log


# ---------- recovery peek (result always carries the current screen) ----------


async def test_run_success_with_text_only_last_step_peeks_for_view() -> None:
    # send_to_clipboard returns no view — the runner must fetch one so the
    # agent never spends a turn peeking after a macro.
    text_only = [{"type": "text", "text": "text staged on clipboard"}]
    peek_view = _gesture("current screen", changed=None, listing="fresh listing")
    mcp = FakeCaller([_gesture("went home"), text_only, peek_view])

    result = await run(_spec(TWO_STEPS), {"msg": "x"}, mcp)

    assert mcp.calls[-1] == ("peek", {})
    assert "fresh `peek` of the current screen" in result.blocks[0]["text"]
    assert any(b["type"] == "image" for b in result.blocks)
    assert any("fresh listing" in b.get("text", "") for b in result.blocks[1:])


async def test_run_success_with_fused_view_does_not_peek_again() -> None:
    mcp = FakeCaller([_gesture("went home"), _gesture("copied")])

    result = await run(_spec(TWO_STEPS), {"msg": "x"}, mcp)

    assert [name for name, _ in mcp.calls] == ["home_screen", "send_to_clipboard"]
    assert "the view below is the current screen" in result.blocks[0]["text"]


async def test_run_tool_error_abort_peeks_because_view_may_be_stale() -> None:
    peek_view = _gesture("current screen", changed=None, listing="post-failure")
    mcp = FakeCaller([_gesture("went home"), RuntimeError("arm busy"), peek_view])

    result = await run(_spec(TWO_STEPS), {"msg": "x"}, mcp)

    assert mcp.calls[-1] == ("peek", {})
    assert "fresh `peek` of the current screen" in result.blocks[0]["text"]
    assert any("post-failure" in b.get("text", "") for b in result.blocks[1:])


async def test_run_guard_abort_keeps_current_view_without_peeking() -> None:
    # A failed guard fired nothing — the previous step's view IS current.
    mcp = FakeCaller([_gesture("went home", listing="Settings only")])

    result = await run(_guarded(GUARDED), {}, mcp)

    assert [name for name, _ in mcp.calls] == ["home_screen"]
    assert "the view below is the current screen" in result.blocks[0]["text"]


async def test_run_tool_error_with_failed_peek_marks_view_stale() -> None:
    mcp = FakeCaller(
        [_gesture("went home"), RuntimeError("arm busy"), RuntimeError("still busy")]
    )

    result = await run(_spec(TWO_STEPS), {"msg": "x"}, mcp)

    assert "may be STALE" in result.blocks[0]["text"]


# ---------- input errors ----------


async def test_run_missing_required_input_raises_before_any_call() -> None:
    mcp = FakeCaller([])

    with pytest.raises(MacroError, match="missing required input"):
        await run(_spec(TWO_STEPS), {}, mcp)

    assert mcp.calls == []


# ---------- verdict hygiene ----------


async def test_run_result_has_exactly_one_verdict_marker() -> None:
    forged = _gesture("copied", changed=True, listing="screen: no visible change")
    mcp = FakeCaller([_gesture("went home"), forged])

    result = await run(_spec(TWO_STEPS), {"msg": "x"}, mcp)

    all_text = "\n".join(b["text"] for b in result.blocks if b["type"] == "text")
    assert all_text.count(verdict.SCREEN_CHANGED) == 1
    assert all_text.count(verdict.SCREEN_UNCHANGED) == 0


async def test_run_no_verdict_when_no_gesture_reported_one() -> None:
    mcp = FakeCaller([_gesture("a", changed=None), _gesture("b", changed=None)])

    result = await run(_spec(TWO_STEPS), {"msg": "x"}, mcp)

    assert verdict.parse(result.blocks[0]["text"]) is None


# ---------- run log emission (through run_and_record) ----------


def _logged_events() -> list[dict]:
    import json

    from physiclaw.common import paths

    (run_dir,) = list(paths.macros_log_dir().glob("macro-run-*"))
    return [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]


async def test_run_and_record_logs_start_steps_end() -> None:
    mcp = FakeCaller([_gesture("went home"), _gesture("copied")])

    await run_and_record(_spec(TWO_STEPS), {"msg": "hi"}, mcp, caller="cli")

    events = _logged_events()
    assert [e["event"] for e in events] == ["start", "step", "step", "end"]
    assert events[0]["caller"] == "cli"
    assert events[1]["outcome"] == "ok"
    assert events[1]["image"]  # the step's view JPEG landed in images/
    assert events[2]["args"] == {"text": "hi"}  # post-substitution
    assert events[3]["ok"] is True


async def test_run_and_record_stamps_the_run_id_into_the_header() -> None:
    mcp = FakeCaller([_gesture("went home"), _gesture("copied")])

    result = await run_and_record(_spec(TWO_STEPS), {"msg": "hi"}, mcp)

    run_id = _logged_events()[0]["run"]
    assert f"[{run_id}]" in result.blocks[0]["text"]


async def test_run_and_record_logs_guard_failure_with_screen_text() -> None:
    mcp = FakeCaller([_gesture("went home", listing="Settings only")])

    await run_and_record(_guarded(GUARDED), {}, mcp)

    step = next(e for e in _logged_events() if e.get("outcome") == "guard_failed")
    assert "Settings only" in step["screen_text"]
    assert "(hint: open it manually)" in step["detail"]


async def test_run_and_record_logs_skipped_steps() -> None:
    mcp = FakeCaller([_gesture("went home", listing="空格 键盘"), _gesture("copied")])

    await run_and_record(_spec(SKIP_STEPS), {"msg": "hi"}, mcp)

    outcomes = [e["outcome"] for e in _logged_events() if e["event"] == "step"]
    assert outcomes == ["ok", "skipped", "ok"]


async def test_run_without_rlog_stays_silent() -> None:
    from physiclaw.common import paths

    mcp = FakeCaller([_gesture("went home"), _gesture("copied")])

    result = await run(_spec(TWO_STEPS), {"msg": "hi"}, mcp)

    assert not paths.macros_log_dir().exists()
    assert "[macro-run-" not in result.blocks[0]["text"]


# ---------- run_and_record ----------


async def test_run_and_record_success_folds_into_stats() -> None:
    mcp = FakeCaller([_gesture("went home"), _gesture("copied")])

    result = await run_and_record(_spec(TWO_STEPS), {"msg": "x"}, mcp)

    assert result.ok is True
    assert macro_stats.load()["demo"]["total_successes"] == 1


async def test_run_and_record_abort_folds_into_stats() -> None:
    mcp = FakeCaller(
        [_gesture("went home"), RuntimeError("arm busy"), RuntimeError("peek down")]
    )

    result = await run_and_record(_spec(TWO_STEPS), {"msg": "x"}, mcp)

    assert result.ok is False
    assert macro_stats.load()["demo"]["last_abort"]["reason"] == REASON_TOOL_ERROR


async def test_run_and_record_keeps_stats_of_disabled_macros() -> None:
    # The prune set is the dirs on disk, not the enabled registry — an
    # engine run must not delete a temporarily-disabled macro's stats.
    from physiclaw.common import paths

    for name in ("demo", "resting"):
        d = paths.macros_dir() / name
        d.mkdir(parents=True)
        (d / "MACRO.yml").write_text(
            f"name: {name}\ndescription: d\nsteps:\n  - name: peek-1\n    tool: peek\n",
            encoding="utf-8",
        )
    macro_stats.record("resting", ok=True, known_names={"demo", "resting"})
    mcp = FakeCaller([_gesture("went home"), _gesture("copied")])

    await run_and_record(_spec(TWO_STEPS), {"msg": "x"}, mcp)

    assert "resting" in macro_stats.load()


async def test_run_and_record_bad_input_records_and_raises() -> None:
    mcp = FakeCaller([])

    with pytest.raises(MacroError, match="missing required input"):
        await run_and_record(_spec(TWO_STEPS), {}, mcp)

    assert macro_stats.load()["demo"]["last_abort"]["reason"] == REASON_BAD_INPUT


# ---------- polling cadence sanity ----------


# ---------- the free check's haystack ----------
#
# A guard/skip_when with no wait polls checks the screen text we already
# hold. Getting "which text" wrong is silent and total: the clauses match
# nothing, so guards abort runs that should pass and skip_when never fires.

# A keyboard-up screen: a letter key (the region-form anchor skip_when uses)
# plus an ordinary multi-char label a whole-screen require can match.
_KEYS = "\n".join(
    [
        format_row(0, "text", "A", [0.05, 0.72, 0.11, 0.78], 0.9),
        format_row(1, "text", "Paste", [0.05, 0.55, 0.12, 0.60], 0.9),
    ]
)

CLIPBOARD_THEN_GUARD = """
name: demo
description: d
steps:
  - name: tap-1
    tool: tap
    with: {bbox: [0.1, 0.1, 0.2, 0.2]}
  - name: send-to-clipboard-1
    tool: send_to_clipboard
    with: {text: hi}
  - name: tap-2
    tool: tap
    with: {bbox: [0.3, 0.3, 0.4, 0.4]}
    guard: {require: "Paste", forbid: "boom"}
"""


@pytest.mark.asyncio
async def test_clipboard_step_does_not_blind_the_next_guard() -> None:
    # send_to_clipboard replies text-only and never touches the screen, so
    # the listing from step 1 IS still the current screen. Overwriting the
    # haystack with the clipboard confirmation would abort this run — the
    # documented stage-then-paste flow — on a screen that satisfies it.
    mcp = FakeCaller(
        [
            _gesture("tapped", listing=_KEYS),
            [{"type": "text", "text": "Copied 2 chars to the clipboard."}],
            _gesture("tapped", listing=_KEYS),
        ]
    )

    result = await run(_spec(CLIPBOARD_THEN_GUARD), {}, mcp)

    assert result.ok, result.blocks[0]["text"]
    # And it cost nothing: the retained listing answered the guard for free.
    assert [name for name, _ in mcp.calls].count("peek") == 0


SKIP_THEN_GUARD = """
name: demo
description: d
steps:
  - name: tap-3
    tool: tap
    with: {bbox: [0.1, 0.9, 0.6, 0.95]}
    skip_when: {text: "A", within: [0.0, 0.5, 1.0, 1.0]}
  - name: tap-4
    tool: tap
    with: {bbox: [0.3, 0.3, 0.4, 0.4]}
    guard: {require: "Paste"}
"""


@pytest.mark.asyncio
async def test_step_one_skip_peek_is_adopted_not_discarded() -> None:
    # A step-1 skip_when peeks. That screen must become the haystack: the
    # skipped step fired nothing, so the very next guard would otherwise
    # judge an EMPTY string and abort on a screen it already read.
    mcp = FakeCaller([_gesture("peeked", listing=_KEYS), _gesture("tapped")])

    result = await run(_spec(SKIP_THEN_GUARD), {}, mcp)

    assert result.ok, result.blocks[0]["text"]
    assert [name for name, _ in mcp.calls] == ["peek", "tap"]  # exactly one peek


GUARDED_FIRST_STEP = """
name: demo
description: d
steps:
  - name: tap-5
    tool: tap
    with: {bbox: [0.1, 0.1, 0.2, 0.2]}
    guard: {require: "Settings"}
"""


@pytest.mark.asyncio
async def test_guard_retries_once_when_the_camera_hiccups(mocker) -> None:
    # A guard checks once, so without a read-failure retry a single camera
    # hiccup would abort the run — and blame a screen nobody ever saw.
    mocker.patch("asyncio.sleep")
    listing = format_row(0, "text", "Settings", [0.1, 0.1, 0.4, 0.2], 0.98)
    mcp = FakeCaller(
        [
            RuntimeError("camera busy"),
            _gesture("peeked", listing=listing),
            _gesture("tapped"),
        ]
    )

    result = await run(_spec(GUARDED_FIRST_STEP), {}, mcp)

    assert result.ok, result.blocks[0]["text"]


@pytest.mark.asyncio
async def test_unreadable_screen_is_not_reported_as_a_missing_element(
    mocker,
) -> None:
    # A failed READ is not a failed guard. Saying "require 'Settings' not on
    # screen" about a screen nobody saw sends the agent to fix the screen
    # instead of retrying the read — and lands in stats' last_abort, the
    # counter the user is told means "the app layout changed, re-rehearse".
    mocker.patch("asyncio.sleep")
    mcp = FakeCaller([RuntimeError("camera busy")] * 4)

    result = await run(_spec(GUARDED_FIRST_STEP), {}, mcp)

    assert not result.ok
    assert result.reason == REASON_GUARD_FAILED
    assert "could not read the screen" in result.detail
    assert "not on screen" not in result.detail


# ---------- guard soundness: which way does it fail? ----------
#
# A guard is a safety gate, so the only acceptable failure direction is
# CLOSED. Both cases below are false-PASSes: the guard says yes and the
# gesture fires on a screen that never matched.

FORBID_ONLY = """
name: demo
description: d
steps:
  - name: tap-6
    tool: tap
    with: {bbox: [0.1, 0.1, 0.2, 0.2]}
    guard: {forbid: "Upgrade now"}
"""


@pytest.mark.asyncio
async def test_forbid_only_guard_fails_closed_on_an_unreadable_screen(
    mocker,
) -> None:
    # `require` fails closed for free (nothing matches ""), but a forbid-only
    # tripwire would PASS on a blank haystack — the popup it exists to catch
    # is invisible, not absent — and the step would fire blind.
    mocker.patch("asyncio.sleep")
    mcp = FakeCaller([RuntimeError("camera busy")] * 4)

    result = await run(_spec(FORBID_ONLY), {}, mcp)

    assert not result.ok
    assert ("tap", {"bbox": [0.1, 0.1, 0.2, 0.2]}) not in mcp.calls


ECHO_GUARD = """
name: demo
description: d
steps:
  - name: tap-7
    tool: tap
    with: {bbox: [0.1, 0.1, 0.2, 0.2]}
  - name: tap-8
    tool: tap
    with: {bbox: [0.3, 0.3, 0.4, 0.4]}
    guard: {require: "Payment confirmed"}
"""


@pytest.mark.asyncio
async def test_guard_cannot_be_satisfied_by_the_action_text_echo() -> None:
    # Block 0 is core-composed action text and echoes the macro's own
    # arguments back. If guards saw it, a `require` naming text an earlier
    # step supplied would match its own echo and gate nothing at all.
    echo = _gesture(
        "Tapped. Payment confirmed",
        listing=format_row(0, "text", "Home", [0.1, 0.1, 0.2, 0.2], 0.9),
    )
    mcp = FakeCaller([echo, echo])

    result = await run(_spec(ECHO_GUARD), {}, mcp)

    assert not result.ok
    assert result.reason == REASON_GUARD_FAILED
    assert ("tap", {"bbox": [0.3, 0.3, 0.4, 0.4]}) not in mcp.calls


def test_screen_text_keeps_a_view_replys_listing() -> None:
    # The rule is "drop block 0 IF it is text" — a gesture replies
    # [action, image, listing], but a view replies [image, listing], where
    # dropping the first text block would throw the screen away entirely.
    assert verdict.screen_text([IMAGE, {"type": "text", "text": "L"}]) == "L"
    assert (
        verdict.screen_text(
            [{"type": "text", "text": "action"}, IMAGE, {"type": "text", "text": "L"}]
        )
        == "L"
    )


# ---------- start_at: resume a macro the caller partly did by hand ----------

RESUMABLE = """
name: demo
description: d
steps:
  - name: go-home
    tool: home_screen
  - name: open-app
    tool: tap
    with: {bbox: [0.1, 0.1, 0.2, 0.2]}
  - name: focus-input-box
    tool: tap
    with: {bbox: [0.1, 0.9, 0.7, 0.95]}
    guard: {require: "Chat"}
"""

_CHAT = format_row(0, "text", "Chat", [0.3, 0.02, 0.5, 0.07], 0.9)


@pytest.mark.asyncio
async def test_start_at_skips_the_prefix_without_executing_it() -> None:
    mcp = FakeCaller([_gesture("peeked", listing=_CHAT), _gesture("tapped")])

    result = await run(_spec(RESUMABLE), {}, mcp, start_at="focus-input-box")

    assert result.ok
    # home_screen and the app tap must NOT have fired — the caller did those.
    assert [name for name, _ in mcp.calls] == ["peek", "tap"]
    assert "↷ 1–2. skipped" in result.blocks[0]["text"]


@pytest.mark.asyncio
async def test_start_at_verifies_the_entry_step_guard() -> None:
    # Resuming lands a rehearsed bbox on a screen this macro did not
    # produce, so the entry step's guard is the thing making it safe. Here
    # the screen does not match and nothing may fire.
    mcp = FakeCaller([_gesture("peeked", listing="somewhere else")])

    result = await run(_spec(RESUMABLE), {}, mcp, start_at="focus-input-box")

    assert not result.ok
    assert result.reason == REASON_GUARD_FAILED
    assert ("tap", {"bbox": [0.1, 0.9, 0.7, 0.95]}) not in mcp.calls


@pytest.mark.asyncio
async def test_start_at_abort_counts_from_the_entry_step_not_from_one() -> None:
    # The skipped prefix was never executed BY THIS RUN. Reporting "steps
    # 1–2 already executed" would send the agent to recover from a state it
    # is not in — MACRO.md tells it to trust this line.
    mcp = FakeCaller([_gesture("peeked", listing="somewhere else")])

    result = await run(_spec(RESUMABLE), {}, mcp, start_at="focus-input-box")

    assert "no steps executed by this run" in result.blocks[0]["text"]


@pytest.mark.asyncio
async def test_start_at_unknown_name_raises_before_any_gesture() -> None:
    mcp = FakeCaller([])

    with pytest.raises(MacroError, match="matches no step"):
        await run(_spec(RESUMABLE), {}, mcp, start_at="nope")

    assert mcp.calls == []  # nothing actuated


def test_duplicate_step_names_are_rejected_at_parse_time() -> None:
    # `start_at` addresses steps by name, so duplicates would make it
    # ambiguous. Caught by `macros check` at authoring time rather than
    # mid-replay with the phone half-way through the flow.
    dupe = RESUMABLE.replace("  - name: go-home\n", "  - name: open-app\n")

    with pytest.raises(MacroError, match="duplicate step name"):
        _spec(dupe)


@pytest.mark.asyncio
async def test_start_at_unguarded_entry_step_says_it_was_unverified() -> None:
    # No guard means nothing checked the screen. Say so rather than letting
    # the log imply the entry state was verified.
    mcp = FakeCaller([_gesture("tapped", listing=_CHAT), _gesture("focused")])

    result = await run(_spec(RESUMABLE), {}, mcp, start_at="open-app")

    assert result.ok
    assert "NOT verified" in result.blocks[0]["text"]


@pytest.mark.asyncio
async def test_no_start_at_runs_the_whole_macro() -> None:
    mcp = FakeCaller(
        [_gesture("home"), _gesture("tapped", listing=_CHAT), _gesture("t")]
    )

    result = await run(_spec(RESUMABLE), {}, mcp)

    assert result.ok
    assert "all 3 steps completed" in result.blocks[0]["text"]
    assert "skipped" not in result.blocks[0]["text"]


# ---------- whole-run wall-clock budget ----------
#
# Declared waits are exact, but they are not the whole cost: every gesture
# step spends real arm-and-camera time that no per-step cap covers. Nothing
# else catches a long run — the MCP read timeout is per request and the
# session deadline is only checked between turns — so a slow app could block
# the engine task for half a session.

BUDGET_STEPS = "name: demo\ndescription: d\nsteps:\n" + "".join(
    f"  - name: s{i}\n    tool: tap\n    with: {{bbox: [0.1, 0.1, 0.2, 0.2]}}\n"
    for i in range(1, 11)
)


@pytest.mark.asyncio
async def test_run_aborts_once_the_wall_clock_budget_is_spent(mocker) -> None:
    base = 1000.0
    # Every monotonic() reading jumps 40s, so the budget runs out mid-macro.
    mocker.patch(
        "physiclaw.agent.macros.runner.time.monotonic",
        side_effect=[base + 40 * i for i in range(400)],
    )
    mcp = FakeCaller([_gesture("tapped", listing="l") for _ in range(10)])

    result = await run(_spec(BUDGET_STEPS), {}, mcp)

    assert not result.ok
    assert result.reason == REASON_TIMEOUT
    assert f"{runner_mod.MAX_RUN_SECONDS}s budget" in result.detail
    # Stopped BETWEEN steps, so the phone is in a known state and the count
    # of fired gestures matches what the abort report claims.
    assert len(mcp.calls) == result.aborted_step - 1


@pytest.mark.asyncio
async def test_a_normal_run_is_untouched_by_the_budget() -> None:
    mcp = FakeCaller([_gesture("tapped", listing="l") for _ in range(10)])

    result = await run(_spec(BUDGET_STEPS), {}, mcp)

    assert result.ok
    assert len(mcp.calls) == 10


@pytest.mark.asyncio
async def test_a_wait_step_counts_against_the_run_budget(mocker) -> None:
    # Guards no longer spend time, so `wait` is the one place a macro can
    # burn the budget on purpose. The check between steps must still catch it.
    mocker.patch("physiclaw.agent.macros.steps.asyncio.sleep")
    base = 1000.0
    mocker.patch(
        "physiclaw.agent.macros.runner.time.monotonic",
        side_effect=[base] + [base + 400] * 400,
    )
    mcp = FakeCaller([_gesture("went home")] * 10)

    result = await run(_spec(WAIT_THEN_GUARD), {}, mcp)

    assert not result.ok
    assert result.reason == REASON_TIMEOUT
    assert f"{runner_mod.MAX_RUN_SECONDS}s budget" in result.detail


# ---------- expect (postcondition) ----------

WAIT_EXPECT = """
name: demo
description: d
steps:
  - name: tap-dock
    tool: tap
    with:
      bbox: [0.1, 0.2, 0.3, 0.4]
  - name: await-app
    tool: wait
    with:
      seconds: 1
    expect: {text: "WeChat", within: [0.15, 0.0, 0.85, 0.15]}
    hint: "WeChat did not open"
"""

_OPEN = format_row(0, "text", "WeChat", [0.3, 0.03, 0.6, 0.09], 0.95)
_LOADING = format_row(0, "text", "Loading", [0.3, 0.03, 0.6, 0.09], 0.95)


@pytest.mark.asyncio
async def test_expect_after_a_wait_costs_exactly_one_peek(mocker) -> None:
    # The whole reason `expect` exists: the old way to assert without acting
    # was a `peek` step carrying a `guard`, which read the screen TWICE —
    # once for the guard, once for the peek that carried it.
    mocker.patch("physiclaw.agent.macros.steps.asyncio.sleep")
    mcp = FakeCaller(
        [_gesture("tapped", listing=_OPEN), _gesture("peeked", listing=_OPEN)]
    )

    result = await run(_spec(WAIT_EXPECT), {}, mcp)

    assert result.ok is True
    assert [n for n, _ in mcp.calls].count("peek") == 1


@pytest.mark.asyncio
async def test_expect_failure_aborts_with_its_own_reason_and_hint(mocker) -> None:
    mocker.patch("physiclaw.agent.macros.steps.asyncio.sleep")
    mcp = FakeCaller(
        [_gesture("tapped", listing=_LOADING), _gesture("peeked", listing=_LOADING)]
    )

    result = await run(_spec(WAIT_EXPECT), {}, mcp)

    assert result.ok is False
    assert result.reason == REASON_EXPECT_FAILED
    assert result.aborted_step == 2
    assert "expected 'WeChat'" in result.detail
    assert "(hint: WeChat did not open)" in result.detail


@pytest.mark.asyncio
async def test_expect_failure_replaces_the_step_line_not_appends(mocker) -> None:
    # One verdict per step: a ✓ and a ✗ sharing step 2 reads as two events.
    mocker.patch("physiclaw.agent.macros.steps.asyncio.sleep")
    mcp = FakeCaller(
        [_gesture("tapped", listing=_LOADING), _gesture("peeked", listing=_LOADING)]
    )

    result = await run(_spec(WAIT_EXPECT), {}, mcp)

    log = result.blocks[0]["text"]
    assert "✗ 2. " in log
    assert "✓ 2. " not in log


@pytest.mark.asyncio
async def test_expect_fails_closed_when_the_screen_cannot_be_read(mocker) -> None:
    # An empty haystack satisfies {not: ...}, so an unreadable screen must
    # never count as a confirmed postcondition.
    mocker.patch("physiclaw.agent.macros.steps.asyncio.sleep")
    mcp = FakeCaller(
        [
            _gesture("tapped", listing=_OPEN),
            RuntimeError("camera busy"),
            RuntimeError("camera busy"),
            RuntimeError("camera busy"),
        ]
    )

    result = await run(_spec(WAIT_EXPECT), {}, mcp)

    assert result.ok is False
    assert result.reason == REASON_EXPECT_FAILED
    assert "could not read the screen" in result.detail


@pytest.mark.asyncio
async def test_expect_adopts_the_screen_it_read_for_the_next_step(mocker) -> None:
    # The expect's peek is a real read — the next step's guard must reuse it
    # rather than buy a second camera cycle.
    mocker.patch("physiclaw.agent.macros.steps.asyncio.sleep")
    spec = _spec(
        "name: demo\ndescription: d\nsteps:\n"
        "  - name: settle\n    tool: wait\n    with: {seconds: 1}\n"
        '    expect: {text: "WeChat", within: [0.15, 0.0, 0.85, 0.15]}\n'
        "  - name: tap-row\n    tool: tap\n    with: {bbox: [0.1, 0.2, 0.3, 0.4]}\n"
        '    guard:\n      require: {text: "WeChat", within: [0.15, 0.0, 0.85, 0.15]}\n'
    )
    mcp = FakeCaller([_gesture("peeked", listing=_OPEN), _gesture("tapped")])

    result = await run(spec, {}, mcp)

    assert result.ok is True
    assert [n for n, _ in mcp.calls] == ["peek", "tap"]  # guard was free


# ---------- operator evaluation ----------

_SCREEN = "\n".join(
    [
        format_row(0, "text", "WeChat", [0.3, 0.02, 0.5, 0.07], 0.9),
        format_row(1, "text", "Chats", [0.1, 0.95, 0.2, 0.99], 0.9),
    ]
)


def _one(body: str):
    spec = _spec(
        "name: demo\ndescription: d\nsteps:\n  - name: s1\n    tool: peek\n"
        "    guard:\n      require:\n" + body
    )
    return spec.steps[0].guard.require


@pytest.mark.parametrize(
    "body, expected",
    [
        ('        {or: ["Weixin", "WeChat"]}\n', True),
        ('        {or: ["Weixin", "Nope"]}\n', False),
        ('        {and: ["WeChat", "Chats"]}\n', True),
        ('        {and: ["WeChat", "Nope"]}\n', False),
        ('        {not: "Upgrade"}\n', True),
        ('        {not: "WeChat"}\n', False),
        ('        {or: [{not: "WeChat"}, {and: ["Chats", "WeChat"]}]}\n', True),
        ('        {and: [{not: "Upgrade"}, {or: ["Nope", "Chats"]}]}\n', True),
    ],
)
def test_clause_operators_evaluate(body: str, expected: bool) -> None:
    assert _one(body).holds(Screen.read(_SCREEN)) is expected


@pytest.mark.asyncio
async def test_skip_when_will_not_skip_on_an_unreadable_screen() -> None:
    # `not` is the one op an EMPTY haystack satisfies, so a camera hiccup
    # would otherwise read as "it's gone, skip the step" and silently drop
    # a gesture. No screen text means no skip.
    spec = _spec(
        "name: demo\ndescription: d\nsteps:\n"
        "  - name: dismiss\n    tool: tap\n    with: {bbox: [0.1, 0.1, 0.2, 0.2]}\n"
        '    skip_when: {not: "popup"}\n'
    )
    mcp = FakeCaller([RuntimeError("camera busy"), _gesture("tapped")])

    result = await run(spec, {}, mcp)

    assert result.ok
    # The tap FIRED rather than being skipped on an unread screen.
    assert ("tap", {"bbox": [0.1, 0.1, 0.2, 0.2]}) in mcp.calls
