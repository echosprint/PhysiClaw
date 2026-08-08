"""Tests for `physiclaw.agent.engine.dispatch` — guards, validation,
local/MCP execution, and result observers. Policy-object units (stuck
guard / plan gate / lint) are exercised through dispatch here — their
judgment is engine-visible behavior. The turn driver's tests live in
`test_loop.py`; session lifecycle in `test_engine.py`."""

from __future__ import annotations

import pytest
from engine_fakes import (
    FakeMcpClient,
    _mk_run,
    _tc,
)

from physiclaw.agent import layout
from physiclaw.agent.engine.builtin_tool import LocalTool
from physiclaw.agent.engine.dispatch import dispatch
from physiclaw.agent.engine.session import Session
from physiclaw.common.config import CONFIG

# ---------- dispatch basics ----------


@pytest.mark.asyncio
async def test_dispatch_unknown_tool_returns_error() -> None:
    call = _tc("does_not_exist")
    run = _mk_run()

    result = await dispatch(run, Session(), call, 0)

    assert result.is_error is True
    assert "unknown tool" in result.content
    assert result.tool_call_id == call.id


@pytest.mark.asyncio
async def test_dispatch_invalid_args_returns_error() -> None:
    schema = {
        "name": "ping",
        "input_schema": {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        },
    }
    run = _mk_run(schema_by_name={"ping": schema})

    result = await dispatch(
        run,
        Session(),
        _tc("ping", {}),
        0,  # missing required "x"
    )

    assert result.is_error is True
    assert "invalid arguments" in result.content


@pytest.mark.asyncio
async def test_dispatch_local_tool_returns_text() -> None:
    async def handler(_session, _args):
        return "hello"

    tool = LocalTool("greet", "say hi", {"type": "object"}, handler)
    schema = {"name": "greet", "input_schema": {"type": "object"}}
    run = _mk_run(schema_by_name={"greet": schema}, local_registry={"greet": tool})

    result = await dispatch(run, Session(), _tc("greet"), 0)

    assert result.is_error is False
    assert result.content == "hello"


@pytest.mark.asyncio
async def test_dispatch_local_tool_returning_blocks_uses_mcp_result_path() -> None:
    # run_macro's shape: a local handler handing back raw MCP-style blocks
    # (step log + final view) must convert like an MCP result — images and
    # all — not stringify.
    async def handler(_session, _args):
        return [
            {"type": "text", "text": "macro demo: all 2 steps completed"},
            {"type": "image", "mime_type": "image/jpeg", "data": "aGk="},
        ]

    tool = LocalTool("run_macro", "x", {"type": "object"}, handler, returns_blocks=True)
    schema = {"name": "run_macro", "input_schema": {"type": "object"}}
    run = _mk_run(
        schema_by_name={"run_macro": schema}, local_registry={"run_macro": tool}
    )

    result = await dispatch(run, Session(), _tc("run_macro"), 0)

    assert result.is_error is False
    assert isinstance(result.content, list)
    kinds = {type(b).__name__ for b in result.content}
    assert "ImageBlock" in kinds


@pytest.mark.asyncio
async def test_dispatch_local_blocks_traced_like_mcp_result() -> None:
    blocks = [{"type": "text", "text": "log line"}]

    async def handler(_session, _args):
        return blocks

    tool = LocalTool("run_macro", "x", {"type": "object"}, handler, returns_blocks=True)
    schema = {"name": "run_macro", "input_schema": {"type": "object"}}
    run = _mk_run(
        schema_by_name={"run_macro": schema}, local_registry={"run_macro": tool}
    )

    await dispatch(run, Session(), _tc("run_macro"), 0)

    traced = [c.args[0] for c in run.tr.write.call_args_list]
    assert any(
        e.get("event") == "tool_result" and e.get("blocks") == blocks for e in traced
    )


@pytest.mark.asyncio
async def test_dispatch_local_text_event_carries_elapsed_without_verdict() -> None:
    async def handler(_session, _args):
        return "noted: ok"

    tool = LocalTool("note", "x", {"type": "object"}, handler)
    schema = {"name": "note", "input_schema": {"type": "object"}}
    run = _mk_run(schema_by_name={"note": schema}, local_registry={"note": tool})

    await dispatch(run, Session(), _tc("note"), 0)

    traced = [c.args[0] for c in run.tr.write.call_args_list]
    ev = next(e for e in traced if e.get("event") == "tool_result")
    assert isinstance(ev["elapsed_ms"], int) and ev["elapsed_ms"] >= 0
    assert "changed" not in ev  # no screen semantics on a local text tool


@pytest.mark.asyncio
async def test_dispatch_tool_error_event_carries_elapsed() -> None:
    schema = {"name": "physiclaw__tap", "input_schema": {"type": "object"}}

    class BadMcp:
        async def call_tool(self, *a, **kw):
            raise RuntimeError("mcp down")

    run = _mk_run(mcp=BadMcp(), schema_by_name={"physiclaw__tap": schema})

    await dispatch(run, Session(), _tc("physiclaw__tap"), 0)

    traced = [c.args[0] for c in run.tr.write.call_args_list]
    ev = next(e for e in traced if e.get("event") == "tool_error")
    assert isinstance(ev["elapsed_ms"], int) and ev["elapsed_ms"] >= 0


@pytest.mark.asyncio
async def test_dispatch_declared_blocks_but_text_is_a_tool_error() -> None:
    # `returns_blocks` and the handler's return type state one fact twice,
    # so dispatch enforces the agreement: a str under the flag would
    # otherwise dissolve into a list of characters.
    async def handler(_session, _args):
        return "oops, plain text"

    tool = LocalTool("run_macro", "x", {"type": "object"}, handler, returns_blocks=True)
    schema = {"name": "run_macro", "input_schema": {"type": "object"}}
    run = _mk_run(
        schema_by_name={"run_macro": schema}, local_registry={"run_macro": tool}
    )

    result = await dispatch(run, Session(), _tc("run_macro"), 0)

    assert result.is_error is True
    assert "returns_blocks" in result.content


@pytest.mark.asyncio
async def test_dispatch_undeclared_blocks_is_a_tool_error() -> None:
    # The mirror mismatch: a block list from a handler that never declared
    # it must not be serialized as its repr.
    async def handler(_session, _args):
        return [{"type": "text", "text": "sneaky blocks"}]

    tool = LocalTool("greet", "x", {"type": "object"}, handler)
    schema = {"name": "greet", "input_schema": {"type": "object"}}
    run = _mk_run(schema_by_name={"greet": schema}, local_registry={"greet": tool})

    result = await dispatch(run, Session(), _tc("greet"), 0)

    assert result.is_error is True
    assert "returns_blocks" in result.content


@pytest.mark.asyncio
async def test_dispatch_mcp_tool_returns_blocks() -> None:
    schema = {"name": "physiclaw__peek", "input_schema": {"type": "object"}}
    mcp = FakeMcpClient()
    run = _mk_run(mcp=mcp, schema_by_name={"physiclaw__peek": schema})

    result = await dispatch(run, Session(), _tc("physiclaw__peek"), 0)

    assert result.is_error is False
    assert mcp.tool_calls == [("physiclaw__peek", {})]


@pytest.mark.asyncio
async def test_dispatch_local_handler_exception_returns_error() -> None:
    async def boom(_session, _args):
        raise RuntimeError("boom")

    tool = LocalTool("bad", "x", {"type": "object"}, boom)
    schema = {"name": "bad", "input_schema": {"type": "object"}}
    run = _mk_run(schema_by_name={"bad": schema}, local_registry={"bad": tool})

    result = await dispatch(run, Session(), _tc("bad"), 0)

    assert result.is_error is True
    assert "boom" in result.content


@pytest.mark.asyncio
async def test_dispatch_mcp_exception_returns_error() -> None:
    schema = {"name": "physiclaw__tap", "input_schema": {"type": "object"}}

    class BadMcp:
        async def call_tool(self, *a, **kw):
            raise RuntimeError("mcp down")

    run = _mk_run(mcp=BadMcp(), schema_by_name={"physiclaw__tap": schema})

    result = await dispatch(run, Session(), _tc("physiclaw__tap"), 0)

    assert result.is_error is True
    assert "mcp down" in result.content


# ---------- dispatch × stuck guard ----------


class VerdictMcpClient:
    """Fake MCP whose gesture results carry the no-change verdict marker —
    the input that drives the stuck guard's counters. `listing` adds a
    second text block (the fused view's OCR listing)."""

    def __init__(
        self, text="Tapped at bbox [...] | screen: no visible change", listing=None
    ):
        self.text = text
        self.listing = listing
        self.tool_calls: list[tuple] = []

    async def call_tool(self, name, arguments):
        self.tool_calls.append((name, arguments))
        blocks = [{"type": "text", "text": self.text}]
        if self.listing is not None:
            blocks.append({"type": "text", "text": self.listing})
        return blocks


_TAP_SCHEMA = {
    "name": "tap",
    "input_schema": {
        "type": "object",
        "properties": {"bbox": {"type": "array"}},
        "required": ["bbox"],
    },
}
_STEPPER = [0.908, 0.526, 0.983, 0.562]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("Tapped at bbox [...] | screen: changed", True, id="changed"),
        pytest.param("Tapped, no verdict marker here", None, id="no-verdict"),
    ],
)
async def test_dispatch_gesture_event_carries_elapsed_and_verdict(
    text, expected
) -> None:
    """Debuggability contract: the tool's own execution time and the
    camera verdict land on the event as DATA — per-step latency and
    miss-rate statistics must never regex the result text."""
    mcp = VerdictMcpClient(text=text)
    run = _mk_run(mcp=mcp, schema_by_name={"tap": _TAP_SCHEMA})

    await dispatch(run, Session(), _tc("tap", {"bbox": _STEPPER}), 0)

    traced = [c.args[0] for c in run.tr.write.call_args_list]
    ev = next(e for e in traced if e.get("event") == "tool_result")
    assert isinstance(ev["elapsed_ms"], int) and ev["elapsed_ms"] >= 0
    assert ev["changed"] is expected


def _all_text(content) -> str:
    """Flatten a tool-result content (str or blocks) for assertions —
    includes appended ⚠ warning blocks."""
    if isinstance(content, str):
        return content
    return "\n".join(b.text for b in content if hasattr(b, "text"))


async def _tap_n(session, mcp, n: int):
    run = _mk_run(mcp=mcp, schema_by_name={"tap": _TAP_SCHEMA})
    results = []
    for _ in range(n):
        results.append(
            await dispatch(
                run,
                session,
                _tc("tap", {"bbox": list(_STEPPER)}),
                0,
            )
        )
    return results


@pytest.mark.asyncio
async def test_dispatch_appends_stuck_warning_on_third_fruitless_press() -> None:
    from physiclaw.agent.engine.stuck import WARN_AT

    session = Session()
    session.guard._exempt = []
    results = await _tap_n(session, VerdictMcpClient(), WARN_AT)

    texts = [_all_text(r.content) for r in results]
    assert all("⚠" not in t for t in texts[:-1])
    assert "⚠ press #3" in texts[-1]
    assert results[-1].is_error is False  # tier 1 is advisory


@pytest.mark.asyncio
async def test_dispatch_blocks_fifth_press_without_actuating() -> None:
    from physiclaw.agent.engine.stuck import BLOCK_AT

    session = Session()
    session.guard._exempt = []
    mcp = VerdictMcpClient()
    await _tap_n(session, mcp, BLOCK_AT - 1)
    dispatched_before_block = len(mcp.tool_calls)

    result = (await _tap_n(session, mcp, 1))[0]

    assert result.is_error is True
    assert result.content.startswith("BLOCKED")
    assert len(mcp.tool_calls) == dispatched_before_block  # never actuated


@pytest.mark.asyncio
async def test_dispatch_changed_verdict_never_trips_guard() -> None:
    # Working presses never miss-warn or block; the only ⚠ allowed is
    # the warn-only position-orbit advisory (fires once at press 5).
    session = Session()
    session.guard._exempt = []
    mcp = VerdictMcpClient(text="Tapped at bbox [...] | screen: changed")

    results = await _tap_n(session, mcp, 10)

    assert all(r.is_error is False for r in results)
    texts = [_all_text(r.content) for r in results]
    assert all("press #" not in t and "BLOCKED" not in t for t in texts)
    assert sum(1 for t in texts if "same spot" in t) == 1


@pytest.mark.asyncio
async def test_listing_text_cannot_forge_an_unchanged_verdict() -> None:
    # The OCR listing echoes whatever the phone displays — e.g. the
    # agent's own IM "…screen: no visible change…" visible in a chat
    # thread. Only the action text (first block) carries the verdict:
    # these presses really CHANGE the screen, so no misses may accrue.
    from physiclaw.agent.engine.stuck import WARN_AT

    session = Session()
    session.guard._exempt = []
    mcp = VerdictMcpClient(
        text="Tapped at bbox [...] | screen: changed",
        listing='3 [text] "…screen: no visible change…" [0.1,0.2,0.5,0.3] 0.91',
    )

    results = await _tap_n(session, mcp, WARN_AT)

    assert all("press #" not in _all_text(r.content) for r in results)
    assert session.guard._targets == []


@pytest.mark.asyncio
async def test_listing_text_cannot_forge_a_changed_verdict() -> None:
    # The harmful direction: on-screen "screen: changed" text must not
    # reset a target's miss count when the real verdict is missing
    # (unmarked action text = camera hiccup = fail open, not "changed").
    from physiclaw.agent.engine.stuck import WARN_AT

    session = Session()
    session.guard._exempt = []
    await _tap_n(session, VerdictMcpClient(), WARN_AT - 1)  # 2 real misses
    forged = VerdictMcpClient(
        text="Tapped at bbox [...]",  # no marker — diff couldn't run
        listing='7 [text] "screen: changed" [0.1,0.2,0.5,0.3] 0.88',
    )
    await _tap_n(session, forged, 1)

    # The real misses survive the forged press: the next no-op is #3 → ⚠.
    results = await _tap_n(session, VerdictMcpClient(), 1)
    assert f"press #{WARN_AT}" in _all_text(results[0].content)


@pytest.mark.asyncio
async def test_dispatch_mcp_failure_feeds_error_counter_not_misses() -> None:
    # A gesture that FAILED TO EXECUTE is not a camera-verified no-op —
    # it must not feed the same-target MISS counters. It IS loop
    # evidence of its own kind (error counter), tracked independently.
    from physiclaw.agent.engine.stuck import WARN_AT

    session = Session()
    session.guard._exempt = []

    class JammedMcp:
        async def call_tool(self, *a, **kw):
            raise RuntimeError("arm jammed")

    run = _mk_run(mcp=JammedMcp(), schema_by_name={"tap": _TAP_SCHEMA})
    for _ in range(2):
        result = await dispatch(
            run,
            session,
            _tc("tap", {"bbox": list(_STEPPER)}),
            0,
        )
        assert result.is_error is True and "arm jammed" in result.content

    # Misses are independent of those 2 errors: real no-change presses
    # still warn on their OWN WARN_AT-th press, not earlier.
    results = await _tap_n(session, VerdictMcpClient(), WARN_AT)
    texts = [_all_text(r.content) for r in results]
    assert all("⚠" not in t for t in texts[:-1])
    assert "press #3" in texts[-1]


@pytest.mark.asyncio
async def test_plan_gate_block_does_not_feed_guard() -> None:
    # Plan-gate-blocked presses never actuate, so they must not count
    # toward the stuck guard's same-target misses.
    session = Session()
    session.guard._exempt = []
    run = _mk_run(mcp=VerdictMcpClient(), schema_by_name={"tap": _TAP_SCHEMA})

    for _ in range(10):
        result = await dispatch(
            run,
            session,
            _tc("tap", {"bbox": list(_STEPPER)}),
            CONFIG.engine.plan_required_after,
        )
        assert result.content.startswith("BLOCKED")

    assert session.guard.should_block("tap", {"bbox": list(_STEPPER)}) is None


@pytest.mark.asyncio
async def test_dispatch_error_repeat_warns_and_blocks() -> None:
    # Identical failing calls (AT-overlap raise, every retry) are loop
    # evidence: warn at WARN_AT, refuse pre-dispatch at BLOCK_AT.
    from physiclaw.agent.engine.stuck import BLOCK_AT, WARN_AT

    session = Session()
    session.guard._exempt = []

    class OverlapMcp:
        async def call_tool(self, *a, **kw):
            raise RuntimeError("target overlaps AssistiveTouch button")

    run = _mk_run(mcp=OverlapMcp(), schema_by_name={"tap": _TAP_SCHEMA})
    results = []
    for _ in range(BLOCK_AT - 1):
        results.append(
            await dispatch(
                run,
                session,
                _tc("tap", {"bbox": list(_STEPPER)}),
                0,
            )
        )
    assert all(r.is_error for r in results)
    assert "⚠" in results[WARN_AT - 1].content  # warning on the WARN_AT-th

    blocked = await dispatch(
        run,
        session,
        _tc("tap", {"bbox": list(_STEPPER)}),
        0,
    )
    assert blocked.content.startswith("BLOCKED")
    assert "failed" in blocked.content


# ---------- dispatch × plan gate ----------


async def _gated_dispatch(session, call, *, turn: int):
    schema = {"name": call.name, "input_schema": {"type": "object"}}
    run = _mk_run(
        mcp=VerdictMcpClient(),
        schema_by_name={call.name: schema, "tap": _TAP_SCHEMA},
    )
    return await dispatch(run, session, call, turn)


@pytest.mark.asyncio
async def test_plan_gate_blocks_action_tools_when_overdue() -> None:
    session = Session()

    result = await _gated_dispatch(
        session,
        _tc("tap", {"bbox": list(_STEPPER)}),
        turn=CONFIG.engine.plan_required_after,
    )

    assert result.is_error is True
    assert result.content.startswith("BLOCKED")
    assert "update_progress" in result.content


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["note", "update_progress", "end_session"])
async def test_plan_gate_exempts_the_way_out(name: str) -> None:
    # Exempt tools reach normal dispatch (anything but the plan-gate
    # BLOCKED text).
    session = Session()

    result = await _gated_dispatch(
        session,
        _tc(name),
        turn=CONFIG.engine.plan_required_after,
    )

    assert not str(result.content).startswith("BLOCKED")


@pytest.mark.asyncio
async def test_plan_gate_open_when_not_overdue() -> None:
    session = Session()

    result = await _gated_dispatch(
        session,
        _tc("tap", {"bbox": list(_STEPPER)}),
        turn=0,
    )

    assert result.is_error is False


# ---------- dispatch: layout lint ----------


_SEQ_SCHEMA = {"name": "sequence", "input_schema": {"type": "object"}}


@pytest.mark.asyncio
async def test_dispatch_blocks_sequence_on_layout_lint(mocker) -> None:
    mocker.patch.object(
        layout,
        "lint_gesture",
        return_value="BLOCKED — not executed: wrong box",
    )
    mcp = FakeMcpClient()
    run = _mk_run(mcp=mcp, schema_by_name={"sequence": _SEQ_SCHEMA})

    result = await dispatch(
        run,
        Session(),
        _tc(
            "sequence",
            {"actions": [{"tool_name": "long_press", "arg": [0, 0.9, 1, 1]}]},
        ),
        0,
    )

    assert result.is_error is True
    assert result.content.startswith("BLOCKED")
    assert mcp.tool_calls == []  # never actuated


@pytest.mark.asyncio
async def test_dispatch_lint_failure_is_fail_open(mocker) -> None:
    # A lint crash must never take down dispatch — the batch runs.
    mocker.patch.object(
        layout,
        "lint_gesture",
        side_effect=RuntimeError("lint bug"),
    )
    mcp = FakeMcpClient()
    run = _mk_run(mcp=mcp, schema_by_name={"sequence": _SEQ_SCHEMA})

    result = await dispatch(
        run,
        Session(),
        _tc("sequence", {"actions": []}),
        0,
    )

    assert result.is_error is not True
    assert len(mcp.tool_calls) == 1


@pytest.mark.asyncio
async def test_dispatch_feeds_keyboard_tracker_the_verdict(mocker) -> None:
    session = Session()
    session.guard._exempt = []
    kb_spy = mocker.patch.object(session.kb, "observe")
    schema = {"name": "tap", "input_schema": {"type": "object"}}
    run = _mk_run(mcp=VerdictMcpClient(), schema_by_name={"tap": schema})

    await dispatch(
        run,
        session,
        _tc("tap", {"bbox": [0.1, 0.9, 0.7, 0.95]}),
        0,
    )

    kb_spy.assert_called_once()
    name, args, changed = kb_spy.call_args.args
    assert name == "tap"
    assert changed in (True, False)  # the parsed verdict, not None
