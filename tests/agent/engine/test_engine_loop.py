"""Integration tests for `physiclaw.agent.engine.engine` — `_loop`,
`_run_session`, `_dispatch`, and `run`.

Phase 5 — exercises the full session lifecycle with a scripted
FakeProvider and FakeMcpClient. Existing pure-helper tests live in
`test_engine.py`; this file owns the loop coverage. Policy-object units
(gates / guards / observers) are exercised through the loop here — their
judgment is engine-visible behavior.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from physiclaw.agent.engine import engine as engine_mod
from physiclaw.agent.engine import (
    builtin_tool as builtin_tool_mod,
    compact as compact_mod,
    jobs as jobs_mod,
    memory as memory_mod,
    plan as plan_mod,
    policy as policy_mod,
    prompt as prompt_mod,
    scratchpad as scratchpad_mod,
    screen_layout as screen_layout_mod,
    skill as skill_mod,
)
from physiclaw.agent.engine.builtin_tool import LocalTool
from physiclaw.agent.engine.dto import (
    AssistantMessage,
    FinishReason,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from physiclaw.agent.engine.session import Session
from physiclaw.agent.runtime.hook import Trigger
from physiclaw.agent.runtime.sentinel import DONE, FAIL, IDLE, STUCK, WAIT
from physiclaw.config import CONFIG


pytestmark = [pytest.mark.slow]


# ---------- Fakes ----------


class FakeProvider:
    """Scripted provider — pops AssistantMessages from a queue."""

    PROVIDER_ID = "fake"
    COLLAPSE_FIRST_AT_TURN = 100
    COLLAPSE_INTERVAL_TURNS = 100
    KEEP_RECENT_TURNS = 10

    def __init__(self, responses: list[Any]):
        self._responses = list(responses)
        self.model = "fake-model"
        self.calls: list[tuple] = []
        self.closed = False

    def serialize_history(self, messages):
        return [{"role": "fake"} for _ in messages]

    async def chat(self, messages, tools):
        self.calls.append((len(messages), len(tools)))
        if not self._responses:
            raise RuntimeError("FakeProvider exhausted")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    async def aclose(self):
        self.closed = True


class FakeMcpClient:
    def __init__(self):
        self.tool_calls: list[tuple] = []

    async def call_tool(self, name, arguments):
        self.tool_calls.append((name, arguments))
        return [{"type": "text", "text": "mcp-ok"}]


def _asst(*, content="", tool_calls=None, finish=FinishReason.TOOL_CALLS,
          usage=None) -> AssistantMessage:
    return AssistantMessage(
        content=content,
        tool_calls=tool_calls or [],
        finish_reason=finish,
        usage=usage or Usage(),
    )


def _tc(name: str, args: dict | None = None, *, tcid: str = None) -> ToolCall:
    import uuid
    return ToolCall(
        id=tcid or f"tc_{uuid.uuid4().hex[:8]}",
        name=name,
        arguments=args or {},
    )


def _settings(**over) -> engine_mod.Settings:
    base = dict(
        max_turns=300, max_session_attempts=3, provider_retry_attempts=3,
        retry_backoff_seconds=0.0, wait_default_minutes=15,
    )
    base.update(over)
    return engine_mod.Settings(**base)


def _mk_run(provider=None, *, mcp=None, tool_schemas=None, schema_by_name=None,
            local_registry=None, tr=None, rlog=None, layout_incomplete=False,
            policies=None, settings=None) -> engine_mod.EngineRun:
    tool_schemas = tool_schemas if tool_schemas is not None else []
    return engine_mod.EngineRun(
        provider=provider,
        mcp=mcp if mcp is not None else FakeMcpClient(),
        tool_schemas=tool_schemas,
        schema_by_name=(
            schema_by_name if schema_by_name is not None
            else {s["name"]: s for s in tool_schemas}
        ),
        local_registry=local_registry if local_registry is not None else {},
        tr=tr if tr is not None else MagicMock(),
        rlog=rlog if rlog is not None else MagicMock(),
        settings=settings or _settings(),
        policies=(
            policies if policies is not None
            else policy_mod.default_policies(layout_incomplete=layout_incomplete)
        ),
        layout_incomplete=layout_incomplete,
    )


# ---------- _dispatch ----------


@pytest.mark.asyncio
async def test_dispatch_unknown_tool_returns_error() -> None:
    call = _tc("does_not_exist")
    run = _mk_run()

    result = await engine_mod._dispatch(run, Session(), call, 0)

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

    result = await engine_mod._dispatch(
        run, Session(), _tc("ping", {}), 0,  # missing required "x"
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

    result = await engine_mod._dispatch(run, Session(), _tc("greet"), 0)

    assert result.is_error is False
    assert result.content == "hello"


@pytest.mark.asyncio
async def test_dispatch_mcp_tool_returns_blocks() -> None:
    schema = {"name": "physiclaw__peek", "input_schema": {"type": "object"}}
    mcp = FakeMcpClient()
    run = _mk_run(mcp=mcp, schema_by_name={"physiclaw__peek": schema})

    result = await engine_mod._dispatch(run, Session(), _tc("physiclaw__peek"), 0)

    assert result.is_error is False
    assert mcp.tool_calls == [("physiclaw__peek", {})]


@pytest.mark.asyncio
async def test_dispatch_local_handler_exception_returns_error() -> None:
    async def boom(_session, _args):
        raise RuntimeError("boom")

    tool = LocalTool("bad", "x", {"type": "object"}, boom)
    schema = {"name": "bad", "input_schema": {"type": "object"}}
    run = _mk_run(schema_by_name={"bad": schema}, local_registry={"bad": tool})

    result = await engine_mod._dispatch(run, Session(), _tc("bad"), 0)

    assert result.is_error is True
    assert "boom" in result.content


@pytest.mark.asyncio
async def test_dispatch_mcp_exception_returns_error() -> None:
    schema = {"name": "physiclaw__tap", "input_schema": {"type": "object"}}

    class BadMcp:
        async def call_tool(self, *a, **kw):
            raise RuntimeError("mcp down")

    run = _mk_run(mcp=BadMcp(), schema_by_name={"physiclaw__tap": schema})

    result = await engine_mod._dispatch(run, Session(), _tc("physiclaw__tap"), 0)

    assert result.is_error is True
    assert "mcp down" in result.content


# ---------- _dispatch × stuck guard ----------


class VerdictMcpClient:
    """Fake MCP whose gesture results carry the no-change verdict marker —
    the input that drives the stuck guard's counters. `listing` adds a
    second text block (the fused view's OCR listing)."""

    def __init__(self, text="Tapped at bbox [...] | screen: no visible change",
                 listing=None):
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
        results.append(await engine_mod._dispatch(
            run, session, _tc("tap", {"bbox": list(_STEPPER)}), 0,
        ))
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
        result = await engine_mod._dispatch(
            run, session, _tc("tap", {"bbox": list(_STEPPER)}), 0,
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
        result = await engine_mod._dispatch(
            run, session, _tc("tap", {"bbox": list(_STEPPER)}),
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
        results.append(await engine_mod._dispatch(
            run, session, _tc("tap", {"bbox": list(_STEPPER)}), 0,
        ))
    assert all(r.is_error for r in results)
    assert "⚠" in results[WARN_AT - 1].content  # warning on the WARN_AT-th

    blocked = await engine_mod._dispatch(
        run, session, _tc("tap", {"bbox": list(_STEPPER)}), 0,
    )
    assert blocked.content.startswith("BLOCKED")
    assert "failed" in blocked.content


# ---------- _dispatch × plan gate ----------


async def _gated_dispatch(session, call, *, turn: int):
    schema = {"name": call.name, "input_schema": {"type": "object"}}
    run = _mk_run(
        mcp=VerdictMcpClient(),
        schema_by_name={call.name: schema, "tap": _TAP_SCHEMA},
    )
    return await engine_mod._dispatch(run, session, call, turn)


@pytest.mark.asyncio
async def test_plan_gate_blocks_action_tools_when_overdue() -> None:
    session = Session()

    result = await _gated_dispatch(
        session, _tc("tap", {"bbox": list(_STEPPER)}),
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
        session, _tc(name), turn=CONFIG.engine.plan_required_after,
    )

    assert not str(result.content).startswith("BLOCKED")


@pytest.mark.asyncio
async def test_plan_gate_open_when_not_overdue() -> None:
    session = Session()

    result = await _gated_dispatch(
        session, _tc("tap", {"bbox": list(_STEPPER)}), turn=0,
    )

    assert result.is_error is False


def test_plan_gate_overdue_predicate() -> None:
    # The gate arms only when: past the threshold turn AND the plan is
    # still undrafted AND not in first-run setup.
    n = CONFIG.engine.plan_required_after
    gate = policy_mod.PlanGate(layout_incomplete=False, required_after=n)
    setup_gate = policy_mod.PlanGate(layout_incomplete=True, required_after=n)

    fresh = Session()
    assert gate.overdue(fresh, n)
    assert not gate.overdue(fresh, n - 1)
    assert not setup_gate.overdue(fresh, n)

    drafted = Session()
    drafted.plan.update(user_said="buy yogurt")
    assert not gate.overdue(drafted, n)

    steps_only = Session()
    steps_only.plan.update(steps=[{"content": "reply to user", "status": "in_progress"}])
    assert not gate.overdue(steps_only, n)


# ---------- _loop ----------


def _peek_tool() -> LocalTool:
    async def handler(_session, _args):
        return "peek-result"

    return LocalTool(
        name="peek",
        description="look",
        input_schema={"type": "object", "additionalProperties": True},
        handler=handler,
    )


def _note_tool() -> LocalTool:
    async def handler(_session, args):
        return f"noted: {args.get('summary', '')}"

    return LocalTool(
        name="note",
        description="note",
        input_schema={"type": "object", "additionalProperties": True},
        handler=handler,
    )


def _end_session_tool() -> LocalTool:
    """Mirrors the real end_session — sets the session sentinel."""

    async def handler(session: Session, args: dict):
        session.sentinel_status = args["status"]
        session.sentinel_recap = args.get("recap", "")
        return f"session closing: {args['status']}"

    return LocalTool(
        name="end_session",
        description="close",
        input_schema={"type": "object", "additionalProperties": True},
        handler=handler,
    )


def _registry() -> dict[str, LocalTool]:
    return {t.name: t for t in (_note_tool(), _peek_tool(), _end_session_tool())}


def _schemas(registry: dict[str, LocalTool]) -> list[dict]:
    return [
        {"name": t.name, "description": t.description,
         "input_schema": t.input_schema}
        for t in registry.values()
    ]


def _loop_run(provider, registry, **over) -> engine_mod.EngineRun:
    schemas = _schemas(registry)
    return _mk_run(
        provider=provider, tool_schemas=schemas, local_registry=registry, **over,
    )


@pytest.fixture
def patched_loop_deps(mocker):
    """Stub out compact / scratchpad / plan tail injection so _loop runs
    in isolation."""
    mocker.patch.object(scratchpad_mod, "inject_tail",
                        side_effect=lambda msgs, _sp: msgs)
    mocker.patch.object(plan_mod, "inject_tail",
                        side_effect=lambda msgs, _p: msgs)
    mocker.patch.object(screen_layout_mod, "inject_tail",
                        side_effect=lambda msgs: msgs)
    mocker.patch.object(compact_mod, "drop_stale_screens")
    mocker.patch.object(compact_mod, "collapse_old_turns")


@pytest.mark.asyncio
async def test_loop_closes_cleanly_on_end_session(patched_loop_deps) -> None:
    registry = _registry()

    asst = _asst(
        tool_calls=[
            _tc("note", {"summary": "closing"}),
            _tc("end_session", {"status": DONE, "recap": "all done"}),
        ],
        finish=FinishReason.TOOL_CALLS,
    )
    provider = FakeProvider([asst])

    session = Session()
    messages: list = [
        SystemMessage(content="sys"),
        UserMessage(content="trig"),
    ]

    await engine_mod._loop(_loop_run(provider, registry), session, messages)

    assert session.sentinel_status == DONE
    assert session.sentinel_recap == "all done"
    # Last messages: assistant + tool_result for note + tool_result for end.
    assert isinstance(messages[-3], AssistantMessage)
    assert isinstance(messages[-2], ToolResultMessage)
    assert isinstance(messages[-1], ToolResultMessage)


@pytest.mark.asyncio
async def test_loop_skips_layout_reminder_when_complete(mocker) -> None:
    # Default (layout_incomplete=False) must NOT call screen_layout.inject_tail
    # — that avoids a per-turn disk read once setup is done.
    mocker.patch.object(scratchpad_mod, "inject_tail",
                        side_effect=lambda msgs, _sp: msgs)
    mocker.patch.object(plan_mod, "inject_tail",
                        side_effect=lambda msgs, _p: msgs)
    mocker.patch.object(compact_mod, "drop_stale_screens")
    mocker.patch.object(compact_mod, "collapse_old_turns")
    spy = mocker.patch.object(screen_layout_mod, "inject_tail",
                              side_effect=lambda msgs: msgs)

    registry = _registry()
    asst = _asst(tool_calls=[
        _tc("note", {"summary": "x"}),
        _tc("end_session", {"status": DONE, "recap": "done"}),
    ])
    session = Session()

    await engine_mod._loop(
        _loop_run(FakeProvider([asst]), registry),  # layout_incomplete defaults False
        session, [SystemMessage(content="s")],
    )
    spy.assert_not_called()

    # With layout_incomplete=True it IS injected each turn.
    session2 = Session()
    await engine_mod._loop(
        _loop_run(FakeProvider([asst]), registry, layout_incomplete=True),
        session2, [SystemMessage(content="s")],
    )
    spy.assert_called()


@pytest.mark.asyncio
async def test_loop_stucks_after_consecutive_correctives(patched_loop_deps) -> None:
    # A model that never holds the [note, one-other] shape must end
    # STUCK after CORRECTIVE_LIMIT consecutive correctives instead of
    # burning max_turns of round-trips.
    registry = _registry()

    bad = [_asst(tool_calls=[_tc("peek")]) for _ in range(engine_mod.CORRECTIVE_LIMIT + 3)]
    provider = FakeProvider(bad)
    session = Session()

    await engine_mod._loop(
        _loop_run(provider, registry), session,
        [SystemMessage(content="sys"), UserMessage(content="trig")],
    )

    assert session.sentinel_status == STUCK
    assert "malformed turns" in session.sentinel_recap
    assert len(provider.calls) == engine_mod.CORRECTIVE_LIMIT


# ---------- pre-compression checkpoint ----------


class CheckpointProvider(FakeProvider):
    """Collapse knobs small enough to hit the checkpoint turn in a short
    scripted session; records every request for tail assertions."""
    COLLAPSE_FIRST_AT_TURN = 2
    COLLAPSE_INTERVAL_TURNS = 100
    KEEP_RECENT_TURNS = 1

    def __init__(self, responses):
        super().__init__(responses)
        self.requests: list[list] = []

    async def chat(self, messages, tools):
        self.requests.append(list(messages))
        return await super().chat(messages, tools)


def _scaffold_messages() -> list:
    """[system, trigger] + the three compaction slots, as bootstrap builds."""
    return [
        SystemMessage(content="sys"),
        UserMessage(content="trig"),
        compact_mod.new_summary_placeholder(),
        UserMessage(content=compact_mod.MEMORY_INITIAL),
        compact_mod.new_skills_placeholder(),
    ]


def _req_texts(request: list) -> str:
    return "\n".join(
        m.content for m in request
        if isinstance(m, UserMessage) and isinstance(m.content, str)
    )


async def _run_checkpoint_session(responses, provider_cls=CheckpointProvider):
    registry = _registry()
    provider = provider_cls(responses)
    session = Session()
    messages = _scaffold_messages()
    await engine_mod._loop(_loop_run(provider, registry), session, messages)
    return provider, session, messages


@pytest.mark.asyncio
async def test_checkpoint_requires_scratchpad_then_collapses() -> None:
    # Turn before the first collapse: the request carries the ⚠ tail; a
    # scratchpad-less note is rejected once; the compliant retry passes
    # and the collapse folds turn 0 to its summary line.
    provider, session, messages = await _run_checkpoint_session([
        _asst(tool_calls=[_tc("note", {"summary": "a"}), _tc("peek")]),
        _asst(tool_calls=[_tc("note", {"summary": "b"}), _tc("peek")]),
        _asst(tool_calls=[
            _tc("note", {"summary": "b", "scratchpad": "cart=3; addr saved"}),
            _tc("peek"),
        ]),
        _asst(tool_calls=[
            _tc("note", {"summary": "closing"}),
            _tc("end_session", {"status": DONE, "recap": "done"}),
        ]),
    ])

    assert session.sentinel_status == DONE
    assert len(provider.calls) == 4
    # ⚠ tail rides the imminent turn's requests only (1 = first attempt,
    # 2 = retry) — not turn 0, not the post-collapse turn.
    tails = ["newest 1 folds" in _req_texts(r) for r in provider.requests]
    assert tails == [False, True, True, False]
    # The retry request carries the checkpoint corrective.
    assert "Re-issue the SAME turn" in _req_texts(provider.requests[2])
    # The collapse fired: turn 0's summary folded into the slot.
    assert "- a" in messages[2].content


@pytest.mark.asyncio
async def test_checkpoint_fails_open_when_reminder_ignored() -> None:
    # A model that never adds the scratchpad is corrected ONCE, then its
    # summary-only note is accepted — compression proceeds on summaries
    # alone rather than stalling the session.
    provider, session, messages = await _run_checkpoint_session([
        _asst(tool_calls=[_tc("note", {"summary": "a"}), _tc("peek")]),
        _asst(tool_calls=[_tc("note", {"summary": "b"}), _tc("peek")]),
        _asst(tool_calls=[_tc("note", {"summary": "b"}), _tc("peek")]),  # still none
        _asst(tool_calls=[
            _tc("note", {"summary": "closing"}),
            _tc("end_session", {"status": DONE, "recap": "done"}),
        ]),
    ])

    assert session.sentinel_status == DONE
    assert len(provider.calls) == 4
    correctives = sum(
        "Re-issue the SAME turn" in _req_texts(r) for r in provider.requests
    )
    assert correctives == 1  # exactly one nudge, no reject loop
    assert "- a" in messages[2].content  # collapse still fired


@pytest.mark.asyncio
async def test_checkpoint_corrective_renews_per_collapse_event() -> None:
    # With a tight interval EVERY turn is collapse-pending — the single
    # retry must renew per collapse event (completed turn), not once per
    # session, or every compression after the first gets no checkpoint.
    class TightProvider(CheckpointProvider):
        COLLAPSE_INTERVAL_TURNS = 1

    def _plain_turn():
        return _asst(tool_calls=[_tc("note", {"summary": "s"}), _tc("peek")])

    provider, session, _ = await _run_checkpoint_session([
        _plain_turn(),  # t0: not pending
        _plain_turn(),  # t1: pending → corrective #1
        _plain_turn(),  # t1 retry: fail open → collapse #1
        _plain_turn(),  # t2: pending again → corrective #2
        _plain_turn(),  # t2 retry: fail open → collapse #2
        _asst(tool_calls=[
            _tc("note", {"summary": "closing"}),
            _tc("end_session", {"status": DONE, "recap": "done"}),
        ]),             # t3: pending but end_session-exempt
    ], provider_cls=TightProvider)

    assert session.sentinel_status == DONE
    assert len(provider.calls) == 6
    correctives = sum(
        "Re-issue the SAME turn" in _req_texts(r) for r in provider.requests
    )
    assert correctives == 2  # one per collapse event, renewed after each


@pytest.mark.asyncio
async def test_checkpoint_skipped_on_end_session_turn() -> None:
    # Closing on the checkpoint turn: nothing to preserve — no corrective.
    provider, session, _ = await _run_checkpoint_session([
        _asst(tool_calls=[_tc("note", {"summary": "a"}), _tc("peek")]),
        _asst(tool_calls=[
            _tc("note", {"summary": "closing"}),
            _tc("end_session", {"status": DONE, "recap": "done"}),
        ]),
    ])

    assert session.sentinel_status == DONE
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_loop_corrective_counter_resets_on_good_turn(patched_loop_deps) -> None:
    # Interleaved good turns reset the counter — occasional shape slips
    # never accumulate into a STUCK.
    registry = _registry()

    def good(i):
        return _asst(tool_calls=[
            _tc("note", {"summary": f"turn {i}"}), _tc("peek"),
        ])

    responses = []
    for i in range(engine_mod.CORRECTIVE_LIMIT - 1):
        responses.append(_asst(tool_calls=[_tc("peek")]))  # bad shape
    responses.append(good(0))  # resets the counter
    for i in range(engine_mod.CORRECTIVE_LIMIT - 1):
        responses.append(_asst(tool_calls=[_tc("peek")]))  # bad again
    responses.append(_asst(tool_calls=[
        _tc("note", {"summary": "closing"}),
        _tc("end_session", {"status": DONE, "recap": "ok"}),
    ]))
    provider = FakeProvider(responses)
    session = Session()

    await engine_mod._loop(
        _loop_run(provider, registry), session,
        [SystemMessage(content="sys"), UserMessage(content="trig")],
    )

    assert session.sentinel_status == DONE


@pytest.mark.asyncio
async def test_loop_routes_content_filter_to_fail(patched_loop_deps) -> None:
    registry = _registry()
    asst = _asst(finish=FinishReason.CONTENT_FILTER)
    provider = FakeProvider([asst])
    session = Session()

    await engine_mod._loop(
        _loop_run(provider, registry), session, [SystemMessage(content="s")],
    )

    assert session.sentinel_status == FAIL
    assert "content filter" in session.sentinel_recap


@pytest.mark.asyncio
async def test_loop_provider_failure_marks_stuck(patched_loop_deps) -> None:
    registry = _registry()
    provider = FakeProvider([RuntimeError("network")])
    session = Session()

    await engine_mod._loop(
        _loop_run(provider, registry), session, [SystemMessage(content="s")],
    )

    assert session.sentinel_status == STUCK
    assert "network" in session.sentinel_recap


@pytest.mark.asyncio
async def test_loop_no_tool_calls_injects_corrective(patched_loop_deps) -> None:
    registry = _registry()
    asst_no_calls = _asst(content="just talking")
    asst_close = _asst(tool_calls=[
        _tc("note", {"summary": "x"}),
        _tc("end_session", {"status": IDLE, "recap": "nothing"}),
    ])
    provider = FakeProvider([asst_no_calls, asst_close])
    session = Session()
    messages: list = [SystemMessage(content="s")]

    await engine_mod._loop(_loop_run(provider, registry), session, messages)

    correctives = [
        m for m in messages
        if isinstance(m, UserMessage) and "no tool_calls" in str(m.content)
    ]
    assert len(correctives) == 1
    assert session.sentinel_status == IDLE


@pytest.mark.asyncio
async def test_loop_bad_turn_shape_injects_corrective(
    patched_loop_deps,
) -> None:
    registry = _registry()
    bad = _asst(tool_calls=[_tc("peek")])
    good = _asst(tool_calls=[
        _tc("note", {"summary": "y"}),
        _tc("end_session", {"status": DONE, "recap": "ok"}),
    ])
    provider = FakeProvider([bad, good])
    session = Session()
    messages: list = [SystemMessage(content="s")]

    await engine_mod._loop(_loop_run(provider, registry), session, messages)

    correctives = [
        m for m in messages
        if isinstance(m, UserMessage) and "without `note`" in str(m.content)
    ]
    assert len(correctives) == 1
    assert session.sentinel_status == DONE


@pytest.mark.asyncio
async def test_loop_max_turns_marks_stuck(patched_loop_deps) -> None:
    registry = _registry()
    provider = FakeProvider([_asst() for _ in range(2)])
    session = Session()

    await engine_mod._loop(
        _loop_run(provider, registry, settings=_settings(max_turns=2)),
        session, [SystemMessage(content="s")],
    )

    assert session.sentinel_status == STUCK
    assert "max turns" in session.sentinel_recap


@pytest.mark.asyncio
async def test_loop_finish_length_logs_warning(patched_loop_deps) -> None:
    registry = _registry()
    asst = _asst(
        tool_calls=[
            _tc("note", {"summary": "x"}),
            _tc("end_session", {"status": DONE, "recap": "ok"}),
        ],
        finish=FinishReason.LENGTH,
    )
    provider = FakeProvider([asst])
    session = Session()
    tr = MagicMock()

    await engine_mod._loop(
        _loop_run(provider, registry, tr=tr), session, [SystemMessage(content="s")],
    )

    events = [c.args[0].get("event") for c in tr.write.call_args_list]
    assert "finish_length_warning" in events


# ---------- pre-close pitfalls gate ----------


def _pitfall_tool() -> LocalTool:
    from physiclaw.agent.engine.builtin_tool import _handle_add_pitfall

    return LocalTool(
        name="add_pitfall",
        description="add_pitfall",
        input_schema={
            "type": "object",
            "properties": {"items": {"type": "array", "items": {"type": "string"}}},
            "required": ["items"],
        },
        handler=_handle_add_pitfall,
    )


def _registry_with_pitfall() -> dict[str, LocalTool]:
    reg = _registry()
    pt = _pitfall_tool()
    reg[pt.name] = pt
    return reg


def _close(status: str) -> AssistantMessage:
    return _asst(tool_calls=[
        _tc("note", {"summary": "closing"}),
        _tc("end_session", {"status": status, "recap": "r"}),
    ])


def _pitfall_asst() -> AssistantMessage:
    return _asst(tool_calls=[
        _tc("note", {"summary": "banking traps"}),
        _tc("add_pitfall", {"items": ["京东: avoid Ai搜索"]}),
    ])


async def _run_close_loop(provider, session, registry):
    messages: list = [SystemMessage(content="s")]
    await engine_mod._loop(_loop_run(provider, registry), session, messages)
    return messages


def _pitfall_correctives(messages) -> list:
    return [m for m in messages
            if isinstance(m, UserMessage) and "add_pitfall(" in str(m.content)]


def _floor0(monkeypatch) -> None:
    # Drop capture_turn_floor to 0 so a short scripted DONE (turn 0) trips the gate.
    from physiclaw import config
    monkeypatch.setattr(config.CONFIG.pitfalls, "capture_turn_floor", 0)


@pytest.mark.asyncio
async def test_loop_forces_pitfall_on_long_done(patched_loop_deps, monkeypatch) -> None:
    # A long DONE → force add_pitfall, then the re-issued close passes.
    from physiclaw.agent.engine import pitfalls

    _floor0(monkeypatch)
    registry = _registry_with_pitfall()
    provider = FakeProvider([_close(DONE), _pitfall_asst(), _close(DONE)])
    session = Session()

    messages = await _run_close_loop(provider, session, registry)

    assert len(_pitfall_correctives(messages)) == 1
    assert session.added_pitfalls is True
    assert session.sentinel_status == DONE
    assert pitfalls.read() == ["京东: avoid Ai搜索"]


@pytest.mark.asyncio
async def test_loop_pitfall_corrective_carries_trajectory(patched_loop_deps, monkeypatch) -> None:
    # The turn-marked trajectory rides the corrective so the agent mines the
    # real turn-wasters even after compaction folds early turns.
    _floor0(monkeypatch)
    registry = _registry_with_pitfall()
    session = Session()
    session.plan.update(
        understanding="buy milk",
        steps=[{"content": "search via 搜索 not Ai搜索", "status": "completed"}],
    )
    provider = FakeProvider([_close(DONE), _pitfall_asst(), _close(DONE)])

    messages = await _run_close_loop(provider, session, registry)

    corrective = _pitfall_correctives(messages)[0].content
    assert "<session-trajectory>" in corrective
    assert "search via 搜索 not Ai搜索" in corrective


@pytest.mark.asyncio
async def test_loop_pitfall_gate_fails_open_after_one_retry(patched_loop_deps, monkeypatch) -> None:
    # Agent ignores the corrective and re-issues end_session — closes on the
    # second attempt (one-shot), never adding.
    _floor0(monkeypatch)
    registry = _registry_with_pitfall()
    provider = FakeProvider([_close(DONE), _close(DONE)])
    session = Session()

    messages = await _run_close_loop(provider, session, registry)

    assert len(_pitfall_correctives(messages)) == 1
    assert session.added_pitfalls is False
    assert session.sentinel_status == DONE


@pytest.mark.asyncio
async def test_loop_no_pitfall_gate_on_stuck(patched_loop_deps, monkeypatch) -> None:
    # A STUCK close never captures — the agent never escaped the trap, so it
    # can't write a real fix. Even with the floor at 0.
    _floor0(monkeypatch)
    registry = _registry_with_pitfall()
    provider = FakeProvider([_close(STUCK)])
    session = Session()

    messages = await _run_close_loop(provider, session, registry)

    assert not _pitfall_correctives(messages)
    assert session.sentinel_status == STUCK


@pytest.mark.asyncio
async def test_loop_no_pitfall_gate_on_short_done(patched_loop_deps) -> None:
    # A short DONE (below capture_turn_floor) sailed through → no gate.
    registry = _registry_with_pitfall()
    provider = FakeProvider([_close(DONE)])  # closes at turn 0, far below the floor
    session = Session()

    messages = await _run_close_loop(provider, session, registry)

    assert not _pitfall_correctives(messages)
    assert session.sentinel_status == DONE


@pytest.mark.asyncio
async def test_loop_no_pitfall_gate_on_idle(patched_loop_deps, monkeypatch) -> None:
    _floor0(monkeypatch)
    registry = _registry_with_pitfall()
    provider = FakeProvider([_close(IDLE)])
    session = Session()

    messages = await _run_close_loop(provider, session, registry)

    assert not _pitfall_correctives(messages)
    assert session.sentinel_status == IDLE


def test_should_capture_matrix() -> None:
    from physiclaw.agent.engine.pitfalls import should_capture

    floor = CONFIG.pitfalls.capture_turn_floor
    s = Session()
    assert should_capture(IDLE, 999, s)[0] is False
    assert should_capture(WAIT, 999, s)[0] is False
    # STUCK / FAIL never capture, however long.
    assert should_capture(STUCK, floor + 5, s)[0] is False
    assert should_capture(FAIL, floor + 5, s)[0] is False
    # DONE captures only past the turn floor.
    assert should_capture(DONE, floor - 1, s)[0] is False  # short DONE
    do, seed = should_capture(DONE, floor, s)               # long DONE
    assert do is True
    assert f"{floor} turns" in seed and "DONE" in seed
    # stuck_events only enriches the seed, doesn't gate.
    looped = Session()
    looped.stuck_events = 2
    assert "2×" in should_capture(DONE, floor, looped)[1]


def test_should_capture_disabled(monkeypatch) -> None:
    from physiclaw import config
    from physiclaw.agent.engine.pitfalls import should_capture

    monkeypatch.setattr(config.CONFIG.pitfalls, "capture_enabled", False)
    assert should_capture(DONE, 999, Session())[0] is False  # long DONE, but off


# ---------- pre-close memory-cue gate ----------


def _save_memory_tool() -> LocalTool:
    async def _h(session, _args):
        session.saved_memory = True
        return "saved"
    return LocalTool(
        name="save_memory", description="save",
        input_schema={"type": "object", "properties": {"text": {"type": "string"}},
                      "required": ["text"]},
        handler=_h,
    )


def _registry_with_save() -> dict[str, LocalTool]:
    reg = _registry()
    st = _save_memory_tool()
    reg[st.name] = st
    return reg


def _note(summary: str, other_name: str, other_args: dict | None = None) -> AssistantMessage:
    return _asst(tool_calls=[_tc("note", {"summary": summary}),
                             _tc(other_name, other_args or {})])


def _memory_correctives(messages) -> list:
    return [m for m in messages
            if isinstance(m, UserMessage) and "flagged something to remember" in str(m.content)]


@pytest.mark.asyncio
async def test_loop_scans_cue_each_turn_and_forces_save_before_close(patched_loop_deps) -> None:
    registry = _registry_with_save()
    # turn 0 banks a cue (note summary) alongside an action; close is IDLE so
    # the pitfalls gate stays out of the way — only the memory gate fires.
    provider = FakeProvider([
        _note("记住 user hates cilantro", "peek"),
        _note("closing", "end_session", {"status": IDLE, "recap": "r"}),
        _note("saving", "save_memory", {"text": "no cilantro"}),
        _note("closing", "end_session", {"status": IDLE, "recap": "r"}),
    ])
    session = Session()
    messages = await _run_close_loop(provider, session, registry)

    assert any("cilantro" in c for c in session.memory_cues)  # scanned turn 0
    assert len(_memory_correctives(messages)) == 1
    assert session.saved_memory is True
    assert session.sentinel_status == IDLE


@pytest.mark.asyncio
async def test_loop_no_memory_gate_when_already_saved(patched_loop_deps) -> None:
    registry = _registry_with_save()
    provider = FakeProvider([
        _note("记住 no cilantro", "save_memory", {"text": "no cilantro"}),
        _note("closing", "end_session", {"status": IDLE, "recap": "r"}),
    ])
    session = Session()
    messages = await _run_close_loop(provider, session, registry)

    assert not _memory_correctives(messages)  # cue present but already saved
    assert session.sentinel_status == IDLE


@pytest.mark.asyncio
async def test_loop_no_memory_gate_without_a_cue(patched_loop_deps) -> None:
    registry = _registry_with_save()
    provider = FakeProvider([_note("tapped the button", "end_session",
                                    {"status": IDLE, "recap": "r"})])
    session = Session()
    messages = await _run_close_loop(provider, session, registry)

    assert not session.memory_cues
    assert not _memory_correctives(messages)
    assert session.sentinel_status == IDLE


@pytest.mark.asyncio
async def test_loop_memory_gate_fails_open_after_one_retry(patched_loop_deps) -> None:
    registry = _registry_with_save()
    provider = FakeProvider([
        _note("记住 the gate code 4021", "peek"),
        _note("closing", "end_session", {"status": IDLE, "recap": "r"}),
        _note("closing anyway", "end_session", {"status": IDLE, "recap": "r"}),
    ])
    session = Session()
    messages = await _run_close_loop(provider, session, registry)

    assert len(_memory_correctives(messages)) == 1  # one nudge, then closes
    assert session.saved_memory is False
    assert session.sentinel_status == IDLE


# ---------- _run_session ----------


def _async_returning(value):
    async def _coro(*a, **kw):
        return value
    return _coro


def _patch_session_deps(mocker):
    """Stub everything _run_session pulls beyond the loop."""
    mocker.patch("physiclaw.config.parse_model_ref",
                 return_value=("fake", "fake-model"))
    mocker.patch.object(engine_mod, "get_mcp",
                        side_effect=_async_returning(FakeMcpClient()))
    mocker.patch.object(engine_mod, "list_tools_cached",
                        side_effect=_async_returning([]))
    mocker.patch.object(skill_mod, "discover_builtin_skills", return_value={})
    mocker.patch.object(skill_mod, "discover_user_skills", return_value={})
    mocker.patch.object(
        builtin_tool_mod, "build_registry", return_value=_registry(),
    )
    mocker.patch.object(
        builtin_tool_mod, "schemas", return_value=_schemas(_registry()),
    )
    mocker.patch.object(memory_mod, "load_persistent", return_value="")
    mocker.patch.object(skill_mod, "render_builtin", return_value="")
    mocker.patch.object(skill_mod, "render_section", return_value="")
    mocker.patch.object(prompt_mod, "render_system_prompts", return_value="SYSTEM")
    mocker.patch.object(prompt_mod, "prefix_hash", return_value="hashX")
    mocker.patch.object(compact_mod, "new_summary_placeholder",
                        return_value=UserMessage(content="<sum>"))
    mocker.patch.object(compact_mod, "new_memory_placeholder",
                        return_value=UserMessage(content="<mem>"))
    mocker.patch.object(compact_mod, "new_skills_placeholder",
                        return_value=UserMessage(content="<skl>"))
    mocker.patch.object(scratchpad_mod, "inject_tail",
                        side_effect=lambda msgs, _sp: msgs)
    mocker.patch.object(plan_mod, "inject_tail",
                        side_effect=lambda msgs, _p: msgs)
    mocker.patch.object(screen_layout_mod, "inject_tail",
                        side_effect=lambda msgs: msgs)
    mocker.patch.object(compact_mod, "drop_stale_screens")
    mocker.patch.object(compact_mod, "collapse_old_turns")
    mocker.patch.object(jobs_mod, "format_fired", return_value="")

    fake_tr = MagicMock()
    fake_rlog = MagicMock()
    mocker.patch.object(engine_mod, "Trace", return_value=fake_tr)
    mocker.patch.object(engine_mod, "RawLog", return_value=fake_rlog)
    return {"tr": fake_tr, "rlog": fake_rlog}


@pytest.mark.asyncio
async def test_run_session_closes_provider_in_finally(mocker) -> None:
    deps = _patch_session_deps(mocker)
    asst = _asst(tool_calls=[
        _tc("note", {"summary": "x"}),
        _tc("end_session", {"status": DONE, "recap": "fin"}),
    ])
    fake_provider = FakeProvider([asst])
    mocker.patch.object(engine_mod, "make_provider", return_value=fake_provider)

    session = Session()
    await engine_mod._run_session(
        [Trigger(description="t")], model_ref="fake/fake-model", session=session,
    )

    assert session.sentinel_status == DONE
    assert fake_provider.closed is True
    deps["tr"].close.assert_called_once()
    deps["rlog"].close.assert_called_once()


@pytest.mark.asyncio
async def test_run_session_crash_marks_stuck(mocker) -> None:
    _patch_session_deps(mocker)
    mocker.patch.object(
        engine_mod, "make_provider", side_effect=RuntimeError("bad provider"),
    )

    session = Session()
    await engine_mod._run_session(
        [Trigger(description="t")], model_ref="fake/fake-model", session=session,
    )

    assert session.sentinel_status == STUCK
    assert "session crashed" in session.sentinel_recap


@pytest.mark.asyncio
async def test_run_session_cancellation_propagates(mocker) -> None:
    _patch_session_deps(mocker)
    mocker.patch.object(
        engine_mod, "make_provider", side_effect=asyncio.CancelledError,
    )

    with pytest.raises(asyncio.CancelledError):
        await engine_mod._run_session(
            [Trigger(description="t")], model_ref="fake/fake-model",
            session=Session(),
        )


@pytest.mark.asyncio
async def test_run_session_wait_without_create_job_auto_schedules(
    mocker,
) -> None:
    _patch_session_deps(mocker)
    asst = _asst(tool_calls=[
        _tc("note", {"summary": "x"}),
        _tc("end_session", {"status": WAIT, "recap": "waiting"}),
    ])
    mocker.patch.object(engine_mod, "make_provider",
                        return_value=FakeProvider([asst]))
    schedule_spy = mocker.patch.object(engine_mod, "_auto_schedule_wait_check")

    session = Session()
    await engine_mod._run_session(
        [Trigger(description="t")], model_ref="fake/fake-model", session=session,
    )

    assert session.sentinel_status == WAIT
    schedule_spy.assert_called_once()


@pytest.mark.asyncio
async def test_run_session_wait_with_create_job_skips_auto_schedule(
    mocker,
) -> None:
    _patch_session_deps(mocker)

    async def _end(session: Session, args: dict):
        session.sentinel_status = args["status"]
        session.sentinel_recap = args.get("recap", "")
        session.sentinel_turn_created_job = True
        return "ok"

    custom = LocalTool(
        "end_session", "x",
        {"type": "object", "additionalProperties": True}, _end,
    )
    registry = {**_registry(), "end_session": custom}
    schemas = _schemas(registry)
    mocker.patch.object(
        builtin_tool_mod, "build_registry", return_value=registry,
    )
    mocker.patch.object(
        builtin_tool_mod, "schemas", return_value=schemas,
    )

    asst = _asst(tool_calls=[
        _tc("note", {"summary": "x"}),
        _tc("end_session", {"status": WAIT, "recap": "scheduled"}),
    ])
    mocker.patch.object(engine_mod, "make_provider",
                        return_value=FakeProvider([asst]))
    schedule_spy = mocker.patch.object(engine_mod, "_auto_schedule_wait_check")

    session = Session()
    await engine_mod._run_session(
        [Trigger(description="t")], model_ref="fake/fake-model", session=session,
    )

    schedule_spy.assert_not_called()


# ---------- run (top-level retry on STUCK) ----------


@pytest.mark.asyncio
async def test_run_retries_on_stuck(mocker) -> None:
    mocker.patch.object(engine_mod.CONFIG.engine, "max_attempts", 3)
    statuses = iter([STUCK, STUCK, DONE])

    async def fake_session(triggers, *, model_ref, session: Session, settings=None):
        session.sentinel_status = next(statuses)
        session.sentinel_recap = "x"

    spy = mocker.patch.object(
        engine_mod, "_run_session", side_effect=fake_session,
    )

    await engine_mod.run([Trigger(description="t")], model_ref="x/y")

    assert spy.call_count == 3


@pytest.mark.asyncio
async def test_run_stops_after_done(mocker) -> None:
    mocker.patch.object(engine_mod.CONFIG.engine, "max_attempts", 5)

    async def fake_session(triggers, *, model_ref, session: Session, settings=None):
        session.sentinel_status = DONE

    spy = mocker.patch.object(
        engine_mod, "_run_session", side_effect=fake_session,
    )

    await engine_mod.run([Trigger(description="t")], model_ref="x/y")

    assert spy.call_count == 1


@pytest.mark.asyncio
async def test_run_gives_up_after_max_stucks(mocker) -> None:
    mocker.patch.object(engine_mod.CONFIG.engine, "max_attempts", 2)

    async def fake_session(triggers, *, model_ref, session: Session, settings=None):
        session.sentinel_status = STUCK
        session.sentinel_recap = "always stuck"

    spy = mocker.patch.object(
        engine_mod, "_run_session", side_effect=fake_session,
    )

    await engine_mod.run([Trigger(description="t")], model_ref="x/y")

    assert spy.call_count == 2


@pytest.mark.asyncio
async def test_run_restarts_once_after_setup_completes(mocker) -> None:
    # Session 1 finishes first-run setup → restart; session 2 does the task.
    outcomes = iter([("setup", True), (DONE, False)])

    async def fake_session(triggers, *, model_ref, session: Session, settings=None):
        status, restart = next(outcomes)
        session.sentinel_status = None if restart else status
        session.restart_for_setup = restart

    spy = mocker.patch.object(engine_mod, "_run_session", side_effect=fake_session)

    await engine_mod.run([Trigger(description="t")], model_ref="x/y")

    assert spy.call_count == 2  # setup session + task session


@pytest.mark.asyncio
async def test_run_no_restart_when_only_first_run_trigger(mocker) -> None:
    # A synthetic first-run wake has no request to resume: after setup
    # completes the layout is saved and loads on the next wake, so the loop
    # must NOT restart (which would replay the stale "learn the layout"
    # trigger and re-enter first-run setup).
    async def fake_session(triggers, *, model_ref, session: Session, settings=None):
        session.sentinel_status = IDLE
        session.restart_for_setup = True

    spy = mocker.patch.object(engine_mod, "_run_session", side_effect=fake_session)

    await engine_mod.run(
        [Trigger(description="learn the layout", source="first-run")],
        model_ref="x/y",
    )

    assert spy.call_count == 1  # no restart — nothing real to resume


@pytest.mark.asyncio
async def test_run_restarts_when_real_trigger_accompanies_first_run(mocker) -> None:
    # first-run + a real request in the same wake → restart so the real
    # request is handled with the layout loaded.
    outcomes = iter([(None, True), (DONE, False)])

    async def fake_session(triggers, *, model_ref, session: Session, settings=None):
        status, restart = next(outcomes)
        session.sentinel_status = status
        session.restart_for_setup = restart

    spy = mocker.patch.object(engine_mod, "_run_session", side_effect=fake_session)

    await engine_mod.run(
        [
            Trigger(description="phone IM arrived", source="phone"),
            Trigger(description="learn the layout", source="first-run"),
        ],
        model_ref="x/y",
    )

    assert spy.call_count == 2  # restarted to handle the phone request


@pytest.mark.asyncio
async def test_run_setup_restart_does_not_consume_stuck_attempt(mocker) -> None:
    # Even with max_attempts=1, the setup restart still allows the task session.
    mocker.patch.object(engine_mod.CONFIG.engine, "max_attempts", 1)
    outcomes = iter([("setup", True), (DONE, False)])

    async def fake_session(triggers, *, model_ref, session: Session, settings=None):
        status, restart = next(outcomes)
        session.sentinel_status = None if restart else status
        session.restart_for_setup = restart

    spy = mocker.patch.object(engine_mod, "_run_session", side_effect=fake_session)

    await engine_mod.run([Trigger(description="t")], model_ref="x/y")

    assert spy.call_count == 2


@pytest.mark.asyncio
async def test_run_setup_restart_fires_at_most_once(mocker) -> None:
    # A pathological session that keeps flagging restart must not loop forever.
    mocker.patch.object(engine_mod.CONFIG.engine, "max_attempts", 3)

    async def fake_session(triggers, *, model_ref, session: Session, settings=None):
        session.sentinel_status = DONE
        session.restart_for_setup = True  # always flags

    spy = mocker.patch.object(engine_mod, "_run_session", side_effect=fake_session)

    await engine_mod.run([Trigger(description="t")], model_ref="x/y")

    # 1 setup restart (uncounted) + then the guard blocks further restarts and
    # the DONE session ends the loop → 2 total.
    assert spy.call_count == 2


# ---------- _dispatch: layout lint ----------


_SEQ_SCHEMA = {"name": "sequence", "input_schema": {"type": "object"}}


@pytest.mark.asyncio
async def test_dispatch_blocks_sequence_on_layout_lint(mocker) -> None:
    mocker.patch.object(
        screen_layout_mod, "lint_gesture",
        return_value="BLOCKED — not executed: wrong box",
    )
    mcp = FakeMcpClient()
    run = _mk_run(mcp=mcp, schema_by_name={"sequence": _SEQ_SCHEMA})

    result = await engine_mod._dispatch(
        run, Session(),
        _tc("sequence", {"actions": [{"tool_name": "long_press", "arg": [0, 0.9, 1, 1]}]}),
        0,
    )

    assert result.is_error is True
    assert result.content.startswith("BLOCKED")
    assert mcp.tool_calls == []  # never actuated


@pytest.mark.asyncio
async def test_dispatch_lint_failure_is_fail_open(mocker) -> None:
    # A lint crash must never take down dispatch — the batch runs.
    mocker.patch.object(
        screen_layout_mod, "lint_gesture",
        side_effect=RuntimeError("lint bug"),
    )
    mcp = FakeMcpClient()
    run = _mk_run(mcp=mcp, schema_by_name={"sequence": _SEQ_SCHEMA})

    result = await engine_mod._dispatch(
        run, Session(), _tc("sequence", {"actions": []}), 0,
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

    await engine_mod._dispatch(
        run, session, _tc("tap", {"bbox": [0.1, 0.9, 0.7, 0.95]}), 0,
    )

    kb_spy.assert_called_once()
    name, args, changed = kb_spy.call_args.args
    assert name == "tap"
    assert changed in (True, False)  # the parsed verdict, not None
