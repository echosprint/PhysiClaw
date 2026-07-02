"""Integration tests for `physiclaw.agent.engine.engine` — `_loop`,
`_run_session`, `_dispatch`, and `run`.

Phase 5 — exercises the full session lifecycle with a scripted
FakeProvider and FakeMcpClient. Existing pure-helper tests live in
`test_engine.py`; this file owns the loop coverage.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from physiclaw.agent.engine import engine as engine_mod
from physiclaw.agent.engine.builtin_tool import LocalTool, Session
from physiclaw.agent.engine.dto import (
    AssistantMessage,
    FinishReason,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from physiclaw.agent.runtime.hook import Trigger
from physiclaw.agent.runtime.sentinel import DONE, FAIL, IDLE, STUCK, WAIT


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


# ---------- _dispatch ----------


@pytest.mark.asyncio
async def test_dispatch_unknown_tool_returns_error() -> None:
    tr = MagicMock()
    call = _tc("does_not_exist")

    result = await engine_mod._dispatch(
        call=call, schema_by_name={}, mcp=FakeMcpClient(),
        local_registry={}, session=Session(), tr=tr, turn=0,
    )

    assert result.is_error is True
    assert "unknown tool" in result.content
    assert result.tool_call_id == call.id


@pytest.mark.asyncio
async def test_dispatch_invalid_args_returns_error() -> None:
    tr = MagicMock()
    schema = {
        "name": "ping",
        "input_schema": {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        },
    }

    result = await engine_mod._dispatch(
        call=_tc("ping", {}),  # missing required "x"
        schema_by_name={"ping": schema},
        mcp=FakeMcpClient(),
        local_registry={},
        session=Session(),
        tr=tr,
        turn=0,
    )

    assert result.is_error is True
    assert "invalid arguments" in result.content


@pytest.mark.asyncio
async def test_dispatch_local_tool_returns_text() -> None:
    tr = MagicMock()

    async def handler(_session, _args):
        return "hello"

    tool = LocalTool("greet", "say hi", {"type": "object"}, handler)
    schema = {"name": "greet", "input_schema": {"type": "object"}}

    result = await engine_mod._dispatch(
        call=_tc("greet"),
        schema_by_name={"greet": schema},
        mcp=FakeMcpClient(),
        local_registry={"greet": tool},
        session=Session(),
        tr=tr,
        turn=0,
    )

    assert result.is_error is False
    assert result.content == "hello"


@pytest.mark.asyncio
async def test_dispatch_mcp_tool_returns_blocks() -> None:
    tr = MagicMock()
    schema = {"name": "physiclaw__peek", "input_schema": {"type": "object"}}
    mcp = FakeMcpClient()

    result = await engine_mod._dispatch(
        call=_tc("physiclaw__peek"),
        schema_by_name={"physiclaw__peek": schema},
        mcp=mcp,
        local_registry={},
        session=Session(),
        tr=tr,
        turn=0,
    )

    assert result.is_error is False
    assert mcp.tool_calls == [("physiclaw__peek", {})]


@pytest.mark.asyncio
async def test_dispatch_local_handler_exception_returns_error() -> None:
    tr = MagicMock()

    async def boom(_session, _args):
        raise RuntimeError("boom")

    tool = LocalTool("bad", "x", {"type": "object"}, boom)
    schema = {"name": "bad", "input_schema": {"type": "object"}}

    result = await engine_mod._dispatch(
        call=_tc("bad"),
        schema_by_name={"bad": schema},
        mcp=FakeMcpClient(),
        local_registry={"bad": tool},
        session=Session(),
        tr=tr,
        turn=0,
    )

    assert result.is_error is True
    assert "boom" in result.content


@pytest.mark.asyncio
async def test_dispatch_mcp_exception_returns_error() -> None:
    tr = MagicMock()
    schema = {"name": "physiclaw__tap", "input_schema": {"type": "object"}}

    class BadMcp:
        async def call_tool(self, *a, **kw):
            raise RuntimeError("mcp down")

    result = await engine_mod._dispatch(
        call=_tc("physiclaw__tap"),
        schema_by_name={"physiclaw__tap": schema},
        mcp=BadMcp(),
        local_registry={},
        session=Session(),
        tr=tr,
        turn=0,
    )

    assert result.is_error is True
    assert "mcp down" in result.content


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


@pytest.fixture
def patched_loop_deps(mocker):
    """Stub out compact / scratchpad / plan tail injection so _loop runs
    in isolation."""
    mocker.patch.object(engine_mod.scratchpad, "inject_tail",
                        side_effect=lambda msgs, _sp: msgs)
    mocker.patch.object(engine_mod.plan, "inject_tail",
                        side_effect=lambda msgs, _p: msgs)
    mocker.patch.object(engine_mod.screen_layout, "inject_tail",
                        side_effect=lambda msgs: msgs)
    mocker.patch.object(engine_mod.compact, "drop_stale_screens")
    mocker.patch.object(engine_mod.compact, "collapse_old_turns")


@pytest.mark.asyncio
async def test_loop_closes_cleanly_on_end_session(patched_loop_deps) -> None:
    registry = _registry()
    schemas = _schemas(registry)
    schema_by_name = {s["name"]: s for s in schemas}

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

    await engine_mod._loop(
        mcp=FakeMcpClient(), provider=provider,
        messages=messages, tool_schemas=schemas,
        schema_by_name=schema_by_name, local_registry=registry,
        session=session, prompt_hash="h",
        tr=MagicMock(), rlog=MagicMock(),
    )

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
    mocker.patch.object(engine_mod.scratchpad, "inject_tail",
                        side_effect=lambda msgs, _sp: msgs)
    mocker.patch.object(engine_mod.plan, "inject_tail",
                        side_effect=lambda msgs, _p: msgs)
    mocker.patch.object(engine_mod.compact, "drop_stale_screens")
    mocker.patch.object(engine_mod.compact, "collapse_old_turns")
    spy = mocker.patch.object(engine_mod.screen_layout, "inject_tail",
                              side_effect=lambda msgs: msgs)

    registry = _registry()
    schemas = _schemas(registry)
    asst = _asst(tool_calls=[
        _tc("note", {"summary": "x"}),
        _tc("end_session", {"status": DONE, "recap": "done"}),
    ])
    session = Session()

    await engine_mod._loop(
        mcp=FakeMcpClient(), provider=FakeProvider([asst]),
        messages=[SystemMessage(content="s")], tool_schemas=schemas,
        schema_by_name={s["name"]: s for s in schemas},
        local_registry=registry, session=session, prompt_hash="h",
        tr=MagicMock(), rlog=MagicMock(),  # layout_incomplete defaults False
    )
    spy.assert_not_called()

    # With layout_incomplete=True it IS injected each turn.
    session2 = Session()
    await engine_mod._loop(
        mcp=FakeMcpClient(), provider=FakeProvider([asst]),
        messages=[SystemMessage(content="s")], tool_schemas=schemas,
        schema_by_name={s["name"]: s for s in schemas},
        local_registry=registry, session=session2, prompt_hash="h",
        tr=MagicMock(), rlog=MagicMock(), layout_incomplete=True,
    )
    spy.assert_called()


@pytest.mark.asyncio
async def test_loop_routes_content_filter_to_fail(patched_loop_deps) -> None:
    registry = _registry()
    schemas = _schemas(registry)
    asst = _asst(finish=FinishReason.CONTENT_FILTER)
    provider = FakeProvider([asst])
    session = Session()

    await engine_mod._loop(
        mcp=FakeMcpClient(), provider=provider,
        messages=[SystemMessage(content="s")], tool_schemas=schemas,
        schema_by_name={s["name"]: s for s in schemas},
        local_registry=registry, session=session, prompt_hash="h",
        tr=MagicMock(), rlog=MagicMock(),
    )

    assert session.sentinel_status == FAIL
    assert "content filter" in session.sentinel_recap


@pytest.mark.asyncio
async def test_loop_provider_failure_marks_stuck(patched_loop_deps) -> None:
    registry = _registry()
    schemas = _schemas(registry)
    provider = FakeProvider([RuntimeError("network")])
    session = Session()

    await engine_mod._loop(
        mcp=FakeMcpClient(), provider=provider,
        messages=[SystemMessage(content="s")], tool_schemas=schemas,
        schema_by_name={s["name"]: s for s in schemas},
        local_registry=registry, session=session, prompt_hash="h",
        tr=MagicMock(), rlog=MagicMock(),
    )

    assert session.sentinel_status == STUCK
    assert "network" in session.sentinel_recap


@pytest.mark.asyncio
async def test_loop_no_tool_calls_injects_corrective(patched_loop_deps) -> None:
    registry = _registry()
    schemas = _schemas(registry)
    asst_no_calls = _asst(content="just talking")
    asst_close = _asst(tool_calls=[
        _tc("note", {"summary": "x"}),
        _tc("end_session", {"status": IDLE, "recap": "nothing"}),
    ])
    provider = FakeProvider([asst_no_calls, asst_close])
    session = Session()
    messages: list = [SystemMessage(content="s")]

    await engine_mod._loop(
        mcp=FakeMcpClient(), provider=provider,
        messages=messages, tool_schemas=schemas,
        schema_by_name={s["name"]: s for s in schemas},
        local_registry=registry, session=session, prompt_hash="h",
        tr=MagicMock(), rlog=MagicMock(),
    )

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
    schemas = _schemas(registry)
    bad = _asst(tool_calls=[_tc("peek")])
    good = _asst(tool_calls=[
        _tc("note", {"summary": "y"}),
        _tc("end_session", {"status": DONE, "recap": "ok"}),
    ])
    provider = FakeProvider([bad, good])
    session = Session()
    messages: list = [SystemMessage(content="s")]

    await engine_mod._loop(
        mcp=FakeMcpClient(), provider=provider,
        messages=messages, tool_schemas=schemas,
        schema_by_name={s["name"]: s for s in schemas},
        local_registry=registry, session=session, prompt_hash="h",
        tr=MagicMock(), rlog=MagicMock(),
    )

    correctives = [
        m for m in messages
        if isinstance(m, UserMessage) and "without `note`" in str(m.content)
    ]
    assert len(correctives) == 1
    assert session.sentinel_status == DONE


@pytest.mark.asyncio
async def test_loop_max_turns_marks_stuck(patched_loop_deps, mocker) -> None:
    mocker.patch.object(engine_mod, "MAX_TURNS", 2)
    registry = _registry()
    schemas = _schemas(registry)
    provider = FakeProvider([_asst() for _ in range(2)])
    session = Session()

    await engine_mod._loop(
        mcp=FakeMcpClient(), provider=provider,
        messages=[SystemMessage(content="s")], tool_schemas=schemas,
        schema_by_name={s["name"]: s for s in schemas},
        local_registry=registry, session=session, prompt_hash="h",
        tr=MagicMock(), rlog=MagicMock(),
    )

    assert session.sentinel_status == STUCK
    assert "max turns" in session.sentinel_recap


@pytest.mark.asyncio
async def test_loop_finish_length_logs_warning(patched_loop_deps) -> None:
    registry = _registry()
    schemas = _schemas(registry)
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
        mcp=FakeMcpClient(), provider=provider,
        messages=[SystemMessage(content="s")], tool_schemas=schemas,
        schema_by_name={s["name"]: s for s in schemas},
        local_registry=registry, session=session, prompt_hash="h",
        tr=tr, rlog=MagicMock(),
    )

    events = [c.args[0].get("event") for c in tr.write.call_args_list]
    assert "finish_length_warning" in events


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
    mocker.patch.object(engine_mod.skill, "discover_builtin_skills", return_value={})
    mocker.patch.object(engine_mod.skill, "discover_user_skills", return_value={})
    mocker.patch.object(
        engine_mod.builtin_tool, "build_registry", return_value=_registry(),
    )
    mocker.patch.object(
        engine_mod.builtin_tool, "schemas", return_value=_schemas(_registry()),
    )
    mocker.patch.object(engine_mod.memory, "load_persistent", return_value="")
    mocker.patch.object(engine_mod.skill, "render_builtin", return_value="")
    mocker.patch.object(engine_mod.skill, "render_section", return_value="")
    mocker.patch.object(engine_mod.prompt, "render_system_prompts", return_value="SYSTEM")
    mocker.patch.object(engine_mod.prompt, "prefix_hash", return_value="hashX")
    mocker.patch.object(engine_mod.compact, "new_summary_placeholder",
                        return_value=UserMessage(content="<sum>"))
    mocker.patch.object(engine_mod.compact, "new_memory_placeholder",
                        return_value=UserMessage(content="<mem>"))
    mocker.patch.object(engine_mod.compact, "new_skills_placeholder",
                        return_value=UserMessage(content="<skl>"))
    mocker.patch.object(engine_mod.scratchpad, "inject_tail",
                        side_effect=lambda msgs, _sp: msgs)
    mocker.patch.object(engine_mod.plan, "inject_tail",
                        side_effect=lambda msgs, _p: msgs)
    mocker.patch.object(engine_mod.screen_layout, "inject_tail",
                        side_effect=lambda msgs: msgs)
    mocker.patch.object(engine_mod.compact, "drop_stale_screens")
    mocker.patch.object(engine_mod.compact, "collapse_old_turns")
    mocker.patch.object(engine_mod.jobs, "format_fired", return_value="")

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
        engine_mod.builtin_tool, "build_registry", return_value=registry,
    )
    mocker.patch.object(
        engine_mod.builtin_tool, "schemas", return_value=schemas,
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
    mocker.patch.object(engine_mod, "MAX_ATTEMPTS", 3)
    statuses = iter([STUCK, STUCK, DONE])

    async def fake_session(triggers, *, model_ref, session: Session):
        session.sentinel_status = next(statuses)
        session.sentinel_recap = "x"

    spy = mocker.patch.object(
        engine_mod, "_run_session", side_effect=fake_session,
    )

    await engine_mod.run([Trigger(description="t")], model_ref="x/y")

    assert spy.call_count == 3


@pytest.mark.asyncio
async def test_run_stops_after_done(mocker) -> None:
    mocker.patch.object(engine_mod, "MAX_ATTEMPTS", 5)

    async def fake_session(triggers, *, model_ref, session: Session):
        session.sentinel_status = DONE

    spy = mocker.patch.object(
        engine_mod, "_run_session", side_effect=fake_session,
    )

    await engine_mod.run([Trigger(description="t")], model_ref="x/y")

    assert spy.call_count == 1


@pytest.mark.asyncio
async def test_run_gives_up_after_max_stucks(mocker) -> None:
    mocker.patch.object(engine_mod, "MAX_ATTEMPTS", 2)

    async def fake_session(triggers, *, model_ref, session: Session):
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

    async def fake_session(triggers, *, model_ref, session: Session):
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
    async def fake_session(triggers, *, model_ref, session: Session):
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

    async def fake_session(triggers, *, model_ref, session: Session):
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
    # Even with MAX_ATTEMPTS=1, the setup restart still allows the task session.
    mocker.patch.object(engine_mod, "MAX_ATTEMPTS", 1)
    outcomes = iter([("setup", True), (DONE, False)])

    async def fake_session(triggers, *, model_ref, session: Session):
        status, restart = next(outcomes)
        session.sentinel_status = None if restart else status
        session.restart_for_setup = restart

    spy = mocker.patch.object(engine_mod, "_run_session", side_effect=fake_session)

    await engine_mod.run([Trigger(description="t")], model_ref="x/y")

    assert spy.call_count == 2


@pytest.mark.asyncio
async def test_run_setup_restart_fires_at_most_once(mocker) -> None:
    # A pathological session that keeps flagging restart must not loop forever.
    mocker.patch.object(engine_mod, "MAX_ATTEMPTS", 3)

    async def fake_session(triggers, *, model_ref, session: Session):
        session.sentinel_status = DONE
        session.restart_for_setup = True  # always flags

    spy = mocker.patch.object(engine_mod, "_run_session", side_effect=fake_session)

    await engine_mod.run([Trigger(description="t")], model_ref="x/y")

    # 1 setup restart (uncounted) + then the guard blocks further restarts and
    # the DONE session ends the loop → 2 total.
    assert spy.call_count == 2

    assert spy.call_count == 2
