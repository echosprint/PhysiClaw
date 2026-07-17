"""Shared fakes and factories for the engine split's test files —
`test_engine.py` (session lifecycle), `test_loop.py` (turn driver), and
`test_dispatch.py` (tool-call execution). Names keep their historical
underscore form from when they were file-local."""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock

from physiclaw.agent.engine import policy as policy_mod
from physiclaw.agent.engine.builtin_tool import LocalTool
from physiclaw.agent.engine.dto import (
    AssistantMessage,
    CollapsePolicy,
    FinishReason,
    ToolCall,
    Usage,
)
from physiclaw.agent.engine.runspec import EngineRun, Settings
from physiclaw.agent.engine.session import Session


class FakeProvider:
    """Scripted provider — pops AssistantMessages from a queue."""

    PROVIDER_ID = "fake"
    COLLAPSE = CollapsePolicy(first_at=100, keep=10, interval=100)

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


def _asst(
    *, content="", tool_calls=None, finish=FinishReason.TOOL_CALLS, usage=None
) -> AssistantMessage:
    return AssistantMessage(
        content=content,
        tool_calls=tool_calls or [],
        finish_reason=finish,
        usage=usage or Usage(),
    )


def _tc(name: str, args: dict | None = None, *, tcid: str = None) -> ToolCall:
    return ToolCall(
        id=tcid or f"tc_{uuid.uuid4().hex[:8]}",
        name=name,
        arguments=args or {},
    )


def _settings(**over) -> Settings:
    base = dict(
        max_turns=300,
        max_session_attempts=3,
        provider_retry_attempts=3,
        retry_backoff_seconds=0.0,
        wait_default_minutes=15,
        max_session_seconds=0,
    )
    base.update(over)
    return Settings(**base)


def _mk_run(
    provider=None,
    *,
    mcp=None,
    tool_schemas=None,
    schema_by_name=None,
    local_registry=None,
    tr=None,
    rlog=None,
    layout_incomplete=False,
    policies=None,
    settings=None,
    deadline=None,
) -> EngineRun:
    tool_schemas = tool_schemas if tool_schemas is not None else []
    return EngineRun(
        provider=provider,
        mcp=mcp if mcp is not None else FakeMcpClient(),
        tool_schemas=tool_schemas,
        schema_by_name=(
            schema_by_name
            if schema_by_name is not None
            else {s["name"]: s for s in tool_schemas}
        ),
        local_registry=local_registry if local_registry is not None else {},
        tr=tr if tr is not None else MagicMock(),
        rlog=rlog if rlog is not None else MagicMock(),
        settings=settings or _settings(),
        policies=(
            policies
            if policies is not None
            else policy_mod.default_policies(layout_incomplete=layout_incomplete)
        ),
        layout_incomplete=layout_incomplete,
        deadline=deadline,
    )


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
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in registry.values()
    ]


def patch_loop_tails(mocker) -> dict:
    """Stub the tail-injection + compaction deps so `loop.drive` runs in
    isolation — shared by `test_loop.py`'s fixture and `test_engine.py`'s
    `_patch_session_deps`. Imports lazily: `compact` pulls cv2/numpy, and
    this module is imported at collection time by files that never drive
    the loop. Returns the mocks keyed by target so tests can spy."""
    from physiclaw.agent.engine import compact, plan, scratchpad, screen_layout

    return {
        "scratchpad": mocker.patch.object(
            scratchpad, "inject_tail", side_effect=lambda msgs, _sp: msgs
        ),
        "plan": mocker.patch.object(
            plan, "inject_tail", side_effect=lambda msgs, _p: msgs
        ),
        "screen_layout": mocker.patch.object(
            screen_layout, "inject_tail", side_effect=lambda msgs: msgs
        ),
        "drop_stale_screens": mocker.patch.object(compact, "drop_stale_screens"),
        "collapse_old_turns": mocker.patch.object(compact, "collapse_old_turns"),
    }
