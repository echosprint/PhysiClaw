"""Tests for the dispatch seam of the e2e debug harness — with
`EngineRun.debug_intercept` set, the call still EXECUTES for real and
only its successful result blocks may be rewritten (traced as
`debug_faked`); None keeps the real result; the production default
(field unset) leaves dispatch untouched."""

from __future__ import annotations

import pytest
from engine_fakes import FakeMcpClient, _mk_run, _tc

from physiclaw.agent.engine.dispatch import dispatch
from physiclaw.agent.engine.session import Session
from physiclaw.debug.thread import TINY_JPEG_B64

PEEK_SCHEMA = {"name": "peek", "input_schema": {"type": "object", "properties": {}}}

FAKED = [
    {"type": "image", "data": TINY_JPEG_B64, "mimeType": "image/jpeg"},
    {"type": "text", "text": 'id [kind] "label" [left,top,right,bottom] conf'},
]


@pytest.mark.asyncio
async def test_rewritten_call_still_executes_and_marks_the_trace() -> None:
    mcp = FakeMcpClient()
    run = _mk_run(mcp=mcp, schema_by_name={"peek": PEEK_SCHEMA})
    seen: list[tuple[bool, list[dict]]] = []
    run.debug_intercept = lambda call, synthesized, blocks: (
        seen.append((synthesized, blocks)),
        FAKED,
    )[1]
    call = _tc("peek", {})
    session = Session()
    session.synthesized_turn = True

    result = await dispatch(run, session, call, 0)

    assert result.is_error is False and result.tool_call_id == call.id
    assert mcp.tool_calls == [("peek", {})]  # the real call happened
    # Structural provenance + the REAL blocks reach the transformer.
    assert seen == [(True, [{"type": "text", "text": "mcp-ok"}])]
    events = [c.args[0]["event"] for c in run.tr.write.call_args_list]
    assert "debug_faked" in events


@pytest.mark.asyncio
async def test_transformer_none_keeps_the_real_result() -> None:
    mcp = FakeMcpClient()
    run = _mk_run(mcp=mcp, schema_by_name={"peek": PEEK_SCHEMA})
    run.debug_intercept = lambda call, synthesized, blocks: None

    result = await dispatch(run, Session(), _tc("peek", {}), 0)

    assert result.is_error is False
    assert result.content == "mcp-ok"


@pytest.mark.asyncio
async def test_production_default_leaves_dispatch_untouched() -> None:
    mcp = FakeMcpClient()
    run = _mk_run(mcp=mcp, schema_by_name={"peek": PEEK_SCHEMA})

    result = await dispatch(run, Session(), _tc("peek", {}), 0)

    assert result.is_error is False
    assert mcp.tool_calls == [("peek", {})]
