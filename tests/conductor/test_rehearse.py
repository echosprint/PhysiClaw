"""Tests for `physiclaw.cli.playbooks`'s rehearsal harness (`playbooks
run`) — the surface that replaced arming as the way to test a walk.

Lives beside the conductor tests, not under `tests/cli/`, because it
drives a real `Program` over the shared pack fixtures in
`conductor_fakes` (a bare sibling import, so only siblings can use it).

What is worth pinning: the routing it reproduces from the engine (local
`run_macro` vs plain MCP) and its never-persist promise. No hardware —
the MCP connection and the macro runner are both faked.
"""

from __future__ import annotations

import pytest
from conductor_fakes import PACK_MACRO, make_screen, write_pack

from physiclaw.cli import playbooks as cli
from physiclaw.common import gesture_vocab
from physiclaw.conductor import suspension
from physiclaw.contract.dto import ToolCall

HOME = make_screen(("Files", 0.5, 0.1)).text
RESULTS = make_screen(("综合", 0.5, 0.1)).text

FLOW = """\
name: flow
description: two legs
inputs:
  keyword:
    description: what to search
nodes:
  - id: open
    type: LEG
    macro: open-app
    with: {message: "{keyword}"}
    verify: home
  - id: search
    type: LEG
    macro: add-cart
    with: {message: "go"}
    enter: home
    verify: results
"""


class _FakeMcp:
    """Records tool calls; answers each with a VIEW-shaped reply.

    Block shape is load-bearing: `verdict.screen_text` drops block 0 when
    it is text, because a gesture replies `[action text, image, listing]`
    while a view replies `[image, listing]`. A fake that returns one bare
    text block would be read as action text and yield nothing."""

    def __init__(self, reply: str = HOME):
        self.calls: list[tuple[str, dict]] = []
        self.reply = reply

    async def call_tool(self, name, args=None):
        self.calls.append((name, args or {}))
        return [{"type": "image", "data": "x"}, {"type": "text", "text": self.reply}]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _registry():
    """The qualified dispatch registry a rehearsal builds — the pack's
    macros under `app/name` keys, exactly like the engine's hidden set."""
    write_pack(playbooks={"flow": FLOW}, macros=("open-app", "add-cart"))
    from physiclaw.conductor.playbook import load_pack, qualified_pack

    return qualified_pack("demo", load_pack("demo"))


# ---------- _dispatch: the engine's routing, reproduced ----------


async def test_dispatch_sends_an_ordinary_gesture_to_mcp() -> None:
    mcp = _FakeMcp()
    call = ToolCall(id="c", name="peek", arguments={})

    text, is_error = await cli._dispatch(mcp, call, _registry())

    assert is_error is False
    assert mcp.calls == [("peek", {})]
    assert "Files" in text


async def test_dispatch_routes_run_macro_to_the_macro_runner(mocker) -> None:
    # run_macro is the LOCAL tool — it must NOT go out as an MCP call.
    mcp = _FakeMcp()
    runner = mocker.patch(
        "physiclaw.macros.runner.run_and_record",
        new=mocker.AsyncMock(
            return_value=mocker.Mock(
                ok=True,
                # The runner contract: composed header + step log is block 0.
                blocks=[
                    {"type": "text", "text": "ran open-app"},
                    {"type": "text", "text": RESULTS},
                ],
            )
        ),
    )
    call = ToolCall(
        id="c",
        name=gesture_vocab.RUN_MACRO,
        arguments={"name": "demo/open-app", "inputs": {"message": "hi"}},
    )

    text, is_error = await cli._dispatch(mcp, call, _registry())

    assert is_error is False and "综合" in text
    assert mcp.calls == []  # never went over MCP
    assert runner.await_count == 1


async def test_dispatch_reports_an_unknown_pack_macro() -> None:
    mcp = _FakeMcp()
    call = ToolCall(
        id="c", name=gesture_vocab.RUN_MACRO, arguments={"name": "demo/nope"}
    )

    text, is_error = await cli._dispatch(mcp, call, _registry())

    assert is_error is True and "nope" in text


async def test_dispatch_turns_a_raised_error_into_a_result(mocker) -> None:
    # A rehearsal reports, it does not crash: the walk must see an errored
    # result and hand over, the way it would in a real session.
    mcp = _FakeMcp()
    mocker.patch.object(mcp, "call_tool", side_effect=RuntimeError("bridge down"))
    call = ToolCall(id="c", name="peek", arguments={})

    text, is_error = await cli._dispatch(mcp, call, _registry())

    assert is_error is True and "bridge down" in text


# ---------- _rehearse: drives, then leaves nothing behind ----------


async def test_rehearse_drives_the_walk_and_persists_nothing(mocker) -> None:
    write_pack(playbooks={"flow": FLOW}, macros=("open-app", "add-cart"))
    mcp = _FakeMcp()
    mocker.patch.object(cli, "McpClient", lambda: mcp, create=True)
    mocker.patch(
        "physiclaw.agent.engine.mcp_tool.McpClient", lambda *a, **k: mcp, create=True
    )
    mocker.patch(
        "physiclaw.macros.runner.run_and_record",
        new=mocker.AsyncMock(
            return_value=mocker.Mock(
                ok=True,
                blocks=[
                    {"type": "text", "text": "ran"},
                    {"type": "text", "text": HOME},
                ],
            )
        ),
    )

    out = await cli._rehearse("demo", "flow", {"keyword": "milk"})

    assert "handed over" in out or "finished" in out
    assert not suspension.suspended_path().exists()


async def test_rehearse_rejects_bad_inputs_before_touching_the_phone() -> None:
    from physiclaw.conductor.playbook import PlaybookError

    write_pack(playbooks={"flow": FLOW}, macros=("open-app", "add-cart"))

    with pytest.raises(PlaybookError, match="unknown input"):
        await cli._rehearse("demo", "flow", {"bogus": "1"})


async def test_rehearse_runs_a_disabled_playbook() -> None:
    # `macros run` rehearses disabled macros for the same reason: you
    # rehearse BEFORE you enable. A disabled playbook must not be refused.
    from physiclaw.conductor import setup as conductor_setup

    write_pack(
        playbooks={
            "flow": FLOW.replace(
                "description: two legs", "enabled: false\ndescription: two legs"
            )
        },
        macros=("open-app", "add-cart"),
    )

    spec, _ = conductor_setup.load_spec("demo", "flow", require_live=False)

    assert spec.enabled is False  # and load_spec did not raise


def test_pack_macro_fixture_is_wired() -> None:
    # Guards the fixture the routing tests lean on: the registry carries
    # the pack's macros under their qualified dispatch keys.
    assert set(_registry()) == {"demo/open-app", "demo/add-cart"}
    assert PACK_MACRO  # the template the fixture writes
