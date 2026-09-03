"""Tests for the rehearsal harness (`conductor.rehearsal` behind
`playbooks run`) — the surface that replaced arming as the way to
test a walk.

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
from physiclaw.conductor import rehearsal, suspension
from physiclaw.contract.dto import ToolCall

HOME = make_screen(("Files", 0.5, 0.1)).text
RESULTS = make_screen(("综合", 0.5, 0.1)).text

FLOW = """\
description: two moves
inputs:
  keyword:
    description: what to search
route:
  - page: home
  - do: open
    macro: open-app
    with: {message: "{inputs.keyword}"}
  - page: home
  - do: search
    macro: add-cart
    with: {message: "go"}
  - page: results
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

    text, is_error = await rehearsal.dispatch(mcp, call, _registry())

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

    text, is_error = await rehearsal.dispatch(mcp, call, _registry())

    assert is_error is False and "综合" in text
    assert mcp.calls == []  # never went over MCP
    assert runner.await_count == 1


async def test_dispatch_reports_an_unknown_pack_macro() -> None:
    mcp = _FakeMcp()
    call = ToolCall(
        id="c", name=gesture_vocab.RUN_MACRO, arguments={"name": "demo/nope"}
    )

    text, is_error = await rehearsal.dispatch(mcp, call, _registry())

    assert is_error is True and "nope" in text


async def test_dispatch_turns_a_raised_error_into_a_result(mocker) -> None:
    # A rehearsal reports, it does not crash: the walk must see an errored
    # result and hand over, the way it would in a real session.
    mcp = _FakeMcp()
    mocker.patch.object(mcp, "call_tool", side_effect=RuntimeError("bridge down"))
    call = ToolCall(id="c", name="peek", arguments={})

    text, is_error = await rehearsal.dispatch(mcp, call, _registry())

    assert is_error is True and "bridge down" in text


async def test_dispatch_transform_replaces_a_successful_result(mocker) -> None:
    # The debug fake-channel seam, reproduced: a transformer may swap the
    # blocks the program reads; None keeps the real ones.
    mcp = _FakeMcp()
    call = ToolCall(id="c", name="peek", arguments={})
    seen: list = []

    def transform(c, blocks):
        seen.append((c.name, len(blocks)))
        return [{"type": "image", "data": "x"}, {"type": "text", "text": RESULTS}]

    text, is_error = await rehearsal.dispatch(
        mcp, call, _registry(), transform=transform
    )

    assert is_error is False and "综合" in text and "Files" not in text
    assert seen == [("peek", 2)]


async def test_dispatch_forwards_macro_step_range_to_the_runner(mocker) -> None:
    mcp = _FakeMcp()
    runner = mocker.patch(
        "physiclaw.macros.runner.run_and_record",
        new=mocker.AsyncMock(
            return_value=mocker.Mock(ok=True, blocks=[{"type": "text", "text": "ran"}])
        ),
    )
    call = ToolCall(
        id="c", name=gesture_vocab.RUN_MACRO, arguments={"name": "demo/open-app"}
    )

    await rehearsal.dispatch(mcp, call, _registry(), start_at="clip", stop_after="clip")

    kwargs = runner.await_args.kwargs
    assert kwargs["start_at"] == "clip" and kwargs["stop_after"] == "clip"


# ---------- walk with step_one: the stepping pause ----------


async def test_walk_pauses_when_the_stepping_cursor_moves(mocker) -> None:
    # `step_one` = run one node only: the first move settles (verify
    # matched), the cursor advances, and the walk pauses BEFORE the next
    # step opens — reporting PAUSED, not a handover — with the position
    # projected for the next invocation.
    from conductor_fakes import build_program

    write_pack(playbooks={"flow": FLOW}, macros=("open-app", "add-cart"))
    mcp = _FakeMcp()
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
    program = build_program(dry=True, keyword="milk")
    program.step_one = True
    from physiclaw.conductor.playbook import load_pack, qualified_pack

    registry = qualified_pack("demo", load_pack("demo"))

    out = await rehearsal.walk(program, registry, mcp, emit=lambda s: None)

    assert out == rehearsal.WALK_PAUSED
    assert program.phase == "paused" and program.outcome is None
    assert program.state()["idx"] == 1
    assert [n for n, _ in mcp.calls] == ["peek", "peek"]  # unlock probe + opening


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
                "description: two moves", "enabled: false\ndescription: two moves"
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


# ---------- the model log: what went to the provider, what came back ----------

AGENT_FLOW = """\
description: a pure-text call then a move
inputs:
  keyword:
    description: what to search
route:
  - agent: parse
    prompt: "keyword for {inputs.keyword}"
    returns:
      keyword: the keyword
  - page: home
  - do: open
    macro: open-app
    with: {message: "{parse.keyword}"}
  - page: home
"""

OPENAI_REPLY = {
    "choices": [{"message": {"content": '{"reason": "r", "answer": "done"}'}}]
}


def _fake_micro(monkeypatch):
    """`micro_caller` → a caller that logs one round-trip to the sink the
    walk wired, then answers `done` with the return field."""
    from physiclaw.conductor.calls import AGENT_DONE
    from physiclaw.conductor.micro import MicroOutcome, MicroResult
    from physiclaw.contract.dto import Usage

    class Caller:
        def __init__(self, rlog):
            self.rlog = rlog

        async def run(self, req):
            self.rlog.write_micro(
                req.call,
                [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "ask"},
                ],
                OPENAI_REPLY,
            )
            return MicroResult(
                outcome=MicroOutcome(
                    out=AGENT_DONE,
                    reason="r",
                    confidence=0.9,
                    payload={"keyword": "milk"},
                ),
                detail="r",
                attempts=1,
                usage=Usage(),
                elapsed_ms=3,
            )

        async def aclose(self):
            pass

    monkeypatch.setattr(rehearsal, "micro_caller", lambda rlog=None: Caller(rlog))


async def test_walk_captures_each_model_round_trip(monkeypatch, mocker) -> None:
    from conductor_fakes import build_program

    write_pack(playbooks={"flow": AGENT_FLOW}, macros=("open-app",))
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
    _fake_micro(monkeypatch)
    from physiclaw.conductor.playbook import load_pack, qualified_pack

    program = build_program(dry=True, keyword="milk")
    registry = qualified_pack("demo", load_pack("demo"))
    lines: list[str] = []
    seen: list[dict] = []

    await rehearsal.walk(
        program,
        registry,
        _FakeMcp(),
        emit=lines.append,
        raw=True,
        on_exchange=seen.append,
    )

    (x,) = seen
    assert x["call"] == "agent_fields" and x["node"] == "parse"
    assert x["request"][0]["role"] == "system" and x["reply"] == OPENAI_REPLY
    assert x["attempt"] == 1 and x["attempts"] == 1 and x["history"] == 0
    assert x["outcome"].startswith("done")
    text = "\n".join(lines)
    assert "── model agent_fields (parse)" in text
    assert "[system]\n      sys" in text and "[user]\n      ask" in text
    assert '── reply\n      {"reason": "r", "answer": "done"}' in text
    assert "── decision: done" in text


async def test_walk_without_raw_still_hands_exchanges_to_the_hook(
    monkeypatch, mocker
) -> None:
    from conductor_fakes import build_program

    write_pack(playbooks={"flow": AGENT_FLOW}, macros=("open-app",))
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
    _fake_micro(monkeypatch)
    from physiclaw.conductor.playbook import load_pack, qualified_pack

    lines: list[str] = []
    seen: list[dict] = []

    await rehearsal.walk(
        build_program(dry=True, keyword="milk"),
        qualified_pack("demo", load_pack("demo")),
        _FakeMcp(),
        emit=lines.append,
        on_exchange=seen.append,
    )

    assert len(seen) == 1
    assert not any("── model" in line for line in lines)


def test_exchange_lines_reads_both_wire_shapes() -> None:
    anthropic = {"content": [{"type": "text", "text": "hi"}]}
    record = {
        "call": "agent_act",
        "node": "pick",
        "attempt": 2,
        "attempts": 2,
        "history": 4,
        "request": [{"role": "user", "content": [{"type": "text", "text": "block"}]}],
        "reply": anthropic,
        "outcome": "act → 'row'",
    }

    lines = rehearsal.exchange_lines(record)

    assert (
        lines[0]
        == "── model agent_act (pick) attempt 2/2 · 4 replayed turn(s) not repeated"
    )
    assert lines[1:3] == ["[user]", "block"]
    assert lines[3:5] == ["── reply", "hi"]
    assert lines[-1] == "── decision: act → 'row'"
    assert rehearsal.reply_text({"odd": 1}) == '{"odd": 1}'
