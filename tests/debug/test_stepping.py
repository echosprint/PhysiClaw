"""Tests for `physiclaw.debug.stepping` — the driver behind
`playbooks step` and the studio's playbook panel: the catalog the
panel lists, the seeded-output fast-forward, and the JSON the CLI
prints for an agent's loop. The CLI's own flag tests live beside the
conductor tests (`tests/conductor/test_step_cli.py`)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "conductor"))
from conductor_fakes import make_screen, write_pack  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from physiclaw.cli import app  # noqa: E402
from physiclaw.debug import stepping  # noqa: E402

runner = CliRunner()
HOME = make_screen(("Files", 0.5, 0.1)).text

FLOW = """\
description: parse then move
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
  - do: search
    macro: add-cart
    with: {message: "go"}
  - page: results
"""


class _FakeMcp:
    def __init__(self, *a, **kw):
        self.calls: list[tuple[str, dict]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def call_tool(self, name, args=None):
        self.calls.append((name, args or {}))
        return [{"type": "image", "data": "x"}, {"type": "text", "text": HOME}]


@pytest.fixture()
def pack(mocker):
    write_pack(playbooks={"flow": FLOW}, macros=("open-app", "add-cart"))
    mocker.patch("physiclaw.agent.engine.mcp_tool.McpClient", _FakeMcp)
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
    calls: list = []

    def no_model(rlog=None):
        calls.append(1)
        raise RuntimeError("no model configured")

    mocker.patch("physiclaw.conductor.rehearsal.micro_caller", no_model)
    return calls


def test_catalog_lists_inputs_route_and_position(pack) -> None:
    (demo,) = stepping.catalog()

    assert demo["app"] == "demo"
    (flow,) = demo["playbooks"]
    assert flow["name"] == "flow" and flow["enabled"] is True
    assert [i["name"] for i in flow["inputs"]] == ["keyword"]
    assert [(n["id"], n["kind"]) for n in flow["nodes"]] == [
        ("parse", "agent"),
        ("open", "do"),
        ("search", "do"),
    ]
    assert flow["nodes"][1]["enter"] == "home" and flow["position"] is None


def test_seeded_output_settles_the_pure_text_node_without_a_call(pack) -> None:
    result = runner.invoke(
        app,
        [
            "playbooks",
            "step",
            "demo/flow",
            "-i",
            "keyword=milk",
            "-o",
            "parse.keyword=milk",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert pack == []  # the model was never asked
    out = json.loads(result.stdout)  # stdout is the JSON alone under --json
    assert out["outcome"] == "paused"
    assert out["position"]["idx"] == 2 and out["position"]["outputs"] == {
        "parse.keyword": "milk"
    }
    assert stepping.catalog()[0]["playbooks"][0]["position"]["idx"] == 2


def test_an_explicit_at_reruns_a_settled_node(pack) -> None:
    runner.invoke(
        app,
        [
            "playbooks",
            "step",
            "demo/flow",
            "-i",
            "keyword=milk",
            "-o",
            "parse.keyword=milk",
        ],
    )

    result = runner.invoke(app, ["playbooks", "step", "demo/flow", "--at", "parse"])

    assert result.exit_code == 1 and "no model configured" in result.output
    assert pack == [1]


def test_status_json_reports_none_before_the_first_step(pack) -> None:
    result = runner.invoke(
        app, ["playbooks", "step", "demo/flow", "--status", "--json"]
    )

    assert result.exit_code == 0 and json.loads(result.output) is None


# ---------- macros: the catalog, the resolver, one gesture ----------


def test_macro_catalog_lists_user_and_pack_macros_with_steps(pack) -> None:
    from physiclaw.macros import store as macro_store

    macro_store.init_macro("mine")
    items = {m["name"]: m for m in stepping.macro_catalog()}

    assert items["mine"]["source"] == "user" and items["mine"]["enabled"] is False
    assert items["demo/open-app"]["source"] == "demo"
    assert [(s["name"], s["tool"]) for s in items["demo/open-app"]["steps"]] == [
        ("go", "tap")
    ]
    assert items["demo/open-app"]["steps"][0]["detail"] == "t"
    assert [i["name"] for i in items["demo/open-app"]["inputs"]] == ["message"]


def test_find_macro_resolves_pack_and_user_names(pack) -> None:
    from physiclaw.macros.model import MacroError

    assert stepping.find_macro("demo/open-app").name == "open-app"
    with pytest.raises(MacroError, match="no macro 'demo/nope'"):
        stepping.find_macro("demo/nope")
    with pytest.raises(MacroError, match="pack 'ghost'"):
        stepping.find_macro("ghost/x")
    with pytest.raises(MacroError, match="no macro named 'mine'"):
        stepping.find_macro("mine")


async def test_run_macro_emits_the_step_log_and_shows_the_view(pack, mocker) -> None:
    from physiclaw.macros import runner as macro_runner

    lines: list[str] = []
    seen: list = []
    mcp = _FakeMcp()

    out = await stepping.run_macro(
        stepping.find_macro("demo/open-app"),
        {"message": "hi"},
        mcp,
        start_at="go",
        stop_after="go",
        emit=lines.append,
        observe=lambda call, blocks: seen.append((call.name, len(blocks))),
    )

    kwargs = macro_runner.run_and_record.await_args.kwargs
    assert kwargs["start_at"] == "go" and kwargs["stop_after"] == "go"
    assert out["ok"] is True and out["message"] == "ran"
    assert lines == ["ran", "(0 rows)"] or lines[0] == "ran"
    assert seen == [("run_macro", 2)]


def test_at_abandons_a_pending_ask(pack) -> None:
    # A position left awaiting a reply, jumped away from: the gate must
    # not resume as a reply check on the wrong node.
    runner.invoke(
        app,
        [
            "playbooks",
            "step",
            "demo/flow",
            "-i",
            "keyword=milk",
            "-o",
            "parse.keyword=milk",
        ],
    )
    state = stepping.load_state("demo", "flow")
    stepping.save_state({**state, "awaiting": True, "yes": ["ok"], "no": ["no"]})

    result = runner.invoke(
        app, ["playbooks", "step", "demo/flow", "--at", "open", "--json"]
    )

    assert result.exit_code == 0, result.output
    out = json.loads(result.stdout)  # stdout is the JSON alone under --json
    assert out["outcome"] == "paused" and out["position"]["awaiting"] is False


# ---------- the boot ----------


def test_catalog_lists_the_channel_boot_with_its_activate_step(pack) -> None:
    from conductor_fakes import write_channel

    write_channel(
        "name: open\ndescription: d\nsteps:\n  - name: go\n    tool: home_screen\n"
    )
    from physiclaw.conductor import scaffold

    scaffold.ensure_channel_boot()

    (channel,) = [p for p in stepping.catalog() if p["app"] == "channel"]
    (boot,) = channel["playbooks"]
    assert boot["name"] == "boot"
    assert [(n["id"], n["kind"], n["enter"]) for n in boot["nodes"]] == [
        ("parse", "activate", "thread")
    ]


def test_stepping_the_boot_reads_the_staged_reply_as_the_request(pack, mocker) -> None:
    # The boot has no inputs and reads the thread first: `--reply` is the
    # user's message there, seeded before the opening read — and the
    # activate step's parse_task runs over the virtual thread.
    from conductor_fakes import write_channel

    from physiclaw.conductor import scaffold
    from physiclaw.conductor.micro import MicroOutcome, MicroResult
    from physiclaw.contract.dto import Usage
    from physiclaw.debug import thread as vthread

    write_channel(
        "name: open\ndescription: d\nsteps:\n  - name: go\n    tool: home_screen\n"
    )
    scaffold.ensure_channel_boot()
    seen: list = []

    class _Micro:
        async def run(self, req):
            seen.append(req)
            outcome = MicroOutcome(out="not_a_task", reason="chat", confidence=0.9)
            return MicroResult(outcome, "chat", 1, Usage(), 1)

        async def aclose(self):
            return None

    mocker.patch(
        "physiclaw.conductor.rehearsal.micro_caller", lambda rlog=None: _Micro()
    )

    result = runner.invoke(
        app, ["playbooks", "step", "channel/boot", "--reply", "买牛奶", "--json"]
    )

    assert result.exit_code == 0, result.output
    assert vthread.load().bubbles[0].text == "买牛奶" and vthread.load().staged == []
    assert (
        len(seen) == 1 and seen[0].call == "parse_task" and "买牛奶" in seen[0].listing
    )
    assert json.loads(result.stdout)["outcome"] == "completed"
