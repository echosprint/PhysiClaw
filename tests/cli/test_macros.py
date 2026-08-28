"""Tests for `physiclaw.cli.macros` — list/check/run/stats over macro dirs
under the per-test `~/.physiclaw` (autouse `physiclaw_home`)."""

from __future__ import annotations

import re

import typer
from typer.testing import CliRunner

from physiclaw.cli.macros import macros_app
from physiclaw.common import paths, verdict
from physiclaw.common.config import CONFIG
from physiclaw.macros import runlog as macro_runlog
from physiclaw.macros import runner as macro_runner
from physiclaw.macros import stats as macro_stats

app = typer.Typer()
app.add_typer(macros_app, name="macros")
runner = CliRunner()

VALID = """
name: {name}
description: Demo macro
enabled: {enabled}

inputs:
  msg:
    description: text

steps:
  - name: send-to-clipboard-1
    tool: send_to_clipboard
    with:
      text: "{{msg}}"
"""


TAP_MACRO = """
name: tap-demo
description: One tap, for reading a bbox back out of the run log

steps:
  - name: tap-1
    tool: tap
    with:
      bbox: [0.12, 0.04, 0.34, 0.10]
"""

NAV_MACRO = """
name: nav-demo
description: One argument-free navigation step

steps:
  - name: home-1
    tool: home_screen
"""


def _write(name: str, text: str | None = None, *, enabled: bool = True) -> None:
    d = paths.macros_dir() / name
    d.mkdir(parents=True)
    body = (
        text
        if text is not None
        else VALID.format(name=name, enabled=str(enabled).lower())
    )
    (d / "MACRO.yml").write_text(body, encoding="utf-8")


class FakeMcp:
    """Async-context MCP stand-in returning one changed-screen gesture."""

    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def call_tool(self, name, args=None):
        return [{"type": "text", "text": verdict.attach(f"{name} ok", True)}]


def _patch_mcp(mocker, client=FakeMcp) -> None:
    mocker.patch("physiclaw.agent.engine.mcp_tool.McpClient", client)


# ---------- init ----------


def test_init_scaffolds_a_checkable_macro() -> None:
    result = runner.invoke(app, ["macros", "init", "my-macro"])

    assert result.exit_code == 0
    assert "MACRO.yml" in result.output
    assert "flip `enabled: false` to true" in result.output  # next-steps guidance
    check = runner.invoke(app, ["macros", "check"])
    assert check.exit_code == 0


def test_init_existing_macro_exits_one() -> None:
    runner.invoke(app, ["macros", "init", "my-macro"])

    result = runner.invoke(app, ["macros", "init", "my-macro"])

    assert result.exit_code == 1
    assert "already exists" in result.output


def test_init_bad_name_exits_one() -> None:
    result = runner.invoke(app, ["macros", "init", "Bad_Name"])

    assert result.exit_code == 1
    assert "lowercase" in result.output


# ---------- list ----------


def test_list_with_no_macros_hints_at_init() -> None:
    result = runner.invoke(app, ["macros", "list"])

    assert result.exit_code == 0
    assert "No macros found" in result.output
    assert "physiclaw macros init" in result.output


def test_list_writes_the_format_readme() -> None:
    runner.invoke(app, ["macros", "list"])

    assert (paths.macros_dir() / "README.md").exists()


def test_list_shows_enabled_disabled_and_invalid() -> None:
    _write("on", enabled=True)
    _write("off", enabled=False)
    _write("bad", text="steps: [not yaml")

    result = runner.invoke(app, ["macros", "list"])

    assert "enabled" in result.output and "on" in result.output
    assert "disabled" in result.output and "off" in result.output
    assert "invalid" in result.output and "bad" in result.output


# ---------- check ----------


def test_check_all_valid_exits_zero() -> None:
    _write("good")

    result = runner.invoke(app, ["macros", "check"])

    assert result.exit_code == 0
    assert "✓ good" in result.output


def test_check_invalid_macro_exits_one_with_reason() -> None:
    _write(
        "bad",
        text="name: other\ndescription: d\nsteps:\n  - name: peek-1\n    tool: peek\n",
    )

    result = runner.invoke(app, ["macros", "check"])

    assert result.exit_code == 1
    assert "must equal the directory name" in result.output


def test_check_reminds_that_a_disabled_macro_is_not_reachable() -> None:
    # A disabled macro passes `check` — the one thing a green checkmark
    # invites you to assume it does not.
    _write("staged", enabled=False)

    result = runner.invoke(app, ["macros", "check"])

    assert result.exit_code == 0
    assert "✓ staged  (disabled)" in result.output
    assert "the agent cannot call: staged" in result.output
    assert "enabled: true" in result.output


def test_check_says_nothing_extra_when_every_macro_is_enabled() -> None:
    _write("live")

    result = runner.invoke(app, ["macros", "check"])

    assert "disabled" not in result.output
    assert "cannot call" not in result.output


def test_check_names_every_disabled_macro() -> None:
    _write("one", enabled=False)
    _write("two", enabled=False)
    _write("three")

    result = runner.invoke(app, ["macros", "check"])

    assert "cannot call: one, two" in result.output
    assert "three" not in result.output.split("cannot call")[1]


def test_check_reports_the_fleet_gate_instead_of_the_per_file_advice(mocker) -> None:
    # The gate overrides every file's own flag, so advising `enabled: true`
    # here would send the user to edit a file that changes nothing.
    mocker.patch.object(CONFIG.macros, "enabled", False)
    _write("live")
    _write("staged", enabled=False)

    result = runner.invoke(app, ["macros", "check"])

    assert "the agent sees no macros at all" in result.output
    assert "cannot call" not in result.output


def test_check_still_exits_one_when_a_disabled_macro_sits_beside_a_broken_one() -> None:
    # The reminder must not swallow the failure it prints after.
    _write("staged", enabled=False)
    _write(
        "bad", text="name: nope\ndescription: d\nsteps:\n  - {name: p, tool: peek}\n"
    )

    result = runner.invoke(app, ["macros", "check"])

    assert result.exit_code == 1
    assert "cannot call: staged" in result.output


# ---------- run ----------


def test_run_unknown_macro_exits_one() -> None:
    result = runner.invoke(app, ["macros", "run", "ghost"])

    assert result.exit_code == 1
    assert "no macro named" in result.output


def test_run_invalid_macro_exits_one_with_reason() -> None:
    _write("bad", text="steps: [not yaml")

    result = runner.invoke(app, ["macros", "run", "bad"])

    assert result.exit_code == 1
    assert "invalid YAML" in result.output


def test_run_bad_input_pair_exits_two() -> None:
    _write("demo")

    result = runner.invoke(app, ["macros", "run", "demo", "-i", "novalue"])

    assert result.exit_code == 2
    assert "key=value" in result.output


def test_run_success_prints_log_and_records_stats(mocker) -> None:
    _write("demo")
    _patch_mcp(mocker)

    result = runner.invoke(app, ["macros", "run", "demo", "-i", "msg=hi"])

    assert result.exit_code == 0
    assert "all 1 steps completed" in result.output
    assert macro_stats.load()["demo"]["total_successes"] == 1


def test_run_works_on_disabled_macro(mocker) -> None:
    # Rehearse-before-enable: `run` must not require enabled: true.
    _write("demo", enabled=False)
    _patch_mcp(mocker)

    result = runner.invoke(app, ["macros", "run", "demo", "-i", "msg=hi"])

    assert result.exit_code == 0


def test_run_missing_required_input_records_bad_input(mocker) -> None:
    _write("demo")
    _patch_mcp(mocker)

    result = runner.invoke(app, ["macros", "run", "demo"])

    assert result.exit_code == 1
    assert "missing required input" in result.output
    assert macro_stats.load()["demo"]["last_abort"]["reason"] == "bad_input"


def test_run_abort_exits_one_and_records(mocker) -> None:
    class DownMcp(FakeMcp):
        async def call_tool(self, name, args=None):
            raise RuntimeError("arm busy")

    _write("demo")
    _patch_mcp(mocker, DownMcp)

    result = runner.invoke(app, ["macros", "run", "demo", "-i", "msg=hi"])

    assert result.exit_code == 1
    assert "ABORTED" in result.output
    assert macro_stats.load()["demo"]["last_abort"]["reason"] == "tool_error"


def test_run_unreachable_server_records_nothing(mocker) -> None:
    class NoServer(FakeMcp):
        async def __aenter__(self):
            # The shape McpClient now raises: a plain Exception naming the
            # URL, not the bare CancelledError the anyio task group used to
            # leak (which no `except Exception` could catch, so this command
            # printed a raw traceback instead of the message below).
            raise ConnectionError(
                "cannot reach the MCP server at http://127.0.0.1:8048/mcp"
            )

    _write("demo")
    _patch_mcp(mocker, NoServer)

    result = runner.invoke(app, ["macros", "run", "demo", "-i", "msg=hi"])

    assert result.exit_code == 1
    assert "cannot reach the MCP server" in result.output
    # `mcp`, not `server`: the latter also spawns the agent runtime, which
    # would drive the phone between the steps being rehearsed.
    assert "Start it first: physiclaw mcp" in result.output
    assert "Traceback" not in result.output
    assert macro_stats.load() == {}  # nothing ran, so nothing is recorded


# ---------- runs (per-step debug log) ----------


def test_run_prints_the_run_id_and_lookup_hint(mocker) -> None:
    _write("demo")
    _patch_mcp(mocker)

    result = runner.invoke(app, ["macros", "run", "demo", "-i", "msg=hi"])

    assert "run: macro-run-" in result.output
    assert "physiclaw macros runs " in result.output


def test_runs_with_no_history_reports_empty() -> None:
    result = runner.invoke(app, ["macros", "runs"])

    assert result.exit_code == 0
    assert "No macro runs logged" in result.output


def test_runs_lists_recent_runs(mocker) -> None:
    _write("demo")
    _patch_mcp(mocker)
    runner.invoke(app, ["macros", "run", "demo", "-i", "msg=hi"])

    result = runner.invoke(app, ["macros", "runs"])

    assert result.exit_code == 0
    assert "macro-run-" in result.output
    assert "demo" in result.output
    assert "cli" in result.output


def test_runs_replays_one_run_by_bare_hex(mocker) -> None:
    _write("demo")
    _patch_mcp(mocker)
    ran = runner.invoke(app, ["macros", "run", "demo", "-i", "msg=hi"])
    hex6 = re.search(r"macro-run-([0-9a-f]{6})", ran.output).group(1)

    result = runner.invoke(app, ["macros", "runs", hex6])

    assert result.exit_code == 0
    assert "✓ 1. send_to_clipboard" in result.output
    assert "[ok]" in result.output


def test_runs_shows_the_arguments_each_step_fired_with(mocker) -> None:
    # The number you edit when a tap lands wrong. It was always recorded;
    # rendering it is what makes the run log answer the rehearsal question
    # without opening MACRO.yml.
    _write("tap-demo", TAP_MACRO)
    _patch_mcp(mocker)
    ran = runner.invoke(app, ["macros", "run", "tap-demo"])
    hex6 = re.search(r"macro-run-([0-9a-f]{6})", ran.output).group(1)

    result = runner.invoke(app, ["macros", "runs", hex6])

    assert result.exit_code == 0
    assert "args: bbox: [0.12, 0.04, 0.34, 0.1]" in result.output


def test_runs_shows_arguments_after_substitution(mocker) -> None:
    # Post-substitution, so the log shows what actually fired rather than
    # the `{msg}` template that produced it.
    _write("demo")
    _patch_mcp(mocker)
    ran = runner.invoke(app, ["macros", "run", "demo", "-i", "msg=hello there"])
    hex6 = re.search(r"macro-run-([0-9a-f]{6})", ran.output).group(1)

    result = runner.invoke(app, ["macros", "runs", hex6])

    assert 'args: text: "hello there"' in result.output
    assert "{msg}" not in result.output


def test_runs_omits_the_args_line_for_a_step_with_no_arguments(mocker) -> None:
    # `home_screen` takes none — an empty `args: ` line would be noise on
    # every navigation step.
    _write("nav-demo", NAV_MACRO)
    _patch_mcp(mocker)
    ran = runner.invoke(app, ["macros", "run", "nav-demo"])
    hex6 = re.search(r"macro-run-([0-9a-f]{6})", ran.output).group(1)

    result = runner.invoke(app, ["macros", "runs", hex6])

    assert "✓ 1. home_screen" in result.output
    assert "args:" not in result.output


def test_runs_clips_an_overlong_argument_value(mocker) -> None:
    _write("long-demo")
    _patch_mcp(mocker)
    ran = runner.invoke(app, ["macros", "run", "long-demo", "-i", f"msg={'x' * 400}"])
    hex6 = re.search(r"macro-run-([0-9a-f]{6})", ran.output).group(1)

    result = runner.invoke(app, ["macros", "runs", hex6])

    args_line = next(ln for ln in result.output.splitlines() if "args:" in ln)

    assert "…" in args_line
    assert len(args_line) < 400


def test_runs_hangs_a_multiline_screen_under_its_own_label() -> None:
    # An element listing carries embedded newlines. Left as-is those rows
    # restart at column 0 and the block stops reading as one field of the
    # step above it.
    rl = macro_runlog.RunLogger("demo", "cli")
    rl.step(1, "tap", "tap-1", "guard_failed", screen_text="row one\nrow two")

    result = runner.invoke(app, ["macros", "runs", rl.run_id[-6:]])

    assert result.exit_code == 0
    assert "      screen: row one" in result.output
    assert "              row two" in result.output


def test_runs_unknown_id_exits_one(mocker) -> None:
    _write("demo")
    _patch_mcp(mocker)
    runner.invoke(app, ["macros", "run", "demo", "-i", "msg=hi"])

    result = runner.invoke(app, ["macros", "runs", "ffffff"])

    assert result.exit_code == 1
    assert "no run matching" in result.output


# ---------- stats ----------


def test_stats_empty_reports_no_runs() -> None:
    result = runner.invoke(app, ["macros", "stats"])

    assert result.exit_code == 0
    assert "No macro runs recorded" in result.output


def test_stats_shows_counters_and_last_abort() -> None:
    macro_stats.record("demo", ok=True, known_names={"demo"})
    macro_stats.record(
        "demo",
        ok=False,
        known_names={"demo"},
        step=2,
        reason="guard_failed",
        detail="'WeChat' not found",
    )

    result = runner.invoke(app, ["macros", "stats"])

    assert "demo: 1/2 ok, 1 abort(s), streak 1" in result.output
    assert "step 2 [guard_failed] 'WeChat' not found" in result.output


def test_run_start_at_is_forwarded_to_the_runner(mocker) -> None:
    # The agent can resume a macro partway in, so the rehearsal command must
    # be able to exercise that same path — otherwise the one thing you are
    # told to test before enabling is untestable.
    _write("demo")
    seen: dict = {}

    async def fake(spec, values, mcp, caller="engine", start_at=""):
        seen["start_at"] = start_at
        return macro_runner.MacroRunResult(blocks=[], ok=True)

    _patch_mcp(mocker, FakeMcp)
    mocker.patch.object(macro_runner, "run_and_record", side_effect=fake)

    result = runner.invoke(
        app, ["macros", "run", "demo", "-i", "msg=hi", "--start-at", "paste"]
    )

    assert result.exit_code == 0
    assert seen["start_at"] == "paste"
