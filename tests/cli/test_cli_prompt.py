"""Tests for `physiclaw.cli.prompt` — the SYSTEM prompt / request dump.

The headline test (`test_cli_request_matches_engine_turn0`) runs the REAL
engine for one turn with only MCP + provider mocked, captures the exact
message array the engine hands the provider, and asserts the CLI's
`prompt system` / `prompt request` output reproduces it byte-for-byte — so the
dump can never drift from what the running agent actually sends.
"""

from __future__ import annotations

import datetime as _dt
from unittest.mock import MagicMock

from typer.testing import CliRunner

from physiclaw.agent.engine import assemble as assemble_mod
from physiclaw.agent.engine import engine as engine_mod
from physiclaw.agent.engine.session import Session
from physiclaw.agent.engine.dto import AssistantMessage, FinishReason, ToolCall, Usage
from physiclaw.agent.runtime.hook import Trigger
from physiclaw.agent.runtime.sentinel import DONE
from physiclaw.cli.prompt import prompt_app

runner = CliRunner()


# ---------- smoke ----------


def test_prompt_system_prints_the_system_prompt() -> None:
    result = runner.invoke(prompt_app, ["system"])

    assert result.exit_code == 0
    assert "# Doctrine" in result.stdout
    assert "## Built-in Skills" in result.stdout


def test_prompt_request_prints_system_then_user_messages() -> None:
    result = runner.invoke(prompt_app, ["request"])

    assert result.exit_code == 0
    assert "===== [system] =====" in result.stdout
    assert "===== [user] =====" in result.stdout
    # a fresh HOME → layout unlearned → the first-run reminder rides the tail
    assert "First-run setup needed" in result.stdout


def test_prompt_system_save_as_writes_file(tmp_path) -> None:
    dest = tmp_path / "sys.md"
    result = runner.invoke(prompt_app, ["system", "--save-as", str(dest)])

    assert result.exit_code == 0
    assert "Wrote " in result.stdout and str(dest) in result.stdout
    assert "# Doctrine" not in result.stdout  # content went to the file, not stdout
    assert dest.read_text(encoding="utf-8").startswith("# Doctrine")


def test_prompt_request_save_as_writes_file(tmp_path) -> None:
    dest = tmp_path / "req.txt"
    result = runner.invoke(prompt_app, ["request", "--save-as", str(dest)])

    assert result.exit_code == 0
    assert str(dest) in result.stdout
    body = dest.read_text(encoding="utf-8")
    assert "===== [system] =====" in body and "===== [user] =====" in body


# ---------- equivalence with the real engine turn-0 ----------


class _FrozenDatetime:
    """Fixed clock so the trigger's `Now:` stamp is identical in the engine
    run and the CLI build (both go through `assemble.dt.datetime.now`)."""

    @classmethod
    def now(cls, tz=None):  # noqa: ARG003
        return _dt.datetime(2026, 7, 2, 10, 0, 0)


class _FakeDtModule:
    datetime = _FrozenDatetime


class _CaptureProvider:
    """Records the turn-0 request, then ends the session on the first turn."""

    PROVIDER_ID = "fake"
    # Collapse knobs read by _loop each turn (provider contract).
    COLLAPSE_FIRST_AT_TURN = 100
    COLLAPSE_INTERVAL_TURNS = 100
    KEEP_RECENT_TURNS = 10

    def __init__(self) -> None:
        self.model = "fake-model"
        self.captured: list | None = None

    def serialize_history(self, messages):
        return [{"role": "fake"} for _ in messages]

    async def chat(self, messages, tools):
        if self.captured is None:
            self.captured = list(messages)
        return AssistantMessage(
            content="",
            tool_calls=[
                ToolCall(id="t1", name="note", arguments={"summary": "done"}),
                ToolCall(
                    id="t2",
                    name="end_session",
                    arguments={"status": DONE, "recap": "fin"},
                ),
            ],
            finish_reason=FinishReason.TOOL_CALLS,
            usage=Usage(),
        )

    async def aclose(self) -> None:
        pass


def _aret(value):
    async def _coro(*_a, **_k):
        return value

    return _coro


def _role(msg) -> str:
    return type(msg).__name__.removesuffix("Message").lower()


def _content(msg) -> str:
    content = msg.content
    if isinstance(content, str):
        return content
    return "\n".join(
        getattr(b, "text", None) or f"[{type(b).__name__}]" for b in content
    )


async def test_cli_request_matches_engine_turn0(monkeypatch) -> None:
    capture = _CaptureProvider()
    # Real assembly (skills, prompt render, placeholders, tails); only the
    # outside world is stubbed.
    monkeypatch.setattr(assemble_mod, "dt", _FakeDtModule)
    monkeypatch.setattr(engine_mod, "get_mcp", _aret(MagicMock()))
    monkeypatch.setattr(engine_mod, "list_tools_cached", _aret([]))
    monkeypatch.setattr(engine_mod, "make_provider", lambda *_a, **_k: capture)
    monkeypatch.setattr(engine_mod, "Trace", lambda *_a, **_k: MagicMock())
    monkeypatch.setattr(engine_mod, "RawLog", lambda *_a, **_k: MagicMock())
    # CLI resolves the provider id from config; align it with the engine's.
    monkeypatch.setattr("physiclaw.common.config.model_ref", lambda: "fake/fake-model")

    trigger = Trigger(description="phone screen changed", source="phone")
    await engine_mod._run_session(
        [trigger],
        model_ref="fake/fake-model",
        session=Session(),
    )

    engine_req = capture.captured
    assert engine_req is not None, "engine never called provider.chat"
    assert _role(engine_req[0]) == "system"

    # `prompt system` == the engine's turn-0 SYSTEM message, exactly.
    sys_out = runner.invoke(prompt_app, ["system"])
    assert sys_out.exit_code == 0
    assert sys_out.stdout.rstrip("\n") == _content(engine_req[0])

    # `prompt request` reproduces the whole turn-0 message array, in order.
    req_out = runner.invoke(prompt_app, ["request"])
    assert req_out.exit_code == 0
    expected = "".join(
        f"\n===== [{_role(m)}] =====\n{_content(m)}\n" for m in engine_req
    )
    assert req_out.stdout == expected
