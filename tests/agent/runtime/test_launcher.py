"""Tests for `physiclaw.agent.runtime.launcher`."""
from __future__ import annotations

import pytest

from physiclaw.agent.runtime import launcher
from physiclaw.agent.runtime.launcher import (
    _claude_available,
    engine_label,
    resolve,
)


# ---------- _claude_available ----------


def test_claude_available_true_when_package_present(mocker) -> None:
    fake = mocker.MagicMock()
    mocker.patch.object(launcher, "_claude_available", wraps=_claude_available)
    mocker.patch("importlib.util.find_spec", return_value=fake)

    assert _claude_available() is True


def test_claude_available_false_when_package_missing(mocker) -> None:
    mocker.patch("importlib.util.find_spec", return_value=None)

    assert _claude_available() is False


# ---------- engine_label ----------


def test_engine_label_for_claude_code() -> None:
    assert engine_label("claude-code/claude-sonnet-4-6") == (
        "engine=claude-code, model=claude-sonnet-4-6"
    )


def test_engine_label_for_in_process_provider() -> None:
    assert engine_label("qwen/qwen3-plus") == (
        "engine=physiclaw, provider=qwen, model=qwen3-plus"
    )


# ---------- resolve ----------


def test_resolve_returns_ref_and_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHYSICLAW_MODEL", "qwen/qwen3-plus")

    ref, source = resolve()

    assert ref == "qwen/qwen3-plus"
    assert source == "PHYSICLAW_MODEL env"


def test_resolve_raises_for_unknown_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHYSICLAW_MODEL", "mystery/x")

    with pytest.raises(RuntimeError, match=r"^unknown provider 'mystery'"):
        resolve()


def test_resolve_raises_for_claude_code_when_unavailable(
    monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    monkeypatch.setenv("PHYSICLAW_MODEL", "claude-code/claude-test")
    mocker.patch.object(launcher, "_claude_available", return_value=False)

    with pytest.raises(
        RuntimeError, match=r"selects claude-code but agent/claude/ is not installed"
    ):
        resolve()


def test_resolve_succeeds_for_claude_code_when_available(
    monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    monkeypatch.setenv("PHYSICLAW_MODEL", "claude-code/claude-test")
    mocker.patch.object(launcher, "_claude_available", return_value=True)

    ref, source = resolve()

    assert ref == "claude-code/claude-test"
    assert source == "PHYSICLAW_MODEL env"


# ---------- launch() ----------


@pytest.mark.integration
def test_launch_runs_engine_path_for_in_process_provider(
    mocker, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHYSICLAW_MODEL", "qwen/qwen3-plus")
    monkeypatch.setattr("sys.argv", ["runtime", "--server", "http://x:9000"])
    mocker.patch.object(launcher, "setup_logging")

    fake_runtime = mocker.MagicMock()

    async def _start():
        return None
    fake_runtime.start.side_effect = _start

    runtime_cls = mocker.patch.object(
        launcher, "Runtime", return_value=fake_runtime,
    )

    async def _close():
        return None
    mocker.patch.object(launcher, "close_mcp", side_effect=_close)

    fake_engine_run = mocker.MagicMock()
    mocker.patch.dict(
        "sys.modules",
        {"physiclaw.agent.engine.engine": mocker.MagicMock(run=fake_engine_run)},
    )

    launcher.launch()

    runtime_cls.assert_called_once()
    kwargs = runtime_cls.call_args.kwargs
    assert kwargs["interval"] == 1.0
    assert kwargs["label"].startswith("engine=physiclaw")


@pytest.mark.integration
def test_launch_runs_claude_path_for_claude_code(
    mocker, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHYSICLAW_MODEL", "claude-code/claude-test")
    monkeypatch.setattr("sys.argv", ["runtime"])
    mocker.patch.object(launcher, "setup_logging")
    mocker.patch.object(launcher, "_claude_available", return_value=True)

    fake_runtime = mocker.MagicMock()

    async def _start():
        return None
    fake_runtime.start.side_effect = _start
    runtime_cls = mocker.patch.object(
        launcher, "Runtime", return_value=fake_runtime,
    )

    async def _close():
        return None
    mocker.patch.object(launcher, "close_mcp", side_effect=_close)

    fake_spawn = mocker.MagicMock()
    mocker.patch.dict(
        "sys.modules",
        {"physiclaw.agent.claude": mocker.MagicMock(spawn_claude=fake_spawn)},
    )

    launcher.launch()

    runtime_cls.assert_called_once()
    assert runtime_cls.call_args.kwargs["label"].startswith("engine=claude-code")


@pytest.mark.integration
def test_launch_normalizes_proxy_env_first(
    mocker, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A direct `python -m physiclaw.agent.runtime` skips the CLI root
    callback, so launch() must normalize `socks://` proxy env itself —
    before anything builds an httpx client (regression: on Linux with a
    VPN on, the poll hook's client raised `Unknown scheme for proxy URL`
    every tick and the agent never woke). Guarded via the earliest exit
    path (no model): env must already be rewritten by then."""
    import os

    monkeypatch.setenv("all_proxy", "socks://127.0.0.1:7897")
    monkeypatch.delenv("PHYSICLAW_MODEL", raising=False)
    from physiclaw import config as _cfg
    monkeypatch.setattr(_cfg.CONFIG.agent, "model", "")
    monkeypatch.setattr("sys.argv", ["runtime"])

    with pytest.raises(SystemExit):
        launcher.launch()

    assert os.environ["all_proxy"] == "socks5://127.0.0.1:7897"


def _stub_asyncio_run(mocker, *, raise_kbd_interrupt: bool = False):
    """Stub `asyncio.run` so it accepts the inner `_main()` coroutine
    without awaiting it. Without `.close()`, Python flags the unawaited
    coro with a RuntimeWarning (often attributed to whichever test
    happens to be running when GC sweeps it)."""

    def _run(coro):
        coro.close()  # release the coroutine so it isn't garbage-collected unawaited
        if raise_kbd_interrupt:
            raise KeyboardInterrupt

    return mocker.patch.object(launcher.asyncio, "run", side_effect=_run)


@pytest.mark.integration
def test_launch_exits_cleanly_when_no_model_configured(
    mocker, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    """First-run UX: runtime subprocess shouldn't dump a Python stack
    trace into the parent shell when no model is set — print a friendly
    message to stderr and exit 1."""
    monkeypatch.delenv("PHYSICLAW_MODEL", raising=False)
    # Ensure the in-config default is also empty.
    from physiclaw import config as _cfg
    monkeypatch.setattr(_cfg.CONFIG.agent, "model", "")
    monkeypatch.setattr("sys.argv", ["runtime"])

    with pytest.raises(SystemExit) as exc_info:
        launcher.launch()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "no model configured" in captured.err
    assert "physiclaw models" in captured.err


@pytest.mark.integration
def test_launch_swallows_keyboard_interrupt(
    mocker, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHYSICLAW_MODEL", "qwen/qwen3-plus")
    monkeypatch.setattr("sys.argv", ["runtime"])
    mocker.patch.object(launcher, "setup_logging")
    _stub_asyncio_run(mocker, raise_kbd_interrupt=True)
    mocker.patch.dict(
        "sys.modules",
        {"physiclaw.agent.engine.engine": mocker.MagicMock(run=mocker.MagicMock())},
    )

    # Must not raise.
    launcher.launch()


@pytest.mark.integration
def test_launch_seeds_physiclaw_server_env(
    mocker, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHYSICLAW_MODEL", "qwen/qwen3-plus")
    monkeypatch.delenv("PHYSICLAW_SERVER", raising=False)
    monkeypatch.setattr("sys.argv", ["runtime", "--server", "http://h:42"])
    mocker.patch.object(launcher, "setup_logging")
    _stub_asyncio_run(mocker)
    mocker.patch.dict(
        "sys.modules",
        {"physiclaw.agent.engine.engine": mocker.MagicMock(run=mocker.MagicMock())},
    )

    launcher.launch()

    import os
    assert os.environ.get("PHYSICLAW_SERVER") == "http://h:42"


@pytest.mark.integration
def test_launch_verbose_flag_sets_debug_level(
    mocker, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import logging as _logging
    monkeypatch.setenv("PHYSICLAW_MODEL", "qwen/qwen3-plus")
    monkeypatch.setattr("sys.argv", ["runtime", "--verbose"])
    setup_spy = mocker.patch.object(launcher, "setup_logging")
    _stub_asyncio_run(mocker)
    mocker.patch.dict(
        "sys.modules",
        {"physiclaw.agent.engine.engine": mocker.MagicMock(run=mocker.MagicMock())},
    )

    launcher.launch()

    setup_spy.assert_called_once_with("runtime", _logging.DEBUG)
