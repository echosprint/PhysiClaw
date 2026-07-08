"""Tests for `physiclaw.cli.server`.

The full `server` command runs an MCP HTTP server in process — heavy
integration. We exercise `_spawn_runtime` directly and the env-var
wiring + warm-start + no-runtime branches with the heavy boundaries
(mcp.run, runtime subprocess, warm_start thread) mocked out.
"""
from __future__ import annotations

import importlib
import logging
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

server_mod = importlib.import_module("physiclaw.cli.server")

runner = CliRunner()


# ---------- _spawn_runtime ----------


def test_spawn_runtime_builds_cmd_with_port_and_label(mocker) -> None:
    fake_proc = MagicMock(pid=4242)
    spy = mocker.patch.object(server_mod.subprocess, "Popen", return_value=fake_proc)

    proc = server_mod._spawn_runtime(8048, verbose=False, label="engine=claude-code")

    assert proc is fake_proc
    args = spy.call_args.args[0]
    # First arg is sys.executable; remaining must include module + server flag.
    assert args[0] == server_mod.sys.executable
    assert "physiclaw.agent.runtime" in args
    assert "--server" in args
    assert "http://127.0.0.1:8048" in args
    assert "--verbose" not in args


def test_spawn_runtime_passes_verbose_flag(mocker) -> None:
    spy = mocker.patch.object(
        server_mod.subprocess, "Popen", return_value=MagicMock(pid=1),
    )

    server_mod._spawn_runtime(9000, verbose=True, label="engine=x")

    assert "--verbose" in spy.call_args.args[0]


def test_spawn_runtime_logs_label(
    mocker, caplog: pytest.LogCaptureFixture,
) -> None:
    mocker.patch.object(
        server_mod.subprocess, "Popen", return_value=MagicMock(pid=99),
    )

    with caplog.at_level(logging.INFO, logger="physiclaw.cli.server"):
        server_mod._spawn_runtime(8048, False, "engine=qwen")

    assert any("engine=qwen" in r.getMessage() for r in caplog.records)


# ---------- server CLI invocation ----------


def _patch_server_runtime_deps(mocker, *, primary: str = "http://device.local:8048",
                                fallback: str = "http://127.0.0.1:8048",
                                resolve_raises: bool = False):
    """Stub out everything heavy `server()` touches once it starts up."""
    mocker.patch.object(server_mod, "_spawn_runtime", return_value=MagicMock())
    mocker.patch("physiclaw.core.logger.setup_logging")
    # The startup auto-update / skills-sync hooks have their own tests and would
    # otherwise shell out / hit the network here — stub them to no-ops.
    mocker.patch("physiclaw.cli.update.notify_staged_update")
    mocker.patch("physiclaw.cli.update.start_stage_update_thread")
    mocker.patch("physiclaw.cli.sync_official_skills.maybe_auto_sync")
    # Never spawn real threads (warm-start / setup-wizard) or open a real
    # browser during tests. Tests that need a thread body re-patch Thread to
    # capture the target and invoke it manually.
    mocker.patch.object(server_mod.threading, "Thread")
    mocker.patch("webbrowser.open")

    fake_mcp = MagicMock()
    fake_shutdown = MagicMock()
    fake_warm = MagicMock()
    fake_warm.wait_for_port.return_value = True
    fake_warm.try_resume.return_value = True
    fake_state = MagicMock()
    fake_launcher = MagicMock()
    if resolve_raises:
        fake_launcher.resolve.side_effect = RuntimeError("no model set")
    else:
        fake_launcher.resolve.return_value = ("openai/gpt-5", "config")
    fake_launcher.engine_label.return_value = "engine=openai"
    fake_bridge = MagicMock()
    fake_bridge.bridge_base_urls.return_value = (primary, fallback)

    # Patch directly on parent packages so attribute lookups resolve to fakes.
    import physiclaw
    mocker.patch.object(physiclaw, "runtime_state", fake_state, create=True)
    fake_core_server = MagicMock(
        mcp=fake_mcp, shutdown=fake_shutdown, warm_start=fake_warm,
    )
    import physiclaw.core
    mocker.patch.object(physiclaw.core, "server", fake_core_server, create=True)
    mocker.patch.dict("sys.modules", {
        "physiclaw.core.server": fake_core_server,
        "physiclaw.core.server.warm_start": fake_warm,
        "physiclaw.agent.runtime.launcher": fake_launcher,
        "physiclaw.core.bridge": fake_bridge,
    })

    return {
        "mcp": fake_mcp, "shutdown": fake_shutdown,
        "state": fake_state, "launcher": fake_launcher,
        "bridge": fake_bridge, "warm_start": fake_warm,
    }


def test_server_opens_setup_wizard_by_default(mocker) -> None:
    deps = _patch_server_runtime_deps(mocker)
    open_spy = mocker.patch("webbrowser.open")
    captured = {}

    def fake_thread(target, daemon=False):
        captured["target"] = target
        return MagicMock()

    mocker.patch.object(server_mod.threading, "Thread", side_effect=fake_thread)

    server_mod.server(
        port=8048, host="127.0.0.1", verbose=False,
        no_runtime=True, warm_start=False, cam_index=None,
        save_tool_calls=False, save_snapshots=False, save_screenshots=False,
    )

    # The wizard opens only after the port is accepting connections.
    captured["target"]()
    deps["warm_start"].wait_for_port.assert_called_once()
    open_spy.assert_called_once_with("http://localhost:8048/setup-hardware")


def test_server_no_setup_hardware_skips_wizard(
    mocker, caplog: pytest.LogCaptureFixture,
) -> None:
    _patch_server_runtime_deps(mocker)
    open_spy = mocker.patch("webbrowser.open")
    thread_spy = mocker.patch.object(server_mod.threading, "Thread")

    with caplog.at_level(logging.INFO, logger="physiclaw.cli.server"):
        server_mod.server(
            port=8048, host="127.0.0.1", verbose=False,
            no_runtime=True, warm_start=False, cam_index=None, no_setup_hardware=True,
            save_tool_calls=False, save_snapshots=False, save_screenshots=False,
        )

    thread_spy.assert_not_called()  # no warm-start, no wizard → no threads
    open_spy.assert_not_called()
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "physiclaw setup hardware" in msgs


def test_server_prints_version_first(mocker, capsys) -> None:
    _patch_server_runtime_deps(mocker)

    server_mod.server(
        port=8048, host="127.0.0.1", verbose=False,
        no_runtime=True, warm_start=False, cam_index=None,
        save_tool_calls=False, save_snapshots=False, save_screenshots=False,
    )

    from physiclaw import __version__
    first_line = capsys.readouterr().out.splitlines()[0]
    assert first_line == f"PhysiClaw {__version__}"


def test_server_default_invocation_runs_mcp(mocker) -> None:
    deps = _patch_server_runtime_deps(mocker)

    server_mod.server(
        port=8048, host="127.0.0.1", verbose=False,
        no_runtime=True, warm_start=False, cam_index=None,
        save_tool_calls=False, save_snapshots=False, save_screenshots=False,
    )

    deps["mcp"].run.assert_called_once_with(transport="streamable-http")
    # State recorded with resolved model.
    deps["state"].write.assert_called_once()


def test_server_save_flags_set_env_vars(
    mocker, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_server_runtime_deps(mocker)
    monkeypatch.delenv("PHYSICLAW_SAVE_TOOL_CALLS", raising=False)
    monkeypatch.delenv("PHYSICLAW_SAVE_SNAPSHOTS", raising=False)
    monkeypatch.delenv("PHYSICLAW_SAVE_SCREENSHOTS", raising=False)

    server_mod.server(
        port=8048, host="127.0.0.1", verbose=False,
        no_runtime=True, warm_start=False, cam_index=None,
        save_tool_calls=True, save_snapshots=True, save_screenshots=True,
    )

    import os
    assert os.environ["PHYSICLAW_SAVE_TOOL_CALLS"] == "1"
    assert os.environ["PHYSICLAW_SAVE_SNAPSHOTS"] == "1"
    assert os.environ["PHYSICLAW_SAVE_SCREENSHOTS"] == "1"


def test_server_records_unset_when_resolve_fails(mocker) -> None:
    deps = _patch_server_runtime_deps(mocker, resolve_raises=True)

    server_mod.server(
        port=8048, host="127.0.0.1", verbose=False,
        no_runtime=True, warm_start=False, cam_index=None,
        save_tool_calls=False, save_snapshots=False, save_screenshots=False,
    )

    # Recorded (host, port, model_ref=None, model_source=None) since resolve failed.
    deps["state"].write.assert_called_once()
    kwargs = deps["state"].write.call_args.kwargs
    assert kwargs.get("model_ref") is None
    assert kwargs.get("model_source") is None


def test_server_skips_runtime_spawn_when_no_model_configured(
    mocker, caplog: pytest.LogCaptureFixture,
) -> None:
    """First-run UX: with no model set, the MCP server still starts but the
    runtime subprocess is not spawned (a stack trace is the bug we're avoiding).
    The user is told why + how to fix."""
    _patch_server_runtime_deps(mocker, resolve_raises=True)
    spawn_spy = mocker.patch.object(server_mod, "_spawn_runtime")

    with caplog.at_level(logging.WARNING, logger="physiclaw.cli.server"):
        server_mod.server(
            port=8048, host="127.0.0.1", verbose=False,
            no_runtime=False, warm_start=False, cam_index=None,
            save_tool_calls=False, save_snapshots=False, save_screenshots=False,
        )

    spawn_spy.assert_not_called()
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "no model configured" in msgs
    assert "physiclaw models" in msgs


def test_server_keyboard_interrupt_is_swallowed(mocker) -> None:
    deps = _patch_server_runtime_deps(mocker)
    deps["mcp"].run.side_effect = KeyboardInterrupt

    # Should NOT raise.
    server_mod.server(
        port=8048, host="127.0.0.1", verbose=False,
        no_runtime=True, warm_start=False, cam_index=None,
        save_tool_calls=False, save_snapshots=False, save_screenshots=False,
    )


def test_server_spawns_runtime_subprocess_by_default(mocker) -> None:
    _patch_server_runtime_deps(mocker)
    spawn_spy = mocker.patch.object(
        server_mod, "_spawn_runtime", return_value=MagicMock(),
    )

    server_mod.server(
        port=8048, host="127.0.0.1", verbose=True,
        no_runtime=False, warm_start=False, cam_index=None,
        save_tool_calls=False, save_snapshots=False, save_screenshots=False,
    )

    spawn_spy.assert_called_once_with(8048, True, "engine=openai")


def test_server_no_runtime_skips_subprocess(mocker) -> None:
    _patch_server_runtime_deps(mocker)
    spawn_spy = mocker.patch.object(server_mod, "_spawn_runtime")

    server_mod.server(
        port=8048, host="127.0.0.1", verbose=False,
        no_runtime=True, warm_start=False, cam_index=None,
        save_tool_calls=False, save_snapshots=False, save_screenshots=False,
    )

    spawn_spy.assert_not_called()


def test_server_logs_single_phone_url_when_no_mdns(
    mocker, caplog: pytest.LogCaptureFixture,
) -> None:
    _patch_server_runtime_deps(
        mocker,
        primary="http://127.0.0.1:8048",
        fallback="http://127.0.0.1:8048",
    )

    with caplog.at_level(logging.INFO, logger="physiclaw.cli.server"):
        server_mod.server(
            port=8048, host="127.0.0.1", verbose=False,
            no_runtime=True, warm_start=False, cam_index=None,
            save_tool_calls=False, save_snapshots=False, save_screenshots=False,
        )

    assert any(
        "stable LocalHostName" in r.getMessage() for r in caplog.records
    )


def test_server_runtime_stop_terminates_subprocess(mocker) -> None:
    _patch_server_runtime_deps(mocker)
    fake_proc = MagicMock()
    fake_proc.poll.return_value = None  # still running
    mocker.patch.object(server_mod, "_spawn_runtime", return_value=fake_proc)
    register_spy = mocker.patch.object(server_mod.atexit, "register")

    server_mod.server(
        port=8048, host="127.0.0.1", verbose=False,
        no_runtime=False, warm_start=False, cam_index=None,
        save_tool_calls=False, save_snapshots=False, save_screenshots=False,
    )

    # Find the registered _stop_runtime closure and run it.
    stop_calls = [
        c for c in register_spy.call_args_list
        if callable(c.args[0]) and getattr(c.args[0], "__name__", "") == "_stop_runtime"
    ]
    assert stop_calls
    stop_calls[0].args[0]()

    fake_proc.terminate.assert_called_once()


def test_server_runtime_stop_kills_after_timeout(mocker) -> None:
    _patch_server_runtime_deps(mocker)
    fake_proc = MagicMock()
    fake_proc.poll.return_value = None
    fake_proc.wait.side_effect = server_mod.subprocess.TimeoutExpired("cmd", 5)
    mocker.patch.object(server_mod, "_spawn_runtime", return_value=fake_proc)
    register_spy = mocker.patch.object(server_mod.atexit, "register")

    server_mod.server(
        port=8048, host="127.0.0.1", verbose=False,
        no_runtime=False, warm_start=False, cam_index=None,
        save_tool_calls=False, save_snapshots=False, save_screenshots=False,
    )

    stop = next(
        c.args[0] for c in register_spy.call_args_list
        if callable(c.args[0]) and getattr(c.args[0], "__name__", "") == "_stop_runtime"
    )
    stop()

    fake_proc.terminate.assert_called_once()
    fake_proc.kill.assert_called_once()


def test_warm_start_thread_exits_on_port_timeout(mocker) -> None:
    deps = _patch_server_runtime_deps(mocker)
    deps["warm_start"].wait_for_port.return_value = False
    captured = {}

    def fake_thread(target, daemon=False):
        captured["target"] = target
        return MagicMock()

    mocker.patch.object(server_mod.threading, "Thread", side_effect=fake_thread)
    interrupt_spy = mocker.patch.object(server_mod._thread, "interrupt_main")

    server_mod.server(
        port=8048, host="127.0.0.1", verbose=False,
        no_runtime=True, warm_start=True, cam_index=None,
        save_tool_calls=False, save_snapshots=False, save_screenshots=False,
    )

    # Run the thread body manually.
    captured["target"]()

    interrupt_spy.assert_called_once_with()


def test_warm_start_thread_exits_on_resume_failure(mocker) -> None:
    deps = _patch_server_runtime_deps(mocker)
    deps["warm_start"].try_resume.return_value = False
    captured = {}

    def fake_thread(target, daemon=False):
        captured["target"] = target
        return MagicMock()

    mocker.patch.object(server_mod.threading, "Thread", side_effect=fake_thread)
    interrupt_spy = mocker.patch.object(server_mod._thread, "interrupt_main")

    server_mod.server(
        port=8048, host="127.0.0.1", verbose=False,
        no_runtime=True, warm_start=True, cam_index=None,
        save_tool_calls=False, save_snapshots=False, save_screenshots=False,
    )

    captured["target"]()

    interrupt_spy.assert_called_once_with()


def test_warm_start_thread_exits_silently_when_resume_succeeds(mocker) -> None:
    deps = _patch_server_runtime_deps(mocker)
    deps["warm_start"].try_resume.return_value = True
    captured = {}

    def fake_thread(target, daemon=False):
        captured["target"] = target
        return MagicMock()

    mocker.patch.object(server_mod.threading, "Thread", side_effect=fake_thread)
    interrupt_spy = mocker.patch.object(server_mod._thread, "interrupt_main")

    server_mod.server(
        port=8048, host="127.0.0.1", verbose=False,
        no_runtime=True, warm_start=True, cam_index=None,
        save_tool_calls=False, save_snapshots=False, save_screenshots=False,
    )

    captured["target"]()

    interrupt_spy.assert_not_called()


def test_server_warm_start_starts_thread(mocker) -> None:
    _patch_server_runtime_deps(mocker)
    thread_spy = mocker.patch.object(server_mod.threading, "Thread")

    server_mod.server(
        port=8048, host="127.0.0.1", verbose=False,
        no_runtime=True, warm_start=True, cam_index=None,
        save_tool_calls=False, save_snapshots=False, save_screenshots=False,
    )

    thread_spy.assert_called_once()
    assert thread_spy.call_args.kwargs.get("daemon") is True
    thread_spy.return_value.start.assert_called_once()


# ---------- auto_calibrate branch (physiclaw auto) ----------


def test_server_auto_calibrate_starts_thread_and_skips_wizard(mocker) -> None:
    _patch_server_runtime_deps(mocker)
    open_spy = mocker.patch("webbrowser.open")
    # The branch spawns a thread that lazily imports the setup module and
    # calls its worker — patch that worker where it lives.
    worker_spy = mocker.patch(
        "physiclaw.cli.setup.hardware.await_bridge_and_calibrate"
    )
    captured = {}

    def fake_thread(target, daemon=False):
        captured["target"] = target
        return MagicMock()

    mocker.patch.object(server_mod.threading, "Thread", side_effect=fake_thread)

    server_mod.server(
        port=8048, host="127.0.0.1", verbose=False,
        no_runtime=True, warm_start=False, cam_index=None, auto_calibrate=True,
        save_tool_calls=False, save_snapshots=False, save_screenshots=False,
    )

    open_spy.assert_not_called()  # no desktop wizard in auto mode
    captured["target"]()          # run the daemon-thread body
    worker_spy.assert_called_once_with("127.0.0.1", 8048)
