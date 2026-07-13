"""Tests for `physiclaw.cli.setup.hardware` — interactive setup helpers.

The full `run()` flow is interactive and 10 steps deep; we cover the
testable helpers (api / ok / lan_ip / calibrate / calibrate_retry /
ask / _done / _fail / _camera_aim_adjust) and the early-exit
branches of `run` (server down, already ready, already calibrated)
plus the `hardware` typer entry point.
"""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock

import pytest
import typer
from typer.testing import CliRunner

hw_mod = importlib.import_module("physiclaw.cli.setup.hardware")

app = typer.Typer()
app.command()(hw_mod.hardware)
runner = CliRunner()


# ---------- api ----------


def test_api_binds_the_module_base(mocker) -> None:
    # The transport is unit-tested in tests/cli/test__http.py — this pins
    # the wiring: the wizard's api() calls _http.api with the mutable
    # module-global BASE.
    spy = mocker.patch.object(hw_mod._http, "api", return_value={"x": 1})
    mocker.patch.object(hw_mod, "BASE", "http://example:1234")

    out = hw_mod.api("GET", "/api/status", timeout=5)

    assert out == {"x": 1}
    spy.assert_called_once_with(
        "http://example:1234", "GET", "/api/status", body=None, timeout=5
    )


# ---------- ok ----------


def test_ok_returns_true_for_ok_dict() -> None:
    assert hw_mod.ok({"status": "ok"}) is True


def test_ok_returns_false_for_none() -> None:
    assert hw_mod.ok(None) is False


def test_ok_returns_false_for_other_status() -> None:
    assert hw_mod.ok({"status": "error"}) is False


# ---------- lan_ip ----------


def test_lan_ip_returns_resolved(mocker) -> None:
    fake_sock = MagicMock()
    fake_sock.getsockname.return_value = ("192.168.1.5", 0)
    mocker.patch.object(hw_mod.socket, "socket", return_value=fake_sock)

    assert hw_mod.lan_ip() == "192.168.1.5"


def test_lan_ip_falls_back_on_failure(mocker) -> None:
    mocker.patch.object(hw_mod.socket, "socket", side_effect=OSError("offline"))

    assert hw_mod.lan_ip() == "127.0.0.1"


# ---------- wait + ask ----------


def test_wait_calls_input(mocker) -> None:
    spy = mocker.patch("builtins.input", return_value="")

    hw_mod.wait("press to continue")

    spy.assert_called_once()


def test_ask_auto_returns_true_without_prompt() -> None:
    assert hw_mod.ask("anything", auto=True) is True


def test_ask_q_quits_the_wizard(mocker, capsys: pytest.CaptureFixture) -> None:
    """Every ask() gates a mandatory step — 'q' must abort the wizard,
    not skip the step and fall through to a false OK."""
    mocker.patch("builtins.input", return_value="q")

    with pytest.raises(SystemExit):
        hw_mod.ask("really?", auto=False)

    assert "aborted" in capsys.readouterr().out


def test_ask_other_returns_true(mocker) -> None:
    mocker.patch("builtins.input", return_value="")

    assert hw_mod.ask("really?", auto=False) is True


def test_base_port_parses_explicit_port(monkeypatch) -> None:
    monkeypatch.setattr(hw_mod, "BASE", "http://localhost:9000")

    assert hw_mod._base_port() == 9000


def test_base_port_defaults_when_port_omitted(monkeypatch) -> None:
    """PHYSICLAW_SERVER without a port must not crash the wizard."""
    monkeypatch.setattr(hw_mod, "BASE", "http://myhost")

    assert hw_mod._base_port() == 8048


# ---------- calibrate / calibrate_retry ----------


def test_calibrate_calls_api(mocker) -> None:
    spy = mocker.patch.object(hw_mod, "api", return_value={"status": "ok"})

    hw_mod.calibrate("arm", timeout=120, body={"fresh": True})

    spy.assert_called_once_with(
        "POST",
        "/api/calibrate/arm",
        body={"fresh": True},
        timeout=120,
    )


def test_calibrate_retry_returns_on_success(mocker) -> None:
    mocker.patch.object(hw_mod, "calibrate", return_value={"status": "ok"})

    out = hw_mod.calibrate_retry(
        "arm",
        "fail",
        "retry?",
        auto=True,
    )

    assert out == {"status": "ok"}


def test_calibrate_retry_auto_exits_on_failure(mocker) -> None:
    mocker.patch.object(
        hw_mod,
        "calibrate",
        return_value={"status": "error", "message": "bad"},
    )

    with pytest.raises(SystemExit):
        hw_mod.calibrate_retry(
            "arm", lambda r: f"x {r['message']}", "retry?", auto=True
        )


def test_calibrate_retry_manual_q_exits(mocker) -> None:
    mocker.patch.object(
        hw_mod,
        "calibrate",
        return_value={"status": "error"},
    )
    mocker.patch.object(hw_mod, "ask", return_value=False)

    with pytest.raises(SystemExit):
        hw_mod.calibrate_retry("arm", "fail", "retry?", auto=False)


def test_calibrate_retry_manual_retries_until_success(mocker) -> None:
    responses = iter(
        [
            {"status": "error"},
            {"status": "ok"},
        ]
    )
    mocker.patch.object(
        hw_mod,
        "calibrate",
        side_effect=lambda *a, **kw: next(responses),
    )
    mocker.patch.object(hw_mod, "ask", return_value=True)

    out = hw_mod.calibrate_retry("arm", "fail", "retry?", auto=False)

    assert out == {"status": "ok"}


def test_calibrate_retry_uses_custom_predicate(mocker) -> None:
    mocker.patch.object(
        hw_mod,
        "calibrate",
        return_value={"status": "ok", "passed": False},
    )

    with pytest.raises(SystemExit):
        hw_mod.calibrate_retry(
            "arm",
            "fail",
            "retry?",
            auto=True,
            predicate=lambda r: bool(r and r.get("passed")),
        )


# ---------- _done / _fail / _warn ----------


def test_done_prints_green(capsys: pytest.CaptureFixture) -> None:
    hw_mod._done("yay")
    out = capsys.readouterr().out
    assert "yay" in out
    assert "\033[32m" in out


def test_fail_prints_red(capsys: pytest.CaptureFixture) -> None:
    hw_mod._fail("bad")
    out = capsys.readouterr().out
    assert "bad" in out
    assert "\033[31m" in out


def test_warn_prints_yellow(capsys: pytest.CaptureFixture) -> None:
    hw_mod._warn("careful")
    out = capsys.readouterr().out
    assert "careful" in out
    assert "\033[33m" in out


# ---------- _camera_aim_adjust ----------


def test_camera_aim_adjust_opens_waits_quits(mocker) -> None:
    open_spy = mocker.patch.object(hw_mod.platform, "open_camera_aim_app")
    quit_spy = mocker.patch.object(hw_mod.platform, "quit_camera_aim_app")
    wait_spy = mocker.patch.object(hw_mod, "wait")
    api_spy = mocker.patch.object(hw_mod, "api", return_value={"status": "ok"})

    hw_mod._camera_aim_adjust("position")

    api_spy.assert_called_once_with("POST", "/api/disconnect-camera")
    open_spy.assert_called_once()
    wait_spy.assert_called_once_with("position")
    quit_spy.assert_called_once()
    # Disconnect must happen before the aim app opens.
    assert api_spy.call_args_list[0] == mocker.call("POST", "/api/disconnect-camera")


# ---------- _step_connect_camera fallback ----------


def _patch_camera_fallback_api(mocker, tmp_path, *, previews, connect_by_index):
    """Stub api() for the camera step: auto-pick always fails, previews and
    the by-index connect come from the arguments. Returns the api mock."""
    mocker.patch.object(hw_mod.tempfile, "gettempdir", return_value=str(tmp_path))

    def fake_api(method, path, body=None, timeout=60):
        if path == "/api/connect-camera":
            if body == {"index": "auto"}:
                return {"status": "error"}
            return connect_by_index
        if path.startswith("/api/camera-preview/"):
            return previews
        return {"status": "ok"}

    return mocker.patch.object(hw_mod, "api", side_effect=fake_api)


def test_step_connect_camera_manual_prompt_picks_index(mocker, tmp_path) -> None:
    """Auto-pick fails interactively: previews are written and opened, and
    the typed index is connected."""
    _patch_camera_fallback_api(
        mocker,
        tmp_path,
        previews={"image": "Zg=="},  # base64 for "f"
        connect_by_index={"status": "ok", "index": 2},
    )
    open_spy = mocker.patch.object(hw_mod.platform, "open_image_files")
    mocker.patch("builtins.input", return_value="2")

    cam = hw_mod._step_connect_camera(auto=False)

    assert cam == 2
    written = sorted(p.name for p in tmp_path.glob("physiclaw_cam*.jpg"))
    assert written == [f"physiclaw_cam{i}.jpg" for i in range(4)]
    open_spy.assert_called_once_with([str(tmp_path / f) for f in written])


def test_step_connect_camera_manual_invalid_input_defaults_to_zero(
    mocker, tmp_path
) -> None:
    """A non-numeric answer at the manual prompt falls back to camera 0,
    and blank preview frames are not written or opened."""
    _patch_camera_fallback_api(
        mocker,
        tmp_path,
        previews=None,  # preview endpoint has nothing to show
        connect_by_index={"status": "ok", "index": 0},
    )
    open_spy = mocker.patch.object(hw_mod.platform, "open_image_files")
    mocker.patch("builtins.input", return_value="not a number")

    cam = hw_mod._step_connect_camera(auto=False)

    assert cam == 0
    assert list(tmp_path.glob("physiclaw_cam*.jpg")) == []
    open_spy.assert_called_once_with([])


def test_step_connect_camera_fallback_connect_failure_exits(mocker, tmp_path) -> None:
    """When the by-index connect also fails, the wizard aborts — and stale
    previews from a previous run are cleaned up first."""
    stale = tmp_path / "physiclaw_cam9.jpg"
    stale.write_bytes(b"old")
    _patch_camera_fallback_api(
        mocker,
        tmp_path,
        previews={"image": ""},  # response without a usable frame
        connect_by_index={"status": "error"},
    )

    with pytest.raises(SystemExit):
        hw_mod._step_connect_camera(auto=True)

    assert not stale.exists()


# ---------- run() early-exit branches ----------


def test_run_exits_when_server_down(mocker) -> None:
    mocker.patch.object(hw_mod, "api", return_value=None)

    with pytest.raises(SystemExit) as exc:
        hw_mod.run()

    assert "Server not running" in str(exc.value)


def test_run_returns_when_already_ready(mocker, capsys: pytest.CaptureFixture) -> None:
    mocker.patch.object(hw_mod, "api", return_value={"ready": True, "calibrated": True})

    hw_mod.run()
    out = capsys.readouterr().out

    assert "already ready" in out


def test_run_finalizes_when_already_calibrated(
    mocker,
    capsys: pytest.CaptureFixture,
) -> None:
    mocker.patch.object(
        hw_mod,
        "api",
        side_effect=[
            {"ready": False, "calibrated": True},  # GET /api/status
            {"status": "ok"},  # POST /api/phone/home
            {"status": "ok"},  # POST /api/ready
        ],
    )
    mocker.patch.object(hw_mod.time, "sleep")

    hw_mod.run()
    out = capsys.readouterr().out

    assert "Already calibrated" in out
    assert "PhysiClaw is ready" in out


# ---------- hardware (typer entry) ----------


def test_hardware_sets_base_and_calls_run(mocker) -> None:
    run_spy = mocker.patch.object(hw_mod, "run")

    result = runner.invoke(app, ["--server-url", "http://example.com:9000"])

    assert result.exit_code == 0
    assert hw_mod.BASE == "http://example.com:9000"
    run_spy.assert_called_once_with(auto=False, trace=False)


def test_hardware_passes_auto_and_trace(mocker) -> None:
    run_spy = mocker.patch.object(hw_mod, "run")

    runner.invoke(app, ["--auto", "--trace"])

    run_spy.assert_called_once_with(auto=True, trace=True)


# ---------- run() full happy-path (auto mode) ----------


def test_run_full_auto_path(mocker, tmp_path) -> None:
    """Walk every step in --auto mode with api() stubbed to always succeed."""
    mocker.patch.object(hw_mod.time, "sleep")
    mocker.patch.object(hw_mod, "_camera_aim_adjust")
    mocker.patch.object(
        hw_mod,
        "_viewport_cache_candidates",
        return_value=[],  # no cache → fresh measurement.
    )

    # Endpoint-specific responses.
    def fake_api(method, path, body=None, timeout=60):
        if path == "/api/status":
            return {
                "ready": False,
                "calibrated": False,
                "bridge": False,
            }
        if path == "/api/connect-arm":
            return {"status": "ok"}
        if path == "/api/connect-camera":
            return {"status": "ok", "index": 2}
        if path.startswith("/api/camera-preview/"):
            return {"image": ""}
        if path == "/api/bridge/switch":
            return {"ok": True}
        if path == "/api/phone/home":
            return {"status": "ok"}
        if path == "/api/ready":
            return {"status": "ok"}
        return {"status": "ok"}

    def fake_calibrate(step, timeout=60, body=None):
        if step == "arm":
            return {
                "status": "ok",
                "pairs": 18,
                "tilt_ratio": 0.001,
                "aligned": True,
            }
        if step == "camera":
            return {
                "status": "ok",
                "rotation_name": "0°",
                "coverage": 0.95,
                "issues": [],
            }
        if step == "validate":
            return {"status": "ok", "calibrated": True}
        if step == "assistive-touch/verify":
            return {"status": "ok", "passed": True, "clipboard": {"fetched": False}}
        return {"status": "ok"}

    mocker.patch.object(hw_mod, "api", side_effect=fake_api)
    mocker.patch.object(hw_mod, "calibrate", side_effect=fake_calibrate)

    hw_mod.run(auto=True, trace=True)


def test_run_full_auto_with_warn_issues(mocker) -> None:
    """The camera-calibration step surfaces issues from calibrate via _warn."""
    mocker.patch.object(hw_mod.time, "sleep")
    mocker.patch.object(hw_mod, "_camera_aim_adjust")
    mocker.patch.object(hw_mod, "_viewport_cache_candidates", return_value=[])

    def fake_api(method, path, body=None, timeout=60):
        if path == "/api/status":
            return {"ready": False, "calibrated": False, "bridge": True}
        if path == "/api/connect-camera":
            return {"status": "ok", "index": 1}
        if path == "/api/bridge/switch":
            return {"ok": True}
        return {"status": "ok"}

    def fake_calibrate(step, timeout=60, body=None):
        if step == "arm":
            return {
                "status": "ok",
                "pairs": 18,
                "tilt_ratio": 0.5,
                "aligned": False,
            }
        if step == "camera":
            return {
                "status": "ok",
                "rotation_name": "0°",
                "coverage": 0.5,
                "issues": ["phone partially out of frame"],
            }
        if step == "validate":
            return {"status": "ok", "calibrated": True}
        if step == "assistive-touch/verify":
            return {
                "status": "ok",
                "passed": True,
                "clipboard": {"fetched": True, "text": "PhysiClaw OK"},
            }
        return {"status": "ok"}

    mocker.patch.object(hw_mod, "api", side_effect=fake_api)
    mocker.patch.object(hw_mod, "calibrate", side_effect=fake_calibrate)

    hw_mod.run(auto=True, trace=False)


def test_run_arm_connect_failure_exits(mocker) -> None:
    mocker.patch.object(hw_mod.time, "sleep")
    mocker.patch.object(hw_mod, "_camera_aim_adjust")
    mocker.patch.object(hw_mod, "_viewport_cache_candidates", return_value=[])

    def fake_api(method, path, body=None, timeout=60):
        if path == "/api/status":
            return {"ready": False, "calibrated": False, "bridge": True}
        if path == "/api/connect-arm":
            return {"status": "error", "message": "no port"}
        return {"status": "ok"}

    mocker.patch.object(hw_mod, "api", side_effect=fake_api)

    with pytest.raises(SystemExit):
        hw_mod.run(auto=True, trace=False)


def test_run_camera_auto_pick_falls_back_to_manual(mocker, tmp_path) -> None:
    mocker.patch.object(hw_mod.time, "sleep")
    mocker.patch.object(hw_mod.tempfile, "gettempdir", return_value=str(tmp_path))
    mocker.patch.object(hw_mod.platform, "open_image_files")
    mocker.patch.object(hw_mod, "_camera_aim_adjust")
    mocker.patch.object(hw_mod, "_viewport_cache_candidates", return_value=[])

    call_count = {"connect": 0}

    def fake_api(method, path, body=None, timeout=60):
        if path == "/api/status":
            return {"ready": False, "calibrated": False, "bridge": True}
        if path == "/api/connect-arm":
            return {"status": "ok"}
        if path == "/api/connect-camera":
            call_count["connect"] += 1
            # First auto-pick fails, second by-index succeeds, third in step 8 succeeds.
            if call_count["connect"] == 1:
                return {"status": "error"}
            return {"status": "ok", "index": 0}
        if path.startswith("/api/camera-preview/"):
            return {"image": "Zg=="}  # base64 for "f"
        if path == "/api/bridge/switch":
            return {"ok": True}
        return {"status": "ok"}

    def fake_calibrate(step, timeout=60, body=None):
        return {
            "status": "ok",
            "pairs": 18,
            "tilt_ratio": 0.001,
            "aligned": True,
            "rotation_name": "0°",
            "coverage": 0.95,
            "issues": [],
            "calibrated": True,
            "passed": True,
            "clipboard": {"fetched": False},
        }

    mocker.patch.object(hw_mod, "api", side_effect=fake_api)
    mocker.patch.object(hw_mod, "calibrate", side_effect=fake_calibrate)

    hw_mod.run(auto=True, trace=False)


def test_run_uses_cached_viewport_in_auto_mode(
    mocker,
    tmp_path,
    capsys: pytest.CaptureFixture,
) -> None:
    cache = tmp_path / "viewport.png"
    cache.write_bytes(b"x")
    mocker.patch.object(hw_mod, "_viewport_cache_candidates", return_value=[cache])
    mocker.patch.object(hw_mod.time, "sleep")
    mocker.patch.object(hw_mod, "_camera_aim_adjust")

    def fake_api(method, path, body=None, timeout=60):
        if path == "/api/status":
            return {"ready": False, "calibrated": False, "bridge": True}
        if path == "/api/bridge/switch":
            return {"ok": True}
        if path == "/api/connect-camera":
            return {"status": "ok", "index": 0}
        return {"status": "ok"}

    def fake_calibrate(step, timeout=60, body=None):
        return {
            "status": "ok",
            "pairs": 18,
            "tilt_ratio": 0.001,
            "aligned": True,
            "rotation_name": "0°",
            "coverage": 0.95,
            "issues": [],
            "calibrated": True,
            "passed": True,
            "clipboard": {"fetched": False},
        }

    mocker.patch.object(hw_mod, "api", side_effect=fake_api)
    mocker.patch.object(hw_mod, "calibrate", side_effect=fake_calibrate)

    hw_mod.run(auto=True, trace=False)
    out = capsys.readouterr().out

    assert "Using cached screenshot" in out


# ---------- await_bridge_and_calibrate ----------


def _patch_auto_worker(mocker, *, port_ok=True):
    """Patch await_bridge_and_calibrate's collaborators: the lazy
    core.server.net.wait_for_port import (faked so the test never loads the
    real core.server package), the HTTP api(), sleep, run(), and the tagged
    logger. `BASE` is patched so the worker's reassignment is restored after
    the test. Returns a dict of the mocks."""
    fake_net = MagicMock()
    fake_net.wait_for_port.return_value = port_ok
    mocker.patch.dict(
        "sys.modules",
        {"physiclaw.core.server.net": fake_net},
    )
    mocker.patch("physiclaw.common.logger.make_tagged_logger", return_value=MagicMock())
    mocker.patch.object(
        hw_mod, "BASE", hw_mod.BASE
    )  # restore after (worker mutates it)
    return {
        "api": mocker.patch.object(hw_mod, "api", return_value={"bridge": True}),
        "sleep": mocker.patch.object(hw_mod.time, "sleep"),
        "run": mocker.patch.object(hw_mod, "run"),
    }


def test_await_bridge_and_calibrate_runs_when_bridge_connects(mocker) -> None:
    m = _patch_auto_worker(mocker)

    hw_mod.await_bridge_and_calibrate("127.0.0.1", 8048)

    assert hw_mod.BASE == "http://localhost:8048"
    m["run"].assert_called_once_with(auto=True)


def test_await_bridge_and_calibrate_skips_on_port_timeout(mocker) -> None:
    m = _patch_auto_worker(mocker, port_ok=False)

    hw_mod.await_bridge_and_calibrate("127.0.0.1", 8048)

    m["run"].assert_not_called()


def test_await_bridge_and_calibrate_polls_until_bridge(mocker) -> None:
    m = _patch_auto_worker(mocker)
    # No response, then not-connected, then connected — waits through both.
    m["api"].side_effect = [None, {"bridge": False}, {"bridge": True}]

    hw_mod.await_bridge_and_calibrate("127.0.0.1", 8048)

    m["run"].assert_called_once_with(auto=True)
    assert m["sleep"].call_count == 2  # one wait per non-connected poll


def test_await_bridge_and_calibrate_starts_on_already_calibrated(mocker) -> None:
    # A resumed server may already be calibrated/ready — proceed (run() itself
    # short-circuits) rather than waiting forever for a fresh bridge flag.
    m = _patch_auto_worker(mocker)
    m["api"].return_value = {"calibrated": True}

    hw_mod.await_bridge_and_calibrate("127.0.0.1", 8048)

    m["run"].assert_called_once_with(auto=True)


def test_await_bridge_and_calibrate_swallows_setup_exit(mocker) -> None:
    # run() calls sys.exit on a step failure — must not crash the server.
    m = _patch_auto_worker(mocker)
    m["run"].side_effect = SystemExit(1)

    hw_mod.await_bridge_and_calibrate("127.0.0.1", 8048)  # must not raise

    m["run"].assert_called_once()


def test_await_bridge_and_calibrate_routes_wizard_output_through_logger(mocker) -> None:
    # The wizard's print() output is badged via LineLogStream, not dumped raw
    # to stdout amid the tagged server logs.
    m = _patch_auto_worker(mocker)
    lines: list[str] = []
    mocker.patch(
        "physiclaw.common.logger.make_tagged_logger",
        return_value=MagicMock(info=lambda x: lines.append(x)),
    )
    m["run"].side_effect = lambda auto: print("  ✓ Phone connected")

    hw_mod.await_bridge_and_calibrate("127.0.0.1", 8048)

    assert "  ✓ Phone connected" in lines
