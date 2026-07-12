"""Tests for `physiclaw.cli.doctor` — health-check command."""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import MagicMock

import urllib.error
import typer
from typer.testing import CliRunner

doctor_mod = importlib.import_module("physiclaw.cli.doctor")

app = typer.Typer()
app.command()(doctor_mod.doctor)
runner = CliRunner()


# ---------- _list_cameras ----------


def test_list_cameras_returns_open_indices(mocker) -> None:
    import cv2

    captures = {}

    def make_cap(i):
        cap = MagicMock()
        cap.isOpened.return_value = i in (0, 2)
        captures[i] = cap
        return cap

    mocker.patch.object(cv2, "VideoCapture", side_effect=make_cap)
    mocker.patch(
        "physiclaw.core.hardware.camera.silenced_stderr",
        return_value=MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None),
    )

    out = doctor_mod._list_cameras(max_index=5)

    assert out == [0, 2]


def test_list_cameras_breaks_after_two_misses(mocker) -> None:
    import cv2

    def make_cap(i):
        cap = MagicMock()
        cap.isOpened.return_value = i == 0
        return cap

    spy = mocker.patch.object(cv2, "VideoCapture", side_effect=make_cap)
    mocker.patch(
        "physiclaw.core.hardware.camera.silenced_stderr",
        return_value=MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None),
    )

    out = doctor_mod._list_cameras(max_index=10)

    assert out == [0]
    # Stops after seeing 2 misses (indices 1 and 2).
    assert spy.call_count == 3


# ---------- _probe_server ----------


def test_probe_server_uses_live_state_when_provided(mocker) -> None:
    mocker.patch.object(doctor_mod._http, "api", return_value={"ready": True})

    host, port, bind_all, status = doctor_mod._probe_server(
        {"host": "127.0.0.1", "port": 9000},
    )

    assert host == "127.0.0.1"
    assert port == 9000
    assert bind_all is False
    assert status == {"ready": True}


def test_probe_server_renames_0_0_0_0_to_localhost(mocker) -> None:
    mocker.patch.object(doctor_mod._http, "api", return_value={"ready": False})

    host, _port, bind_all, _status = doctor_mod._probe_server(
        {"host": "0.0.0.0", "port": 8048},
    )

    assert host == "localhost"
    assert bind_all is True


def test_probe_server_falls_back_to_config(mocker) -> None:
    mocker.patch.object(doctor_mod._http, "api", return_value={"ready": True})

    host, port, bind_all, status = doctor_mod._probe_server(None)

    # Just verify the call worked — host/port come from CONFIG.
    assert isinstance(host, str)
    assert isinstance(port, int)


def test_probe_server_returns_none_on_http_error(mocker) -> None:
    # _http.api is fail-soft (None on refused/timeout) — pin the pass-through.
    mocker.patch.object(doctor_mod._http, "api", return_value=None)

    host, port, bind_all, status = doctor_mod._probe_server(
        {"host": "x", "port": 1},
    )

    assert status is None


# ---------- _probe_vision_model_deep ----------


def test_probe_vision_model_deep_ok(mocker) -> None:
    fake_input = MagicMock()
    fake_input.shape = [1, 3, 1280, 1280]
    fake_session = MagicMock()
    fake_session.get_inputs.return_value = [fake_input]
    fake_ort = MagicMock()
    fake_ort.InferenceSession.return_value = fake_session
    mocker.patch.dict("sys.modules", {"onnxruntime": fake_ort})

    out = doctor_mod._probe_vision_model_deep()

    assert "loads OK" in out


def test_probe_vision_model_deep_failure(mocker) -> None:
    fake_ort = MagicMock()
    fake_ort.InferenceSession.side_effect = RuntimeError("bad model")
    mocker.patch.dict("sys.modules", {"onnxruntime": fake_ort})

    out = doctor_mod._probe_vision_model_deep()

    assert "load failed" in out
    assert "RuntimeError" in out


# ---------- _probe_camera_frame ----------


def test_probe_camera_frame_ok(mocker) -> None:
    import cv2
    import numpy as np

    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    fake_cap = MagicMock()
    fake_cap.read.return_value = (True, fake_frame)
    mocker.patch.object(cv2, "VideoCapture", return_value=fake_cap)
    mocker.patch(
        "physiclaw.core.hardware.camera.silenced_stderr",
        return_value=MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None),
    )

    out = doctor_mod._probe_camera_frame(0)

    assert "frame OK" in out
    assert "640x480" in out


def test_probe_camera_frame_no_frame(mocker) -> None:
    import cv2

    fake_cap = MagicMock()
    fake_cap.read.return_value = (False, None)
    mocker.patch.object(cv2, "VideoCapture", return_value=fake_cap)
    mocker.patch(
        "physiclaw.core.hardware.camera.silenced_stderr",
        return_value=MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None),
    )

    out = doctor_mod._probe_camera_frame(2)

    assert "no frame" in out


# ---------- _probe_calibration_deep ----------


def test_probe_calibration_deep_no_bundle(mocker) -> None:
    mocker.patch.object(doctor_mod.paths, "load_calibration_bundle", return_value=None)

    out = doctor_mod._probe_calibration_deep()

    assert "parse failed" in out


def test_probe_calibration_deep_unreadable(mocker) -> None:
    mocker.patch.object(
        doctor_mod.paths,
        "load_calibration_bundle",
        return_value={"x": 1},
    )
    fake_cal_module = MagicMock()
    fake_cal_module.Calibration.from_dict.side_effect = ValueError("bad")
    mocker.patch.dict(
        "sys.modules",
        {"physiclaw.core.calibration.state": fake_cal_module},
    )

    out = doctor_mod._probe_calibration_deep()

    assert "unreadable" in out


def test_probe_calibration_deep_complete(mocker) -> None:
    mocker.patch.object(
        doctor_mod.paths,
        "load_calibration_bundle",
        return_value={"a": 1},
    )
    fake_cal = MagicMock(complete=True)
    fake_cal_module = MagicMock()
    fake_cal_module.Calibration.from_dict.return_value = fake_cal
    mocker.patch.dict(
        "sys.modules",
        {"physiclaw.core.calibration.state": fake_cal_module},
    )

    out = doctor_mod._probe_calibration_deep()

    assert "bundle complete" in out


def test_probe_calibration_deep_partial_with_missing(mocker) -> None:
    mocker.patch.object(
        doctor_mod.paths,
        "load_calibration_bundle",
        return_value={"a": None, "b": 1, "c": None},
    )
    fake_cal = MagicMock(complete=False)
    fake_cal_module = MagicMock()
    fake_cal_module.Calibration.from_dict.return_value = fake_cal
    mocker.patch.dict(
        "sys.modules",
        {"physiclaw.core.calibration.state": fake_cal_module},
    )

    out = doctor_mod._probe_calibration_deep()

    assert "partial" in out
    assert "'a'" in out and "'c'" in out


def test_probe_calibration_deep_partial_no_missing(mocker) -> None:
    mocker.patch.object(
        doctor_mod.paths,
        "load_calibration_bundle",
        return_value={"a": 1, "b": 2},
    )
    fake_cal = MagicMock(complete=False)
    fake_cal_module = MagicMock()
    fake_cal_module.Calibration.from_dict.return_value = fake_cal
    mocker.patch.dict(
        "sys.modules",
        {"physiclaw.core.calibration.state": fake_cal_module},
    )

    out = doctor_mod._probe_calibration_deep()

    assert "keys:" in out


# ---------- _probe_bridge_deep ----------


def test_probe_bridge_deep_connected(mocker) -> None:
    mocker.patch.object(
        doctor_mod._http, "fetch_json", return_value={"connected": True}
    )

    out = doctor_mod._probe_bridge_deep("localhost", 8048)

    assert "phone connected" in out


def test_probe_bridge_deep_not_paired(mocker) -> None:
    mocker.patch.object(
        doctor_mod._http, "fetch_json", return_value={"connected": False}
    )

    out = doctor_mod._probe_bridge_deep("localhost", 8048)

    assert "not paired" in out


def test_probe_bridge_deep_http_error(mocker) -> None:
    # The warn line carries the failure detail (exception type + message).
    mocker.patch.object(
        doctor_mod._http,
        "fetch_json",
        side_effect=urllib.error.URLError("x"),
    )

    out = doctor_mod._probe_bridge_deep("localhost", 8048)

    assert "URLError" in out


# ---------- _skills_lines ----------


def test_skills_lines_empty(mocker, tmp_path: Path) -> None:
    mocker.patch.object(doctor_mod.paths, "skills_dir", return_value=tmp_path)

    fake_skills = MagicMock()
    fake_skills.installed_skill_dirs.return_value = []
    fake_skills.read_provenance.return_value = None
    fake_skills.PROVENANCE_FILE = ".installed-from"
    mocker.patch.dict("sys.modules", {"physiclaw.cli.skills": fake_skills})

    lines = doctor_mod._skills_lines()

    assert len(lines) == 1
    assert "none installed" in lines[0]


def test_skills_lines_with_provenance(mocker, tmp_path: Path) -> None:
    skill_dir = tmp_path / "alpha"
    skill_dir.mkdir()

    fake_skills = MagicMock()
    fake_skills.installed_skill_dirs.return_value = [skill_dir]
    fake_skills.read_provenance.return_value = {
        "source": "owner/repo",
        "ref": "main",
        "sha": "abc1234",
    }
    fake_skills.PROVENANCE_FILE = ".installed-from"
    mocker.patch.dict("sys.modules", {"physiclaw.cli.skills": fake_skills})

    lines = doctor_mod._skills_lines()

    assert any("alpha" in line and "owner/repo" in line for line in lines)


def test_skills_lines_uses_sha_when_no_ref(mocker, tmp_path: Path) -> None:
    skill_dir = tmp_path / "alpha"
    skill_dir.mkdir()

    fake_skills = MagicMock()
    fake_skills.installed_skill_dirs.return_value = [skill_dir]
    fake_skills.read_provenance.return_value = {
        "source": "owner/repo",
        "ref": "",
        "sha": "abcdef0123",
    }
    fake_skills.PROVENANCE_FILE = ".installed-from"
    mocker.patch.dict("sys.modules", {"physiclaw.cli.skills": fake_skills})

    lines = doctor_mod._skills_lines()

    assert any("abcdef0" in line for line in lines)


def test_skills_lines_local_skill(mocker, tmp_path: Path) -> None:
    skill_dir = tmp_path / "myown"
    skill_dir.mkdir()

    fake_skills = MagicMock()
    fake_skills.installed_skill_dirs.return_value = [skill_dir]
    fake_skills.read_provenance.return_value = None
    fake_skills.PROVENANCE_FILE = ".installed-from"
    mocker.patch.dict("sys.modules", {"physiclaw.cli.skills": fake_skills})

    lines = doctor_mod._skills_lines()

    assert any("myown" in line and "(local)" in line for line in lines)


def test_skills_lines_unparseable_provenance(mocker, tmp_path: Path) -> None:
    skill_dir = tmp_path / "broken"
    skill_dir.mkdir()
    (skill_dir / ".installed-from").write_text("not json")

    fake_skills = MagicMock()
    fake_skills.installed_skill_dirs.return_value = [skill_dir]
    fake_skills.read_provenance.return_value = None
    fake_skills.PROVENANCE_FILE = ".installed-from"
    mocker.patch.dict("sys.modules", {"physiclaw.cli.skills": fake_skills})

    lines = doctor_mod._skills_lines()

    assert any("provenance unreadable" in line for line in lines)


# ---------- doctor command (smoke) ----------


def _patch_doctor_environment(
    mocker,
    *,
    server_status: dict | None = None,
    model_exists: bool = False,
    bundle_exists: bool = False,
    live_ref: str | None = None,
    key_unset: bool = True,
) -> None:
    """Stub everything `doctor()` touches so the command runs to completion."""
    mocker.patch.object(doctor_mod.paths, "ensure_dirs")
    mocker.patch.object(doctor_mod.paths, "HOME", Path("/fake/home"))
    cp = MagicMock(spec=Path)
    cp.exists.return_value = True
    cp.__str__ = lambda s: "/fake/config.toml"
    mocker.patch("physiclaw.common.config.config_path", return_value=cp)

    onnx = MagicMock(spec=Path)
    onnx.exists.return_value = model_exists
    onnx.stat.return_value = MagicMock(st_size=1024 * 1024)
    onnx.__str__ = lambda s: "/fake/model.onnx"
    mocker.patch.object(doctor_mod.paths, "omniparser_onnx", return_value=onnx)

    bundle = MagicMock(spec=Path)
    bundle.exists.return_value = bundle_exists
    bundle.__str__ = lambda s: "/fake/bundle.json"
    mocker.patch.object(doctor_mod.paths, "calibration_bundle", return_value=bundle)

    fake_state = MagicMock()
    if live_ref:
        fake_state.read_live.return_value = {
            "host": "127.0.0.1",
            "port": 8048,
            "model_ref": live_ref,
            "model_source": "config",
        }
    else:
        fake_state.read_live.return_value = None
    mocker.patch.object(doctor_mod, "runtime_state", fake_state)

    mocker.patch.object(
        doctor_mod,
        "_probe_server",
        return_value=("localhost", 8048, False, server_status),
    )
    mocker.patch.object(doctor_mod, "_list_cameras", return_value=[0])
    mocker.patch.object(
        doctor_mod, "_skills_lines", return_value=["  (none installed)"]
    )

    fake_launcher = MagicMock()
    fake_launcher.engine_label.return_value = "engine=openai"
    fake_launcher.resolve.return_value = ("openai/gpt-5", "config")
    mocker.patch.dict(
        "sys.modules",
        {"physiclaw.agent.runtime.launcher": fake_launcher},
    )

    fake_provider = MagicMock()
    fake_provider.CLAUDE_CODE_ID = "claude-code"
    fake_provider.provider_key_status.return_value = (
        (None, "config") if key_unset else ("sk-***x", "env")
    )
    mocker.patch.dict(
        "sys.modules",
        {"physiclaw.agent.provider": fake_provider},
    )

    fake_grbl = MagicMock()
    fake_grbl.detect_grbl.return_value = "/dev/cu.usbserial-X"
    fake_grbl.candidate_ports.return_value = ["/dev/cu.usbserial-X"]
    mocker.patch.dict(
        "sys.modules",
        {"physiclaw.core.hardware.grbl": fake_grbl},
    )


def test_doctor_server_running_and_ready(mocker) -> None:
    _patch_doctor_environment(
        mocker,
        server_status={
            "arm": True,
            "camera": True,
            "calibrated": True,
            "ready": True,
        },
        model_exists=True,
        bundle_exists=True,
        key_unset=False,
    )

    result = runner.invoke(app, [])

    assert result.exit_code == 0
    assert "PhysiClaw doctor" in result.output
    assert "All set" in result.output


def test_doctor_server_not_running_probes_locally(mocker) -> None:
    _patch_doctor_environment(
        mocker,
        server_status=None,  # server offline
        model_exists=False,
        bundle_exists=False,
    )

    result = runner.invoke(app, [])

    assert result.exit_code == 0
    assert "server: not running" in result.output
    assert "GRBL arm" in result.output
    assert "physiclaw setup local-vision-model" in result.output
    assert "starts the server and opens the hardware-setup wizard" in result.output


def test_doctor_no_cameras_detected_offline(mocker) -> None:
    _patch_doctor_environment(mocker, server_status=None)
    mocker.patch.object(doctor_mod, "_list_cameras", return_value=[])

    result = runner.invoke(app, [])

    assert "cameras: none detected" in result.output


def test_doctor_no_grbl_detected(mocker) -> None:
    _patch_doctor_environment(mocker, server_status=None)
    fake_grbl = MagicMock()
    fake_grbl.detect_grbl.return_value = None
    fake_grbl.candidate_ports.return_value = []
    mocker.patch.dict(
        "sys.modules",
        {"physiclaw.core.hardware.grbl": fake_grbl},
    )

    result = runner.invoke(app, [])

    assert "no serial ports detected" in result.output


def test_doctor_no_grbl_with_candidate_ports(mocker) -> None:
    _patch_doctor_environment(mocker, server_status=None)
    fake_grbl = MagicMock()
    fake_grbl.detect_grbl.return_value = None
    fake_grbl.candidate_ports.return_value = ["/dev/cu.usbserial-A"]
    mocker.patch.dict(
        "sys.modules",
        {"physiclaw.core.hardware.grbl": fake_grbl},
    )

    result = runner.invoke(app, [])

    assert "no GRBL detected" in result.output


def test_doctor_warns_about_bind_all(mocker) -> None:
    _patch_doctor_environment(
        mocker,
        server_status={
            "arm": True,
            "camera": True,
            "calibrated": True,
            "ready": True,
        },
    )
    mocker.patch.object(
        doctor_mod,
        "_probe_server",
        return_value=(
            "localhost",
            8048,
            True,
            {
                "arm": True,
                "camera": True,
                "calibrated": True,
                "ready": True,
            },
        ),
    )

    result = runner.invoke(app, [])

    assert "0.0.0.0" in result.output


def test_doctor_engine_invalid_when_resolve_fails(mocker) -> None:
    _patch_doctor_environment(
        mocker,
        server_status=None,
    )
    fake_launcher = MagicMock()
    fake_launcher.resolve.side_effect = RuntimeError("no model set")
    mocker.patch.dict(
        "sys.modules",
        {"physiclaw.agent.runtime.launcher": fake_launcher},
    )

    result = runner.invoke(app, [])

    assert "engine: invalid" in result.output


def test_doctor_uses_live_engine_ref(mocker) -> None:
    _patch_doctor_environment(
        mocker,
        server_status={"arm": True, "camera": True, "calibrated": True, "ready": True},
        live_ref="openai/gpt-5",
    )

    result = runner.invoke(app, [])

    assert "live server" in result.output


def test_doctor_no_config_warns(mocker) -> None:
    _patch_doctor_environment(mocker, server_status=None)
    cp = MagicMock(spec=Path)
    cp.exists.return_value = False
    mocker.patch("physiclaw.common.config.config_path", return_value=cp)

    result = runner.invoke(app, [])

    assert "no config yet" in result.output


def test_doctor_deep_runs_probes(mocker) -> None:
    _patch_doctor_environment(
        mocker,
        server_status={"arm": True, "camera": True, "calibrated": True, "ready": True},
        model_exists=True,
        bundle_exists=True,
        key_unset=False,
    )
    vision_spy = mocker.patch.object(
        doctor_mod,
        "_probe_vision_model_deep",
        return_value="✓ vision OK",
    )
    bridge_spy = mocker.patch.object(
        doctor_mod,
        "_probe_bridge_deep",
        return_value="✓ bridge OK",
    )
    calib_spy = mocker.patch.object(
        doctor_mod,
        "_probe_calibration_deep",
        return_value="✓ calib OK",
    )
    provider_spy = mocker.patch.object(
        doctor_mod,
        "_probe_provider_deep",
        return_value="✓ provider OK",
    )

    runner.invoke(app, ["--deep"])

    vision_spy.assert_called_once()
    bridge_spy.assert_called_once()
    calib_spy.assert_called_once()
    provider_spy.assert_called_once()


# ---------- _probe_provider_deep ----------


def _make_async(value):
    async def _coro(*a, **kw):
        return value

    return _coro


def test_probe_provider_deep_setup_error_returns_warn(mocker) -> None:
    fake_provider = MagicMock()
    fake_provider.make_provider.side_effect = ValueError("missing api key")
    mocker.patch.dict(
        "sys.modules",
        {"physiclaw.agent.provider": fake_provider},
    )

    out = doctor_mod._probe_provider_deep("openai", "gpt-5")

    assert "setup" in out
    assert "missing api key" in out


def test_probe_provider_deep_chat_exception_returns_warn(mocker) -> None:
    """Non-empty reply check happens before this; the chat itself raises."""

    fake_prov = MagicMock()
    fake_prov.chat = mocker.MagicMock(side_effect=RuntimeError("network"))
    fake_prov.aclose = _make_async(None)

    fake_module = MagicMock()
    fake_module.make_provider.return_value = fake_prov
    mocker.patch.dict(
        "sys.modules",
        {"physiclaw.agent.provider": fake_module},
    )

    out = doctor_mod._probe_provider_deep("openai", "gpt-5")

    assert "RuntimeError" in out
    assert "network" in out


def test_probe_provider_deep_empty_reply_returns_warn(mocker) -> None:
    from physiclaw.agent.engine.dto import (
        AssistantMessage,
        FinishReason,
    )

    asst = AssistantMessage(
        content="",
        tool_calls=[],
        finish_reason=FinishReason.STOP,
    )
    fake_prov = MagicMock()
    fake_prov.chat = _make_async(asst)
    fake_prov.aclose = _make_async(None)

    fake_module = MagicMock()
    fake_module.make_provider.return_value = fake_prov
    mocker.patch.dict(
        "sys.modules",
        {"physiclaw.agent.provider": fake_module},
    )

    out = doctor_mod._probe_provider_deep("openai", "gpt-5")

    assert "empty reply" in out


def test_probe_provider_deep_vision_check_fails(mocker) -> None:
    from physiclaw.agent.engine.dto import (
        AssistantMessage,
        FinishReason,
        Usage,
    )

    asst = AssistantMessage(
        content="something else",  # no "physiclaw" in reply
        tool_calls=[],
        finish_reason=FinishReason.STOP,
        usage=Usage(prompt_tokens=100, completion_tokens=10),
    )
    fake_prov = MagicMock()
    fake_prov.chat = _make_async(asst)
    fake_prov.aclose = _make_async(None)

    fake_module = MagicMock()
    fake_module.make_provider.return_value = fake_prov
    mocker.patch.dict(
        "sys.modules",
        {"physiclaw.agent.provider": fake_module},
    )

    out = doctor_mod._probe_provider_deep("openai", "gpt-5")

    assert "vision check failed" in out
    assert "PhysiClaw" in out


def test_probe_provider_deep_success_with_usage(mocker) -> None:
    from physiclaw.agent.engine.dto import (
        AssistantMessage,
        FinishReason,
        Usage,
    )

    asst = AssistantMessage(
        content="PhysiClaw",
        tool_calls=[],
        finish_reason=FinishReason.STOP,
        usage=Usage(prompt_tokens=200, completion_tokens=2),
    )
    fake_prov = MagicMock()
    fake_prov.chat = _make_async(asst)
    fake_prov.aclose = _make_async(None)

    fake_module = MagicMock()
    fake_module.make_provider.return_value = fake_prov
    mocker.patch.dict(
        "sys.modules",
        {"physiclaw.agent.provider": fake_module},
    )

    out = doctor_mod._probe_provider_deep("openai", "gpt-5")

    # Success rendered as ok.
    assert "PhysiClaw" in out
    assert "200p+2c" in out


def test_probe_provider_deep_success_no_usage(mocker) -> None:
    from physiclaw.agent.engine.dto import (
        AssistantMessage,
        FinishReason,
        Usage,
    )

    asst = AssistantMessage(
        content="PhysiClaw",
        tool_calls=[],
        finish_reason=FinishReason.STOP,
        usage=Usage(prompt_tokens=0, completion_tokens=0),
    )
    fake_prov = MagicMock()
    fake_prov.chat = _make_async(asst)
    fake_prov.aclose = _make_async(None)

    fake_module = MagicMock()
    fake_module.make_provider.return_value = fake_prov
    mocker.patch.dict(
        "sys.modules",
        {"physiclaw.agent.provider": fake_module},
    )

    out = doctor_mod._probe_provider_deep("openai", "gpt-5")

    assert "no usage" in out


def test_doctor_invalid_active_model_ref(mocker) -> None:
    _patch_doctor_environment(
        mocker,
        server_status=None,
        live_ref="garbage-no-slash",
    )
    fake_cfg = MagicMock()
    fake_cfg.parse_model_ref.side_effect = ValueError("bad ref")
    mocker.patch.dict("sys.modules", {"physiclaw.common.config": fake_cfg})

    result = runner.invoke(app, [])

    # Doesn't crash even with unparseable ref.
    assert result.exit_code == 0
