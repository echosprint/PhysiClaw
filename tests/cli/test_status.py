"""Tests for `physiclaw.cli.status`."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

status_mod = importlib.import_module("physiclaw.cli.status")
status = status_mod.status


def test_status_reports_vision_model_ok(
    tmp_path: Path,
    mocker,
    capsys: pytest.CaptureFixture,
) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"x" * 1024 * 1024)  # 1 MB
    mocker.patch.object(status_mod.paths, "omniparser_onnx", return_value=model)
    mocker.patch.object(status_mod.paths, "load_calibration_bundle", return_value=None)
    mocker.patch.object(status_mod.paths, "jobs_file", return_value=tmp_path / "no.md")

    status()
    out = capsys.readouterr().out

    assert "vision model" in out
    assert "ok" in out
    assert "1 MB" in out


def test_status_reports_vision_model_missing(
    tmp_path: Path,
    mocker,
    capsys: pytest.CaptureFixture,
) -> None:
    mocker.patch.object(
        status_mod.paths,
        "omniparser_onnx",
        return_value=tmp_path / "missing.onnx",
    )
    mocker.patch.object(status_mod.paths, "load_calibration_bundle", return_value=None)
    mocker.patch.object(status_mod.paths, "jobs_file", return_value=tmp_path / "no.md")

    status()
    out = capsys.readouterr().out

    assert "vision model" in out
    assert "missing" in out


def test_status_reports_calibration_complete(
    tmp_path: Path,
    mocker,
    capsys: pytest.CaptureFixture,
) -> None:
    mocker.patch.object(
        status_mod.paths,
        "omniparser_onnx",
        return_value=tmp_path / "missing.onnx",
    )
    mocker.patch.object(
        status_mod.paths,
        "load_calibration_bundle",
        return_value={"complete": True},
    )
    mocker.patch.object(status_mod.paths, "jobs_file", return_value=tmp_path / "no.md")

    status()
    out = capsys.readouterr().out

    assert "calibration" in out
    assert "complete" in out


def test_status_reports_calibration_partial(
    tmp_path: Path,
    mocker,
    capsys: pytest.CaptureFixture,
) -> None:
    mocker.patch.object(
        status_mod.paths,
        "omniparser_onnx",
        return_value=tmp_path / "missing.onnx",
    )
    mocker.patch.object(
        status_mod.paths,
        "load_calibration_bundle",
        return_value={"complete": False},
    )
    mocker.patch.object(status_mod.paths, "jobs_file", return_value=tmp_path / "no.md")

    status()
    out = capsys.readouterr().out

    assert "partial" in out


def test_status_reports_calibration_missing(
    tmp_path: Path,
    mocker,
    capsys: pytest.CaptureFixture,
) -> None:
    mocker.patch.object(
        status_mod.paths,
        "omniparser_onnx",
        return_value=tmp_path / "missing.onnx",
    )
    mocker.patch.object(status_mod.paths, "load_calibration_bundle", return_value=None)
    mocker.patch.object(status_mod.paths, "jobs_file", return_value=tmp_path / "no.md")

    status()
    out = capsys.readouterr().out

    assert "calibration" in out
    assert "missing" in out


def test_status_reports_jobs_file_present(
    tmp_path: Path,
    mocker,
    capsys: pytest.CaptureFixture,
) -> None:
    jobs = tmp_path / "jobs.md"
    jobs.write_text("")
    mocker.patch.object(
        status_mod.paths,
        "omniparser_onnx",
        return_value=tmp_path / "missing.onnx",
    )
    mocker.patch.object(status_mod.paths, "load_calibration_bundle", return_value=None)
    mocker.patch.object(status_mod.paths, "jobs_file", return_value=jobs)

    status()
    out = capsys.readouterr().out

    assert "jobs file" in out
    assert str(jobs) in out


def test_status_reports_jobs_file_missing(
    tmp_path: Path,
    mocker,
    capsys: pytest.CaptureFixture,
) -> None:
    jobs = tmp_path / "jobs.md"
    mocker.patch.object(
        status_mod.paths,
        "omniparser_onnx",
        return_value=tmp_path / "missing.onnx",
    )
    mocker.patch.object(status_mod.paths, "load_calibration_bundle", return_value=None)
    mocker.patch.object(status_mod.paths, "jobs_file", return_value=jobs)

    status()
    out = capsys.readouterr().out

    assert "none yet" in out


def test_status_shows_doctor_hint_on_tty(
    tmp_path: Path,
    mocker,
    capsys: pytest.CaptureFixture,
) -> None:
    mocker.patch.object(
        status_mod.paths,
        "omniparser_onnx",
        return_value=tmp_path / "missing.onnx",
    )
    mocker.patch.object(status_mod.paths, "load_calibration_bundle", return_value=None)
    mocker.patch.object(status_mod.paths, "jobs_file", return_value=tmp_path / "no.md")
    mocker.patch.object(status_mod.sys.stdout, "isatty", return_value=True)

    status()
    out = capsys.readouterr().out

    assert "physiclaw doctor" in out


def test_status_suppresses_doctor_hint_on_pipe(
    tmp_path: Path,
    mocker,
    capsys: pytest.CaptureFixture,
) -> None:
    mocker.patch.object(
        status_mod.paths,
        "omniparser_onnx",
        return_value=tmp_path / "missing.onnx",
    )
    mocker.patch.object(status_mod.paths, "load_calibration_bundle", return_value=None)
    mocker.patch.object(status_mod.paths, "jobs_file", return_value=tmp_path / "no.md")
    mocker.patch.object(status_mod.sys.stdout, "isatty", return_value=False)

    status()
    out = capsys.readouterr().out

    assert "physiclaw doctor" not in out
