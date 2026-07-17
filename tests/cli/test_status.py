"""Tests for `physiclaw.cli.status`."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

status_mod = importlib.import_module("physiclaw.cli.status")
status = status_mod.status


def _bundle(**overrides) -> dict:
    """A complete v1 bundle shaped like Calibration.to_dict() writes it —
    `complete` is a @property there, so no such key exists on disk."""
    base = {
        "version": 1,
        "viewport_shift": {
            "offset_x": 0.0,
            "offset_y": 120.0,
            "dpr": 3.0,
            "screenshot_width": 1170,
            "screenshot_height": 2532,
        },
        "cam_rotation": 2,
        "pct_to_grbl": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "pct_to_cam": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "cam_size": [1920, 1080],
        "cam_index": 0,
        "screen_dimension": {"width": 390, "height": 844},
    }
    base.update(overrides)
    return base


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
        return_value=_bundle(),
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
        return_value=_bundle(pct_to_cam=None),  # camera mapping step not done
    )
    mocker.patch.object(status_mod.paths, "jobs_file", return_value=tmp_path / "no.md")

    status()
    out = capsys.readouterr().out

    assert "partial" in out


def test_status_reports_unreadable_bundle_as_partial(
    tmp_path: Path,
    mocker,
    capsys: pytest.CaptureFixture,
) -> None:
    # from_dict raises on a schema-version mismatch — status degrades to
    # "partial" rather than crashing (doctor carries the deep diagnosis).
    mocker.patch.object(
        status_mod.paths,
        "omniparser_onnx",
        return_value=tmp_path / "missing.onnx",
    )
    mocker.patch.object(
        status_mod.paths,
        "load_calibration_bundle",
        return_value=_bundle(version=99),
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
