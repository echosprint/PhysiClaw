"""Tests for `physiclaw.cli.setup.phone` — keyboard learning command."""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import typer
from typer.testing import CliRunner

phone_mod = importlib.import_module("physiclaw.cli.setup.phone")

app = typer.Typer()
app.command()(phone_mod.phone)
runner = CliRunner()


def _write_image(path: Path) -> None:
    path.write_bytes(b"fake image")


def _patch_pipeline(
    mocker,
    *,
    rows=None,
    boxes=None,
    label_returns_none: bool = False,
    cv_imread_returns_none: bool = False,
) -> dict:
    """Stub out cv2 + keyboard module pieces."""
    fake_frame = np.zeros((100, 50, 3), dtype=np.uint8)
    fake_cv = MagicMock()
    fake_cv.imread.return_value = None if cv_imread_returns_none else fake_frame
    fake_cv.imwrite = MagicMock()
    mocker.patch.dict("sys.modules", {"cv2": fake_cv})

    fake_kbd = MagicMock()
    fake_kbd.label_keyboard.return_value = (
        None
        if label_returns_none
        else (
            rows
            or [
                [{"element": "q", "position": [0, 0, 0.1, 0.1]}],
            ]
        )
    )
    fake_kbd.detect_key_boxes.return_value = (boxes or [[0.0, 0.0, 0.1, 0.1]], 50)
    fake_kbd.draw_detected_keys.return_value = fake_frame
    fake_kbd.generate_preset.return_value = "## Alpha Keyboard\n\nfoo\n"
    mocker.patch.dict("sys.modules", {"physiclaw.core.vision.keyboard": fake_kbd})

    return {"cv2": fake_cv, "keyboard": fake_kbd}


def test_phone_no_images_skipped(
    tmp_path: Path,
    mocker,
) -> None:
    img = tmp_path / "shot.png"
    _write_image(img)
    _patch_pipeline(mocker, cv_imread_returns_none=True)

    preset = tmp_path / "preset.md"
    bbox_dir = tmp_path / "bboxes"

    result = runner.invoke(
        app,
        [
            str(img),
            "--preset",
            str(preset),
            "--bbox-dir",
            str(bbox_dir),
        ],
    )

    assert result.exit_code == 1
    assert "cannot read" in result.output


def test_phone_label_keyboard_returns_none(
    tmp_path: Path,
    mocker,
) -> None:
    img = tmp_path / "shot.png"
    _write_image(img)
    _patch_pipeline(mocker, label_returns_none=True)

    result = runner.invoke(
        app,
        [
            str(img),
            "--preset",
            str(tmp_path / "p.md"),
            "--bbox-dir",
            str(tmp_path / "b"),
        ],
    )

    assert result.exit_code == 1
    assert "No keys detected" in result.output


def test_phone_alpha_keyboard_writes_preset(
    tmp_path: Path,
    mocker,
) -> None:
    img = tmp_path / "alpha.png"
    _write_image(img)
    _patch_pipeline(mocker)

    preset = tmp_path / "out" / "preset.md"
    bbox_dir = tmp_path / "bbox"

    result = runner.invoke(
        app,
        [
            str(img),
            "--preset",
            str(preset),
            "--bbox-dir",
            str(bbox_dir),
        ],
    )

    assert result.exit_code == 0
    assert preset.read_text() == "## Alpha Keyboard\n\nfoo\n"
    assert (bbox_dir / "system-keyboard.ref.md").exists()
    assert "Alpha Keyboard" in result.output


def test_phone_numeric_keyboard_detected(
    tmp_path: Path,
    mocker,
) -> None:
    img = tmp_path / "num.png"
    _write_image(img)
    # Numeric layout: row 1 = 10, row 2 = 10.
    rows = [
        [{"element": str(i), "position": [0, 0, 0, 0]} for i in range(10)],
        [{"element": "?", "position": [0, 0, 0, 0]} for _ in range(10)],
        [{"element": "?", "position": [0, 0, 0, 0]} for _ in range(7)],
        [{"element": "?", "position": [0, 0, 0, 0]} for _ in range(3)],
    ]
    _patch_pipeline(mocker, rows=rows)

    result = runner.invoke(
        app,
        [
            str(img),
            "--preset",
            str(tmp_path / "p.md"),
            "--bbox-dir",
            str(tmp_path / "b"),
        ],
    )

    assert result.exit_code == 0
    assert "Numeric Keyboard" in result.output


def test_phone_skips_duplicate_page(tmp_path: Path, mocker) -> None:
    img1 = tmp_path / "alpha1.png"
    img2 = tmp_path / "alpha2.png"
    _write_image(img1)
    _write_image(img2)
    _patch_pipeline(mocker)

    result = runner.invoke(
        app,
        [
            str(img1),
            str(img2),
            "--preset",
            str(tmp_path / "p.md"),
            "--bbox-dir",
            str(tmp_path / "b"),
        ],
    )

    assert result.exit_code == 0
    assert "already captured" in result.output


def test_phone_no_keyboard_detected_anywhere(tmp_path: Path, mocker) -> None:
    img = tmp_path / "blank.png"
    _write_image(img)
    _patch_pipeline(mocker, label_returns_none=True)

    result = runner.invoke(
        app,
        [
            str(img),
            "--preset",
            str(tmp_path / "p.md"),
            "--bbox-dir",
            str(tmp_path / "b"),
        ],
    )

    assert result.exit_code == 1
    assert "No keyboard detected" in result.output
