"""Tests for `physiclaw.core.server.warm_start` — the `try_resume`
happy-path / early-return branches with elaborate mocking. The `_sanity`
helper is integration-only and lives behind hardware fakes. `wait_for_port`
now lives in `core.server.net` (tested in test_net.py); we only pin the
re-export here.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from physiclaw.core.server import net, warm_start

# ---------- wait_for_port re-export ----------


def test_wait_for_port_is_re_exported_from_net() -> None:
    # Callers still do `warm_start.wait_for_port` / `from ...warm_start import
    # wait_for_port`; keep that pointing at the real implementation in net.
    assert warm_start.wait_for_port is net.wait_for_port


# ---------- try_resume: early-return branches ----------


def test_try_resume_returns_false_when_no_bundle(
    mocker, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    mocker.patch("physiclaw.core.calibration.state.Calibration.load", return_value=None)

    with caplog.at_level(logging.ERROR, logger="physiclaw.core.server.warm_start"):
        result = warm_start.try_resume(
            cam_index_override=None,
            physiclaw=MagicMock(),
            calib=MagicMock(),
            phone=MagicMock(),
        )

    assert result is False
    assert any(
        "no calibration bundle on disk" in r.getMessage() for r in caplog.records
    )


def test_try_resume_returns_false_when_bundle_incomplete(
    mocker, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    fake_cal = MagicMock()
    fake_cal.complete = False
    fake_cal.viewport_shift = None
    fake_cal.screen_dimension = None
    fake_cal.cam_rotation = 0
    mocker.patch(
        "physiclaw.core.calibration.state.Calibration.load",
        return_value=fake_cal,
    )
    fake_app = MagicMock()

    with caplog.at_level(logging.ERROR, logger="physiclaw.core.server.warm_start"):
        result = warm_start.try_resume(
            cam_index_override=None,
            physiclaw=fake_app,
            calib=MagicMock(),
            phone=MagicMock(),
        )

    assert result is False
    assert any("bundle on disk is incomplete" in r.getMessage() for r in caplog.records)


def test_try_resume_returns_false_when_hardware_connect_raises(
    mocker, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    fake_cal = MagicMock()
    fake_cal.complete = True
    fake_cal.viewport_shift = None
    fake_cal.screen_dimension = None
    fake_cal.cam_index = 0
    fake_cal.cam_rotation = 0
    mocker.patch(
        "physiclaw.core.calibration.state.Calibration.load",
        return_value=fake_cal,
    )
    fake_app = MagicMock()
    fake_app.rig.connect_arm.side_effect = RuntimeError("port unavailable")

    with caplog.at_level(logging.ERROR, logger="physiclaw.core.server.warm_start"):
        result = warm_start.try_resume(
            cam_index_override=None,
            physiclaw=fake_app,
            calib=MagicMock(),
            phone=MagicMock(),
        )

    assert result is False
    assert any(
        "hardware reconnect failed: port unavailable" in r.getMessage()
        for r in caplog.records
    )


# ---------- try_resume: post-connect flow ----------


def _ready_bundle() -> MagicMock:
    """A complete-and-loaded Calibration mock."""
    cal = MagicMock()
    cal.complete = True
    cal.viewport_shift = MagicMock()
    cal.screen_dimension = (390, 844)
    cal.cam_index = 1
    cal.cam_rotation = 0
    cal.pct_to_grbl = MagicMock()
    cal.pct_to_cam = MagicMock()
    cal.cam_size = (1920, 1080)
    cal.effective_rotation.return_value = 0
    return cal


def _ready_app(cal) -> MagicMock:
    app = MagicMock()
    app.rig.calibration = cal
    app.rig.assistive_touch = MagicMock()
    # A rotated live frame whose (w, h) matches the bundle's cam_size, so
    # the resolution reconcile step sees "same size" on the clean path.
    app.rig.cam.peek.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
    return app


def _patch_resume_env(mocker, cal, app, sanity: bool = True) -> MagicMock:
    """Install the try_resume environment patches; returns the _sanity mock."""
    mocker.patch(
        "physiclaw.core.calibration.state.Calibration.load",
        return_value=cal,
    )
    mocker.patch.object(warm_start.sys.stdin, "isatty", return_value=False)
    return mocker.patch.object(warm_start, "_sanity", return_value=sanity)


def _resume(app, cam_index_override=None, calib=None, **kwargs) -> bool:
    """Call try_resume with the fake assembly's state objects."""
    return warm_start.try_resume(
        cam_index_override=cam_index_override,
        physiclaw=app,
        calib=calib if calib is not None else MagicMock(),
        phone=MagicMock(),
        **kwargs,
    )


def test_try_resume_succeeds_on_clean_path(mocker) -> None:
    cal = _ready_bundle()
    app = _ready_app(cal)
    mocker.patch(
        "physiclaw.core.calibration.state.Calibration.load",
        return_value=cal,
    )
    fake_calib_state = MagicMock()
    mocker.patch.object(warm_start, "_sanity", return_value=True)
    mocker.patch.object(warm_start.sys.stdin, "isatty", return_value=False)

    result = _resume(app, calib=fake_calib_state)

    assert result is True
    # Bundle replaced into the app + viewport_shift mirrored to bridge state.
    assert app.rig.calibration is cal
    assert fake_calib_state.viewport_shift is cal.viewport_shift
    assert fake_calib_state.screen_dimension == (390, 844)
    app.rig.assistive_touch.compute_at_screen_pos.assert_called_once_with(
        cal.viewport_shift,
    )
    app.rig.connect_arm.assert_called_once()
    app.rig.connect_camera.assert_called_once_with(1)
    # Origin re-pinned from the park spot so the bundle's affine stays valid.
    app.rig.restore_park_origin.assert_called_once()
    app.home_screen.assert_called_once()
    # become_ready owns the settle-then-mark order (camera settle between
    # home — dark scene showing — and the ready flip).
    app.become_ready.assert_called_once()


def test_try_resume_uses_cam_index_override(mocker) -> None:
    cal = _ready_bundle()
    app = _ready_app(cal)
    mocker.patch(
        "physiclaw.core.calibration.state.Calibration.load",
        return_value=cal,
    )
    mocker.patch.object(warm_start, "_sanity", return_value=True)
    mocker.patch.object(warm_start.sys.stdin, "isatty", return_value=False)

    _resume(app, cam_index_override=3)

    app.rig.connect_camera.assert_called_once_with(3)


def test_try_resume_falls_back_to_cam_index_zero(mocker) -> None:
    cal = _ready_bundle()
    cal.cam_index = None
    app = _ready_app(cal)
    mocker.patch(
        "physiclaw.core.calibration.state.Calibration.load",
        return_value=cal,
    )
    mocker.patch.object(warm_start, "_sanity", return_value=True)
    mocker.patch.object(warm_start.sys.stdin, "isatty", return_value=False)

    _resume(app)

    app.rig.connect_camera.assert_called_once_with(0)


def test_try_resume_returns_false_when_bridge_never_connects(
    mocker,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    cal = _ready_bundle()
    app = _ready_app(cal)
    app.rig.bridge.wait_for_connection.return_value = False
    mocker.patch(
        "physiclaw.core.calibration.state.Calibration.load",
        return_value=cal,
    )
    mocker.patch.object(warm_start.sys.stdin, "isatty", return_value=True)

    with caplog.at_level(logging.ERROR, logger="physiclaw.core.server.warm_start"):
        result = _resume(app)

    assert result is False
    assert any("/bridge page not polling" in r.getMessage() for r in caplog.records)


def test_try_resume_returns_false_when_sanity_fails(mocker) -> None:
    cal = _ready_bundle()
    app = _ready_app(cal)
    mocker.patch(
        "physiclaw.core.calibration.state.Calibration.load",
        return_value=cal,
    )
    mocker.patch.object(warm_start, "_sanity", return_value=False)
    mocker.patch.object(warm_start.sys.stdin, "isatty", return_value=False)

    result = _resume(app)

    assert result is False
    app.become_ready.assert_not_called()


def test_try_resume_no_verify_skips_everything_that_touches_the_phone(
    mocker,
) -> None:
    # --hot-start: even on an interactive tty there is no bridge wait, no
    # sanity tap, and no home-screen swipe — hardware reconnects, the origin
    # is re-pinned from the park spot, and ready flips immediately.
    cal = _ready_bundle()
    app = _ready_app(cal)
    sanity = _patch_resume_env(mocker, cal, app)
    mocker.patch.object(warm_start.sys.stdin, "isatty", return_value=True)

    result = _resume(app, verify=False)

    assert result is True
    sanity.assert_not_called()
    app.rig.bridge.wait_for_connection.assert_not_called()
    app.home_screen.assert_not_called()
    app.rig.restore_park_origin.assert_called_once()
    app.become_ready.assert_called_once()


def test_try_resume_no_verify_still_requires_a_complete_bundle(
    mocker, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    cal = _ready_bundle()
    cal.complete = False
    app = _ready_app(cal)
    _patch_resume_env(mocker, cal, app)

    with caplog.at_level(logging.ERROR, logger="physiclaw.core.server.warm_start"):
        result = _resume(app, verify=False)

    assert result is False
    app.become_ready.assert_not_called()
    assert any(
        "--hot-start: bundle on disk is incomplete" in r.getMessage()
        for r in caplog.records
    )


def test_try_resume_reconciles_live_resolution_with_bundle(mocker) -> None:
    # The camera negotiated 4K but the bundle was calibrated at 1080p —
    # try_resume must offer the live ROTATED (w, h) to the bundle's
    # reconcile before running sanity.
    cal = _ready_bundle()
    app = _ready_app(cal)
    app.rig.cam.peek.return_value = np.zeros((2160, 3840, 3), dtype=np.uint8)
    _patch_resume_env(mocker, cal, app)

    result = _resume(app)

    assert result is True
    cal.reconcile_cam_size.assert_called_once_with((3840, 2160))


def test_try_resume_returns_false_when_aspect_changed(mocker) -> None:
    cal = _ready_bundle()
    cal.reconcile_cam_size.return_value = False
    app = _ready_app(cal)
    sanity = _patch_resume_env(mocker, cal, app)

    result = _resume(app)

    assert result is False
    sanity.assert_not_called()
    app.become_ready.assert_not_called()


def test_try_resume_returns_false_when_camera_has_no_frame(mocker) -> None:
    cal = _ready_bundle()
    app = _ready_app(cal)
    app.rig.cam.peek.return_value = None
    _patch_resume_env(mocker, cal, app)

    result = _resume(app)

    assert result is False
    app.become_ready.assert_not_called()


# ---------- _sanity ----------


def test_sanity_passes_when_all_taps_within_tolerance(mocker) -> None:
    fake_validate = mocker.patch(
        "physiclaw.core.calibration.calibrate.validate_calibration",
        return_value=[
            {"passed": True, "error": 0.5},
            {"passed": True, "error": 0.6},
        ],
    )
    pl = MagicMock()
    pl.rig.calibration = _ready_bundle()
    phone = MagicMock()

    out = warm_start._sanity(pl, MagicMock(), phone)

    assert out is True
    fake_validate.assert_called_once()
    # Bridge mode restored on success.
    assert phone.set_mode.call_args_list[-1].args == ("bridge",)


def test_sanity_fails_when_no_taps_received(
    mocker,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    mocker.patch(
        "physiclaw.core.calibration.calibrate.validate_calibration",
        return_value=[
            {"passed": False, "error": 999},
            {"passed": False, "error": 999},
        ],
    )
    pl = MagicMock()
    pl.rig.calibration = _ready_bundle()

    with caplog.at_level(logging.ERROR, logger="physiclaw.core.server.warm_start"):
        out = warm_start._sanity(pl, MagicMock(), MagicMock())

    assert out is False
    assert any("no taps registered" in r.getMessage() for r in caplog.records)


def test_sanity_fails_when_taps_received_but_off(
    mocker,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    mocker.patch(
        "physiclaw.core.calibration.calibrate.validate_calibration",
        return_value=[
            {"passed": False, "error": 12.5},
            {"passed": True, "error": 1.0},
        ],
    )
    pl = MagicMock()
    pl.rig.calibration = _ready_bundle()

    with caplog.at_level(logging.ERROR, logger="physiclaw.core.server.warm_start"):
        out = warm_start._sanity(pl, MagicMock(), MagicMock())

    assert out is False
    assert any("looks stale" in r.getMessage() for r in caplog.records)


def test_sanity_restores_bridge_mode_on_validate_exception(mocker) -> None:
    mocker.patch(
        "physiclaw.core.calibration.calibrate.validate_calibration",
        side_effect=RuntimeError("hardware down"),
    )
    pl = MagicMock()
    pl.rig.calibration = _ready_bundle()
    phone = MagicMock()

    with pytest.raises(RuntimeError):
        warm_start._sanity(pl, MagicMock(), phone)

    # Phone goes calibrate → bridge even on exception.
    assert phone.set_mode.call_args_list[-1].args == ("bridge",)
