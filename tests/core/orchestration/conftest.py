"""Shared hardware doubles for the orchestration suite.

Factory fixtures, not instances: several tests need two doubles of the
same device (the connect_* tests replace an existing one), so each
fixture returns the factory itself. `wire_rig` wires any HardwareRig —
a bare one or a facade's `pc.rig` — with spec'd doubles and a fake
calibration, so the three test files share one definition of the
mock-hardware contract.
"""

from __future__ import annotations

from unittest.mock import MagicMock, create_autospec

import numpy as np
import pytest

from physiclaw.core.bridge.state import BridgeState
from physiclaw.core.hardware.arm import StylusArm
from physiclaw.core.hardware.camera import Camera


def _bridge_double():
    """Autospec'd BridgeState double — nonexistent attribute access or a
    wrong-signature call fails loudly instead of silently passing."""
    return create_autospec(BridgeState, instance=True)


def _arm_double():
    """Spec'd StylusArm double — nonexistent attribute access fails loudly."""
    return MagicMock(spec=StylusArm)


def _cam_double():
    """Spec'd Camera double. ``rotation`` is set in ``Camera.__init__``, so
    it's not in the class spec — seed it here so every double exposes the
    real attribute (at its real initial value)."""
    cam = MagicMock(spec=Camera)
    cam.rotation = -1
    return cam


def _at_double():
    """AssistiveTouch double: calibrated, parked clear of every test bbox."""
    at = MagicMock()
    at.ready = True
    at.at_screen = (0.05, 0.1)
    at.overlaps_at.return_value = False
    at.swipe_crosses_at.return_value = False
    return at


def _diag_10_20_affine() -> np.ndarray:
    """The suite's stock pct→GRBL affine: diag(10, 20). Assertions like
    ``rapid_to(-1.0, -1.0)`` for park (-0.1, -0.05) derive from it."""
    return np.array(
        [
            [10.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _fake_transforms(*, swipe_end=(0.5, 0.6)):
    t = MagicMock()
    t.bbox_center_pct.side_effect = lambda bbox: (
        (bbox[0] + bbox[2]) / 2,
        (bbox[1] + bbox[3]) / 2,
    )
    t.pct_to_grbl_mm.side_effect = lambda x, y: (x * 10, y * 20)
    t.swipe_end_pct.return_value = swipe_end
    t.pct_to_grbl = _diag_10_20_affine()
    return t


def _wire_rig(rig, *, transforms=None):
    """Wire spec'd arm/cam doubles + a fake calibration into `rig`."""
    rig._arm = _arm_double()
    rig._arm.MOVE_DIRECTIONS = {"up": "x"}
    rig._arm.SWIPE_SPEEDS = {"slow": 100, "medium": 500, "fast": 1500}
    rig._cam = _cam_double()
    rig.calibration = MagicMock()
    rig.calibration.transforms_ready = True
    rig.calibration.transforms.return_value = transforms or _fake_transforms()
    rig.calibration.summary.return_value = {"step1": "OK"}
    rig.calibration.cam_rotation = None
    rig.calibration.pct_to_grbl = None
    rig.calibration.pct_to_grbl_mm.return_value = (5.0, 6.0)


@pytest.fixture
def bridge_double():
    return _bridge_double


@pytest.fixture
def arm_double():
    return _arm_double


@pytest.fixture
def cam_double():
    return _cam_double


@pytest.fixture
def at_double():
    return _at_double


@pytest.fixture
def diag_10_20_affine():
    return _diag_10_20_affine()


@pytest.fixture
def fake_transforms():
    return _fake_transforms


@pytest.fixture
def wire_rig():
    return _wire_rig
