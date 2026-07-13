"""Pin for the `physiclaw.core.calibration.calibrate` re-export facade.

The step implementations moved to viewport / camera_frame / arm_cal /
camera_map / validate / at_verify (+ `_common`); their behavior is
tested next to each module. What still rides on the facade is its
import surface: `handler.py`'s top-level import list, the lazy
`trace_screen_edge` / `validate_calibration` imports (whose tests patch
the *facade* attribute, relying on the lazy `from … import` re-reading
it at call time), and any external code importing the historical path.
This pin fails loudly if a re-export is dropped or stops aliasing the
real implementation.
"""

from __future__ import annotations

import pytest

from physiclaw.core.calibration import (
    _common,
    arm_cal,
    at_verify,
    calibrate,
    camera_frame,
    camera_map,
    validate,
    viewport,
)

# name → module that owns the implementation the facade must alias.
_EXPECTED_HOME = {
    "CAL_STRIKE_DURATION": _common,
    "grid_positions": _common,
    "_tap_once": _common,
    "_tap_and_read": _common,
    "VIEWPORT_CACHE_STEM": viewport,
    "_find_viewport_cache": viewport,
    "_decode_screenshot_square": viewport,
    "measure_viewport_shift": viewport,
    "_pick_rotation_from_markers": camera_frame,
    "calibrate_camera_frame": camera_frame,
    "PROBE_D": arm_cal,
    "TILT_ALIGNED_THRESHOLD": arm_cal,
    "_tilt_from_affine": arm_cal,
    "calibrate_arm": arm_cal,
    "GRID_FIT_MIN_INLIERS": camera_map,
    "GRID_FIT_MAX_RESIDUAL": camera_map,
    "SCREEN_POLY_MARGIN_FRAC": camera_map,
    "_detect_screen_region": camera_map,
    "_mask_outside": camera_map,
    "_fit_grid_mapping": camera_map,
    "compute_camera_mapping": camera_map,
    "DOT_MATCH_MAX_DIST_FRAC": validate,
    "DOT_OFFSCREEN_LO": validate,
    "DOT_OFFSCREEN_HI": validate,
    "validate_calibration": validate,
    "trace_screen_edge": validate,
    "verify_assistive_touch": at_verify,
}


def test_facade_all_matches_expected_surface() -> None:
    assert sorted(calibrate.__all__) == sorted(_EXPECTED_HOME)


@pytest.mark.parametrize("name", sorted(_EXPECTED_HOME))
def test_facade_reexports_alias_the_implementation(name: str) -> None:
    assert getattr(calibrate, name) is getattr(_EXPECTED_HOME[name], name)
