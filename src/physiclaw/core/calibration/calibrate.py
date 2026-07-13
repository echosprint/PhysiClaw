"""Facade over the calibration pipeline — split along its tested seams.

This module used to hold all seven calibration steps in one ~1100-line
file; the steps now live in focused modules and this facade re-exports
every previously-public name (plus the private helpers external tests
historically patched) so existing import sites — ``handler.py``'s
top-level import, its lazy trace-edge import, ``warm_start.py``'s lazy
``validate_calibration`` import — keep working unchanged:

- ``_common.py``      — shared tap cycle (``_tap_once``, ``_tap_and_read``,
                        ``CAL_STRIKE_DURATION``) + the canonical 15-point
                        ``grid_positions`` generator
- ``viewport.py``     — pre-cal screenshot mapping (``measure_viewport_shift``
                        and its cache/decode helpers)
- ``camera_frame.py`` — camera setup check + rotation from UP/RIGHT markers
                        (``calibrate_camera_frame``)
- ``arm_cal.py``      — probe triangle + grid → screen↔arm affine and tilt
                        (``calibrate_arm``)
- ``camera_map.py``   — 15 red dots → screen↔camera affine
                        (``compute_camera_mapping``) with fit gates
- ``validate.py``     — full-chain validation (``validate_calibration``) and
                        the visual ``trace_screen_edge``
- ``at_verify.py``    — AssistiveTouch gesture verification
                        (``verify_assistive_touch``)

Note: attributes are looked up in the modules above, not here — patch a
name where it is *used* (e.g. ``arm_cal._tap_and_read``), not on this
facade, unless the code under test imports it lazily from this module.
"""

from physiclaw.core.calibration._common import (
    CAL_STRIKE_DURATION,
    _tap_and_read,
    _tap_once,
    grid_positions,
)
from physiclaw.core.calibration.arm_cal import (
    PROBE_D,
    TILT_ALIGNED_THRESHOLD,
    _tilt_from_affine,
    calibrate_arm,
)
from physiclaw.core.calibration.at_verify import verify_assistive_touch
from physiclaw.core.calibration.camera_frame import (
    _pick_rotation_from_markers,
    calibrate_camera_frame,
)
from physiclaw.core.calibration.camera_map import (
    GRID_FIT_MAX_RESIDUAL,
    GRID_FIT_MIN_INLIERS,
    SCREEN_POLY_MARGIN_FRAC,
    _detect_screen_region,
    _fit_grid_mapping,
    _mask_outside,
    compute_camera_mapping,
)
from physiclaw.core.calibration.validate import (
    DOT_MATCH_MAX_DIST_FRAC,
    DOT_OFFSCREEN_HI,
    DOT_OFFSCREEN_LO,
    trace_screen_edge,
    validate_calibration,
)
from physiclaw.core.calibration.viewport import (
    VIEWPORT_CACHE_STEM,
    _decode_screenshot_square,
    _find_viewport_cache,
    measure_viewport_shift,
)

__all__ = [
    "CAL_STRIKE_DURATION",
    "DOT_MATCH_MAX_DIST_FRAC",
    "DOT_OFFSCREEN_HI",
    "DOT_OFFSCREEN_LO",
    "GRID_FIT_MAX_RESIDUAL",
    "GRID_FIT_MIN_INLIERS",
    "PROBE_D",
    "SCREEN_POLY_MARGIN_FRAC",
    "TILT_ALIGNED_THRESHOLD",
    "VIEWPORT_CACHE_STEM",
    "_decode_screenshot_square",
    "_detect_screen_region",
    "_find_viewport_cache",
    "_fit_grid_mapping",
    "_mask_outside",
    "_pick_rotation_from_markers",
    "_tap_and_read",
    "_tap_once",
    "_tilt_from_affine",
    "calibrate_arm",
    "calibrate_camera_frame",
    "compute_camera_mapping",
    "grid_positions",
    "measure_viewport_shift",
    "trace_screen_edge",
    "validate_calibration",
    "verify_assistive_touch",
]
