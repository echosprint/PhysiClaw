"""Platform-specific helpers — single source of truth for OS branching.

Imports the right backend at import time and re-exports a flat API.
Callers do ``from physiclaw.common import platform`` and call
``platform.local_hostname()`` etc.; they never check ``sys.platform``
themselves. New helpers go in ``darwin.py`` + ``windows.py`` (matching
``sys.platform`` literals) and get re-exported here.
"""

import sys

if sys.platform == "darwin":
    from . import darwin as _impl
elif sys.platform == "win32":
    from . import windows as _impl
elif sys.platform.startswith("linux"):
    from . import linux as _impl
else:
    raise RuntimeError(
        f"PhysiClaw does not support sys.platform={sys.platform!r}. "
        "Supported: 'darwin', 'win32', 'linux'."
    )

ensure_camera_permission = _impl.ensure_camera_permission
local_hostname = _impl.local_hostname
camera_exposure_tunable = _impl.camera_exposure_tunable
camera_size_cap = _impl.camera_size_cap
camera_backend = _impl.camera_backend
camera_set_auto_exposure = _impl.camera_set_auto_exposure
camera_set_manual_exposure = _impl.camera_set_manual_exposure
open_camera_aim_app = _impl.open_camera_aim_app
quit_camera_aim_app = _impl.quit_camera_aim_app
open_image_files = _impl.open_image_files
camera_denied_hint = _impl.camera_denied_hint
camera_aim_hint = _impl.camera_aim_hint
opencv_import_hint = _impl.opencv_import_hint
hardware_permission_hints = _impl.hardware_permission_hints
TRUST_PROXY_ENV = _impl.TRUST_PROXY_ENV

__all__ = [
    "ensure_camera_permission",
    "local_hostname",
    "camera_exposure_tunable",
    "camera_size_cap",
    "camera_backend",
    "camera_set_auto_exposure",
    "camera_set_manual_exposure",
    "open_camera_aim_app",
    "quit_camera_aim_app",
    "open_image_files",
    "camera_denied_hint",
    "camera_aim_hint",
    "opencv_import_hint",
    "hardware_permission_hints",
    "TRUST_PROXY_ENV",
]
