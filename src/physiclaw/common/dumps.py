"""Optional debug dumps to the user data dir.

Each helper is a no-op unless its env var is set (typically via a CLI
flag on ``physiclaw server``):

| env var                    | CLI flag            | subdir         |
| -------------------------- | ------------------- | -------------- |
| PHYSICLAW_SAVE_TOOL_CALLS  | --save-tool-calls   | tool_calls/    |
| PHYSICLAW_SAVE_SNAPSHOTS   | --save-snapshots    | snapshots/     |
| PHYSICLAW_SAVE_SCREENSHOTS | --save-screenshots  | screenshots/   |
| PHYSICLAW_SAVE_RAW_CAMERA  | --save-raw-camera   | raw_camera/    |

Filenames are millisecond-precision timestamps so rapid-fire calls
don't collide.
"""

import os
from datetime import datetime
from pathlib import Path

from physiclaw.common import paths
from physiclaw.common.text import write_text

_ENSURED: set[Path] = set()


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def _mkdir(d: Path) -> Path:
    if d not in _ENSURED:
        d.mkdir(parents=True, exist_ok=True)
        _ENSURED.add(d)
    return d


# The view-producing tool being dumped: "peek", "screenshot", or any
# gesture name — every gesture attaches a fused post-action view now.
ToolKind = str


def save_tool_call(kind: ToolKind, listing: str, jpeg: bytes | None = None) -> None:
    if not os.environ.get("PHYSICLAW_SAVE_TOOL_CALLS"):
        return
    d = _mkdir(paths.tool_calls_dir())
    stamp = _stamp()
    write_text(d / f"{stamp}_{kind}.txt", listing)
    if jpeg is not None:
        (d / f"{stamp}_{kind}.jpg").write_bytes(jpeg)


def save_snapshot(frame) -> None:
    if not os.environ.get("PHYSICLAW_SAVE_SNAPSHOTS"):
        return
    import cv2

    d = _mkdir(paths.snapshots_dir())
    cv2.imwrite(str(d / f"{_stamp()}.jpg"), frame)


def save_screenshot(data: bytes) -> None:
    if not os.environ.get("PHYSICLAW_SAVE_SCREENSHOTS"):
        return
    d = _mkdir(paths.screenshots_dir())
    (d / f"{_stamp()}.jpg").write_bytes(data)


def save_raw_camera(frame) -> None:
    if not os.environ.get("PHYSICLAW_SAVE_RAW_CAMERA"):
        return
    import cv2

    d = _mkdir(paths.raw_camera_dir())
    cv2.imwrite(str(d / f"{_stamp()}.jpg"), frame)
