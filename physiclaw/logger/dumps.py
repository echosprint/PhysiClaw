"""Optional debug dumps to data/ subdirectories.

Each helper is a no-op unless its env var is set (typically via a CLI
flag on the ``physiclaw`` server):

| env var                    | CLI flag            | dir                |
| -------------------------- | ------------------- | ------------------ |
| PHYSICLAW_SAVE_TOOL_CALLS  | --save-tool-calls   | data/tool_calls/   |
| PHYSICLAW_SAVE_SNAPSHOTS   | --save-snapshots    | data/snapshots/    |
| PHYSICLAW_SAVE_SCREENSHOTS | --save-screenshots  | data/screenshots/  |

Filenames are millisecond-precision timestamps so rapid-fire calls
don't collide.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Literal

import cv2

_DATA = Path("data")
_ENSURED: set[str] = set()  # subdirs we've already mkdir'd this process


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def _mkdir(sub: str) -> Path:
    d = _DATA / sub
    if sub not in _ENSURED:
        d.mkdir(parents=True, exist_ok=True)
        _ENSURED.add(sub)
    return d


ToolKind = Literal["scan", "peek", "screenshot"]


def save_tool_call(kind: ToolKind, listing: str, jpeg: bytes | None = None) -> None:
    """Dump a processed view-tool output to data/tool_calls/."""
    if not os.environ.get("PHYSICLAW_SAVE_TOOL_CALLS"):
        return
    d = _mkdir("tool_calls")
    stamp = _stamp()
    (d / f"{stamp}_{kind}.txt").write_text(listing, encoding="utf-8")
    if jpeg is not None:
        (d / f"{stamp}_{kind}.jpg").write_bytes(jpeg)


def save_snapshot(frame) -> None:
    """Dump a raw camera frame (BGR ndarray) to data/snapshots/."""
    if not os.environ.get("PHYSICLAW_SAVE_SNAPSHOTS"):
        return
    d = _mkdir("snapshots")
    cv2.imwrite(str(d / f"{_stamp()}.jpg"), frame)


def save_screenshot(data: bytes) -> None:
    """Dump raw phone-own screenshot bytes to data/screenshots/."""
    if not os.environ.get("PHYSICLAW_SAVE_SCREENSHOTS"):
        return
    d = _mkdir("screenshots")
    (d / f"{_stamp()}.jpg").write_bytes(data)
