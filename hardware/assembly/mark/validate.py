"""Validation + snapping for browser-supplied input."""

from __future__ import annotations

import re
from typing import List, Tuple

from hardware.assembly.mark.svg import Color, DEFAULT_COLOR


_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
_VALID_TYPES = {"polygon", "rect", "circle", "ellipse", "line", "arrow"}


def vertex_from_click(x: float, y: float) -> Tuple[float, float]:
    """Map a raw click to a polygon vertex. Identity today; this is the
    isolated hook for future snapping (nearest edge / vertex)."""
    return (x, y)


def validate_color(raw) -> Color | None:
    """Coerce ``{fill, opacity}`` into a ``Color`` dict, or ``None`` if
    ``raw`` is missing. Raises ``ValueError`` on malformed input."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("color must be an object {fill, opacity}")
    fill = raw.get("fill")
    opacity = raw.get("opacity")
    if not isinstance(fill, str) or not _HEX_RE.match(fill):
        raise ValueError("color.fill must be a #rrggbb hex string")
    if not isinstance(opacity, (int, float)) or not (0.0 <= opacity <= 1.0):
        raise ValueError("color.opacity must be a number in [0, 1]")
    return {"fill": fill, "opacity": float(opacity)}


def _num(raw, name: str) -> float:
    if not isinstance(raw, (int, float)):
        raise ValueError(f"{name} must be a number")
    return float(raw)


def _validate_polygon(c: dict) -> dict | None:
    points = c.get("points")
    if not isinstance(points, list) or len(points) < 3:
        return None
    verts: List[Tuple[float, float]] = []
    for pt in points:
        if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
            raise ValueError("polygon vertex must be [x, y]")
        verts.append(vertex_from_click(float(pt[0]), float(pt[1])))
    return {"points": verts}


def _validate_rect(c: dict) -> dict | None:
    w, h = _num(c.get("w"), "w"), _num(c.get("h"), "h")
    if w <= 0 or h <= 0:
        return None
    return {
        "x": _num(c.get("x"), "x"),
        "y": _num(c.get("y"), "y"),
        "w": w,
        "h": h,
        "rx": max(0.0, _num(c.get("rx", 0), "rx")),
    }


def _validate_circle(c: dict) -> dict | None:
    r = _num(c.get("r"), "r")
    if r <= 0:
        return None
    return {
        "cx": _num(c.get("cx"), "cx"),
        "cy": _num(c.get("cy"), "cy"),
        "r": r,
    }


def _validate_ellipse(c: dict) -> dict | None:
    rx = _num(c.get("rx"), "rx")
    ry = _num(c.get("ry"), "ry")
    if rx <= 0 or ry <= 0:
        return None
    return {
        "cx": _num(c.get("cx"), "cx"),
        "cy": _num(c.get("cy"), "cy"),
        "rx": rx,
        "ry": ry,
    }


def _validate_line_like(c: dict) -> dict | None:
    """Shared body for ``line`` and ``arrow``."""
    x1, y1 = _num(c.get("x1"), "x1"), _num(c.get("y1"), "y1")
    x2, y2 = _num(c.get("x2"), "x2"), _num(c.get("y2"), "y2")
    if (x1, y1) == (x2, y2):
        return None
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


_DISPATCH = {
    "polygon": _validate_polygon,
    "rect": _validate_rect,
    "circle": _validate_circle,
    "ellipse": _validate_ellipse,
    "line": _validate_line_like,
    "arrow": _validate_line_like,
}


def validate_shapes(raw) -> List[dict]:
    """Return shapes as ``[{type, geom, color, outlined}, ...]``.

    Each shape carries its own ``color`` (locked at draw time in the
    UI) and ``outlined`` flag, plus a ``geom`` dict holding the
    type-specific geometry (``{points}`` for polygon, ``{x, y, w, h,
    rx}`` for rect, ``{cx, cy, r}`` for circle, etc.). Degenerate
    shapes (zero-size rects, coincident line endpoints, polygons < 3
    vertices) are dropped silently; malformed shapes raise
    ``ValueError``."""
    if not isinstance(raw, list):
        raise ValueError("shapes must be a list")
    out: List[dict] = []
    for s in raw:
        if not isinstance(s, dict):
            raise ValueError("each shape must be an object")
        t = s.get("type")
        if t not in _VALID_TYPES:
            raise ValueError(f"unknown shape type {t!r}")
        geom_raw = s.get("geom")
        if not isinstance(geom_raw, dict):
            raise ValueError(f"{t}.geom must be an object")
        geom = _DISPATCH[t](geom_raw)
        if geom is None:
            continue
        out.append(
            {
                "type": t,
                "geom": geom,
                "color": validate_color(s.get("color")) or {**DEFAULT_COLOR},
                "outlined": bool(s.get("outlined", False)),
            }
        )
    return out
