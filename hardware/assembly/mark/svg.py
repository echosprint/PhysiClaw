"""Build the snapshot SVG: original line art + a layer of marked shapes
(polygon / rectangle / circle / ellipse / line / arrow) + optional
cropped viewBox. The original file is read once as bytes; the output
is returned as bytes so the caller controls where it lands on disk."""

from __future__ import annotations

import math
import re
from typing import Iterable, List, Literal, TypedDict

from hardware.assembly.svg_utils import get_root_viewbox, set_root_viewbox

FILL_GROUP_ID = "manual-fill"


class Color(TypedDict):
    """Polygon style — one of the six palette presets in the UI."""

    fill: str  # e.g. "#1e88ff"
    opacity: float  # 0.0 .. 1.0


DEFAULT_COLOR: Color = {"fill": "#1e88ff", "opacity": 0.35}

ShapeType = Literal["polygon", "rect", "circle", "ellipse", "line", "arrow"]

# Stroke width (SVG units) for outlined shapes + line / arrow.
STROKE_WIDTH = 2

# Outlined rendering is the natural form for these — they have no fill
# area — so the filled / outlined toggle has no effect on them.
_LINE_LIKE = ("line", "arrow")

# Arrow head shape: half-angle (radians) from the line direction, and
# cap on head length so it stays proportional to the line on short
# arrows and doesn't grow unbounded on long ones.
_ARROW_HEAD_HALF_ANGLE = math.radians(20)
_ARROW_HEAD_LEN_MAX = 12.0
_ARROW_HEAD_FRACTION = 0.25


def _arrow_geometry(x1: float, y1: float, x2: float, y2: float):
    """Return ``(stem_end, head_tri_points)`` where ``stem_end`` is the
    ``(x, y)`` where the line stem terminates (so it doesn't poke past
    the head) and ``head_tri_points`` is the three vertices of the
    filled triangle head ``[(tip), (base1), (base2)]``. Returns
    ``None`` for a zero-length arrow.

    The stem terminates at the triangle base's projection on the line
    (``hl * cos(half_angle)`` from the tip) — NOT at distance ``hl`` —
    so the line meets the back edge of the triangle without a gap."""
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    if L == 0:
        return None
    ang = math.atan2(dy, dx)
    hl = min(L * _ARROW_HEAD_FRACTION, _ARROW_HEAD_LEN_MAX)
    back = hl * math.cos(_ARROW_HEAD_HALF_ANGLE)
    stem_end = (x2 - back * math.cos(ang), y2 - back * math.sin(ang))
    base1 = (
        x2 - hl * math.cos(ang - _ARROW_HEAD_HALF_ANGLE),
        y2 - hl * math.sin(ang - _ARROW_HEAD_HALF_ANGLE),
    )
    base2 = (
        x2 - hl * math.cos(ang + _ARROW_HEAD_HALF_ANGLE),
        y2 - hl * math.sin(ang + _ARROW_HEAD_HALF_ANGLE),
    )
    return stem_end, [(x2, y2), base1, base2]


def _shape_bbox(s: dict) -> tuple[float, float, float, float] | None:
    """Tight bounding box of one shape in source-SVG units, or ``None``
    when the shape has no spatial extent (a zero-length arrow). Stroke
    widths are pixel-fixed (``non-scaling-stroke``) so they don't
    contribute to the bbox."""
    t = s["type"]
    c = s["geom"]
    if t == "polygon":
        xs, ys = zip(*c["points"])
        return min(xs), min(ys), max(xs), max(ys)
    if t == "rect":
        return c["x"], c["y"], c["x"] + c["w"], c["y"] + c["h"]
    if t == "circle":
        return c["cx"] - c["r"], c["cy"] - c["r"], c["cx"] + c["r"], c["cy"] + c["r"]
    if t == "ellipse":
        return (
            c["cx"] - c["rx"],
            c["cy"] - c["ry"],
            c["cx"] + c["rx"],
            c["cy"] + c["ry"],
        )
    if t == "line":
        return (
            min(c["x1"], c["x2"]),
            min(c["y1"], c["y2"]),
            max(c["x1"], c["x2"]),
            max(c["y1"], c["y2"]),
        )
    if t == "arrow":
        geo = _arrow_geometry(c["x1"], c["y1"], c["x2"], c["y2"])
        if geo is None:
            return None
        _, tri = geo
        xs = [c["x1"], c["x2"], *(p[0] for p in tri)]
        ys = [c["y1"], c["y2"], *(p[1] for p in tri)]
        return min(xs), min(ys), max(xs), max(ys)
    raise ValueError(f"unknown shape type {t!r}")


def _shapes_bbox(shapes) -> tuple[float, float, float, float] | None:
    """Union bbox of every shape, or ``None`` if no shape has extent."""
    bboxes = [b for b in (_shape_bbox(s) for s in shapes) if b is not None]
    if not bboxes:
        return None
    return (
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    )


def _expand_viewbox(vb: str | None, bb: tuple[float, float, float, float]) -> str:
    """Return a viewBox that contains both ``vb`` (if not None) and the
    shapes bounding box ``bb``. Without ``vb`` the bbox is the new
    viewBox outright."""
    sx0, sy0, sx1, sy1 = bb
    if vb is None:
        x0, y0, x1, y1 = sx0, sy0, sx1, sy1
    else:
        cx0, cy0, cw, ch = (float(p) for p in vb.split())
        x0, y0 = min(cx0, sx0), min(cy0, sy0)
        x1, y1 = max(cx0 + cw, sx1), max(cy0 + ch, sy1)
    return f"{x0:.4f} {y0:.4f} {x1 - x0:.4f} {y1 - y0:.4f}"


# Last ``</svg>`` closer in the file. The marked-shapes group is
# inserted just before it; trailing comments or nested ``</svg>``
# inside CDATA can't trick us into inserting too early.
_SVG_CLOSE_RE = re.compile(r"</\s*svg\s*>", re.IGNORECASE)


def _style_key(s: dict) -> tuple:
    """Group key: shapes sharing this tuple can sit inside one wrapper
    ``<g>`` and skip declaring their own style attrs."""
    color = s.get("color") or DEFAULT_COLOR
    outlined = bool(s.get("outlined")) or s["type"] in _LINE_LIKE
    return (color["fill"], color["opacity"], outlined)


def _group_attrs(key: tuple) -> str:
    """Style attrs to put on the wrapping ``<g>`` for a style run.
    Includes constants (``stroke-width``, ``vector-effect``) so per-
    element tags stay attr-free."""
    fill, opacity, outlined = key
    if outlined:
        return (
            f'stroke="{fill}" stroke-opacity="{opacity}" '
            f'stroke-width="{STROKE_WIDTH}" vector-effect="non-scaling-stroke" '
            f'fill="none"'
        )
    return f'fill="{fill}" fill-opacity="{opacity}" stroke="none"'


def _shape_tag(s: dict) -> str:
    """Bare SVG tag for one shape (style lives on the wrapping ``<g>``)."""
    t = s["type"]
    c = s["geom"]
    if t == "polygon":
        pts = " ".join(f"{x:.4f},{y:.4f}" for x, y in c["points"])
        return f'<polygon points="{pts}"/>'
    if t == "rect":
        rx = c.get("rx", 0)
        return (
            f'<rect x="{c["x"]:.4f}" y="{c["y"]:.4f}" '
            f'width="{c["w"]:.4f}" height="{c["h"]:.4f}" '
            f'rx="{rx:.4f}" ry="{rx:.4f}"/>'
        )
    if t == "circle":
        return f'<circle cx="{c["cx"]:.4f}" cy="{c["cy"]:.4f}" r="{c["r"]:.4f}"/>'
    if t == "ellipse":
        return (
            f'<ellipse cx="{c["cx"]:.4f}" cy="{c["cy"]:.4f}" '
            f'rx="{c["rx"]:.4f}" ry="{c["ry"]:.4f}"/>'
        )
    if t == "line":
        return (
            f'<line x1="{c["x1"]:.4f}" y1="{c["y1"]:.4f}" '
            f'x2="{c["x2"]:.4f}" y2="{c["y2"]:.4f}"/>'
        )
    if t == "arrow":
        # Stem (line) inherits stroke from the wrapping <g>; head
        # (polygon) overrides ``fill="none"`` with the same colour so
        # the filled triangle matches the stem.
        geo = _arrow_geometry(c["x1"], c["y1"], c["x2"], c["y2"])
        if geo is None:
            return ""
        (sx, sy), tri = geo
        color = s.get("color") or DEFAULT_COLOR
        pts = " ".join(f"{x:.4f},{y:.4f}" for x, y in tri)
        return (
            f'<line x1="{c["x1"]:.4f}" y1="{c["y1"]:.4f}" '
            f'x2="{sx:.4f}" y2="{sy:.4f}"/>\n      '
            f'<polygon points="{pts}" fill="{color["fill"]}" '
            f'fill-opacity="{color["opacity"]}" stroke="none"/>'
        )
    raise ValueError(f"unknown shape type {t!r}")


def build_shapes_svg(
    original: bytes,
    shapes: Iterable[dict],
    viewbox: str | None = None,
) -> bytes:
    """Return the original SVG with a marked-shapes group appended just
    before the closing ``</svg>``. ``viewbox`` rewrites the root
    ``<svg>``'s ``viewBox`` when given. Each shape is a dict with a
    ``type`` discriminator, a ``geom`` dict with type-specific
    coordinates, a ``color``, and an ``outlined`` flag (ignored for
    line / arrow).

    The viewBox is always expanded to include every drawn shape — a
    shape placed outside the current frame would otherwise be clipped
    when the snapshot is embedded as ``<img>``."""
    text = original.decode("utf-8")
    matches = list(_SVG_CLOSE_RE.finditer(text))
    if not matches:
        raise ValueError("input does not contain </svg>")
    insert_at = matches[-1].start()

    shapes = list(shapes)

    shapes_bb = _shapes_bbox(shapes)
    if shapes_bb is not None:
        viewbox = _expand_viewbox(viewbox or get_root_viewbox(text), shapes_bb)

    lines: List[str] = [
        "  <!-- manual-fill: appended by hardware/assembly/mark -->",
        f'  <g id="{FILL_GROUP_ID}" class="{FILL_GROUP_ID}">',
    ]

    # Run-length group consecutive shapes by style so the wrapping
    # ``<g>`` declares fill / stroke / etc. once for a run. Runs (not a
    # dict) so the original draw order — and thus z-order — is preserved.
    run_key: tuple | None = None
    for s in shapes:
        key = _style_key(s)
        if key != run_key:
            if run_key is not None:
                lines.append("    </g>")
            lines.append(f"    <g {_group_attrs(key)}>")
            run_key = key
        lines.append(f"      {_shape_tag(s)}")
    if run_key is not None:
        lines.append("    </g>")
    lines.append("  </g>")
    block = "\n" + "\n".join(lines) + "\n"

    text = text[:insert_at] + block + text[insert_at:]
    if viewbox is not None:
        text = set_root_viewbox(text, viewbox)
    return text.encode("utf-8")
