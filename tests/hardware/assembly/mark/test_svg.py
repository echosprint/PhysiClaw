"""Tests for hardware/assembly/mark/svg.py — snapshot SVG composition."""

from __future__ import annotations

import pytest

from hardware.assembly.mark.svg import FILL_GROUP_ID, build_shapes_svg
from hardware.assembly.svg_utils import get_root_viewbox

SRC = b'<svg viewBox="0 0 100 100"><g id="art"/></svg>'
RED = {"fill": "#ff0000", "opacity": 0.5}


def rect(x=10.0, color=RED, outlined=False) -> dict:
    return {
        "type": "rect",
        "geom": {"x": x, "y": 10.0, "w": 5.0, "h": 5.0},
        "color": color,
        "outlined": outlined,
    }


def test_build_shapes_svg_appends_group_before_closing_tag():
    out = build_shapes_svg(SRC, [rect()]).decode()

    assert (
        f'<g id="{FILL_GROUP_ID}"' in out
        and out.index(FILL_GROUP_ID) < out.rindex("</svg>")
        and out.index(FILL_GROUP_ID) > out.index('<g id="art"/>')
    )


def test_build_shapes_svg_without_closing_tag_raises():
    with pytest.raises(ValueError, match="</svg>"):
        build_shapes_svg(b"<svg>", [rect()])


def test_build_shapes_svg_applies_caller_viewbox():
    out = build_shapes_svg(SRC, [], viewbox="0 0 50 50").decode()

    assert get_root_viewbox(out) == "0 0 50 50"


def test_build_shapes_svg_expands_viewbox_to_contain_outlying_shape():
    out = build_shapes_svg(SRC, [rect(x=195.0)]).decode()

    assert get_root_viewbox(out) == "0.0000 0.0000 200.0000 100.0000"


def test_build_shapes_svg_groups_consecutive_same_style_shapes_in_one_g():
    out = build_shapes_svg(SRC, [rect(), rect(x=20.0)]).decode()

    assert out.count('<g fill="#ff0000"') == 1


def test_build_shapes_svg_starts_new_group_on_style_change():
    blue = {"fill": "#0000ff", "opacity": 0.5}

    out = build_shapes_svg(SRC, [rect(), rect(color=blue)]).decode()

    assert '<g fill="#0000ff"' in out


def test_build_shapes_svg_outlined_shape_renders_stroke_only():
    out = build_shapes_svg(SRC, [rect(outlined=True)]).decode()

    assert 'stroke="#ff0000"' in out and 'fill="none"' in out


def test_build_shapes_svg_zero_length_arrow_adds_no_geometry():
    arrow = {
        "type": "arrow",
        "geom": {"x1": 5.0, "y1": 5.0, "x2": 5.0, "y2": 5.0},
        "color": RED,
        "outlined": False,
    }

    out = build_shapes_svg(SRC, [arrow]).decode()

    assert "<line" not in out and get_root_viewbox(out) == "0 0 100 100"


def test_build_shapes_svg_unknown_shape_type_raises():
    with pytest.raises(ValueError, match="unknown shape type"):
        build_shapes_svg(SRC, [{"type": "star", "geom": {}}])
