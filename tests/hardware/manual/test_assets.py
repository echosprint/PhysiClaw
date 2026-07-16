"""Tests for hardware/manual/assets.py — image-asset strategies."""

from __future__ import annotations

import base64

import pytest

from hardware.manual import assets


@pytest.fixture
def svg_pool(tmp_path, monkeypatch):
    """A per-test SVG render pool; clears the module's read cache."""
    monkeypatch.setattr(assets, "SVG_DIR", tmp_path)
    assets._read_svg.cache_clear()
    (tmp_path / "frame_10_base_exploded_cam0.svg").write_text(
        "<svg>frame</svg>", encoding="utf-8"
    )
    return tmp_path


# ── inline strategy ───────────────────────────────────────────────────────────


def test_inline_figure_embeds_the_svg_as_a_data_uri(svg_pool):
    src = assets.InlineAssets().figure("frame_10_base_exploded_cam0.svg")

    prefix, b64 = src.split(",", 1)
    assert (prefix, base64.b64decode(b64)) == (
        "data:image/svg+xml;base64",
        b"<svg>frame</svg>",
    )


def test_inline_preload_is_empty():
    assert assets.InlineAssets().preload("anything") == ""


def test_inline_emit_writes_nothing(tmp_path):
    out = tmp_path / "out"
    out.mkdir()

    assets.InlineAssets().emit(out)

    assert list(out.iterdir()) == []


# ── external strategy ─────────────────────────────────────────────────────────


def test_external_figure_returns_relative_assets_url(svg_pool):
    ext = assets.ExternalAssets()

    src = ext.figure("frame_10_base_exploded_cam0.svg")

    assert (src, ext.referenced) == (
        "assets/frame_10_base_exploded_cam0.svg",
        {"frame_10_base_exploded_cam0.svg"},
    )


def test_external_preload_emits_a_link_tag():
    assert (
        assets.ExternalAssets().preload("assets/cover.svg")
        == '<link rel="preload" as="image" href="assets/cover.svg">'
    )


def test_external_emit_copies_referenced_renders_and_the_crab(svg_pool, tmp_path):
    out = tmp_path / "out"
    ext = assets.ExternalAssets()
    ext.figure("frame_10_base_exploded_cam0.svg")

    ext.emit(out)

    assert sorted(p.name for p in (out / "assets").iterdir()) == [
        "crab.svg",
        "frame_10_base_exploded_cam0.svg",
    ]


def test_external_emit_with_missing_render_raises_build_error(svg_pool, tmp_path):
    ext = assets.ExternalAssets()
    ext.figure("belt_20_clamp_assembled_cam1.svg")

    with pytest.raises(assets.BuildError, match="belt_20_clamp_assembled_cam1.svg"):
        ext.emit(tmp_path / "out")


def test_require_renders_error_names_the_stems_to_rebuild(svg_pool):
    with pytest.raises(
        assets.BuildError, match="--stems belt_20_clamp camera_40_frame"
    ):
        assets._require_renders(
            [
                "camera_40_frame_assembled_cam0_ljek.svg",
                "belt_20_clamp_exploded_cam1.svg",
            ]
        )
