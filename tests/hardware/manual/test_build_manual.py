"""Tests for hardware/manual/build_manual.py — the pure document helpers."""

from __future__ import annotations

import pytest

from hardware.manual import build_manual
from hardware.manual.assets import InlineAssets

# ── figure_alt ────────────────────────────────────────────────────────────────


def test_figure_alt_prefers_a_pinned_alt():
    assert build_manual.figure_alt({"src": "x.svg", "alt": "the frame"}) == "the frame"


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        (
            "frame_10_extrusion_tnut_exploded_cam0_rbun.svg",
            "frame_10_extrusion_tnut exploded",
        ),
        ("belt_20_clamp_assembled_cam1.svg", "belt_20_clamp assembled"),
    ],
    ids=["with-hash-token", "without-hash-token"],
)
def test_figure_alt_derives_readable_text_from_the_src(src, expected):
    assert build_manual.figure_alt({"src": src}) == expected


# ── cover helpers ─────────────────────────────────────────────────────────────


def test_cover_title_reads_the_cover_pages_localized_title():
    pages = [{"type": "cover", "title": {"en": "Assembly Manual", "zh": "装配手册"}}]

    title = build_manual.cover_title(pages, build_manual.Ctx("zh", InlineAssets()))

    assert title == "装配手册"


def test_cover_title_without_a_cover_falls_back_to_the_default():
    title = build_manual.cover_title([], build_manual.Ctx("en", InlineAssets()))

    assert title == "PhysiClaw Assembly Manual"
