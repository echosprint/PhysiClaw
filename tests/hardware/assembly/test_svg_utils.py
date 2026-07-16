"""Tests for hardware/assembly/svg_utils.py — root-tag SVG rewriting."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from hardware.assembly import svg_utils

SVG = '<svg xmlns="x" width="80mm" height="60mm" viewBox="0 0 80 60"><rect width="5"/></svg>'

finite = st.integers(-9999, 9999)


# ── strip_root_dims ───────────────────────────────────────────────────────────


def test_strip_root_dims_removes_width_and_height_from_root():
    out = svg_utils.strip_root_dims(SVG)

    assert out.startswith('<svg xmlns="x" viewBox="0 0 80 60">')


def test_strip_root_dims_leaves_nested_element_dims_alone():
    out = svg_utils.strip_root_dims(SVG)

    assert '<rect width="5"/>' in out


def test_strip_root_dims_without_svg_tag_returns_text_unchanged():
    assert svg_utils.strip_root_dims("<p>hi</p>") == "<p>hi</p>"


def test_strip_root_dims_rewrites_only_the_first_svg_tag():
    nested = '<svg width="1"><svg width="2"></svg></svg>'

    out = svg_utils.strip_root_dims(nested)

    assert out == '<svg><svg width="2"></svg></svg>'


# ── inject_non_scaling_strokes ────────────────────────────────────────────────


def test_inject_non_scaling_strokes_inserts_style_after_root_tag():
    out = svg_utils.inject_non_scaling_strokes('<svg viewBox="0 0 1 1"><g/></svg>')

    assert out.startswith(
        '<svg viewBox="0 0 1 1"><style>svg * { vector-effect: non-scaling-stroke; }'
    )


def test_inject_non_scaling_strokes_injects_exactly_once():
    out = svg_utils.inject_non_scaling_strokes('<svg><svg id="inner"/></svg>')

    assert out.count("<style>") == 1


# ── get_root_viewbox / set_root_viewbox ───────────────────────────────────────


@pytest.mark.parametrize(
    "tag",
    ['<svg viewBox="1 2 3 4">', "<svg viewBox='1 2 3 4'>"],
    ids=["double-quoted", "single-quoted"],
)
def test_get_root_viewbox_reads_either_quote_style(tag):
    assert svg_utils.get_root_viewbox(f"{tag}</svg>") == "1 2 3 4"


def test_get_root_viewbox_without_attribute_returns_none():
    assert svg_utils.get_root_viewbox("<svg></svg>") is None


def test_get_root_viewbox_without_svg_tag_returns_none():
    assert svg_utils.get_root_viewbox("plain text") is None


def test_set_root_viewbox_replaces_existing_attribute():
    out = svg_utils.set_root_viewbox('<svg viewBox="0 0 1 1"></svg>', "5 6 7 8")

    assert out == '<svg viewBox="5 6 7 8"></svg>'


def test_set_root_viewbox_inserts_when_absent():
    out = svg_utils.set_root_viewbox('<svg xmlns="x"></svg>', "5 6 7 8")

    assert out == '<svg xmlns="x" viewBox="5 6 7 8"></svg>'


def test_set_root_viewbox_inserts_before_self_closing_end():
    out = svg_utils.set_root_viewbox("<svg/>", "5 6 7 8")

    assert out == '<svg viewBox="5 6 7 8"/>'


@given(nums=st.tuples(finite, finite, finite, finite))
def test_set_then_get_root_viewbox_round_trips(nums):
    viewbox = " ".join(str(n) for n in nums)

    out = svg_utils.set_root_viewbox('<svg viewBox="0 0 1 1"></svg>', viewbox)

    assert svg_utils.get_root_viewbox(out) == viewbox


# ── validate_viewbox ──────────────────────────────────────────────────────────


def test_validate_viewbox_none_passes_through():
    assert svg_utils.validate_viewbox(None) is None


@pytest.mark.parametrize("raw", ["0 0 80 60", "  -1.5 2e3 3.25 4E-2  ", "0\t0 1 1"])
def test_validate_viewbox_accepts_four_numbers(raw):
    assert svg_utils.validate_viewbox(raw) == raw.strip()


@pytest.mark.parametrize("raw", ["0 0 80", "0 0 80 60 1", "a b c d", 42, "1,2,3,4"])
def test_validate_viewbox_rejects_non_four_number_input(raw):
    with pytest.raises(ValueError, match="four numbers"):
        svg_utils.validate_viewbox(raw)
