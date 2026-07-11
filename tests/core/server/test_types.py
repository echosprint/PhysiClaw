"""Tests for `physiclaw.core.server.types` — Pydantic-Annotated tool types.

`Bbox` and `ClipboardText` are `Annotated` aliases that FastMCP
propagates into JSONSchema. The constraints (ge/le/min_length/
max_length) are validated at the tool boundary by FastMCP's Pydantic
adapter. Per-element constraints (left < right, top < bottom) live in
`agent/engine/validator.py` and `core/vision/util.validate_bbox` —
those are tested separately.
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from physiclaw.core.server.types import Bbox, ClipboardText


# ---------- Bbox ----------


def _bbox_adapter() -> TypeAdapter:
    return TypeAdapter(Bbox)


def test_bbox_accepts_valid_unit_box() -> None:
    out = _bbox_adapter().validate_python([0.0, 0.0, 1.0, 1.0])

    assert out == [0.0, 0.0, 1.0, 1.0]


def test_bbox_accepts_interior_box() -> None:
    out = _bbox_adapter().validate_python([0.1, 0.2, 0.3, 0.4])

    assert out == [0.1, 0.2, 0.3, 0.4]


def test_bbox_rejects_too_few_coords() -> None:
    with pytest.raises(ValidationError):
        _bbox_adapter().validate_python([0.0, 0.0, 1.0])


def test_bbox_rejects_too_many_coords() -> None:
    with pytest.raises(ValidationError):
        _bbox_adapter().validate_python([0.0, 0.0, 1.0, 1.0, 0.5])


def test_bbox_rejects_empty_list() -> None:
    with pytest.raises(ValidationError):
        _bbox_adapter().validate_python([])


@pytest.mark.parametrize("bad", [-0.01, -1.0, 1.01, 2.0, 100.0])
def test_bbox_rejects_coord_below_zero_or_above_one(bad: float) -> None:
    with pytest.raises(ValidationError):
        _bbox_adapter().validate_python([bad, 0.0, 1.0, 1.0])


def test_bbox_rejects_non_numeric_coord() -> None:
    with pytest.raises(ValidationError):
        _bbox_adapter().validate_python(["zero", 0.0, 1.0, 1.0])


def test_bbox_rejects_string_input() -> None:
    """The whole `Bbox` must be a list, not a JSON-encoded string."""
    with pytest.raises(ValidationError):
        _bbox_adapter().validate_python("[0, 0, 1, 1]")


def test_bbox_accepts_boundary_coords() -> None:
    """Each coord ∈ [0, 1] inclusive — `ge=0.0` and `le=1.0`."""
    a = _bbox_adapter().validate_python([0.0, 0.0, 0.0, 0.0])
    b = _bbox_adapter().validate_python([1.0, 1.0, 1.0, 1.0])

    # Note: Pydantic's per-coord validation passes even for degenerate
    # boxes here — the left<right / top<bottom invariant is enforced
    # downstream by validate_bbox in the engine validator. This test
    # documents that division of responsibility.
    assert a == [0.0, 0.0, 0.0, 0.0]
    assert b == [1.0, 1.0, 1.0, 1.0]


def test_bbox_coerces_int_to_float() -> None:
    """Pydantic by default coerces compatible numerics; `0` → `0.0`."""
    out = _bbox_adapter().validate_python([0, 0, 1, 1])

    assert out == [0.0, 0.0, 1.0, 1.0]


# ---------- ClipboardText ----------


def _clip_adapter() -> TypeAdapter:
    return TypeAdapter(ClipboardText)


def test_clipboard_text_accepts_short_string() -> None:
    out = _clip_adapter().validate_python("hello")

    assert out == "hello"


def test_clipboard_text_accepts_unicode() -> None:
    out = _clip_adapter().validate_python("你好 🌍")

    assert out == "你好 🌍"


def test_clipboard_text_rejects_empty_string() -> None:
    """`min_length=1` — empty paste is almost always a bug; reject loud."""
    with pytest.raises(ValidationError):
        _clip_adapter().validate_python("")


def test_clipboard_text_rejects_dict_misuse() -> None:
    """A common LLM mistake: wrapping the value as `{"text": "..."}`.
    Pydantic must refuse to coerce a dict to a string here."""
    with pytest.raises(ValidationError):
        _clip_adapter().validate_python({"text": "hello"})


def test_clipboard_text_rejects_list() -> None:
    with pytest.raises(ValidationError):
        _clip_adapter().validate_python(["hello"])


def test_clipboard_text_rejects_over_10kb() -> None:
    """`max_length=10_000` — guards against pathological inputs."""
    too_long = "x" * 10_001

    with pytest.raises(ValidationError):
        _clip_adapter().validate_python(too_long)


def test_clipboard_text_accepts_at_max_length() -> None:
    at_cap = "x" * 10_000

    out = _clip_adapter().validate_python(at_cap)

    assert len(out) == 10_000
