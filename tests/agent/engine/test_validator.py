"""Tests for `physiclaw.agent.engine.validator`.

Validates JSONSchema-shaped tool arguments coming back from an LLM.
Coverage targets:

  - top-level: arguments must be a JSON object; required fields present
  - bbox special-case: 5 distinct error families, identical to
    `core/vision/util.validate_bbox` (the validator runs first)
  - per-property: enum, number bounds, string lengths, type matching
    (including bool ≠ integer/number), type-list, unknown types
  - array: per-item recursion, minItems
  - object: nested recursion via re-entering `validate_arguments`
  - Hypothesis: any dict against an empty schema never raises

All `pytest.raises(match=...)` regexes are `^`-anchored so a mutmut
`XX…XX` wrap on the message would no longer substring-match.

Accepted equivalent mutants (cannot be killed by behavior tests):

  - `matched = False` ↔ `matched = None` initialization — both falsy,
    `if not matched:` evaluates identically.
  - Unknown-type `break` ↔ `continue` after `matched = True` — the
    post-loop `if not matched:` is False either way; the loop only
    ever sets `matched=True`, never back to False.
  - `t == "boolean"` ↔ `t == "XXbooleanXX"` — when the boolean arm is
    skipped, control falls through to the generic
    `isinstance(value, py_type)` check with `py_type = bool`, giving
    the same matched/not-matched outcome.
  - `break` ↔ `continue` immediately after `matched = True` — same
    rationale; matched cannot regress.
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from physiclaw.agent.engine.validator import ValidationError, validate_arguments


# ---------- top-level shape ----------


@pytest.mark.parametrize(
    "bad",
    ["a string", ["list", "item"], None, 42, True, 3.14],
)
def test_top_level_non_dict_raises(bad: Any) -> None:
    with pytest.raises(ValidationError, match=r"^arguments must be a JSON object"):
        validate_arguments(bad, {"type": "object", "properties": {}})


def test_top_level_empty_dict_against_no_required_passes() -> None:
    validate_arguments({}, {"type": "object", "properties": {}})


def test_top_level_missing_required_field_raises_with_field_name() -> None:
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
    }

    with pytest.raises(ValidationError, match=r"^missing required argument\(s\): a$"):
        validate_arguments({}, schema)


def test_top_level_multiple_missing_required_fields_listed_sorted() -> None:
    schema = {
        "type": "object",
        "properties": {"a": {}, "b": {}, "c": {}},
        "required": ["b", "a", "c"],
    }

    with pytest.raises(
        ValidationError, match=r"^missing required argument\(s\): a, b, c$"
    ):
        validate_arguments({}, schema)


def test_unknown_property_is_tolerated_when_no_schema_match() -> None:
    schema = {"type": "object", "properties": {"known": {"type": "string"}}}

    validate_arguments({"unknown_key": 42}, schema)


def test_unknown_property_does_not_short_circuit_later_known_validation() -> None:
    # Order matters: an unknown key first must `continue`, not `break`,
    # so the later known key still gets validated.
    schema = {
        "type": "object",
        "properties": {"known": {"type": "string"}},
    }

    with pytest.raises(ValidationError, match=r"^known: expected string, got int"):
        validate_arguments({"unknown": 1, "known": 42}, schema)


# ---------- bbox special path (mirrors validate_bbox in core/vision/util.py) ----------


@pytest.mark.parametrize(
    "bbox",
    [
        "not a list",
        [0.0, 0.0, 1.0],  # too short
        [0.0, 0.0, 1.0, 1.0, 0.5],  # too long
        [],  # empty
        {"l": 0, "t": 0, "r": 1, "b": 1},  # dict, not list
    ],
)
def test_bbox_wrong_shape_raises_with_anchored_message(bbox: Any) -> None:
    with pytest.raises(
        ValidationError,
        match=r"^bbox: must be \[left, top, right, bottom\]; got",
    ):
        validate_arguments({"bbox": bbox}, {"type": "object", "properties": {}})


@pytest.mark.parametrize(
    "bbox",
    [
        ["x", 0.5, 0.6, 0.7],
        [0.1, None, 0.6, 0.7],
        [0.1, 0.2, "y", 0.7],
        [0.1, 0.2, 0.3, [0.4]],
    ],
)
def test_bbox_non_number_coord_raises_with_anchored_message(bbox: list[Any]) -> None:
    with pytest.raises(ValidationError, match=r"^bbox: each coord must be a number;"):
        validate_arguments({"bbox": bbox}, {"type": "object", "properties": {}})


@pytest.mark.parametrize(
    "bbox",
    [
        [-0.1, 0.0, 0.5, 0.5],
        [0.0, -0.1, 0.5, 0.5],
        [0.0, 0.0, 1.1, 0.5],
        [0.0, 0.0, 0.5, 1.1],
    ],
)
def test_bbox_coord_out_of_unit_range_raises(bbox: list[float]) -> None:
    with pytest.raises(
        ValidationError, match=r"^bbox: each coord must be in \[0, 1\];"
    ):
        validate_arguments({"bbox": bbox}, {"type": "object", "properties": {}})


@pytest.mark.parametrize(
    "bbox",
    [
        [0.5, 0.0, 0.4, 1.0],  # left > right
        [0.5, 0.0, 0.5, 1.0],  # left == right
        [0.0, 0.5, 1.0, 0.4],  # top > bottom
        [0.0, 0.5, 1.0, 0.5],  # top == bottom
    ],
)
def test_bbox_inverted_or_degenerate_raises(bbox: list[float]) -> None:
    with pytest.raises(ValidationError, match=r"^bbox: left < right, top < bottom;"):
        validate_arguments({"bbox": bbox}, {"type": "object", "properties": {}})


def test_bbox_valid_passes() -> None:
    validate_arguments(
        {"bbox": [0.0, 0.0, 1.0, 1.0]},
        {"type": "object", "properties": {}},
    )


def test_bbox_tuple_form_also_accepted() -> None:
    validate_arguments(
        {"bbox": (0.1, 0.2, 0.3, 0.4)},
        {"type": "object", "properties": {}},
    )


def test_bbox_absent_skips_special_check() -> None:
    # No "bbox" key — special-case is bypassed entirely; ordinary schema
    # rules still apply.
    validate_arguments({"x": 1}, {"type": "object", "properties": {}})


# ---------- enum ----------


def test_enum_value_not_in_list_raises_with_offending_value() -> None:
    schema = {"type": "object", "properties": {"mode": {"enum": ["a", "b", "c"]}}}

    with pytest.raises(ValidationError, match=r"^mode: value 'd' is not one of"):
        validate_arguments({"mode": "d"}, schema)


def test_enum_value_in_list_passes() -> None:
    schema = {"type": "object", "properties": {"mode": {"enum": ["a", "b"]}}}

    validate_arguments({"mode": "a"}, schema)


# ---------- number bounds ----------


def test_number_below_minimum_raises() -> None:
    schema = {
        "type": "object",
        "properties": {"n": {"type": "number", "minimum": 0}},
    }

    with pytest.raises(ValidationError, match=r"^n: -1 < minimum 0$"):
        validate_arguments({"n": -1}, schema)


def test_number_above_maximum_raises() -> None:
    schema = {
        "type": "object",
        "properties": {"n": {"type": "number", "maximum": 10}},
    }

    with pytest.raises(ValidationError, match=r"^n: 11 > maximum 10$"):
        validate_arguments({"n": 11}, schema)


def test_string_value_with_numeric_minimum_in_schema_does_not_apply_bound() -> None:
    # `_is_number` gates the min/max check — it must reject non-numbers
    # (a flipped-operator mutation would let `"abc" < 5` raise TypeError).
    schema = {"type": "object", "properties": {"v": {"type": "string", "minimum": 5}}}

    validate_arguments({"v": "hello"}, schema)


@pytest.mark.parametrize("value", [0, 5, 10, 5.5])
def test_number_within_inclusive_bounds_passes(value: int | float) -> None:
    schema = {
        "type": "object",
        "properties": {"n": {"type": "number", "minimum": 0, "maximum": 10}},
    }

    validate_arguments({"n": value}, schema)


# ---------- string lengths ----------


def test_string_below_minLength_raises() -> None:
    schema = {
        "type": "object",
        "properties": {"s": {"type": "string", "minLength": 3}},
    }

    with pytest.raises(ValidationError, match=r"^s: string length 2 < minLength 3$"):
        validate_arguments({"s": "ab"}, schema)


def test_string_above_maxLength_raises() -> None:
    schema = {
        "type": "object",
        "properties": {"s": {"type": "string", "maxLength": 3}},
    }

    with pytest.raises(ValidationError, match=r"^s: string length 4 > maxLength 3$"):
        validate_arguments({"s": "abcd"}, schema)


def test_string_at_minLength_boundary_passes() -> None:
    # `<` not `<=` — len exactly equal to minLength is valid.
    schema = {"type": "object", "properties": {"s": {"type": "string", "minLength": 3}}}

    validate_arguments({"s": "abc"}, schema)


def test_string_at_maxLength_boundary_passes() -> None:
    # `>` not `>=` — len exactly equal to maxLength is valid.
    schema = {"type": "object", "properties": {"s": {"type": "string", "maxLength": 3}}}

    validate_arguments({"s": "abc"}, schema)


# ---------- type checks ----------


@pytest.mark.parametrize(
    "type_str, value",
    [
        ("string", "hello"),
        ("integer", 42),
        ("number", 3.14),
        ("number", 7),  # int is also valid number
        ("boolean", True),
        ("array", [1, 2]),
        ("object", {"k": 1}),
        ("null", None),
    ],
)
def test_type_match_passes(type_str: str, value: Any) -> None:
    schema = {"type": "object", "properties": {"v": {"type": type_str}}}

    validate_arguments({"v": value}, schema)


@pytest.mark.parametrize(
    "type_str, bad_value",
    [
        ("string", 42),
        ("integer", "42"),
        ("number", "3.14"),
        ("boolean", 1),  # int is not bool here
        ("array", "list"),
        ("object", []),
        ("null", 0),
    ],
)
def test_type_mismatch_raises(type_str: str, bad_value: Any) -> None:
    schema = {"type": "object", "properties": {"v": {"type": type_str}}}

    with pytest.raises(ValidationError, match=rf"^v: expected {type_str}, got"):
        validate_arguments({"v": bad_value}, schema)


@pytest.mark.parametrize("value", ["abc", None])
def test_type_list_accepts_either_member(value: Any) -> None:
    schema = {
        "type": "object",
        "properties": {"v": {"type": ["string", "null"]}},
    }

    validate_arguments({"v": value}, schema)


def test_type_list_boolean_then_string_with_string_value_falls_through_to_string() -> (
    None
):
    # The boolean-arm `continue` (after the inner block) must be `continue`
    # not `break`; otherwise the loop would exit and the later "string"
    # type wouldn't get checked.
    schema = {
        "type": "object",
        "properties": {"v": {"type": ["boolean", "string"]}},
    }

    validate_arguments({"v": "abc"}, schema)


def test_type_list_integer_then_boolean_with_bool_value_falls_through_to_boolean() -> (
    None
):
    # The integer/number bool-guard `continue` must be `continue` not
    # `break`; otherwise a bool wouldn't reach the later "boolean" arm.
    schema = {
        "type": "object",
        "properties": {"v": {"type": ["integer", "boolean"]}},
    }

    validate_arguments({"v": True}, schema)


def test_type_list_rejects_non_member_with_pipe_separated_want() -> None:
    schema = {
        "type": "object",
        "properties": {"v": {"type": ["string", "null"]}},
    }

    with pytest.raises(ValidationError, match=r"v: expected string\|null, got int"):
        validate_arguments({"v": 42}, schema)


def test_bool_value_rejected_for_integer_type() -> None:
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}}

    with pytest.raises(ValidationError, match=r"^n: expected integer, got bool"):
        validate_arguments({"n": True}, schema)


def test_bool_value_rejected_for_number_type() -> None:
    schema = {"type": "object", "properties": {"n": {"type": "number"}}}

    with pytest.raises(ValidationError, match=r"^n: expected number, got bool"):
        validate_arguments({"n": False}, schema)


def test_unknown_type_in_schema_passes_silently() -> None:
    # Per the inline comment in validator.py: unknown types are skipped
    # rather than rejected, so forward-compatible schemas don't break.
    schema = {"type": "object", "properties": {"v": {"type": "exotic-future-type"}}}

    validate_arguments({"v": "anything"}, schema)


def test_property_with_no_type_passes_anything() -> None:
    schema = {"type": "object", "properties": {"v": {}}}

    validate_arguments({"v": object()}, schema)


# ---------- array ----------


def test_array_items_schema_recurses_and_raises_on_bad_element() -> None:
    schema = {
        "type": "object",
        "properties": {
            "xs": {"type": "array", "items": {"type": "integer"}},
        },
    }

    with pytest.raises(ValidationError, match=r"^xs\[1\]: expected integer, got str"):
        validate_arguments({"xs": [1, "two", 3]}, schema)


def test_array_below_minItems_raises() -> None:
    schema = {
        "type": "object",
        "properties": {"xs": {"type": "array", "minItems": 2}},
    }

    with pytest.raises(ValidationError, match=r"^xs: array has 1 items, min 2$"):
        validate_arguments({"xs": [1]}, schema)


def test_array_at_minItems_boundary_passes() -> None:
    # `<` not `<=` — len exactly equal to minItems is valid.
    schema = {"type": "object", "properties": {"xs": {"type": "array", "minItems": 2}}}

    validate_arguments({"xs": [1, 2]}, schema)


def test_array_with_items_schema_and_empty_value_passes() -> None:
    # Covers the for-loop fall-through branch in `_check_value`: items
    # schema is set but the array is empty, so iteration never runs.
    schema = {
        "type": "object",
        "properties": {"xs": {"type": "array", "items": {"type": "integer"}}},
    }

    validate_arguments({"xs": []}, schema)


def test_array_minItems_only_enforced_when_int() -> None:
    # `isinstance(min_items, int)` guard — non-int values are silently
    # ignored, not rejected.
    schema = {
        "type": "object",
        "properties": {"xs": {"type": "array", "minItems": "two"}},
    }

    validate_arguments({"xs": []}, schema)


# ---------- object recursion ----------


def test_object_nested_recurses_and_raises_on_inner_missing_required() -> None:
    schema = {
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "properties": {"inner": {"type": "string"}},
                "required": ["inner"],
            },
        },
    }

    with pytest.raises(
        ValidationError, match=r"^missing required argument\(s\): inner$"
    ):
        validate_arguments({"outer": {}}, schema)


def test_object_nested_passes_when_inner_satisfies_required() -> None:
    schema = {
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "properties": {"inner": {"type": "string"}},
                "required": ["inner"],
            },
        },
    }

    validate_arguments({"outer": {"inner": "ok"}}, schema)


def test_object_with_no_properties_or_required_skips_recursion() -> None:
    schema = {"type": "object", "properties": {"v": {"type": "object"}}}

    validate_arguments({"v": {"anything": [1, 2]}}, schema)


def test_nested_object_with_only_properties_still_recurses() -> None:
    # `properties OR required` — having properties alone must trigger
    # recursion (mutating the OR to AND would skip recursion).
    schema = {
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "properties": {"inner": {"type": "string"}},
            },
        },
    }

    with pytest.raises(ValidationError, match=r"^inner: expected string, got int"):
        validate_arguments({"outer": {"inner": 42}}, schema)


def test_nested_object_with_only_required_still_recurses() -> None:
    # Symmetric: having required alone must trigger recursion.
    schema = {
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "required": ["inner"],
            },
        },
    }

    with pytest.raises(
        ValidationError, match=r"^missing required argument\(s\): inner$"
    ):
        validate_arguments({"outer": {}}, schema)


# ---------- Hypothesis ----------


_PRIMITIVE = st.one_of(
    st.text(),
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans(),
    st.none(),
    st.lists(st.integers(), max_size=5),
)


@given(args=st.dictionaries(st.text(min_size=1), _PRIMITIVE, max_size=8))
def test_arbitrary_dict_against_empty_schema_never_raises(
    args: dict[str, Any],
) -> None:
    # `bbox` may be drawn as a non-list/short-list value; the special
    # path will reject those legitimately — Hypothesis would shrink to
    # one. Strip the key so the property covers the schema-only path.
    args.pop("bbox", None)

    validate_arguments(args, {"type": "object", "properties": {}})
