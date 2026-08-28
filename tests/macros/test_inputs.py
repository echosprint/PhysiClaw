"""Tests for `physiclaw.macros.inputs` — input resolution and
`{name}` placeholder templating."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from physiclaw.macros.inputs import resolve_inputs, substitute
from physiclaw.macros.model import Macro, MacroError, MacroInput
from physiclaw.macros.template import TemplateError, placeholders


def _spec(*inputs: MacroInput) -> Macro:
    return Macro(
        name="demo", description="d", enabled=True, inputs=tuple(inputs), steps=()
    )


TWO_INPUTS = _spec(
    MacroInput(name="message", description="text"),
    MacroInput(name="greeting", description="opening", default="hi"),
)


# ---------- resolve_inputs ----------


def test_resolve_inputs_applies_defaults_and_overrides() -> None:
    values = resolve_inputs(TWO_INPUTS, {"message": "hello"})

    assert values == {"message": "hello", "greeting": "hi"}


def test_resolve_inputs_missing_required_raises() -> None:
    with pytest.raises(MacroError, match="missing required input 'message'"):
        resolve_inputs(TWO_INPUTS, {})


def test_resolve_inputs_unknown_name_raises() -> None:
    with pytest.raises(MacroError, match="unknown input.*typo"):
        resolve_inputs(TWO_INPUTS, {"message": "m", "typo": "x"})


def test_resolve_inputs_non_string_value_raises() -> None:
    with pytest.raises(MacroError, match="input 'message' must be a string"):
        resolve_inputs(TWO_INPUTS, {"message": 42})


# ---------- substitute / placeholders ----------


def test_substitute_fills_placeholders_and_unescapes_braces() -> None:
    args = {"text": "{msg} {{literal}}", "bbox": [0.1, "{msg}"]}

    out = substitute(args, {"msg": "hi"})

    assert out == {"text": "hi {literal}", "bbox": [0.1, "hi"]}


def test_substitute_leaves_non_strings_untouched() -> None:
    args = {"bbox": [0.1, 0.2, 0.3, 0.4], "flag": True}

    out = substitute(args, {})

    assert out == args


def test_placeholders_extracts_names_and_skips_escapes() -> None:
    names = placeholders("{a} {{not_one}} {b_2}")

    assert names == {"a", "b_2"}


@pytest.mark.parametrize("bad", ["{Bad}", "half {", "} half", "{1x}"])
def test_placeholders_stray_brace_raises(bad: str) -> None:
    with pytest.raises(TemplateError, match="stray"):
        placeholders(bad)


@given(st.text(alphabet=st.characters(blacklist_characters="{}"), max_size=80))
def test_substitute_string_without_braces_is_identity(text: str) -> None:
    assert substitute({"v": text}, {}) == {"v": text}


def test_substitute_reaches_placeholders_nested_in_a_mapping() -> None:
    # `parse._check_placeholders` walks dicts, so a nested placeholder
    # validates clean at load. If `substitute` did not walk them too, it
    # would replay as the literal "{msg}".
    args = {"text": "{msg}", "nested": {"inner": "{msg}"}, "lst": ["{msg}"]}

    assert substitute(args, {"msg": "HELLO"}) == {
        "text": "HELLO",
        "nested": {"inner": "HELLO"},
        "lst": ["HELLO"],
    }
