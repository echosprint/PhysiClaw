"""`{name}` placeholder syntax — the DSL's one dynamic construct.

The true leaf of the package: pure string computation, no imports beyond
the error type, so both `model` (clause text) and `inputs` (`with:`
tables) can template without either depending on the other.

Inputs are the ONLY dynamic thing in a macro — flat strings, no
expressions — so this is a tokenizer and a dict lookup, nothing more.
"""

import re


class TemplateError(ValueError):
    """A malformed template. `model.MacroError` subclasses this, so parse
    reports it like any other load failure while this module stays free of
    macro-specific imports."""


# One pass tokenizes escapes, placeholders, and stray braces together —
# `{{`/`}}` must win over `{name}` so an escaped brace never half-matches.
TOKEN_RE = re.compile(r"\{\{|\}\}|\{([a-z][a-z0-9_]*)\}|[{}]")


def fill(text: str, values: dict[str, str]) -> str:
    """One string with its placeholders resolved. Every name was validated
    against the declared inputs at parse time, so a missing key here is a
    programming error, not user input."""
    return TOKEN_RE.sub(lambda m: _replace_token(m, values), text)


def placeholders(value: str) -> set[str]:
    """The input names referenced by `{name}` in one string. Raises on a
    stray `{`/`}` that is neither an escape nor a valid placeholder — a
    typo'd template must fail at load, not replay wrong."""
    names: set[str] = set()
    for m in TOKEN_RE.finditer(value):
        if m.group(0) in ("{{", "}}"):
            continue
        if m.group(1) is None:
            raise TemplateError(
                f"stray {m.group(0)!r} in {value!r} — use {{name}} for a "
                "declared input, or {{{{ / }}}} for a literal brace"
            )
        names.add(m.group(1))
    return names


def _replace_token(m: re.Match[str], values: dict[str, str]) -> str:
    token = m.group(0)
    if token == "{{":
        return "{"
    if token == "}}":
        return "}"
    return values[m.group(1)]
