"""Input resolution and `{name}` placeholder templating.

The runtime half of a macro's parameterization: `parse` validated at
load time that every placeholder names a declared input, so by the time
`resolve_inputs` + `substitute` run here, a missing key is a programming
error rather than user input. Inputs are the ONLY dynamic thing in a
macro — flat strings, no expressions — so this module is a dict merge
over `template`'s tokenizer, nothing more. Clause text templates itself
(`model.Clause.substituted`); this half handles `with:` tables.
"""

from typing import Any

from physiclaw.agent.macros.model import Macro, MacroError
from physiclaw.agent.macros.template import fill


def resolve_inputs(spec: Macro, provided: dict[str, Any]) -> dict[str, str]:
    """Merge provided values over declared defaults. Raises MacroError on an
    unknown name, a missing required input, or a non-string value — before
    any gesture fires."""
    declared = {i.name: i for i in spec.inputs}
    unknown = sorted(provided.keys() - declared.keys())
    if unknown:
        raise MacroError(
            f"unknown input(s): {', '.join(unknown)}. "
            f"Declared: {', '.join(sorted(declared)) or '(none)'}"
        )
    values: dict[str, str] = {}
    for inp in spec.inputs:
        if inp.name in provided:
            value = provided[inp.name]
            if not isinstance(value, str):
                raise MacroError(f"input {inp.name!r} must be a string")
            values[inp.name] = value
        elif inp.default is not None:
            values[inp.name] = inp.default
        else:
            raise MacroError(f"missing required input {inp.name!r}")
    return values


def substitute(args: dict[str, Any], values: dict[str, str]) -> dict[str, Any]:
    """Fill `{name}` placeholders in the string leaves of a step's `with:`
    table. Every placeholder was validated against the declared inputs at
    parse time, so a missing key here is a programming error, not user
    input."""
    return {k: _fill(v, values) for k, v in args.items()}


def _fill(value: Any, values: dict[str, str]) -> Any:
    if isinstance(value, str):
        return fill(value, values)
    if isinstance(value, list):
        return [_fill(v, values) for v in value]
    if isinstance(value, dict):
        # Mirrors `parse._check_placeholders`, which walks dicts too. Without
        # this the two disagree: a placeholder nested in a mapping validates
        # clean at load and then replays as the literal `{name}`.
        return {k: _fill(v, values) for k, v in value.items()}
    return value
