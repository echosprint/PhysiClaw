"""The ref grammar — `{inputs.name}` and `{move.field}`, validated at
parse (`refs_in`, `check_refs`) and filled at run (`fill_refs`).

Every ref is dotted: `inputs.<name>` reads a declared input, `<move>.<field>`
an EARLIER agent step's declared return field, `ask.total` a payment
ask's quoted total. A bare `{name}` is a load error. Refs are
playbook-level — resolved to plain strings before any macro sees them,
so pack macros keep the stock single-name template grammar.
"""

import re
from typing import Any

from physiclaw.conductor.spec import specfile
from physiclaw.conductor.spec.model import INPUTS_ROOT, PlaybookError

# `{root.name}` — the playbook's own ref grammar, always dotted
# (`inputs.` / an earlier agent step's id). The root follows the
# move-name grammar (hyphens included — `{pick-into-cart.message}`);
# the field stays input-shaped. Dotted refs are deliberately NOT part
# of the macro template layer (its tokenizer rejects them); same
# `{{`/`}}` escapes, same fail-at-load-on-stray-brace rule.
REF_RE = re.compile(r"\{\{|\}\}|\{([a-z0-9][a-z0-9_-]*\.[a-z][a-z0-9_]*)\}|[{}]")


def field_name(name: Any, what: str) -> str:
    """The one naming rule for the values `{x.y}` refs read — an agent's
    return fields, spelled like the inputs they sit beside."""
    if not isinstance(name, str) or not specfile.INPUT_NAME_RE.match(name):
        raise PlaybookError(
            f"{what} {name!r} must be lowercase, start with a "
            "letter, and contain only letters/digits/underscores"
        )
    return name


def refs_in(text: str, where: str) -> set[str]:
    """Every dotted ref a template names; a stray brace is a load error."""
    names: set[str] = set()
    for m in REF_RE.finditer(text):
        if m.group(0) in ("{{", "}}"):
            continue
        if m.group(1) is None:
            raise PlaybookError(
                f"{where}: stray {m.group(0)!r} — every ref is dotted: "
                "{inputs.name} for an input, {node.field} for an earlier "
                "agent step's output, {{ / }} for a literal brace"
            )
        names.add(m.group(1))
    return names


def check_refs(
    refs: set[str],
    input_names: set[str],
    payloads: dict[str, tuple[str, ...]],
    where: str,
) -> None:
    """Every ref resolves: an `inputs.*` to a declared input, anything
    else to an EARLIER move's declared output field."""
    for ref in sorted(refs):
        root, _, fld = ref.partition(".")
        if root == INPUTS_ROOT:
            if fld not in input_names:
                raise PlaybookError(f"{where}: {{{ref}}} not declared under `inputs`")
        elif root not in payloads:
            raise PlaybookError(
                f"{where}: {{{ref}}} references move {root!r}, which "
                "is not an EARLIER agent step — outputs wire forward "
                "only, in route order"
            )
        elif fld not in payloads[root]:
            raise PlaybookError(
                f"{where}: {{{ref}}}: move {root!r} has no output "
                f"{fld!r} (has: {', '.join(payloads[root]) or '(none)'})"
            )


def check_arg_refs(
    value: Any,
    input_names: set[str],
    payloads: dict[str, tuple[str, ...]],
    where: str,
) -> None:
    """`check_refs` over a `with:` value — recursing into lists and
    mappings exactly as `fill_refs` fills them."""
    if isinstance(value, str):
        check_refs(refs_in(value, where), input_names, payloads, where)
    elif isinstance(value, list):
        for v in value:
            check_arg_refs(v, input_names, payloads, where)
    elif isinstance(value, dict):
        for v in value.values():
            check_arg_refs(v, input_names, payloads, where)


def fill_refs(value: Any, values: dict[str, str], where: str) -> Any:
    """A `with:` value with its refs resolved from `values` — the runtime
    half of the ref grammar, beside REF_RE so no consumer re-derives the
    braces. `values` is keyed by the dotted ref spellings themselves
    (`inputs.name`, `node.field`). Recurses into lists/dicts exactly as
    `check_arg_refs` validates them. Raises PlaybookError on a ref with
    no value (e.g. an agent output not yet recorded)."""
    if isinstance(value, list):
        return [fill_refs(v, values, where) for v in value]
    if isinstance(value, dict):
        return {k: fill_refs(v, values, where) for k, v in value.items()}
    if not isinstance(value, str):
        return value

    def repl(m: re.Match[str]) -> str:
        token = m.group(0)
        if token == "{{":
            return "{"
        if token == "}}":
            return "}"
        ref = m.group(1)
        if ref is None:
            raise PlaybookError(f"{where}: stray {token!r} in {value!r}")
        if ref not in values:
            raise PlaybookError(f"{where}: no value for {{{ref}}}")
        return values[ref]

    return REF_RE.sub(repl, value)
