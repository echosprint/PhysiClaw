"""A macro file (`macros/<name>.yml`) → a validated `Macro`.

The file shape follows GitHub Actions conventions at the top (``name`` /
``description`` / ``inputs.<id>.{description, default}``); each step is
one ``verb: object`` line plus its qualifiers. Parsed with a YAML 1.2
loader so ``yes/no/on/off`` stay strings. What 1.2 still coerces
(unquoted ``true``/``false``, bare numbers like ``007``) is caught by
the strict type check here: every string-position value that arrives as
a bool/int/float raises a MacroError naming the YAML gotcha and the
quoted fix — silent type flips become loud ``macros check`` failures.

The grammar, top-down — the functions below follow it in this order,
recursive-descent style, sharing the scalar terminals at the bottom:

    macro     ::= name description [enabled] [inputs] steps
    inputs    ::= {id: {description, [default], [example]}}     # ≤ MAX_INPUTS
    steps     ::= [step, ...]                                   # 1–MAX_STEPS
    step      ::= verb                                          # argless, bare word
                | {verb: object, [at], [when | skip_when],
                   [require], [forbid], [expect [hint]]}
    verb      ::= tap | double_tap | long_press    object = label, at REQUIRED
                | swipe                            object = up|down|left|right, at REQUIRED,
                                                  [size] [speed] off the ladder
                | send_to_clipboard                object = text
                | wait                             object = seconds  (expect lives here)
                | home_screen | go_back | force_quit | peek       no object
    label     ::= "text" | ["text", ...]          # ≤ MAX_LABEL_READINGS readings of ONE target
    at        ::= [left, top, right, bottom]      # unit floats, l<r, t<b
    check     ::= clause          # require / forbid / expect / when / skip_when
    clause    ::= "text"                          # whole-screen substring
                | ["text", ...]                   # any of them
                | {text: "t" | [alts], within: band | bbox}   # element-granular
                | {and|or: [clause × 2+]} | {not: clause}     # may carry within
                  # combinators nest ≤ MAX_CLAUSE_DEPTH levels
    band      ::= top | bottom | left | right     # common.bbox.BANDS

The `at:` box never travels alone: the verb's object says what the
coordinates ARE (on-screen text heals the press to where that text sits
today; a description merely documents).

Validation is all-or-nothing (a file failing ANY check is excluded
whole, never partially loaded), so the runner never meets an unknown
tool, a bad guard shape, or a dangling placeholder mid-replay. The
format is deliberately logic-free — fixed linear steps, string-only
inputs, ``{name}`` substitution, and per-step checks that pass or
abort — plus one sanctioned conditional, ``when`` / ``skip_when``, an
idempotence postcondition rather than general branching. A macro's
robustness comes from staying a dumb replay of a rehearsed path.

A step carries no name: its handle (`idx3-tap-paste`) is derived here
from its position and its verb line (`model.step_handle`), so every
step is addressable and nothing in the file exists for the address.
One rule exists for reasons outside this module. The
prose fields — ``description`` and each input's ``description`` /
``example`` — are single-line and length-capped because
`store.render_section` renders them verbatim into the CACHED system
prefix; YAML anchors/aliases are rejected outright so no consumer has
to be alias-safe.
"""

import io
from collections.abc import Callable
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from physiclaw.common import bbox, gesture_vocab
from physiclaw.common.placeholders import resolve_placeholders
from physiclaw.macros.model import (
    ALLOWED_STEP_TOOLS,
    ARGLESS_TOOLS,
    BOXED_TOOLS,
    COMBINATORS,
    INPUT_NAME_RE,
    MAX_CLAUSE_DEPTH,
    MAX_INPUTS,
    MAX_PROSE_LEN,
    MAX_RUN_SECONDS,
    MAX_STEPS,
    MAX_WAIT_SECONDS,
    OBJECT_ARG,
    TARGET_BBOX,
    WAIT,
    WAIT_SECONDS_ARG,
    Bbox,
    Clause,
    Macro,
    MacroError,
    MacroGuard,
    MacroInput,
    NotClause,
    OrClause,
    TextClause,
    check_name,
    checked_readings,
    step_handle,
)
from physiclaw.macros.steps import GestureStep, Step, WaitStep
from physiclaw.macros.template import TemplateError, placeholders

# The key vocabulary of each mapping production, in grammar order.
_TOP_KEYS = {"name", "description", "enabled", "inputs", "steps"}
_INPUT_KEYS = {"description", "default", "example"}
# A step is one verb key plus these qualifiers.
_QUALIFIER_KEYS = {
    "at",
    "when",
    "skip_when",
    "require",
    "forbid",
    "expect",
    "hint",
}
# A swipe's stroke: `size` off the ladder, `speed` off the names — the
# server's own optional arguments, spelled beside the direction.
_SWIPE_LADDER: dict[str, tuple[str, ...]] = {
    "size": tuple(gesture_vocab.SWIPE_DISTANCES),
    "speed": gesture_vocab.SWIPE_SPEEDS,
}
_STEP_KEYS = ALLOWED_STEP_TOOLS | _QUALIFIER_KEYS
_SWIPE_STEP_KEYS = _STEP_KEYS | _SWIPE_LADDER.keys()


def _press_object(tool: str, obj: Any, where: str, step: dict) -> dict:
    args = {OBJECT_ARG[tool]: obj}
    checked_readings(args, where, _require_str, MacroError)
    return args


def _swipe_object(tool: str, obj: Any, where: str, step: dict) -> dict:
    if obj not in gesture_vocab.SWIPE_DIRECTIONS:
        raise MacroError(
            f"{where}: `swipe` takes a direction "
            f"({' / '.join(gesture_vocab.SWIPE_DIRECTIONS)}), got {obj!r}"
        )
    args = {OBJECT_ARG[tool]: obj}
    for name, ladder in _SWIPE_LADDER.items():
        if name in step:
            if step[name] not in ladder:
                raise MacroError(
                    f"{where}: `{name}` must be one of {', '.join(ladder)} "
                    f"(got {step[name]!r})"
                )
            args[name] = step[name]
    return args


def _clipboard_object(tool: str, obj: Any, where: str, step: dict) -> dict:
    return {OBJECT_ARG[tool]: _require_str(obj, f"{where}: `{tool}`")}


def _wait_object(tool: str, obj: Any, where: str, step: dict) -> dict:
    return {OBJECT_ARG[tool]: _wait_seconds(obj, where, "expect" in step)}


# One record per object-taking verb: how its object (and, for a swipe,
# the stroke keys beside it) becomes wire arguments, and how the object
# is spelled — for the error that asks for one. Keyed exactly like
# `OBJECT_ARG`, which names the wire key; the pin below keeps the two
# tables the same set of verbs.
_OBJECTS: dict[str, tuple[Callable[[str, Any, str, dict], dict], str]] = {
    **{
        t: (_press_object, f'`- {t}: "Paste"` with `at: [...]`')
        for t in gesture_vocab.PRESS_TOOLS
    },
    gesture_vocab.SWIPE: (_swipe_object, "`- swipe: up` with `at: [...]`"),
    gesture_vocab.SEND_TO_CLIPBOARD: (
        _clipboard_object,
        '`- send_to_clipboard: "{message}"`',
    ),
    WAIT: (_wait_object, "`- wait: 2`"),
}
assert _OBJECTS.keys() == OBJECT_ARG.keys()

# YAML 1.2 safe loader, pure-python. One instance, load-only.
_yaml = YAML(typ="safe", pure=True)


def parse_macro(text: str, stem: str) -> Macro:
    """Parse + validate one macro file; `stem` is its file name without
    the suffix, which `name:` must equal. Raises MacroError with a
    message that names the offending field — never a partially-valid
    spec."""
    text = resolve_placeholders(text, MacroError)
    try:
        data = _yaml.load(io.StringIO(text))
    except YAMLError as e:
        raise MacroError(f"invalid YAML: {e}") from e
    if not isinstance(data, dict):
        raise MacroError("a macro file must be a YAML mapping (key: value pairs)")

    reject_aliases(data)

    unknown = sorted(set(data.keys()) - _TOP_KEYS)
    if unknown:
        raise MacroError(f"unknown key(s): {', '.join(map(str, unknown))}")

    name = _require_str(data.get("name"), "`name`")
    if name != stem:
        raise MacroError(f"name {name!r} must equal the file name {stem!r}")
    check_name(name)

    description = _prose(data.get("description"), "`description`")

    # Absent → enabled: a hand-written macro is live once valid, no extra
    # ceremony. The `init` scaffold writes an explicit `enabled: false` so
    # an unrehearsed scaffold still can't go live by accident.
    enabled = data.get("enabled", True)
    if not isinstance(enabled, bool):
        raise MacroError("`enabled` must be true or false")

    inputs = parse_inputs(data.get("inputs", {}))
    steps = _parse_steps(data.get("steps"), {i.name for i in inputs})
    return Macro(
        name=name,
        description=description,
        enabled=enabled,
        inputs=inputs,
        steps=tuple(steps),
    )


# What an embedded body must NOT carry: `name` (the caller synthesizes
# it), `description` (the node it sits on is the context), `enabled` (an
# inline macro goes live with its playbook, never on its own). The inline
# vocabulary is derived by subtraction so a key added to the macro-file
# grammar reaches the inline form without a second edit.
_IDENTITY_KEYS = frozenset({"name", "description", "enabled"})
_INLINE_KEYS = frozenset(_TOP_KEYS) - _IDENTITY_KEYS


def parse_inline_macro(data: Any, name: str) -> Macro:
    """A macro embedded where a name was expected (a playbook move's
    `macro:` mapping) → a validated `Macro` under the caller-synthesized
    `name`. Same `inputs`/`steps` grammar and budgets as a macro file —
    one step parser, so the two homes can never drift — and always
    enabled (the embedding playbook's own `enabled:` is its gate).
    Raises MacroError; the caller frames it with the node's address."""
    if not isinstance(data, dict):
        raise MacroError(
            "an inline macro must be a YAML mapping with `steps:` "
            "(a string names a directory macro instead)"
        )
    reject_aliases(data)
    unknown = sorted(set(map(str, data.keys())) - _INLINE_KEYS)
    if unknown:
        raise MacroError(
            f"unknown key(s): {', '.join(unknown)} — an inline macro takes "
            "only `steps` and `inputs`; name, description and enabled "
            "belong to a directory macro"
        )
    inputs = parse_inputs(data.get("inputs", {}))
    steps = _parse_steps(data.get("steps"), {i.name for i in inputs})
    return Macro(
        name=name,
        description=f"inline macro {name}",
        enabled=True,
        inputs=inputs,
        steps=tuple(steps),
    )


# ---------- inputs ----------


def parse_inputs(raw: Any) -> tuple[MacroInput, ...]:
    """`inputs:` → declared inputs — the one authoring grammar a macro
    and a playbook share (the conductor delegates here and translates
    the error class at its seam)."""
    if not isinstance(raw, dict):
        raise MacroError("`inputs` must be a mapping of input names")
    if len(raw) > MAX_INPUTS:
        raise MacroError(f"too many inputs ({len(raw)} > {MAX_INPUTS})")
    out: list[MacroInput] = []
    for name, spec in raw.items():
        if not isinstance(name, str) or not INPUT_NAME_RE.match(name):
            raise MacroError(
                f"input name {name!r} must be lowercase, start with a "
                "letter, and contain only letters/digits/underscores"
            )
        if not isinstance(spec, dict):
            raise MacroError(f"input {name!r} must be a mapping")
        unknown = sorted(set(spec.keys()) - _INPUT_KEYS)
        if unknown:
            raise MacroError(f"input {name!r}: unknown key(s): {', '.join(unknown)}")
        out.append(
            MacroInput(
                name=name,
                description=_prose(
                    spec.get("description"), f"input {name!r}: `description`"
                ),
                default=_opt_prose(spec.get("default"), f"input {name!r}: `default`"),
                example=_opt_prose(spec.get("example"), f"input {name!r}: `example`"),
            )
        )
    return tuple(out)


# ---------- steps ----------


def _parse_steps(raw: Any, input_names: set[str]) -> list[Step]:
    """The step list: shape and size here, each step in `_parse_step`,
    then the one check that only the WHOLE list can answer (the wait
    budget)."""
    if not isinstance(raw, list) or not raw:
        raise MacroError("`steps` must be a non-empty list")
    if len(raw) > MAX_STEPS:
        raise MacroError(f"too many steps ({len(raw)} > {MAX_STEPS})")
    out = [_parse_step(i, step, input_names) for i, step in enumerate(raw, start=1)]
    _check_wait_budget(out)
    return out


def _parse_step(i: int, step: Any, input_names: set[str]) -> Step:
    """One step: the verb and its object, `at`, its checks — in
    the order that yields the most specific error first (an unknown key
    beats a bad verb beats a malformed check).

    Two spellings: a bare word for an argless verb (`- home_screen`), or
    a mapping whose ONE verb key carries the object (`- tap: "Paste"`)
    beside the qualifiers."""
    where = f"step {i}"
    if isinstance(step, str):
        return _argless(i, step)
    if not isinstance(step, dict):
        raise MacroError(
            f"{where} must be a verb (`- home_screen`) or a mapping (`- tap: ...`)"
        )
    keys = set(map(str, step.keys()))
    verbs = sorted(keys & ALLOWED_STEP_TOOLS)
    allowed = _SWIPE_STEP_KEYS if gesture_vocab.SWIPE in verbs else _STEP_KEYS
    unknown = sorted(keys - allowed)
    if unknown:
        raise MacroError(
            f"{where}: unknown key(s): {', '.join(unknown)} — a step is one of "
            f"{', '.join(sorted(ALLOWED_STEP_TOOLS))} plus "
            f"{', '.join(sorted(_QUALIFIER_KEYS))}"
        )
    if len(verbs) != 1:
        raise MacroError(
            f"{where}: exactly one verb per step (one of "
            f"{', '.join(sorted(ALLOWED_STEP_TOOLS))}); got "
            f"{', '.join(verbs) or 'none'}"
        )
    tool = verbs[0]
    args = _step_args(tool, step, where, input_names)
    name = step_handle(i, tool, args)
    skip_when = _parse_skip_when(step, where, input_names)
    hint = _opt_str(step.get("hint"), f"{where}: `hint`") or ""
    guard = _parse_guard(step, where, input_names, hint)
    expect = _parse_expect(step, tool, where, input_names)
    # One `hint` per step: it steers the recovery when THIS step's checks
    # fail, so it rides the guard and the expect alike — and needs one.
    if hint and guard is None and expect is None:
        raise MacroError(
            f"{where}: `hint` steers the recovery when a check fails, so it "
            "needs a `require`, `forbid` or `expect` to belong to"
        )
    if expect is not None and skip_when is not None:
        # A skipped step never reaches its `expect`, so the weaker check
        # would silently disable the stronger one — and they are judged
        # on different frames besides (skip on the previous step's,
        # expect on a fresh post-sleep peek).
        raise MacroError(
            f"{where}: `when`/`skip_when` and `expect` on one step contradict "
            "each other — a skipped step never runs its `expect`. Split "
            "them into two steps."
        )
    # The one place the DSL's two step kinds are chosen. Everything
    # downstream is polymorphic, so this `if` has no siblings.
    if tool == WAIT:
        return WaitStep(
            name=name,
            guard=guard,
            skip_when=skip_when,
            seconds=int(args[WAIT_SECONDS_ARG]),
            expect=expect,
            hint=hint,
        )
    return GestureStep(
        name=name, guard=guard, skip_when=skip_when, mcp_tool=tool, args=args
    )


def _argless(i: int, verb: str) -> Step:
    """The bare-word step: a verb that takes no object and no qualifier."""
    where = f"step {i}"
    if verb not in ALLOWED_STEP_TOOLS:
        raise MacroError(
            f"{where}: {verb!r} is not a step verb — one of "
            f"{', '.join(sorted(ALLOWED_STEP_TOOLS))}"
        )
    if verb not in ARGLESS_TOOLS:
        raise MacroError(f"{where}: `{verb}` takes an object — write {_example(verb)}")
    return GestureStep(name=step_handle(i, verb, {}), mcp_tool=verb)


def _step_args(tool: str, step: dict, where: str, input_names: set[str]) -> dict:
    """The verb's object and `at:` → the wire arguments, validated per
    verb: a press wants a label (one string or readings of ONE target)
    with `at`; a swipe a direction with `at`; the clipboard its text;
    `wait` its seconds; an argless verb nothing at all."""
    obj = step[tool]
    at = step.get("at")
    if "at" in step and tool not in BOXED_TOOLS:
        raise MacroError(
            f"{where}: `at` belongs to a press or swipe — {tool!r} takes no box"
        )
    if tool in BOXED_TOOLS and at is None:
        raise MacroError(
            f"{where}: `{tool}` needs `at: [left, top, right, bottom]` — the box "
            "the gesture lands on (the object says what sits there)"
        )
    if tool in ARGLESS_TOOLS:
        if obj is not None:
            raise MacroError(
                f"{where}: `{tool}` takes no object — write `- {tool}` (or "
                f"`{tool}:` with nothing after it when it carries qualifiers)"
            )
        return {}
    if obj is None:
        raise MacroError(f"{where}: `{tool}` needs its object — e.g. {_example(tool)}")
    # Placeholders in the object are vetted here, once — and the
    # unquoted-`{placeholder}` trap (a mapping) gets its own hint before
    # the string check could call it "not a string".
    _check_placeholders(obj, input_names, where)
    parse_object, _ = _OBJECTS[tool]
    args = parse_object(tool, obj, where, step)
    if at is not None:
        if not (isinstance(at, list) and any(isinstance(v, str) for v in at)):
            # The runner READS this value (healing measures drift from
            # its center), so a literal box must not reach it malformed.
            # A box carrying `{placeholder}` strings stays server-checked
            # (the placeholder pass below vets the names).
            _box(at, f"{where}: `at`")
        _check_placeholders(at, input_names, where)
        args[TARGET_BBOX] = at
    return args


def _example(tool: str) -> str:
    return _OBJECTS[tool][1]


def _parse_skip_when(step: dict, where: str, input_names: set[str]) -> Clause | None:
    """`when` / `skip_when` → the one skip clause. `skip_when: X` skips the
    step while X shows; `when: X` runs it only while X shows, i.e. skips
    unless X — stored as `{not: X}` so the runner has one rule."""
    if "when" in step and "skip_when" in step:
        raise MacroError(
            f"{where}: `when` and `skip_when` on one step contradict each other — "
            "keep the one that reads naturally"
        )
    if "skip_when" in step:
        return _parse_check(step["skip_when"], f"{where}: `skip_when`", input_names)
    if "when" in step:
        return NotClause(
            child=_parse_check(step["when"], f"{where}: `when`", input_names)
        )
    return None


def _parse_guard(
    step: dict, where: str, input_names: set[str], hint: str
) -> MacroGuard | None:
    """`require` / `forbid` beside the verb → the pre-step gate, the step's
    `hint` riding it."""
    require = (
        _parse_check(step["require"], f"{where}: `require`", input_names)
        if "require" in step
        else None
    )
    forbid = (
        _parse_check(step["forbid"], f"{where}: `forbid`", input_names)
        if "forbid" in step
        else None
    )
    if require is None and forbid is None:
        return None
    return MacroGuard(require=require, forbid=forbid, hint=hint)


def _parse_expect(
    step: dict, tool: str, where: str, input_names: set[str]
) -> Clause | None:
    """`expect`, a wait step's post-sleep check."""
    if "expect" not in step:
        return None
    if tool != WAIT:
        # Deliberately wait-only. A gesture's own view is captured ~2s
        # after the touch and is the SAME frame the next step's checks
        # read for free — so `expect` there asserts nothing new, just
        # a second name for one check on one frame. Forcing a `wait`
        # step also forces real settle time, which ~2s often isn't.
        raise MacroError(
            f"{where}: `expect` belongs to a `wait` step, not to {tool!r} — to "
            "check what this step produced, add a step after it: "
            '`- wait: 1` with `expect: "..."`'
        )
    return _parse_check(step["expect"], f"{where}: `expect`", input_names)


def _wait_seconds(raw: Any, where: str, has_expect: bool) -> int:
    """`wait` is the one step the MCP server never sees, so nothing
    downstream will reject a malformed one — validate it here or it fails
    on the rig mid-run."""
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise MacroError(
            f"{where}: `wait` takes a whole number of seconds (got {raw!r})"
        )
    if not 0 <= raw <= MAX_WAIT_SECONDS:
        raise MacroError(f"{where}: `wait` must be 0–{MAX_WAIT_SECONDS} (got {raw})")
    if raw == 0 and not has_expect:
        # 0 earns its place only as "check now": a `wait` with neither sleep
        # nor assertion is a step that does nothing at all.
        raise MacroError(
            f"{where}: `wait: 0` is only meaningful with an `expect` — a wait "
            "that neither sleeps nor checks does nothing"
        )
    return raw


def _check_wait_budget(steps: list[Step]) -> None:
    """Declared sleep alone must fit the run budget. `MAX_WAIT_SECONDS` x
    `MAX_STEPS` is 1500s against a 300s cap, and the cap is only checked
    BETWEEN steps — so without this a macro can be authored that always
    times out, after physically executing half its gestures."""
    total = sum(s.declared_seconds for s in steps)
    if total >= MAX_RUN_SECONDS:
        raise MacroError(
            f"`wait` steps declare {total}s of sleep, but a whole run is "
            f"capped at {MAX_RUN_SECONDS}s — this macro would always time "
            "out partway through. Shorten the waits."
        )


# ---------- checks and the clause grammar ----------


def _parse_check(raw: Any, where: str, input_names: set[str]) -> Clause:
    """THE check shape — `require`, `forbid`, `expect`, `when`, `skip_when`
    all take exactly one clause, so there is nothing positional to remember. Parses
    the clause expression, then runs the leaf-level validation passes.

    A list is any-of (the alternate-readings shape the whole format uses);
    conjunction is spelled `{and: [...]}`."""
    if raw is None:
        # The key was written and left empty. Silently reading that as
        # "no check" would drop a guard the author believed they had.
        raise MacroError(
            f"{where} is empty — write one clause, e.g. "
            f'{where.rsplit(".", 1)[-1]}: "WeChat"'
        )
    clause = _clause_expr(raw, where)
    _check_single_chars(clause, where)
    _check_clause_placeholders(clause, input_names, where)
    return clause


def _clause_expr(
    v: Any, where: str, scope: Bbox | None = None, depth: int = 0
) -> Clause:
    """The recursive clause production, in one of five forms:

        "WeChat"                                whole-screen substring
        ["WeChat", "Weixin"]                    any of them
        {text: "微信", within: top}              element-granular, scoped to a
                                                band or a [l,t,r,b] box; text
                                                may list alternates
        {or: [c, c, ...]}  {and: [c, c, ...]}   combinators, spelled out
        {not: c}                                negation

    The combinators nest, so `{or: [{not: x}, {and: [y, z]}]}` is legal and
    means what it reads as. A bare list is `or` — the same "alternate
    readings" shape a press label and a page anchor take, so one list
    means one thing everywhere in the format.

    Any of them may also carry `within`, which SCOPES the whole subtree —
    `{or: ["space", "空格"], within: [...]}` beats repeating the same bbox
    on every alternative. It is pure sugar: `scope` is pushed down here so
    the runner only ever sees fully-resolved leaves and never has to walk
    back up for an inherited region. Innermost wins, so a leaf may override
    the scope it sits in.

    `depth` counts the combinators enclosing this expression, capped at
    `MAX_CLAUSE_DEPTH` — see there for why. Leaves are free: the cap is on
    nesting, not on how many alternatives an `or` lists."""
    if isinstance(v, list):
        # A list is `or` without the word — and counts as a level like
        # one, so dropping the operator name cannot dodge the cap.
        depth += 1
        if depth > MAX_CLAUSE_DEPTH:
            raise MacroError(
                f"{where}: clauses may nest at most {MAX_CLAUSE_DEPTH} levels — "
                "flatten the list"
            )
        return _any_of([_clause_expr(k, where, scope, depth) for k in v], where)
    if not isinstance(v, dict):
        # A string leaf, or the YAML coercion trap (`require: [true]`,
        # `require: [007]`) — the strict string check accepts the first and
        # rejects the second with the quote-it hint, rather than a
        # structural "not a clause" that hides the real cause.
        return TextClause(text=_require_str(v, f"{where} item"), within=scope)
    ops = [k for k in COMBINATORS if k in v]
    if ops:
        extra = sorted(set(v.keys()) - {*ops, "within"})
        if len(ops) > 1 or extra:
            raise MacroError(
                f"{where}: a clause takes exactly one of "
                f"{_operator_list()} (plus an optional `within`); got: "
                f"{', '.join(map(str, sorted(v)))} — nest them instead of "
                "listing them side by side"
            )
        op = ops[0]
        # Checked AFTER the shape check above, so a malformed clause reports
        # its shape rather than its depth — the more specific error wins.
        depth += 1
        if depth > MAX_CLAUSE_DEPTH:
            raise MacroError(
                f"{where}.{op}: clauses may nest at most {MAX_CLAUSE_DEPTH} "
                f"levels of {_operator_list()} — flatten it. `and`/`or` take "
                "any number of children, so conditions that are merely "
                "siblings never need a level of their own; a check that "
                "genuinely needs more belongs in two steps."
            )
        build, arity = COMBINATORS[op]
        inner = _within(v["within"], f"{where}.within") if "within" in v else scope
        if arity == 1:
            return build((_clause_expr(v[op], f"{where}.{op}", inner, depth),))
        kids = v[op]
        if not isinstance(kids, list) or len(kids) < arity:
            raise MacroError(
                f"{where}.{op} must be a list of at least {arity} clauses "
                "(one child would just be the child)"
            )
        return build(
            tuple(_clause_expr(k, f"{where}.{op}", inner, depth) for k in kids)
        )
    return _region_clause(v, where, scope)


def _region_clause(raw: dict, where: str, scope: Bbox | None = None) -> Clause:
    unknown = sorted(set(raw.keys()) - {"text", "within"})
    if unknown:
        raise MacroError(
            f"{where} mapping item: unknown key(s): {', '.join(map(str, unknown))} "
            f"(want `text` and `within`, or {_operator_list()})"
        )
    if "text" not in raw:
        raise MacroError(f"{where} mapping item needs `text`")
    if "within" not in raw and scope is None:
        raise MacroError(
            f"{where} mapping item needs `within` — either on the item or on "
            f"an enclosing {_operator_list()}"
        )
    within = _within(raw["within"], f"{where}.within") if "within" in raw else scope
    readings = checked_readings(raw, where, _require_str, MacroError, key="text")
    return _any_of([TextClause(text=t, within=within) for t in readings], where)


def _any_of(kids: list[Clause], where: str) -> Clause:
    """A list of alternates → the one clause that means any of them: the
    child itself when there is one, else `or`."""
    if not kids:
        raise MacroError(f"{where}: an empty list matches nothing — list a reading")
    return kids[0] if len(kids) == 1 else OrClause(children=tuple(kids))


def _within(value: Any, where: str) -> Bbox:
    """A check's `within:` — a band name or a box, read by the one
    shared parser (`common.bbox.parse_within`)."""
    try:
        return bbox.parse_within(value)
    except (ValueError, TypeError) as e:
        raise MacroError(f"{where}: {e}") from e


def _box(value: Any, where: str) -> Bbox:
    try:
        return bbox.parse_box(value)
    except (ValueError, TypeError) as e:
        raise MacroError(f"{where}: {e}") from e


def _operator_list() -> str:
    """The operator names, derived from the registry so an error can never
    advertise a set the parser does not accept."""
    return " / ".join(f"`{k}`" for k in COMBINATORS)


# ---------- validation passes over a parsed clause ----------


def _check_single_chars(clause: Clause, where: str) -> None:
    """Single characters match whole element labels (keyboard keys), which
    only works with element-granular matching — the region form. As a plain
    whole-screen substring a single char matches almost anything, so reject
    it loudly instead of letting the clause silently always pass. Walks the
    tree: a bare single char is just as wrong three levels inside an `or`."""
    bad = [
        c.text
        for c in clause.walk()
        if isinstance(c, TextClause) and len(c.text) == 1 and c.within is None
    ]
    if bad:
        raise MacroError(
            f"{where}: single-character text(s) "
            f"{', '.join(repr(t) for t in bad)} match almost anything as a "
            "substring — use the region form: "
            '{text: "…", within: [left, top, right, bottom]}'
        )


def _check_clause_placeholders(
    clause: Clause, input_names: set[str], where: str
) -> None:
    """Every `{name}` in a check must name a declared input, exactly as in
    a step's arguments. Undeclared names used to be accepted and then matched
    literally at replay — silently inverting `forbid`/`skip_when`, which a
    never-present string satisfies. `placeholders` also raises on a stray
    brace, so a typo'd template fails at load rather than on the rig."""
    for leaf in clause.walk():
        if not isinstance(leaf, TextClause):
            continue
        undeclared = sorted(_names_in(leaf.text, where) - input_names)
        if undeclared:
            raise MacroError(
                f"{where}: placeholder(s) {', '.join(undeclared)} "
                "not declared under `inputs`"
            )


# ---------- placeholder validation for step arguments ----------


def _check_placeholders(value: Any, input_names: set[str], where: str) -> None:
    """Validate every `{name}` in a step's arguments, walking each
    container at most once.

    A repeat is REJECTED, not skipped: sharing identity means an alias (two
    look-alike nodes parse to two distinct objects), aliases buy a flat
    20-step step list nothing, and refusing them keeps every later consumer
    a plain tree walk instead of making each one alias-safe —
    `inputs.substitute` walks this same structure at replay time and would
    blow up identically, on the phone, mid-run. `_reject_aliases` already
    refused anchors document-wide, so this walk only checks names."""
    if isinstance(value, str):
        undeclared = sorted(_names_in(value, where) - input_names)
        if undeclared:
            raise MacroError(
                f"{where}: placeholder(s) {', '.join(undeclared)} "
                "not declared under `inputs`"
            )
    elif isinstance(value, list):
        for v in value:
            _check_placeholders(v, input_names, where)
    elif isinstance(value, dict):
        # An unquoted `text: {message}` is YAML flow-mapping syntax, not a
        # placeholder — the classic mistake. Detect the exact shape and
        # point at the fix instead of failing later at the server.
        if len(value) == 1:
            (k, v), *_ = value.items()
            if v is None and isinstance(k, str) and INPUT_NAME_RE.match(k):
                raise MacroError(
                    f"{where}: {{{k}}} was parsed as a YAML mapping — "
                    f'quote placeholder strings: "{{{k}}}"'
                )
        for v in value.values():
            _check_placeholders(v, input_names, where)


def _names_in(text: str, where: str) -> set[str]:
    """`template.placeholders`, reported as a load failure. The template
    layer knows nothing about macros, so its error is translated once here
    rather than leaking a second exception type to `macros check`."""
    try:
        return placeholders(text)
    except TemplateError as e:
        raise MacroError(f"{where}: {e}") from e


# ---------- scalar terminals and YAML strictness ----------


def _require_str(value: Any, where: str) -> str:
    """A string-position value. Non-string scalars get the quoting hint:
    the YAML 1.2 loader already keeps yes/no/on/off as strings, but
    unquoted true/false and bare numbers still coerce — this check turns
    that silent flip into a loud, fixable load error."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    if value is None or isinstance(value, str):  # missing, empty, whitespace
        raise MacroError(f"{where} is required and must be a non-empty string")
    raise MacroError(
        f"{where} must be a string but YAML parsed {value!r} "
        f'({type(value).__name__}) — quote it: "{value}"'
    )


def _opt_str(value: Any, where: str) -> str | None:
    if value is None:
        return None
    return _require_str(value, where)


def _prose(value: Any, where: str) -> str:
    """A human-readable field rendered into the agent's SYSTEM prompt —
    `description` and input `description` / `example`.

    Single line, bounded length. This is a trust boundary, not tidiness:
    `store.render_section` interpolates these verbatim into `## Available
    Macros`, which lives in the CACHED SYSTEM prefix, so a multi-line value
    can forge headings at the same depth as real doctrine and a huge one is
    billed on every wake. Skills cannot do this — their frontmatter parser
    splits on lines — but macros parse full YAML, block scalars included, so
    the guarantee has to be enforced here.

    Rejected, never silently collapsed: these strings are instructions the
    agent reads, and quietly rewriting them is its own trap."""
    text = _require_str(value, where)
    if "\n" in text or "\r" in text:
        raise MacroError(
            f"{where} must be a single line — it is rendered into the agent's "
            "prompt, where extra lines can forge headings. Keep it to one line."
        )
    if len(text) > MAX_PROSE_LEN:
        raise MacroError(
            f"{where} is {len(text)} characters (max {MAX_PROSE_LEN}). It rides "
            "in the cached system prompt on every wake — keep it to one line "
            "that says when to use this macro."
        )
    return text


def _opt_prose(value: Any, where: str) -> str | None:
    """`_prose`, but absent is allowed (input `example`)."""
    return None if value is None else _prose(value, where)


def reject_aliases(
    value: Any, seen: dict[int, Any] | None = None, path: str = ""
) -> None:
    """Refuse YAML anchors/aliases anywhere in the document, once.

    Load-bearing, not tidiness. An alias makes one object reachable by many
    paths, so a naive walk treats a DAG as a tree — and clause parsing
    MATERIALIZES a `Clause` per path, so a 792-byte file nested 20 deep
    built 8.4M clauses in 30s and 24 deep is an OOM. That happens on the
    session-startup path (`store.scan` → `discover_enabled` →
    `build_prompt_bundle`), where `store.scan`'s broad `except` cannot help
    because nothing is raised.

    Done here, before validation, rather than at each recursive site: the
    invariant is file-wide, so making it opt-in per call site meant every
    new nested construct was a new place to forget it — `inputs:` was
    already uncovered. Keyed by id, but `seen` holds the reference, which is
    what keeps the key stable: some containers are temporaries, and a freed
    one could otherwise have its address recycled into the next.

    Rejected rather than skipped: sharing identity means an alias (two
    look-alike nodes parse to two distinct objects), aliases buy a
    20-step file nothing, and refusing them keeps every later consumer a
    plain tree walk instead of making each one alias-safe.

    Public because the guard is document-shaped, not macro-shaped: the
    conductor's spec doors run it over PLAYBOOK.yml too (inline macros
    put clause parsing — the materializer the bomb rides — behind every
    spec door; `specfile.load_yaml` calls it after every load, and
    `parse_inline_macro` guards its own data-shaped door)."""
    if not isinstance(value, (dict, list)):
        return
    seen = {} if seen is None else seen
    if id(value) in seen:
        raise MacroError(
            f"{path or 'the document'}: YAML anchors/aliases (`&name` / `*name`) "
            "are not supported — write the value out in full"
        )
    seen[id(value)] = value
    items = value.items() if isinstance(value, dict) else enumerate(value)
    for key, child in items:
        reject_aliases(child, seen, f"{path}.{key}" if path else str(key))
