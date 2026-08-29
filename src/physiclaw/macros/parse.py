"""MACRO.yml → a validated `Macro`.

The file shape follows GitHub Actions conventions (``name`` /
``description`` / ``inputs.<id>.{description, default}`` at the top,
``steps`` with per-step ``name`` and ``with:``), parsed with a YAML 1.2
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
    step      ::= name tool [with] [guard] [skip_when]          # gesture
                | name "wait" with:{seconds} [guard] [skip_when]
                  [expect [hint]]                               # settle
    guard     ::= {[require: check] [forbid: check] [hint]}     # ≥1 check
    check     ::= clause          # also the shape of expect / skip_when
    clause    ::= "text"                          # whole-screen substring
                | {text, within: bbox}            # element-granular
                | {and|or: [clause × 2+]} | {not: clause}   # may carry within
                  # combinators nest ≤ MAX_CLAUSE_DEPTH levels
    bbox      ::= [left, top, right, bottom]      # unit floats, l<r, t<b

Validation is all-or-nothing (a file failing ANY check is excluded
whole, never partially loaded), so the runner never meets an unknown
tool, a bad guard shape, or a dangling placeholder mid-replay. The
format is deliberately logic-free — fixed linear steps, string-only
inputs, ``{name}`` substitution, and per-step guards that pass or
abort — plus one sanctioned conditional, ``skip_when``, an idempotence
postcondition rather than general branching. A macro's robustness comes
from staying a dumb replay of a rehearsed path.

Two rules exist for reasons outside this module. Step names are
identifiers (required, lowercase/digits/hyphens, unique) because
``start_at`` addresses steps by name. The prose fields — ``description``
and each input's ``description`` / ``example`` — are single-line and
length-capped because `store.render_section` renders them verbatim into
the CACHED system prefix; YAML anchors/aliases are rejected outright so
no consumer has to be alias-safe.
"""

import io
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from physiclaw.common.placeholders import resolve_placeholders
from physiclaw.macros.model import (
    ALLOWED_STEP_TOOLS,
    COMBINATORS,
    INPUT_NAME_RE,
    MAX_CLAUSE_DEPTH,
    MAX_INPUTS,
    MAX_PROSE_LEN,
    MAX_RUN_SECONDS,
    MAX_STEPS,
    MAX_WAIT_SECONDS,
    WAIT,
    WAIT_SECONDS_ARG,
    Bbox,
    Clause,
    Macro,
    MacroError,
    MacroGuard,
    MacroInput,
    TextClause,
    check_name,
)
from physiclaw.macros.steps import GestureStep, Step, WaitStep
from physiclaw.macros.template import TemplateError, placeholders

# The key vocabulary of each mapping production, in grammar order.
_TOP_KEYS = {"name", "description", "enabled", "inputs", "steps"}
_INPUT_KEYS = {"description", "default", "example"}
_STEP_KEYS = {"tool", "with", "name", "guard", "skip_when", "expect", "hint"}
_GUARD_KEYS = {"require", "forbid", "hint"}

# YAML 1.2 safe loader, pure-python. One instance, load-only.
_yaml = YAML(typ="safe", pure=True)


def parse_macro(text: str, dir_name: str) -> Macro:
    """Parse + validate one MACRO.yml. Raises MacroError with a message
    that names the offending field — never a partially-valid spec."""
    text = resolve_placeholders(text, MacroError)
    try:
        data = _yaml.load(io.StringIO(text))
    except YAMLError as e:
        raise MacroError(f"invalid YAML: {e}") from e
    if not isinstance(data, dict):
        raise MacroError("MACRO.yml must be a YAML mapping (key: value pairs)")

    _reject_aliases(data)

    unknown = sorted(set(data.keys()) - _TOP_KEYS)
    if unknown:
        raise MacroError(f"unknown key(s): {', '.join(map(str, unknown))}")

    name = _require_str(data.get("name"), "`name`")
    if name != dir_name:
        raise MacroError(f"name {name!r} must equal the directory name {dir_name!r}")
    check_name(name)

    description = _prose(data.get("description"), "`description`")

    # Absent → enabled: a hand-written macro is live once valid, no extra
    # ceremony. The `init` scaffold writes an explicit `enabled: false` so
    # an unrehearsed scaffold still can't go live by accident.
    enabled = data.get("enabled", True)
    if not isinstance(enabled, bool):
        raise MacroError("`enabled` must be true or false")

    inputs = _parse_inputs(data.get("inputs", {}))
    steps = _parse_steps(data.get("steps"), {i.name for i in inputs})
    return Macro(
        name=name,
        description=description,
        enabled=enabled,
        inputs=inputs,
        steps=tuple(steps),
    )


# ---------- inputs ----------


def _parse_inputs(raw: Any) -> tuple[MacroInput, ...]:
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
    budget). `seen_names` is the symbol table `start_at` resolution
    relies on — duplicates are caught at the step that reuses a name, so
    the error can cite both occupants."""
    if not isinstance(raw, list) or not raw:
        raise MacroError("`steps` must be a non-empty list")
    if len(raw) > MAX_STEPS:
        raise MacroError(f"too many steps ({len(raw)} > {MAX_STEPS})")
    seen_names: dict[str, int] = {}
    out = [
        _parse_step(i, step, input_names, seen_names)
        for i, step in enumerate(raw, start=1)
    ]
    _check_wait_budget(out)
    return out


def _parse_step(
    i: int, step: Any, input_names: set[str], seen_names: dict[str, int]
) -> Step:
    """One step: keys, tool, args, name, its checks — in the order that
    yields the most specific error first (an unknown key beats a bad
    tool beats a malformed check)."""
    if not isinstance(step, dict):
        raise MacroError(f"step {i} must be a mapping")
    unknown = sorted(set(step.keys()) - _STEP_KEYS)
    if unknown:
        raise MacroError(f"step {i}: unknown key(s): {', '.join(map(str, unknown))}")
    tool = step.get("tool")
    if not isinstance(tool, str) or tool not in ALLOWED_STEP_TOOLS:
        raise MacroError(
            f"step {i}: `tool` must be one of "
            f"{', '.join(sorted(ALLOWED_STEP_TOOLS))} (got {tool!r})"
        )
    args = step.get("with", {})
    if not isinstance(args, dict):
        raise MacroError(f"step {i}: `with` must be a mapping of arguments")
    _check_placeholders(args, input_names, i)
    if tool == WAIT:
        _check_wait_args(args, i, "expect" in step)
    # Required, identifier-shaped, and unique. A step name is what
    # `start_at` addresses and what the step log, abort report and run
    # log show, so an unnamed step is unreachable and unreadable. Same
    # lowercase/hyphen rule as macro names: it is an identifier, not
    # prose — no spaces to quote at a shell prompt. Duplicates are
    # caught here rather than mid-replay.
    name = _require_str(step.get("name"), f"step {i}: `name`")
    check_name(
        name,
        f"step {i}: `name`",
        " — it is the identifier `start_at` uses (e.g. `focus-input-box`)",
    )
    if name in seen_names:
        raise MacroError(
            f"step {i}: duplicate step name {name!r} (step "
            f"{seen_names[name]} already uses it) — `start_at` addresses "
            "steps by name, so they must be unique"
        )
    seen_names[name] = i
    if "expect" in step and tool != WAIT:
        # Deliberately wait-only. A gesture's own view is captured ~2s
        # after the touch and is the SAME frame the next step's guard
        # reads for free — so `expect` there asserts nothing new, just
        # a second name for one check on one frame. Forcing a `wait`
        # step also forces real settle time, which ~2s often isn't.
        raise MacroError(
            f"step {i}: `expect` belongs to a `wait` step, not to "
            f"{tool!r} — to check what this step produced, add a step "
            "after it: `- {name: confirm, tool: wait, with: {seconds: 1}, "
            'expect: "..."}`'
        )
    if "expect" in step and "skip_when" in step:
        # A skipped step never reaches its `expect`, so the weaker check
        # would silently disable the stronger one — and they are judged
        # on different frames besides (skip on the previous step's,
        # expect on a fresh post-sleep peek).
        raise MacroError(
            f"step {i}: `skip_when` and `expect` on one step contradict "
            "each other — a skipped step never runs its `expect`. Split "
            "them into two steps."
        )
    expect = (
        _parse_check(step.get("expect"), f"step {i}: `expect`", input_names)
        if "expect" in step
        else None
    )
    hint = _opt_str(step.get("hint"), f"step {i}: `hint`") or ""
    if hint and expect is None:
        raise MacroError(
            f"step {i}: `hint` steers the recovery when `expect` fails, "
            "so it needs an `expect` to belong to (a guard carries its "
            "own `hint` inside `guard`)"
        )
    guard = _parse_guard(step["guard"], i, input_names) if "guard" in step else None
    skip_when = (
        _parse_check(step.get("skip_when"), f"step {i}: `skip_when`", input_names)
        if "skip_when" in step
        else None
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
        name=name, guard=guard, skip_when=skip_when, mcp_tool=tool, args=dict(args)
    )


def _check_wait_args(args: dict, step_no: int, has_expect: bool) -> None:
    """`wait` is the one step the MCP server never sees, so nothing
    downstream will reject a malformed one — validate it here or it fails on
    the rig mid-run."""
    where = f"step {step_no}: `wait`"
    unknown = sorted(set(args.keys()) - {WAIT_SECONDS_ARG})
    if unknown:
        raise MacroError(
            f"{where}: unknown argument(s): {', '.join(map(str, unknown))} "
            f"(want only `{WAIT_SECONDS_ARG}`)"
        )
    if WAIT_SECONDS_ARG not in args:
        raise MacroError(
            f"{where} needs `{WAIT_SECONDS_ARG}` — e.g. "
            f"`with: {{{WAIT_SECONDS_ARG}: 3}}`"
        )
    seconds = args[WAIT_SECONDS_ARG]
    if isinstance(seconds, bool) or not isinstance(seconds, int):
        raise MacroError(
            f"{where}.{WAIT_SECONDS_ARG} must be a whole number of seconds "
            f"(got {seconds!r})"
        )
    if not 0 <= seconds <= MAX_WAIT_SECONDS:
        raise MacroError(
            f"{where}.{WAIT_SECONDS_ARG} must be 0–{MAX_WAIT_SECONDS} (got {seconds})"
        )
    if seconds == 0 and not has_expect:
        # 0 earns its place only as "check now": a `wait` with neither sleep
        # nor assertion is a step that does nothing at all.
        raise MacroError(
            f"{where}.{WAIT_SECONDS_ARG}: 0 is only meaningful with an "
            "`expect` — a wait that neither sleeps nor checks does nothing"
        )


def _check_wait_budget(steps: list[Step]) -> None:
    """Declared sleep alone must fit the run budget. `MAX_WAIT_SECONDS` x
    `MAX_STEPS` is 600s against a 300s cap, and the cap is only checked
    BETWEEN steps — so without this a macro can be authored that always
    times out, after physically executing half its gestures."""
    total = sum(s.declared_seconds for s in steps)
    if total >= MAX_RUN_SECONDS:
        raise MacroError(
            f"`wait` steps declare {total}s of sleep, but a whole run is "
            f"capped at {MAX_RUN_SECONDS}s — this macro would always time "
            "out partway through. Shorten the waits."
        )


# ---------- guards ----------


def _parse_guard(raw: Any, step_no: int, input_names: set[str]) -> MacroGuard | None:
    if raw is None:
        raise MacroError(
            f"step {step_no}: `guard` is empty — write `require` and/or "
            '`forbid`, e.g. `guard: {require: "WeChat"}`. Delete the key '
            "entirely if this step needs no gate."
        )
    where = f"step {step_no}: `guard`"
    if not isinstance(raw, dict):
        raise MacroError(f"{where} must be a mapping")
    if "wait_seconds" in raw:
        # Named explicitly rather than falling into "unknown key": this key
        # was real, and the replacement is a different shape, so say so.
        raise MacroError(
            f"{where}.wait_seconds was removed — a guard now only checks. "
            "To settle before the check, put a step before it: "
            "`- {name: settle, tool: wait, with: {seconds: 3}}`"
        )
    unknown = sorted(set(raw.keys()) - _GUARD_KEYS)
    if unknown:
        raise MacroError(f"{where}: unknown key(s): {', '.join(unknown)}")
    require = (
        _parse_check(raw["require"], f"{where}.require", input_names)
        if "require" in raw
        else None
    )
    forbid = (
        _parse_check(raw["forbid"], f"{where}.forbid", input_names)
        if "forbid" in raw
        else None
    )
    if require is None and forbid is None:
        raise MacroError(f"{where} needs `require` and/or `forbid`")
    hint = _opt_str(raw.get("hint"), f"{where}.hint") or ""
    return MacroGuard(require=require, forbid=forbid, hint=hint)


# ---------- checks and the clause grammar ----------


def _parse_check(raw: Any, where: str, input_names: set[str]) -> Clause:
    """THE check shape — `require`, `forbid`, `expect`, `skip_when` all take
    exactly one clause, so there is nothing positional to remember. Parses
    the clause expression, then runs the leaf-level validation passes.

    They were lists once (implicitly AND-ed) and `forbid` was a flat list of
    bare strings. That made bracket shape carry meaning: a list meant AND at
    the top level and was an error one level down. Spelling conjunction
    `{and: [...]}` costs six characters and removes the rule."""
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
    """The recursive clause production, in one of four forms:

        "WeChat"                                whole-screen substring
        {text: "微信", within: [l,t,r,b]}        element-granular, region-scoped
        {or: [c, c, ...]}  {and: [c, c, ...]}   combinators, spelled out
        {not: c}                                negation

    The combinators nest, so `{or: [{not: x}, {and: [y, z]}]}` is legal and
    means what it reads as.

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
        raise MacroError(
            f"{where}: a bare list is not a clause — write the operator: "
            "{or: [...]} for any-of, {and: [...]} for all-of"
        )
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
        inner = _bbox(v["within"], f"{where}.within") if "within" in v else scope
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
    if isinstance(raw["text"], list):
        raise MacroError(
            f"{where}.text must be a single string — for alternatives use "
            '{or: ["a", "b"], within: [...]}'
        )
    within = _bbox(raw["within"], f"{where}.within") if "within" in raw else scope
    return TextClause(text=_require_str(raw["text"], f"{where}.text"), within=within)


def _bbox(value: Any, where: str) -> Bbox:
    ok = (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
    )
    if not ok:
        raise MacroError(f"{where} must be [left, top, right, bottom] numbers (0–1)")
    left, top, right, bottom = (float(v) for v in value)
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise MacroError(
            f"{where} must satisfy 0 ≤ left < right ≤ 1 and 0 ≤ top < bottom ≤ 1"
        )
    return (left, top, right, bottom)


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
    """Every `{name}` in a check must name a declared input, exactly as in a
    `with:` table. Undeclared names used to be accepted and then matched
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


# ---------- placeholder validation for `with:` tables ----------


def _check_placeholders(value: Any, input_names: set[str], step_no: int) -> None:
    """Validate every `{name}` in a step's `with:` table, walking each
    container at most once.

    A repeat is REJECTED, not skipped: sharing identity means an alias (two
    look-alike nodes parse to two distinct objects), aliases buy a flat
    20-step step list nothing, and refusing them keeps every later consumer
    a plain tree walk instead of making each one alias-safe —
    `inputs.substitute` walks this same structure at replay time and would
    blow up identically, on the phone, mid-run. `_reject_aliases` already
    refused anchors document-wide, so this walk only checks names."""
    if isinstance(value, str):
        undeclared = sorted(_names_in(value, f"step {step_no}") - input_names)
        if undeclared:
            raise MacroError(
                f"step {step_no}: placeholder(s) {', '.join(undeclared)} "
                "not declared under `inputs`"
            )
    elif isinstance(value, list):
        for v in value:
            _check_placeholders(v, input_names, step_no)
    elif isinstance(value, dict):
        # An unquoted `text: {message}` is YAML flow-mapping syntax, not a
        # placeholder — the classic mistake. Detect the exact shape and
        # point at the fix instead of failing later at the server.
        if len(value) == 1:
            (k, v), *_ = value.items()
            if v is None and isinstance(k, str) and INPUT_NAME_RE.match(k):
                raise MacroError(
                    f"step {step_no}: {{{k}}} was parsed as a YAML mapping — "
                    f'quote placeholder strings: "{{{k}}}"'
                )
        for v in value.values():
            _check_placeholders(v, input_names, step_no)


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


def _reject_aliases(
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
    plain tree walk instead of making each one alias-safe."""
    if not isinstance(value, (dict, list)):
        return
    seen = {} if seen is None else seen
    if id(value) in seen:
        raise MacroError(
            f"{path or 'MACRO.yml'}: YAML anchors/aliases (`&name` / `*name`) "
            "are not supported — write the value out in full"
        )
    seen[id(value)] = value
    items = value.items() if isinstance(value, dict) else enumerate(value)
    for key, child in items:
        _reject_aliases(child, seen, f"{path}.{key}" if path else str(key))
