"""Macro shapes — the constants, the clause algebra, and the screen it is
evaluated against.

Leaf of the package's internal split (only `template` sits below):
`parse` (MACRO.yml → Macro), `steps` (the executable step hierarchy),
`inputs`, `store`, `runner` and `stats` all depend on this module, never
on each other through it.

Behaviour lives here only where it is PURE. A clause deciding whether it
holds against a screen is computation, so `Clause.holds` belongs on the
clause; issuing a camera read to obtain that screen is I/O, so it lives
in `steps`/`runner`. That line is what keeps this module importable from
anywhere — nothing here reads files, YAML, or the rig.
"""

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar, TypeVar

from physiclaw.agent.macros.template import TemplateError, fill
from physiclaw.common import gesture_vocab
from physiclaw.common.listing import LISTING_HEADER, ROW_PARSE_RE

MAX_STEPS = 20
MAX_INPUTS = 8
MAX_NAME_LEN = 64
# `description` / input `description` / `example` are rendered verbatim into
# the cached SYSTEM prefix, so they are bounded like any other prompt input.
MAX_PROSE_LEN = 200
# One `wait` step's ceiling. A settle is a pause between rehearsed
# gestures, not a way to sit on the rig: anything longer than this is the
# agent's problem, not a macro's.
MAX_WAIT_SECONDS = 30
# Whole-run wall-clock ceiling, and the only bound that actually holds.
# Declared waits are exact now, but they are not the whole cost: every
# gesture step spends real arm-and-camera time (measured 5-8s per tap on the
# reference rig), which no per-step cap covers. Nothing else in the stack
# catches a long run either — the MCP client's read timeout is per request,
# and the session deadline is only checked between turns, so a macro this
# long blocks the engine task and an incoming IM cannot land a turn. One
# macro must never own half a session.
MAX_RUN_SECONDS = 300
# How deep the combinators may nest inside ONE check. Two, because that is
# where real macros already sit: the shipped `init` scaffold and the docs
# example both top out at ONE level, and `{not: {or: ["Upgrade", "升级"]}}`
# — no language variant of the popup — is the deepest idiom worth writing.
# Three means tracing brackets, and an unreadable check is one nobody
# verifies against the screen; the failure mode is a guard that silently
# always passes, which reads exactly like a guard that passed.
#
# The cap costs authors nothing because the format flattens for them: a
# guard conjoins `require` and `forbid` for free, `forbid: X` IS
# `require: {not: X}`, and `within` scopes a whole subtree so alternatives
# need no wrapper each. What is left over belongs in two steps.
#
# `not` counts like the binary operators: exempting it would leave
# `{not: {not: ...}}` as an unbounded escape hatch around the cap. The cap
# is on NESTING only — an `or` may list as many alternatives as it likes.
MAX_CLAUSE_DEPTH = 2

MACRO_FILENAME = "MACRO.yml"

# Abort reasons. They live in the leaf module rather than in `runner`
# because `stats` needs to tell them apart (a `bad_input` run never reached
# the screen, so it must not move the re-rehearse streak) and `runner`
# already imports `stats`. `runner` re-exports these under the same names.
REASON_TOOL_ERROR = "tool_error"
REASON_GUARD_FAILED = "guard_failed"
# A step's `expect` postcondition did not hold. Distinct from
# `guard_failed` because the two say different things to whoever reads the
# stats: a guard means the macro was not safe to CONTINUE, an expect means
# the step did not DO what it was written to do.
REASON_EXPECT_FAILED = "expect_failed"
REASON_BAD_INPUT = "bad_input"
REASON_TIMEOUT = "timeout"

# The macro-local settle step. NOT an MCP tool — the runner sleeps
# in-process rather than paying a round trip to block the arm server — and
# NOT in the shared gesture vocabulary: no classifier outside this package
# names it, and the engine has an unrelated LOCAL tool that also answers to
# "wait", so a vocab constant would be ambiguous in the very module that
# exists to disambiguate names. It exists because a guard should be a pure
# predicate: waiting is a separate concern from checking, and fusing them
# into `wait_seconds` made a duration that was really a poll count.
WAIT = "wait"
# The `wait` argument shape: {"seconds": N}. `expect`/`hint` are step-level
# keys, not arguments — see `parse`.
WAIT_SECONDS_ARG = "seconds"

# The gesture/perception surface a step may call — rehearsable physical
# actions only, composed from the shared tool-name vocabulary (no private
# name copies). Session control, memory, jobs (local tools) and
# `unlock_phone` / `sequence` / `screenshot` stay with the agent: unlock is
# security-sensitive, `sequence` is the anonymous server-side batch this
# layer replaces, and `screenshot` rides the slow AssistiveTouch upload
# path.
ALLOWED_STEP_TOOLS = frozenset(
    gesture_vocab.PRESS_TOOLS
    | gesture_vocab.NAV_TOOLS
    | {
        gesture_vocab.SWIPE,
        gesture_vocab.SEND_TO_CLIPBOARD,
        gesture_vocab.PEEK,
        WAIT,
    }
)

# `wait` never reaches the MCP server — the runner sleeps in-process. Every
# other allowed step is a real server tool, which `tests/agent/macros/
# test_parse.py` pins against the live registration.
LOCAL_STEP_TOOLS = frozenset({WAIT})

# Macro names follow the skill-folder convention (lowercase/digits/hyphens,
# no leading/trailing/consecutive hyphens); input names double as `{name}`
# placeholders, so they follow identifier rules instead.
NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
INPUT_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class MacroError(TemplateError):
    """A macro definition or invocation is invalid. Subclasses
    `TemplateError` so a malformed `{name}` reports as one of these. Message is user-facing:
    `physiclaw macros check` prints it verbatim, and the engine returns it
    as the tool error on a bad `run_macro` call."""


@dataclass(frozen=True)
class MacroInput:
    name: str
    description: str
    default: str | None = None  # present → the input is optional
    example: str | None = None

    @property
    def required(self) -> bool:
        return self.default is None


if TYPE_CHECKING:  # `Macro` names its steps; `steps` imports this module.
    from physiclaw.agent.macros.steps import Step


# The YAML keys for the three combinators. They name the GRAMMAR; the
# classes below carry the behaviour, so there is no runtime tag to switch
# on. `COMBINATORS` (defined under the classes) is the one map from key to
# constructor, so a new operator is a class plus one entry — never a
# dispatch edited in several places, and never an error message listing
# operators by hand.
AND, OR, NOT = "and", "or", "not"

Bbox = tuple[float, float, float, float]


@dataclass(frozen=True)
class Screen:
    """One reading of the phone screen, parsed once and asked many times.

    Built from the element listing (`id [kind] "label" [l,t,r,b] conf`).
    Two haystacks fall out of it, and the distinction is load-bearing:

    `content` is what a WHOLE-SCREEN clause matches — labels only, with the
    listing's own syntax removed. Matching the raw listing meant
    `require: "conf"` hit the header on every screen and `forbid: "12"`
    tripped on the coordinate `0.128`; both directions are silent, and a
    guard that always passes reads exactly like a guard that passed.

    `rows` is what a REGION clause matches — label plus box, so a match can
    be required to sit where it was rehearsed."""

    text: str
    content: str
    rows: tuple[tuple[str, Bbox], ...]

    @classmethod
    def read(cls, text: str) -> "Screen":
        content: list[str] = []
        rows: list[tuple[str, Bbox]] = []
        for line in text.splitlines():
            row = ROW_PARSE_RE.match(line)
            if row:
                content.append(row.group(1))
                try:
                    left, top, right, bottom = (
                        float(v) for v in row.group(2).split(",")
                    )
                except ValueError:
                    continue  # malformed coords — not a real row
                rows.append((row.group(1), (left, top, right, bottom)))
            elif line.strip() and line.strip() != LISTING_HEADER:
                # Not a listing at all (a plain text block) — keep it, or a
                # clause could never match non-listing content.
                content.append(line)
        return cls(text=text, content="\n".join(content), rows=tuple(rows))

    @property
    def readable(self) -> bool:
        """False for a failed camera read. Every consumer must decide what
        an unreadable screen means BEFORE evaluating, because `not` is
        satisfied by an empty haystack."""
        return bool(self.text)


BLANK_SCREEN = Screen(text="", content="", rows=())


class Clause(ABC):
    """A boolean expression over the screen — the one shape `require`,
    `forbid`, `expect` and `skip_when` all take.

    The three combinators are spelled out rather than implied by nesting
    depth. An earlier grammar made a bare list mean "any of these", so
    ``[["a","b"]]`` was OR while ``["a","b"]`` was AND — one bracket apart,
    opposite meanings, and no way to say `not` at all.

    Subclasses answer three questions and nothing else: does it hold, how
    does it read in an abort report, and what does it look like with inputs
    filled in. Adding an operator means adding a class, not editing a
    dispatch in four places."""

    @abstractmethod
    def holds(self, screen: Screen) -> bool:
        """Whether this clause is satisfied by `screen`."""

    @abstractmethod
    def display(self) -> str:
        """How the clause reads in a step log or abort detail."""

    @abstractmethod
    def substituted(self, values: dict[str, str]) -> "Clause":
        """A copy with `{name}` placeholders resolved in every text leaf."""

    def walk(self) -> "list[Clause]":
        """This clause and every descendant — for validation passes that
        must reach leaves wherever they are nested."""
        return [self]


@dataclass(frozen=True)
class TextClause(Clause):
    """A leaf: `text` must appear on screen. With `within`, the match is
    element-granular — a listing row whose label matches AND whose CENTER
    falls inside the region (how taps target elements, and tolerant of OCR
    box jitter)."""

    text: str
    within: Bbox | None = None

    def holds(self, screen: Screen) -> bool:
        if self.within is None:
            return self.text in screen.content
        left, top, right, bottom = self.within
        for label, (el, et, er, eb) in screen.rows:
            if not self._label_matches(label):
                continue
            cx, cy = (el + er) / 2, (et + eb) / 2
            if left <= cx <= right and top <= cy <= bottom:
                return True
        return False

    def _label_matches(self, label: str) -> bool:
        """Substring for normal texts; WHOLE-label equality for single
        characters. Keyboard keys OCR as standalone one-letter elements,
        while a single char as a substring would match inside almost any
        chat bubble — exact matching is what makes letter-key anchors
        usable (parse rejects single chars outside the region form for the
        same reason)."""
        if len(self.text) == 1:
            return label.strip() == self.text
        return self.text in label

    def display(self) -> str:
        if self.within is None:
            return repr(self.text)
        coords = ",".join(f"{v:g}" for v in self.within)
        return f"{self.text!r} within [{coords}]"

    def substituted(self, values: dict[str, str]) -> "Clause":
        return replace(self, text=fill(self.text, values))


@dataclass(frozen=True)
class _Combinator(Clause):
    """Shared plumbing for `and`/`or` — two or more children, joined by a
    word. One child would just be the child, which parse rejects."""

    children: tuple[Clause, ...]
    JOINER: ClassVar[str]  # set by each concrete operator

    def walk(self) -> list[Clause]:
        out: list[Clause] = [self]
        for c in self.children:
            out.extend(c.walk())
        return out

    def display(self) -> str:
        return "(" + self.JOINER.join(c.display() for c in self.children) + ")"

    def substituted(self, values: dict[str, str]) -> "Clause":
        return replace(
            self, children=tuple(c.substituted(values) for c in self.children)
        )


@dataclass(frozen=True)
class AndClause(_Combinator):
    JOINER = " and "

    def holds(self, screen: Screen) -> bool:
        return all(c.holds(screen) for c in self.children)


@dataclass(frozen=True)
class OrClause(_Combinator):
    JOINER = " or "

    def holds(self, screen: Screen) -> bool:
        return any(c.holds(screen) for c in self.children)


@dataclass(frozen=True)
class NotClause(Clause):
    """The one operator an EMPTY haystack satisfies — which is why every
    consumer decides what an unreadable screen means before evaluating:
    guards and `expect` fail closed, `skip_when` declines to skip."""

    child: Clause

    def holds(self, screen: Screen) -> bool:
        return not self.child.holds(screen)

    def walk(self) -> list[Clause]:
        return [self, *self.child.walk()]

    def display(self) -> str:
        return f"not {self.child.display()}"

    def substituted(self, values: dict[str, str]) -> "Clause":
        return replace(self, child=self.child.substituted(values))


# key → (constructor over the parsed children, minimum children). `NOT`
# takes exactly one child; the binary operators need two, since one child
# would just be the child. One uniform signature — children in, clause
# out — so `parse` builds every operator the same way and a new one is
# still a class plus one entry.
COMBINATORS: dict[str, tuple[Callable[[tuple[Clause, ...]], Clause], int]] = {
    AND: (lambda kids: AndClause(children=kids), 2),
    OR: (lambda kids: OrClause(children=kids), 2),
    NOT: (lambda kids: NotClause(child=kids[0]), 1),
}


# The two shapes carrying `{name}` placeholders through `sub`.
_S = TypeVar("_S", "Clause", "MacroGuard")


def sub(value: _S | None, values: dict[str, str]) -> _S | None:
    """`value.substituted(values)`, passing None through. Every optional
    check field needs it, so the ternary is written once."""
    return None if value is None else value.substituted(values)


@dataclass(frozen=True)
class MacroGuard:
    """Pre-step gate: `require` must hold AND `forbid` must not, else abort.
    `hint` rides into the abort report to steer the agent's recovery.

    Both are ONE `Clause`, the shape every check in the format takes — no
    per-field shape to remember. They were lists once (implicitly AND-ed),
    and `forbid` was a flat list of bare strings; that made bracket shape
    carry meaning, which is the trap the explicit combinators exist to
    kill. Two conditions are now spelled `{and: [...]}`, and `forbid: X`
    stays exactly `require: {not: X}` — kept because the popup-tripwire
    reading earns a name, not because it needs a different shape.

    A guard is a pure predicate: it checks once and either passes or
    aborts. Waiting is a separate step (`WaitStep`)."""

    require: Clause | None = None
    forbid: Clause | None = None
    hint: str = ""

    def check(self, screen: Screen) -> str:
        """ "" when the gate opens, else why it did not — the detail that
        rides into the step log, the abort report, and stats."""
        parts = []
        if self.require is not None and not self.require.holds(screen):
            parts.append(f"require {self.require.display()} not on screen")
        if self.forbid is not None and self.forbid.holds(screen):
            parts.append(f"forbid {self.forbid.display()} on screen")
        detail = "; ".join(parts)
        if detail and self.hint:
            detail = f"{detail} (hint: {self.hint})"
        return detail

    def substituted(self, values: dict[str, str]) -> "MacroGuard":
        return replace(
            self,
            require=sub(self.require, values),
            forbid=sub(self.forbid, values),
            hint=fill(self.hint, values),
        )


@dataclass(frozen=True)
class Macro:
    """One validated macro. `parse.parse_macro` is the only producer, so a
    Macro is correct by construction: the runner never meets an unknown
    tool, a bad clause shape, or a dangling placeholder mid-replay."""

    name: str
    description: str
    enabled: bool
    inputs: tuple[MacroInput, ...]
    steps: tuple["Step", ...]


def check_name(name: str, where: str = "name", extra: str = "") -> None:
    """Enforce the identifier rule (the skill-folder convention). Shared by
    `parse.parse_macro`, step names, and `macros init`, so a name that
    scaffolds also parses and the three can never drift. Raises
    MacroError."""
    if len(name) > MAX_NAME_LEN or not NAME_RE.match(name):
        raise MacroError(
            f"{where} {name!r} must be lowercase letters/digits/hyphens, "
            f"≤{MAX_NAME_LEN} chars, no leading/trailing/consecutive "
            f"hyphens{extra}"
        )
