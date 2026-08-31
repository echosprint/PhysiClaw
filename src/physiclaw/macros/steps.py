"""The executable step hierarchy — one class per kind of thing a macro
does, each answering `execute`.

This is where the DSL stops being data and starts touching the rig, so it
sits above `model` (pure clause algebra) and below `runner` (which owns
the loop and the run-wide state). The split is what keeps `model`
importable from anywhere.

Two kinds, and the difference is not cosmetic:

    GestureStep   one MCP call — the rehearsed physical action
    WaitStep      an in-process sleep, then one assertion

`expect` lives on `WaitStep` alone, as a field rather than a validation
rule, so "expect is wait-only" is a fact about the type instead of a
check someone could forget to run. It is wait-only because a gesture's
own view is captured ~2s after the touch and is the SAME frame the next
step's guard reads for free — an `expect` there would assert nothing new.

Every `execute` returns a `StepOutcome` rather than raising, because a
mid-run failure is a substantive result the agent must read (with the
screen to work from), not an error to retry.
"""

import asyncio
import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from physiclaw.common import gesture_vocab, verdict
from physiclaw.common.bbox import center_of
from physiclaw.common.listing import label_hit
from physiclaw.macros.inputs import substitute
from physiclaw.macros.model import (
    BLANK_SCREEN,
    REASON_EXPECT_FAILED,
    REASON_GUARD_FAILED,
    REASON_TOOL_ERROR,
    TARGET_BBOX,
    TARGET_LABEL,
    WAIT,
    Clause,
    MacroGuard,
    Screen,
    label_readings,
    sub,
)
from physiclaw.macros.template import fill

log = logging.getLogger(__name__)

# How far (normalized screen distance between centers) a label match may
# sit from the recorded bbox and still count as the SAME element. The
# healing rule from the self-healing-locator literature, adapted: heal
# small drift (a banner pushed content down, a layout shifted), refuse
# teleports — a matching label far across the screen is more likely a
# DIFFERENT element (another dialog's confirm button) than a moved one,
# and off-radius falls back to the recorded coordinates exactly like a
# miss. Fixed, not authorable: the trust boundary is the format's, not a
# per-macro knob.
HEAL_RADIUS = 0.2


class McpCaller(Protocol):
    async def call_tool(
        self, name: str, args: dict[str, Any] | None = None
    ) -> list[dict]: ...


@dataclass
class RunContext:
    """Everything one run carries between steps.

    The mutable half of the runner, kept in one object so a step's
    `execute` can read and update it without the loop threading eight
    locals through every call. Two invariants live here and nowhere else.

    `screen` is the most recent reading, or BLANK when we hold none — every
    check is keyed on "do we know the screen", never on the step index, so
    a check never rules on an empty haystack.

    `view_stale` says whether `last_view` still depicts the CURRENT screen.
    A `wait` sleeps precisely because the screen is expected to change, so
    the frame held afterwards is out of date by construction; shipping it
    under "the view below is the current screen" would hand the agent — and
    through dispatch's screen supersede, its whole notion of "now" — a
    frame that is `seconds` old."""

    mcp: McpCaller
    deadline: float
    screen: Screen = BLANK_SCREEN
    last_view: list[dict] = field(default_factory=list)
    view_stale: bool = False
    last_verdict: bool | None = None
    reads: int = 0  # camera cycles this step spent, whatever the outcome

    def adopt_view(self, blocks: list[dict], *, touched_screen: bool = True) -> None:
        """Take a step's fused view as the current screen."""
        self.last_view = blocks
        self.view_stale = False
        if verdict.has_image(blocks):
            self.screen = Screen.read(verdict.screen_text(blocks))
        elif touched_screen:
            # A gesture that came back without a view still actuated, so the
            # retained listing is now stale — drop it and let the next check
            # read. A step that never touches the screen keeps it, which is
            # what makes the documented paste flow's next check free.
            self.screen = BLANK_SCREEN

    def invalidate_screen(self) -> None:
        """The held listing is no longer trustworthy, and neither is the
        view we would otherwise ship as current."""
        self.screen = BLANK_SCREEN
        self.view_stale = True

    async def read_screen(self, *, retry: bool = False) -> Screen:
        """The screen — free when we already hold one, else one camera read.

        `retry` gives a failed READ a second chance: a camera hiccup is not
        a failed check, and reporting one as `… not on screen` sends the
        agent to fix a screen nobody managed to look at."""
        if self.screen.readable:
            return self.screen
        for _ in range(2 if retry else 1):
            self.reads += 1
            try:
                view = await self.mcp.call_tool(gesture_vocab.PEEK, {})
            except Exception as e:
                log.warning("macro peek failed: %s", e)
                continue
            self.adopt_view(view)
            if self.screen.readable:
                return self.screen
        return BLANK_SCREEN

    @property
    def has_view(self) -> bool:
        """Whether the held view carries an image worth shipping."""
        return verdict.has_image(self.last_view)

    @property
    def out_of_time(self) -> bool:
        return time.monotonic() > self.deadline


# The two outcomes that do not stop the run; everything else is a REASON_*.
_CONTINUES = frozenset({"ok", "skipped"})


@dataclass(frozen=True)
class StepOutcome:
    """What one step did. `reason` is set iff the run must stop here."""

    log_line: str
    # The run-log verb, and — for a stopping outcome — the abort reason.
    # They were two fields set to the same string at every site; one name
    # for one value means they cannot disagree.
    outcome: str  # ok | skipped | REASON_*
    detail: str = ""
    verdict: bool | None = None
    view: list[dict] | None = None  # blocks worth logging for this step
    screen_text: str = ""  # the haystack a failed check actually saw
    # Set when the step FIRED with different coordinates than authored
    # (a healed press) — the forensic run log records these instead, so
    # `macros runs` answers "where did the tap actually land". None =
    # the authored args are the truth.
    fired_args: dict[str, Any] | None = None

    @property
    def reason(self) -> str | None:
        return None if self.outcome in _CONTINUES else self.outcome

    @property
    def stop(self) -> bool:
        return self.reason is not None


@dataclass(frozen=True)
class Step(ABC):
    """A named step with its optional checks.

    `guard` runs BEFORE the step (a precondition), `skip_when` runs before
    that (idempotence — if the postcondition already holds, the step is
    not needed). Both are one clause; see `model.Clause`."""

    name: str  # required and unique per macro — what `start_at` addresses
    guard: MacroGuard | None = None
    # Idempotence, the Ansible creates/unless model, NOT general branching:
    # when already true the step is skipped, because executing it would be
    # redundant or harmful (e.g. tapping the keyboard-hidden input-box
    # position while the keyboard is up hits the keys). Author contract:
    # skip state == post-execution state, so later bboxes stay synchronized.
    skip_when: Clause | None = None

    @property
    @abstractmethod
    def tool(self) -> str:
        """The name shown in logs and reports. For a gesture it is the MCP
        tool; `wait` never reaches the server."""

    @abstractmethod
    async def execute(self, ctx: RunContext) -> StepOutcome:
        """Do the thing, having already passed `guard`."""

    @abstractmethod
    def substituted(self, values: dict[str, str]) -> "Step":
        """A copy with `{name}` resolved everywhere — args AND checks. A
        check about an input is the whole point of parameterizing: a macro
        that pastes to `{contact}` wants to verify it landed in
        `{contact}`'s chat."""

    def display(self) -> str:
        return f'{self.tool} "{self.name}"'

    @property
    def log_args(self) -> dict[str, Any]:
        """What the run log records for this step. A step with no wire
        arguments records none — the loop must not reach into a subclass."""
        return {}

    @property
    def declared_seconds(self) -> int:
        """Sleep this step promises up front, summed by `parse` against the
        run budget. A future sleeping step kind is counted automatically."""
        return 0


@dataclass(frozen=True)
class GestureStep(Step):
    """One rehearsed physical action: exactly one MCP call, whose fused
    view (image + listing) becomes the screen the next check reads."""

    mcp_tool: str = ""
    args: dict[str, Any] = field(default_factory=dict)  # the `with:` table

    @property
    def tool(self) -> str:
        return self.mcp_tool

    @property
    def log_args(self) -> dict[str, Any]:
        return self.args

    async def execute(self, ctx: RunContext) -> StepOutcome:
        args, heal_note = self._lowered(ctx.screen)
        try:
            blocks = await ctx.mcp.call_tool(self.mcp_tool, args)
        except Exception as e:
            log.warning("macro step %s (%s) failed: %s", self.name, self.tool, e)
            return StepOutcome(
                log_line=f"✗ {self.display()} — {e}",
                outcome=REASON_TOOL_ERROR,
                detail=str(e),
            )
        # Verdict only from the first text block (`verdict.action_text`) —
        # core-composed action text, the one haystack on-screen content
        # cannot forge.
        changed = verdict.parse(verdict.action_text(blocks))
        ctx.adopt_view(blocks, touched_screen=self.touches_screen)
        healed = args.get(TARGET_BBOX) != self.args.get(TARGET_BBOX)
        return StepOutcome(
            log_line=f"✓ {self.display()}{_verdict_note(changed)}{heal_note}",
            outcome="ok",
            verdict=changed,
            view=blocks,
            # Label kept beside the HEALED bbox: the forensic record stays
            # an annotated target, just at the coordinates that fired.
            fired_args={**self.args, TARGET_BBOX: args[TARGET_BBOX]}
            if healed
            else None,
        )

    def _lowered(self, screen: Screen) -> tuple[dict[str, Any], str]:
        """The wire arguments plus the log note explaining them.

        `label` is the author's half of the target — the server knows
        only `bbox`, so it is always stripped. For a PRESS gesture with a
        readable screen it first gets its chance to HEAL: if a text row
        matching a declared reading sits within HEAL_RADIUS of the
        recorded center, the press goes where that text is TODAY (the
        recorded bbox is the fallback, never the assertion — asserting
        presence is `guard`'s job). A heal is always noted in the step
        log: a step that heals every run is a step waiting to be
        re-recorded.

        Matching is the base rule (`label_hit`) ONLY — deliberately the
        same regime every `guard`/`skip_when` clause sees, so a heal can
        never match text a guard could not. The richer tiers (normalize,
        variants, fuzzy) are the conductor's, behind the macros-never-
        import-conductor boundary; an OCR-mangled label simply doesn't
        heal, which the silent fallback absorbs by design."""
        if TARGET_LABEL not in self.args:
            return self.args, ""
        args = dict(self.args)
        del args[TARGET_LABEL]
        if self.mcp_tool not in gesture_vocab.PRESS_TOOLS or not screen.readable:
            return args, ""
        readings = label_readings(self.args)
        recorded = center_of(args[TARGET_BBOX])
        if recorded is None:
            # Only a substituted-placeholder bbox can get here (literal
            # targets are `_bbox`-validated at parse) — no target center,
            # no healing; the server judges the value.
            return args, ""
        best: tuple[float, Any] | None = None
        for row in screen.rows:
            if row.kind != "text":
                continue
            if not any(label_hit(t, row.label) for t in readings):
                continue
            c = center_of(row.bbox)
            assert c is not None  # Element bboxes are valid by construction
            d = math.dist(c, recorded)
            if best is None or d < best[0]:
                best = (d, row)
        if best is None:
            return args, ""
        dist, row = best
        if dist > HEAL_RADIUS:
            # A matching label far from the recorded spot is suspicious,
            # not a heal — noted so a wrong-element story is debuggable.
            return args, f" (label {row.label.strip()!r} off-radius — recorded bbox)"
        args[TARGET_BBOX] = list(row.bbox)
        return args, f" (label {row.label.strip()!r} healed, drift {dist:.2f})"

    @property
    def touches_screen(self) -> bool:
        """`send_to_clipboard` writes the pasteboard and nothing else, so
        the listing already held stays valid across it — that is what makes
        the documented paste flow's next check free."""
        return self.mcp_tool != gesture_vocab.SEND_TO_CLIPBOARD

    def substituted(self, values: dict[str, str]) -> "Step":
        return replace(
            self,
            args=substitute(self.args, values),
            guard=sub(self.guard, values),
            skip_when=sub(self.skip_when, values),
        )


@dataclass(frozen=True)
class WaitStep(Step):
    """A settle: sleep exactly `seconds`, then optionally confirm what
    arrived.

    Waiting used to live inside the guard as `wait_seconds`, which polled
    `ceil(n/2)` times at one camera cycle each — so the number was a poll
    budget wearing a duration's name (8 meant ~15s on the reference rig)
    and odd values aliased to even ones. Splitting them made both honest.

    `seconds: 0` is legal only WITH an `expect`, where it reads as "check
    now"; a wait that neither sleeps nor checks does nothing at all."""

    seconds: int = 0
    expect: Clause | None = None
    hint: str = ""

    @property
    def tool(self) -> str:
        return WAIT

    @property
    def declared_seconds(self) -> int:
        return self.seconds

    async def execute(self, ctx: RunContext) -> StepOutcome:
        await asyncio.sleep(self.seconds)
        # The held listing is now `seconds` out of date BY CONSTRUCTION —
        # waiting exists precisely because the screen is expected to change.
        # Keeping it would hand the next check the very screen this step was
        # written to move past.
        ctx.invalidate_screen()
        waited = f"✓ {self.display()} — waited {self.seconds}s"
        if self.expect is None:
            return StepOutcome(log_line=waited, outcome="ok")

        screen = await ctx.read_screen(retry=True)
        if not screen.readable:
            # An unreadable screen is never a satisfied assertion: `not` is
            # satisfied by an empty haystack, so a camera hiccup would
            # otherwise "confirm" an absence nobody saw.
            detail = "could not read the screen (`peek` failed)"
        elif self.expect.holds(screen):
            return StepOutcome(
                log_line=f"{waited}, expect ok", outcome="ok", view=ctx.last_view
            )
        else:
            detail = f"expected {self.expect.display()} — not on screen"
            if self.hint:
                detail = f"{detail} (hint: {self.hint})"
        # Replace the step's ✓ rather than adding a ✗ beside it: the step
        # DID run, but one step gets one verdict.
        return StepOutcome(
            log_line=f"✗ {self.display()} — {detail}",
            outcome=REASON_EXPECT_FAILED,
            detail=detail,
            view=ctx.last_view,
            screen_text=screen.text,
        )

    def substituted(self, values: dict[str, str]) -> "Step":
        return replace(
            self,
            expect=sub(self.expect, values),
            hint=fill(self.hint, values),
            guard=sub(self.guard, values),
            skip_when=sub(self.skip_when, values),
        )


def guard_outcome(step: Step, detail: str, screen_text: str) -> StepOutcome:
    """The shared shape of a failed precondition, so the two callers (the
    guard check and its report) cannot word it differently."""
    return StepOutcome(
        log_line=f"✗ {step.display()} — guard: {detail}",
        outcome=REASON_GUARD_FAILED,
        detail=detail,
        screen_text=screen_text,
    )


def _verdict_note(changed: bool | None) -> str:
    if changed is None:
        return ""
    return " (changed)" if changed else " (no change)"
