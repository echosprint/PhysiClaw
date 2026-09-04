"""Synthesized turns — the shape the conductor speaks to the loop in.

Everything the conductor does to the phone, it does by MINTING A TURN:
a ``[note, one-other]`` assistant message, the shape the loop enforces
on model turns, so dispatch, the phone-protection guards, compaction,
and the wire log see an ordinary turn (`AssistantMessage.synthesized`
skips only the judgment gates). The loop dispatches the tool calls and
the results come back as history, read by call id (`views.result_for`).

`Turnsmith` owns the round trip — mint (`synth`) and settle (`settle`)
— and `Pending` is the memo of the one action in flight: its kind, its
call id, and whether it lands on the user thread (the synth site
declares that). What a kind MEANS and what to do about a failure are
the walk's.
"""

from dataclasses import dataclass

from physiclaw.conductor.walk import views
from physiclaw.contract.dto import (
    AssistantMessage,
    FinishReason,
    Message,
    ToolCall,
    ToolResultMessage,
)

# Call-id prefix for every synthesized action. Stable on purpose: it is
# what lets a reader of a recorded session (wire.jsonl, traces) tell
# conductor-made tool calls apart — in-process code never parses ids;
# provenance there is the session's structural `synthesized` bit.
# The full id is `conductor-<scope>-<seq>-<role>`: each walk mints
# under its own scope (its playbook ref), so ids stay unique across the
# session even though the boot and the program it activates each run a
# private Turnsmith — without the scope, both sequences start at 1 and
# `views.result_for` (exact-match, newest-first) could hand a walk the
# OTHER walk's stale result whenever its own failed to land.
CALL_PREFIX = "conductor"

# Where a scroll swipe originates: a mid-content band, clear of top
# chrome and the tab bar, so the drag scrolls the list rather than
# dismissing or paging anything. Stylus up → page scrolls down. Shared
# by the boot's history scroll and an episode's scroll verb.
SCROLL_BBOX = (0.2, 0.35, 0.8, 0.65)


def scroll_args(*, down: bool) -> dict:
    """The `swipe` arguments that scroll the content one notch: `down`
    = see what lies further down (the stylus drags up)."""
    return {"bbox": list(SCROLL_BBOX), "direction": "up" if down else "down"}


# How much of a blocked call's text a hand-over reason quotes. Enough to
# name the cause, short enough not to paste a screen into a log line.
MAX_ERROR_CHARS = 200


@dataclass(frozen=True)
class Pending:
    """The synthesized action whose result the next advance() must read.

    The walk's cursor never moves while an action is pending, so a
    pending move is always the node the cursor sits on. `channel` marks
    actions that land on the user thread —
    the synth site declares it, so the reader never re-derives it from
    kind names."""

    kind: str
    call_id: str
    channel: bool = False


class Turnsmith:
    """Mints synthesized turns and remembers the one action in flight.

    `scope` names the walk this smith belongs to (its playbook ref) and
    rides every call id — required rather than defaulted so two walks
    in one session cannot silently collide on one sequence."""

    def __init__(self, scope: str) -> None:
        self.scope = scope
        self.pending: Pending | None = None
        self._seq = 0

    def settle(
        self, history: list[Message]
    ) -> "tuple[Pending, ToolResultMessage | None, str | None]":
        """The other half of the contract: read the pending action's
        result out of the transcript and retire it.

        Returns `(what was pending, its result, why there is none)` —
        exactly one of the last two is set. The action is cleared only
        when a result actually landed, so a walk that failed still
        knows what it was waiting on. The caller keeps its own policy
        (the walk drops a suspension, hands over); what lives here is
        where a result is found and how a missing or blocked one is
        phrased, so no caller words — or truncates — the same failure
        differently."""
        pending = self.pending
        assert pending is not None, "settle() with no action in flight"
        result = views.result_for(history, pending.call_id)
        if result is None:
            return (
                pending,
                None,
                f"the result of {pending.kind} never arrived in history",
            )
        if result.is_error:
            return (
                pending,
                None,
                f"{pending.kind} was blocked or failed: "
                f"{views.text_of(result)[:MAX_ERROR_CHARS]}",
            )
        self.pending = None
        return pending, result, None

    def synth(
        self, kind: str, summary: str, tool: str, args: dict, *, channel: bool = False
    ) -> AssistantMessage:
        """One ``[note, one-other]`` turn, with its action registered as
        the pending one. The note carries `summary` verbatim — callers
        compose their own prose, so this stays free of the walk's
        vocabulary."""
        self._seq += 1
        cid = f"{CALL_PREFIX}-{self.scope}-{self._seq}"
        self.pending = Pending(kind=kind, call_id=f"{cid}-act", channel=channel)
        return AssistantMessage(
            content="",
            tool_calls=[
                ToolCall(id=f"{cid}-note", name="note", arguments={"summary": summary}),
                ToolCall(id=f"{cid}-act", name=tool, arguments=args),
            ],
            finish_reason=FinishReason.TOOL_CALLS,
            synthesized=True,
        )
