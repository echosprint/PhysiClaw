"""Synthesized turns — the shape the conductor speaks to the loop in.

Everything the conductor does to the phone, it does by MINTING A TURN: a
``[note, one-other]`` assistant message, exactly the shape the loop
enforces on model turns, so dispatch, the phone-protection guards,
compaction, and the wire log all see an ordinary turn (`loop.py` skips
only the judgment gates, off `AssistantMessage.synthesized`). The turn
is not sent anywhere — the loop dispatches its tool calls and the
results come back as ordinary history, which the next ``advance()``
reads by call id (`views.result_for`).

That makes one action outstanding at a time, and `Pending` is the memo:
which action, under which call id, and whether it lands on the user
thread rather than the pack's own pages (the synth site declares that —
a reader must never re-derive it from kind names).

`Turnsmith` owns the whole round trip — mint (`synth`) and settle
(`settle`) — because both conductor drivers need it: the playbook walk
(`program.py`) and the boot to the thread (`overture.py`). Neither one
should re-spell the call-id convention, the message shape, or how a
result is found; a drift there is invisible until the loop rejects a
turn or a lookup silently misses.

Deliberately NOT here: what a kind MEANS, the decision journal, whether
the walk has acted yet, and what to DO about a failure. Those are the
drivers' vocabulary and policy; this module knows only how to mint a
turn, remember the one in flight, and find what it came back as.
"""

from dataclasses import dataclass

from physiclaw.conductor import views
from physiclaw.contract.dto import (
    AssistantMessage,
    FinishReason,
    Message,
    ToolCall,
    ToolResultMessage,
)

# Call-id prefix for every synthesized action. Stable on purpose: it is
# how a recorded session's tool calls are recognized as conductor-made.
CALL_PREFIX = "conductor"

# How much of a blocked call's text a hand-over reason quotes. Enough to
# name the cause, short enough not to paste a screen into a log line.
MAX_ERROR_CHARS = 200


@dataclass(frozen=True)
class Pending:
    """The synthesized action whose result the next advance() must read.

    A driver's cursor never moves while an action is pending, so a
    pending leg (or the decide a "swipe" re-asks) is always the node the
    cursor sits on; a pending "tap" already had its cursor routed at
    resolve time. `channel` marks actions that land on the user thread —
    the synth site declares it, so the reader never re-derives it from
    kind names."""

    kind: str
    call_id: str
    channel: bool = False


class Turnsmith:
    """Mints synthesized turns and remembers the one action in flight."""

    def __init__(self) -> None:
        self.pending: Pending | None = None
        self._seq = 0

    def settle(
        self, history: list[Message]
    ) -> "tuple[Pending, ToolResultMessage | None, str | None]":
        """The other half of the contract: read the pending action's
        result out of the transcript and retire it.

        Returns `(what was pending, its result, why there is none)` —
        exactly one of the last two is set. The action is cleared only
        when a result actually landed, so a driver that failed still
        knows what it was waiting on. Callers keep their own policy
        (the walk drops a suspension; the boot spends no retry budget);
        what lives here is where a result is found and how a missing or
        blocked one is phrased, so the two drivers cannot word — or
        truncate — the same failure differently."""
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
        compose their own prose, so this stays free of any driver's
        vocabulary."""
        self._seq += 1
        cid = f"{CALL_PREFIX}-{self._seq}"
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
