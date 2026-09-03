"""Offline replay — the real walk over recorded screens, no phone.

Every session records its listings; a replay feeds them, in order, to
a dry `Program` as the results of the actions it synthesizes, and
reports what the walk would have done: which node acted, on which
verdict, with which tool, and where it stopped. Nothing is simulated —
the same executors, matcher, lints, and money predicates run — so a
replay answers "where would this playbook hand over on that session"
without the arm, and every real failure becomes a regression case.

A pure-text agent step resolves from `outputs` (`node.field` → value,
what the author says the model would have answered); any other model
decision stops the replay at that node — the deterministic prefix is
what a replay proves. The program is built `dry`: no runs.jsonl line,
no daily-log entry, no suspension file.
"""

from dataclasses import dataclass

from physiclaw.conductor.calls import AGENT_DONE
from physiclaw.conductor.limits import MAX_REPLAY_TURNS
from physiclaw.conductor.micro import AGENT_FIELDS, DecisionRequest, MicroOutcome
from physiclaw.conductor.playbook import AgentNode
from physiclaw.conductor.program import Program
from physiclaw.conductor.step import Paused
from physiclaw.contract.dto import SystemMessage, ToolResultMessage, UserMessage


@dataclass(frozen=True)
class ReplayTurn:
    """One synthesized action: the node at the cursor, what the screen
    read as when it was decided, the tool it dispatched, and the note."""

    node: str | None
    verdict: str
    tool: str
    note: str


@dataclass(frozen=True)
class Replay:
    turns: tuple[ReplayTurn, ...]
    outcome: str  # completed | handover | suspended | stopped
    detail: str  # the brief, or why the replay stopped


def replay(
    program: Program, screens: list[str], outputs: dict[str, str] | None = None
) -> Replay:
    """Drive `program` (built dry) over `screens`, each the result of the
    walk's next action, until it finishes, hands over, suspends, needs a
    decision the replay cannot supply, or runs out of screens."""
    outputs = outputs or {}
    history: list = [
        SystemMessage(content="replay"),
        UserMessage(content="replay the walk over recorded screens"),
    ]
    turns: list[ReplayTurn] = []
    feed = iter(screens)
    for _ in range(MAX_REPLAY_TURNS):
        step = program.advance(history)
        while isinstance(step, DecisionRequest):
            answer = _answer(program, step, outputs)
            if answer is None:
                return Replay(
                    tuple(turns),
                    "stopped",
                    f"needs a model decision at agent {step.node_id!r} "
                    f"({step.call}) — supply its outputs, or stop here",
                )
            step = program.resolve(answer)
        if step is None or isinstance(step, Paused):
            return Replay(tuple(turns), program.outcome or "stopped", "went quiet")
        note, act = step.tool_calls
        summary = str(note.arguments.get("summary", ""))
        node = program.node
        turns.append(
            ReplayTurn(
                node=node.id if node is not None else None,
                verdict=_verdict_text(program),
                tool=act.name,
                note=summary,
            )
        )
        if program.outcome is not None:
            # A terminal turn (the brief's peek, the suspension's
            # end_session): the walk's outcome is already recorded.
            return Replay(tuple(turns), program.outcome, summary)
        screen = next(feed, None)
        if screen is None:
            return Replay(
                tuple(turns),
                "stopped",
                f"screens exhausted before the walk ended (at {act.name})",
            )
        history.append(step)
        history.append(
            ToolResultMessage(tool_call_id=act.id, content=screen, is_error=False)
        )
    return Replay(tuple(turns), "stopped", f"{MAX_REPLAY_TURNS} turns without an end")


def _answer(
    program: Program, req: DecisionRequest, outputs: dict[str, str]
) -> MicroOutcome | None:
    """A pure-text agent's answer from the supplied outputs — every
    declared return field present — else None (the replay stops)."""
    node = program.node
    if (
        req.call != AGENT_FIELDS
        or not isinstance(node, AgentNode)
        or node.id != req.node_id
    ):
        return None
    payload = {}
    for field in node.return_fields:
        value = outputs.get(f"{node.id}.{field}")
        if value is None:
            return None
        payload[field] = value
    return MicroOutcome(
        out=AGENT_DONE, reason="replayed", confidence=1.0, payload=payload
    )


def _verdict_text(program: Program) -> str:
    v = program.verdict
    if v is None:
        return "(no screen yet)"
    return f"{v.kind} {v.page_id or '-'}"
