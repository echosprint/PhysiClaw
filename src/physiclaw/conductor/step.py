"""The step executor — what every route entry kind implements.

The walk (`program.py`) owns the cursor, the one action in flight, the
page verdict, and recovery; a step owns everything about ONE node at
the cursor. The walk asks a step three questions and routes by the
pending-action kinds the step declares:

  - `open()`       the cursor arrived — the step's first turn
  - `landed(kind)` one of its actions landed; the result view is in
                   `walk.screen` / `walk.verdict`
  - `resolve()`    the micro request it returned came back

plus `failed(kind, error, raw)` when one of its actions was blocked or
errored — None (the default) lets the walk hand over with the error; a
step that can retry (a move under a popup, a locked phone) answers a
turn instead. A step ends by `walk.advance_cursor()` or
`walk.handover(...)`; the walk never inspects a step's private state.
"""

from typing import TYPE_CHECKING, Generic, TypeVar

from physiclaw.conductor.micro import DecisionRequest, MicroOutcome
from physiclaw.contract.dto import AssistantMessage, ToolResultMessage

if TYPE_CHECKING:
    from physiclaw.conductor.program import Program

# What every walk call answers: a synthesized turn, a micro request for
# the conductor to broker, or None — hand over.
Turn = AssistantMessage | DecisionRequest | None

N = TypeVar("N")


class Step(Generic[N]):
    kinds: frozenset[str] = frozenset()

    def __init__(self, walk: "Program", node: N) -> None:
        self.walk = walk
        self.node = node

    @property
    def name(self) -> str:
        node_id = getattr(self.node, "id", None)
        return node_id if node_id is not None else type(self).__name__

    def open(self) -> Turn:
        raise NotImplementedError

    def landed(self, kind: str) -> Turn:
        raise NotImplementedError

    def failed(self, kind: str, error: str, raw: ToolResultMessage | None) -> Turn:
        return None

    def resolve(self, outcome: MicroOutcome | None) -> Turn:
        return self.walk.handover(f"{self.name!r} brokered no decision")
