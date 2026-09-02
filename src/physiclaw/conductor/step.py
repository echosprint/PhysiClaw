"""The step executor — what every route entry kind implements.

The walk (`program.py`) owns the cursor, the one action in flight, the
page verdict, and recovery; a step owns everything about ONE node at
the cursor. The walk asks a step three questions and routes by the
pending-action kinds the step declares:

  - `open()`       the cursor arrived — the step's first turn
  - `landed(kind)` one of its actions landed; the result view is in
                   `walk.screen` / `walk.verdict`
  - `resolve()`    the micro request it returned came back

A blocked or errored action is not a question: the walk hands over
with the error (nothing retries in the background). A step ends by
`walk.advance_cursor()` or `walk.handover(...)`; the walk never
inspects a step's private state.
"""

from typing import TYPE_CHECKING, Generic, TypeVar

from physiclaw.conductor.micro import DecisionRequest, MicroOutcome
from physiclaw.contract.dto import AssistantMessage

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

    def open(self) -> Turn:
        raise NotImplementedError

    def landed(self, kind: str) -> Turn:
        raise NotImplementedError

    def resolve(self, outcome: MicroOutcome | None) -> Turn:
        name = getattr(self.node, "id", type(self).__name__)
        return self.walk.handover(f"{name!r} brokered no decision")
