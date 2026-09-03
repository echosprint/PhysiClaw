"""The step executor — what every route entry kind implements, and the
`Walk` contract a step may rely on.

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

`Walk` is the whole surface a step sees of the program — listed here
so the boundary is explicit and type-checked, and so a reader of any
executor knows exactly what it may read and call without opening
`program.py`.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

from physiclaw.conductor.micro import DecisionRequest, MicroOutcome
from physiclaw.contract.dto import AssistantMessage

if TYPE_CHECKING:
    from physiclaw.common.listing import Screen
    from physiclaw.conductor.channel import Channel
    from physiclaw.conductor.gate import Gate
    from physiclaw.conductor.match import Verdict
    from physiclaw.conductor.pages import Landmark
    from physiclaw.conductor.playbook import AgentNode, DoNode, Playbook
    from physiclaw.conductor.recover import Mode


@dataclass(frozen=True)
class Paused:
    """The stepping pause: the walk's cursor left the node this run was
    for, and the walk answers nothing more THIS run. A later run
    rebuilds it from its projection. Distinct from None on purpose —
    None is the driver spent for good (handed over, completed,
    crashed): the conductor drops it and the model speaks."""


# What every walk call answers: a synthesized turn, a micro request for
# the conductor to broker, a stepping pause, or None — spent.
Turn = AssistantMessage | DecisionRequest | Paused | None


class Walk(Protocol):
    """The program as a step sees it. State the step reads, and the
    walk-level moves it calls; nothing else of the program is a step's
    business."""

    app: str
    idx: int
    spec: "Playbook"
    gate: "Gate"
    screen: "Screen | None"
    verdict: "Verdict | None"
    outputs: dict[str, str]
    landmarks: "dict[str, Landmark]"
    channel: "Channel | None"

    def ref_values(self) -> dict[str, str]: ...

    def mismatch(self, verdict: "Verdict", expected_id: str) -> str | None: ...

    def money_page_block(self, what: str) -> str | None: ...

    def spend_consent(self) -> None: ...

    def log_purchase(self) -> None: ...

    def enter_gate(self, node: "DoNode | AgentNode") -> Turn: ...

    def journal(self, text: str) -> None: ...

    def peek(self) -> AssistantMessage: ...

    def synth(
        self, kind: str, summary: str, tool: str, args: dict, *, channel: bool = False
    ) -> AssistantMessage: ...

    def handover(self, reason: str) -> AssistantMessage: ...

    def advance_cursor(self) -> Turn: ...

    def suspend(self, *, resume_idx: int, awaiting: bool) -> AssistantMessage: ...

    def recover_or_handover(
        self, node: "DoNode | AgentNode", expected_id: str, mode: "Mode", reason: str
    ) -> Turn: ...


N = TypeVar("N")


class Step(Generic[N]):
    kinds: frozenset[str] = frozenset()

    def __init__(self, walk: Walk, node: N) -> None:
        self.walk = walk
        self.node = node

    def open(self) -> Turn:
        raise NotImplementedError

    def landed(self, kind: str) -> Turn:
        raise NotImplementedError

    def resolve(self, outcome: MicroOutcome | None) -> Turn:
        name = getattr(self.node, "id", type(self).__name__)
        return self.walk.handover(f"{name!r} brokered no decision")
