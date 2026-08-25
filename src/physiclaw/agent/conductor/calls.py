"""DECIDE call-type declarations — the code-owned decision vocabulary.

Playbooks parameterize decision calls; they never define new prompt
shapes. Each call type declares here what the playbook parser needs to
validate against: the routing arms its `on:` map must cover in full, the
payload fields downstream `{node.field}` refs may read, and the `with:`
parameters it accepts. Execution (`micro.py` prompts + `program.py`
routing) sits behind these same declarations, so parser and runner can
never disagree about a call's vocabulary.

`decide` is the one open call: its outs are authored per node (the
question defines the answers), so the node declares `outs:` itself —
with `escalate` mandatory, because every closed choice needs the
concrete escape arm.
"""

from dataclasses import dataclass

ESCALATE = "escalate"


@dataclass(frozen=True)
class CallDecl:
    outs: tuple[str, ...]  # fixed routing arms; () = node-declared (decide)
    payload: tuple[str, ...]  # fields `{node.field}` refs may read
    params: tuple[str, ...]  # `with:` keys — all required until a call
    #   actually grows an optional one
    # The arm produced when the model answers with a candidate key (the
    # conductor then taps that candidate); None = the call has no
    # candidate list.
    pick_arm: str | None = None
    # The one arm whose SELF-route refreshes the screen between re-asks
    # (the conductor swipes); the parser sanctions self-loops only here.
    reask_arm: str | None = None

    @property
    def escapes(self) -> tuple[str, ...]:
        """The non-candidate answers a pick-style call offers the model —
        every out except the pick arm and the routing-only escalate."""
        return tuple(o for o in self.outs if o not in (self.pick_arm, ESCALATE))


CALLS: dict[str, CallDecl] = {
    "choose_item": CallDecl(
        outs=("pick", "scroll", "none_fit", ESCALATE),
        payload=("pick",),
        params=("criteria",),
        pick_arm="pick",
        reask_arm="scroll",
    ),
    "decide": CallDecl(
        outs=(),  # node-declared; parser enforces `escalate` membership
        payload=(),
        params=("question",),
    ),
}
