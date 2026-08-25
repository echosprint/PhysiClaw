"""DECIDE call-type declarations — the code-owned decision vocabulary.

Playbooks parameterize decision calls; they never define new prompt
shapes. Each call type declares here what the playbook parser needs to
validate against: the routing arms its `on:` map must cover in full, the
payload fields downstream `{node.field}` refs may read, and the `with:`
parameters it accepts. Execution lands in a later phase behind these
same declarations, so a playbook that parses today runs unchanged then.

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


CALLS: dict[str, CallDecl] = {
    "choose_item": CallDecl(
        outs=("pick", "scroll", "none_fit", ESCALATE),
        payload=("pick",),
        params=("criteria",),
    ),
    "decide": CallDecl(
        outs=(),  # node-declared; parser enforces `escalate` membership
        payload=(),
        params=("question",),
    ),
}
