"""The `do` step — one recorded macro, framed by its pages.

Before the move: its enter page (when declared — a `start` has none
and runs unconditionally) must match the current screen. After it: its
verify page must match the screen the macro result carries. A payment
move additionally runs the fire-time money predicates in code on the
current screen, after the human's consent (`money.py` owns the rules),
and consumes that consent by firing.

A macro that fails hands over — nothing re-runs in the background;
the author who wants a popup cleared or a locked phone woken declares
it (a `skip_when` in the macro, a `recover:` on the page).
"""

from physiclaw.common import gesture_vocab
from physiclaw.conductor.spec.conventions import page_id
from physiclaw.conductor.spec.model import DoNode
from physiclaw.conductor.spec.pack import qualified_macro
from physiclaw.conductor.spec.refs import fill_refs
from physiclaw.conductor.walk import money, recover
from physiclaw.conductor.walk.step import Step, Turn

KIND_RUN = "do"


class DoStep(Step[DoNode]):
    kinds = frozenset({KIND_RUN})

    def open(self) -> Turn:
        walk, node = self.walk, self.node
        gate = walk.enter_gate(node)
        if gate is not None:
            return gate
        if node.irreversible == "payment":
            blocked = walk.money_page_block(f"payment move {node.id!r}")
            if blocked is None:
                assert walk.screen is not None  # a matched verdict was read off it
                blocked = money.fire_block(
                    consented=walk.gate.consented,
                    seen=walk.gate.seen,
                    screen=walk.screen,
                )
            if blocked is not None:
                return walk.handover(f"payment move {node.id!r}: {blocked}")
        vals = walk.ref_values()
        inputs = {
            k: fill_refs(v, vals, where=f"move {node.id!r} `with.{k}`")
            for k, v in node.args.items()
        }
        args: dict = {"name": qualified_macro(walk.app, node.macro)}
        if inputs:
            args["inputs"] = inputs
        if node.irreversible == "payment":
            walk.spend_consent()
        nodes = len(walk.spec.nodes)
        return walk.synth(
            KIND_RUN,
            f"conductor: move {node.id} ({walk.idx + 1}/{nodes}) — "
            f"macro {node.macro}, verify {node.verify}",
            gesture_vocab.RUN_MACRO,
            args,
        )

    def landed(self, kind: str) -> Turn:
        walk, node = self.walk, self.node
        # Whatever the verify check says next, money may have moved:
        # the purchase line lands the moment the result does.
        walk.log_purchase()
        assert walk.verdict is not None
        expected = page_id(walk.app, node.verify)
        wrong = walk.mismatch(walk.verdict, expected)
        if wrong is not None:
            return walk.recover_or_handover(
                node,
                expected,
                recover.Mode.VERIFY,
                f"move {node.id!r} did not land on {node.verify!r} ({wrong})",
            )
        return walk.advance_cursor()
