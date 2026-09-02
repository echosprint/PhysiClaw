"""The `do` step — one recorded macro, framed by its pages.

Before the move: its enter page (when declared — a `start` has none
and runs unconditionally) must match the current screen. After it: its
verify page must match the screen the macro result carries. A payment
move additionally runs the fire-time money predicates in code on the
current screen, after the human's consent (`money.py` owns the rules),
and consumes that consent by firing.

Two one-shot retries, once per move per walk, cover the two ways a
macro fails without moving anything: an abort BEFORE any gesture ran
under a popup band (recover the page, re-run), and a run against a
locked phone (unlock, re-run). Everything else keeps the hard handover.
"""

from physiclaw.common import gesture_vocab
from physiclaw.common.listing import Screen
from physiclaw.conductor import money, recover, views
from physiclaw.conductor.match import match_screen, reads_as_locked
from physiclaw.conductor.pages import page_id
from physiclaw.conductor.playbook import DoNode, fill_refs, qualified_macro
from physiclaw.conductor.step import Step, Turn
from physiclaw.contract.dto import ToolResultMessage
from physiclaw.macros.model import NO_GESTURES_NOTE

KIND_RUN = "do"
KIND_UNLOCK = "do-unlock"


class DoStep(Step[DoNode]):
    kinds = frozenset({KIND_RUN, KIND_UNLOCK})

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
                    consented=walk.gate.consented, cap=walk.gate.cap, screen=walk.screen
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
        if kind == KIND_UNLOCK:
            # The locked-move retry's unlock landed — re-run the move;
            # its own enter check judges the fresh world.
            return self.open()
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
                recover.MODE_VERIFY,
                f"move {node.id!r} did not land on {node.verify!r} ({wrong})",
            )
        return walk.advance_cursor()

    def failed(self, kind: str, error: str, raw: ToolResultMessage | None) -> Turn:
        """The macro's run failed. The two one-shot retries below, once
        per move per walk; None = not a case they cover, the walk hands
        over with `error`."""
        walk, node = self.walk, self.node
        if kind != KIND_RUN or raw is None or node.id in walk.retried_moves:
            return None
        screen = views.screen_of(raw)
        retry = self._retry_after_abort(raw, screen)
        if retry is None:
            retry = self._retry_locked(screen)
        return retry

    def _retry_after_abort(self, raw: ToolResultMessage, screen: Screen) -> Turn:
        """A macro that aborted BEFORE any gesture ran (the runner's
        marker) moved nothing — when the abort's own view reads as an
        overlay on a known page, recover the page and re-run the move.

        The marker is read off the RAW result text, not the settle
        failure string — that one is clipped (`turns.MAX_ERROR_CHARS`)
        and the marker sits late in a header that grows with macro name
        and view notes."""
        walk, node = self.walk, self.node
        if NO_GESTURES_NOTE not in views.text_of(raw):
            return None
        verdict = match_screen(screen, walk.prints)
        if verdict.kind != "occluded" or verdict.page_id is None:
            return None
        walk.retried_moves.add(node.id)
        walk.turns.drop()  # the abort is handled, not retried blind
        walk.screen, walk.verdict = screen, verdict
        return walk.recover_or_handover(
            node,
            verdict.page_id,
            recover.MODE_ENTER,
            f"move {node.id!r} aborted before any gesture under an overlay "
            f"on {verdict.page_id}",
        )

    def _retry_locked(self, screen: Screen) -> Turn:
        """A macro that FAILED while the screen reads locked (hint text
        or the cover's hero clock) moved nothing — a locked phone
        swallows gestures. Unlock and re-run. The field case: the phone
        auto-locks during a long model call before the first gesture (a
        route opening with a pure-text agent reasons for tens of
        seconds). The unlock's own result view is the next reading."""
        walk, node = self.walk, self.node
        if not reads_as_locked(screen):
            return None
        walk.retried_moves.add(node.id)
        walk.turns.drop()  # the failure is handled, not retried blind
        walk.journal(f"move {node.id} ran against a locked phone")
        return walk.synth(
            KIND_UNLOCK,
            f"conductor: phone locked during move {node.id} — unlocking to retry",
            gesture_vocab.UNLOCK_PHONE,
            {},
        )
