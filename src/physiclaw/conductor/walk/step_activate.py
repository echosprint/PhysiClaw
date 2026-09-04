"""The `select` step — the channel boot's own, and its last: read the
thread, select the playbook it asks for.

At wake the conductor walks the channel pack's boot playbook
(`channel/boot.yml`): the thread page with the hands it declares for
a locked phone or a stray screen, then this step. On the thread it
fires ONE `parse_task` micro-call over the menu of enabled playbooks
(`activation.Activation`); a positive answer becomes the walk's BATON —
the program the conductor drives next — and anything else ends the
boot quietly, the model already standing on the thread. The invariant
is `agent/context/AGENT.md`'s "open the user's chat thread every
wake"; the boot route is who executes it, in synthesized turns and no
provider calls.

One thing learned from recorded sessions lives here: the request may
sit above the fold. A wake is usually the user's SECOND prod — the
newest bubble is a nudge and the request it refers to sits above it —
so parse_task's `scroll_up` escape swipes the thread up (bounded by
the step's `limit: {scrolls}`) and the re-ask reads the ACCUMULATED
listing, the newly revealed older labels seamed onto the ones already
seen (`merge_labels`).
"""

import logging
from dataclasses import replace

from physiclaw.common import gesture_vocab
from physiclaw.conductor.spec.model import ActivateNode
from physiclaw.conductor.walk.micro import SCROLL_UP, MicroOutcome
from physiclaw.conductor.walk.step import Step, Turn, Walk
from physiclaw.conductor.walk.turns import scroll_args

log = logging.getLogger(__name__)

KIND_SCROLL = "activate-scroll"


class ActivateStep(Step[ActivateNode]):
    kinds = frozenset({KIND_SCROLL})

    def __init__(self, walk: Walk, node: ActivateNode) -> None:
        super().__init__(walk, node)
        # The scroll-for-history accumulation: how many scroll rounds
        # ran, and the merged thread labels (oldest first) the re-ask
        # reads — None until the first scroll.
        self.scrolls = 0
        self.merged: list[str] | None = None

    def open(self) -> Turn:
        walk = self.walk
        if walk.activation is None:
            return walk.handover(
                f"activate {self.node.id!r}: no playbooks to offer — the boot "
                "activates only with an enabled playbook on disk"
            )
        gate = walk.enter_gate(self.node)
        if gate is not None:
            return gate
        return self._request()

    def landed(self, kind: str) -> Turn:
        # The history scroll landed: its result view is the thread one
        # notch up — re-ask over the seamed listing.
        return self._request()

    def resolve(self, outcome: MicroOutcome | None) -> Turn:
        node, walk = self.node, self.walk
        assert walk.activation is not None  # open() handed over without one
        if outcome is not None and outcome.out == SCROLL_UP:
            self.scrolls += 1
            if self.scrolls <= node.max_scrolls:
                if self.merged is None and walk.screen is not None:
                    self.merged = walk.screen.labels
                return walk.synth(
                    KIND_SCROLL,
                    "conductor: the request sits above the fold — scrolling up",
                    gesture_vocab.SWIPE,
                    scroll_args(down=False),
                    channel=True,
                )
            # Budget spent: the cautious read — a request we cannot fully
            # see activates nothing, exactly the `not_a_task` disposition.
            outcome = None
        program = walk.activation.build(outcome)
        if program is None:
            walk.conclude("no playbook covers the thread — the model takes it")
            return None
        assert outcome is not None  # build answers None without one
        # The record: what the boot decided, readable off the walk's
        # outputs (a stepping tool's position, a replay's report).
        walk.outputs[f"{node.id}.playbook"] = outcome.out
        walk.baton = program
        walk.conclude(f"hands over to {program.app}/{program.spec.name}")
        return None

    def _request(self) -> Turn:
        """The one micro-call this whole walk exists for. After a scroll
        round the request reads the ACCUMULATED listing, not just the
        current viewport."""
        walk = self.walk
        assert walk.activation is not None and walk.screen is not None
        req = walk.activation.request(walk.screen, self.node.id)
        if self.merged is not None:
            self.merged = merge_labels(walk.screen.labels, self.merged)
            req = replace(req, listing="\n".join(self.merged))
        return req


def merge_labels(newer: list[str], previous: list[str]) -> list[str]:
    """Seam two thread readings. Two-step because chrome pins: the
    contact-name header tops BOTH readings, so first strip the common
    PREFIX (rows the scroll did not move — chrome, or an unmoved top),
    then seam the scrollable remainders: after a scroll UP the screen
    shows OLDER content whose bottom overlaps the previous remainder's
    top — the longest suffix-equals-prefix run is the seam. Overlap
    matching (not set dedup) on purpose: a thread legitimately repeats
    labels, and dedup would eat the second of two identical bubbles."""
    p = 0
    while p < min(len(newer), len(previous)) and newer[p] == previous[p]:
        p += 1
    head, newer_rest, prev_rest = newer[:p], newer[p:], previous[p:]
    for k in range(min(len(newer_rest), len(prev_rest)), 0, -1):
        if newer_rest[-k:] == prev_rest[:k]:
            return head + newer_rest + prev_rest[k:]
    return head + newer_rest + prev_rest
