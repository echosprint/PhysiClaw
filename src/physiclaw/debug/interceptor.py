"""The fake-channel interceptor — the dispatch seam's debug half.

A result TRANSFORMER, not a bypass: every call dispatches for real —
the gate's `channel/send` genuinely drives the phone and the ask really
lands in the IM thread — and only the OBSERVATION of conductor channel
actions is rewritten, so the conversation the walk reads stays the
scripted one. One rule set, three lines of truth:

  - A conductor-minted `run_macro` naming a `channel/*` macro ran for
    real; its result is replaced with a gesture-shaped reply whose
    listing is the virtual thread render. A `channel/send` also appends
    the agent's bubble to the virtual thread. The conductor is now "on
    the thread".
  - A conductor-minted `peek` while on the thread ran for real too; its
    listing is replaced with the same render — the read that also
    releases a staged user reply when an ask is outstanding
    (`thread.py` owns the timing rule).
  - Everything else passes through untouched — and any phone-moving
    tool drops the on-thread state, so the next thread action must
    re-enter through `channel/open` exactly as the walk already does
    (`tried_open` self-heals a mismatch, in debug as in production).

Errors stay real: dispatch transforms only successful block results, so
a failed send propagates and the walk hands over exactly as it would in
production. Model-minted calls are never touched: hidden channel macros
already error for model turns, and the model's own navigation stays
real. Provenance is structural — dispatch passes the session's
`synthesized_turn` bit, the same one the phone-protection policies key
on — never inferred from call-id strings.

State is per-session and deliberately minimal: a fresh interceptor
starts off-thread until a conductor peek reads as the thread, so a
resumed suspension's first gate-peek reads the real phone and, unless
the IM app is already showing the thread, mismatches and re-enters via
the open macro — no cross-wake state to corrupt. A REAL listing that
already reads as the channel thread (the previous session's wrap-up
left the IM app open on it) is virtualized on that peek, or the boot
would parse the real thread — where the last word is the assistant's
own completion report — instead of the scripted one. The thread file is re-read on every render; only the
channel fingerprint is cached (it cannot change mid-session).
"""

import logging
from functools import cached_property

from physiclaw.common import gesture_vocab, verdict
from physiclaw.common.listing import Screen
from physiclaw.conductor.spec.conventions import CHANNEL_APP, SEND_MACRO
from physiclaw.conductor.spec.match import match_screen
from physiclaw.conductor.spec.pack import macro_app, qualified_macro
from physiclaw.conductor.spec.pages import PagePrint
from physiclaw.contract.dto import ToolCall
from physiclaw.debug import thread as vthread

log = logging.getLogger(__name__)

# Tools whose result cannot mean the phone left the thread — the
# on-thread state survives them; ANY other transformed-or-passed tool
# drops it (conservative: a wrong drop costs one open-macro retry, a
# wrong keep would read a fake thread over a real app screen). Local
# text-returning tools (note, end_session, …) never reach the
# transformer at all — none of them can move the phone.
_KEEP_STATE = frozenset({gesture_vocab.PEEK, "wait"})


def build() -> "FakeChannel":
    """The factory `agent.engine.plugins.load_debug_intercept` names by
    dotted path — the engine never imports this package."""
    return FakeChannel()


class FakeChannel:
    """One session's virtual user channel behind the dispatch seam."""

    def __init__(self) -> None:
        self._on_thread = False

    @cached_property
    def _pp(self) -> PagePrint | None:
        # Per-session cache — the channel fingerprint cannot change
        # mid-session (None, a valid answer, is cached too).
        return vthread.thread_print()

    def intercept(
        self, call: ToolCall, synthesized: bool, blocks: list[dict]
    ) -> list[dict] | None:
        """Replacement blocks for a REAL result just produced by `call`,
        or None — keep it untouched. `synthesized` is the session's
        plugin-minted-turn bit. Never raises: a harness bug must fail
        open into the real observation, not kill the session."""
        try:
            return self._intercept(call, synthesized, blocks)
        except Exception:
            log.exception("debug fake-channel crashed — keeping the real result")
            return None

    def _render(self, bubbles: list[vthread.Bubble]) -> str:
        return vthread.render_listing(bubbles, self._pp)

    def _intercept(
        self, call: ToolCall, synthesized: bool, blocks: list[dict]
    ) -> list[dict] | None:
        args = call.arguments or {}
        macro = str(args.get("name") or "")
        if (
            synthesized
            and call.name == gesture_vocab.RUN_MACRO
            and macro_app(macro) == CHANNEL_APP
        ):
            if macro == qualified_macro(CHANNEL_APP, SEND_MACRO):
                message = str((args.get("inputs") or {}).get("message") or "")
                bubbles = vthread.record_send(message)
                action = f"debug: {macro} ran on the phone — observation virtualized"
            else:
                bubbles = vthread.load().bubbles
                action = f"debug: {macro} ran on the phone — now on the virtual thread"
            self._on_thread = True
            log.info("debug fake-channel: %s", action)
            return vthread.gesture_blocks(
                verdict.attach(action, True), self._render(bubbles)
            )
        if synthesized and call.name == gesture_vocab.PEEK:
            if not self._on_thread and self._reads_as_thread(blocks):
                self._on_thread = True
                log.info(
                    "debug fake-channel: the real thread is on screen — virtualized"
                )
            if self._on_thread:
                log.info("debug fake-channel: virtualized peek of the thread")
                return vthread.view_blocks(self._render(vthread.peek_bubbles()))
        if call.name not in _KEEP_STATE:
            self._on_thread = False
        return None

    def _reads_as_thread(self, blocks: list[dict]) -> bool:
        """Whether a REAL peek shows the channel thread — judged by the
        same matcher and fingerprint the conductor would use on it."""
        if self._pp is None:
            return False
        screen = Screen.read(verdict.screen_text(blocks))
        return match_screen(screen, [self._pp]).matches(self._pp.page_id)
