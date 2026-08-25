"""Armed playbook → synthesized turns — the conductor's first interception.

One `Program` is one explicitly armed playbook mid-walk. The conductor
asks it for each turn; it answers with a synthesized ``[note, one-other]``
assistant turn (a LEG as ``run_macro``, plus one opening ``peek`` to see
where the phone is) or with ``None`` — "I hand over". ``None`` is
permanent for the session: the conductor goes quiet and the model takes
over with the transcript as the handoff, because every synthesized turn
and every tool result is ordinary history the model can read.

This phase executes LEG nodes only, strictly:

  - Before a leg: its ``enter:`` page (when declared) must match the
    current screen. After a leg: its ``verify:`` page must match the
    screen the macro result carries. Anything else — wrong page,
    occluded, unknown, a blocked or errored call, a node type this phase
    cannot drive (DECIDE/CONFIRM/HUMAN_GATE), a reserved ``ios.*`` /
    ``channel.*`` page — hands over. No retries, no recovery legs yet.
  - The opening peek doubles as resume: a killed session's next wake
    fast-forwards past every node whose ``verify`` page already matches
    the screen, so completed gestures are never replayed.

Arming is a manual testing surface (``physiclaw playbooks arm``): one
``playbooks/armed.json`` names the playbook and its input values. Loading
is fail-open — a missing, stale, or invalid arm file degrades to a
normal session, never takes one down. Automatic activation from the
task itself (parse_task) is a later phase.
"""

import json
import logging
from dataclasses import dataclass

from physiclaw.agent.conductor.match import Verdict, match_screen
from physiclaw.agent.conductor.pages import PagePrint, prints_for_app
from physiclaw.agent.conductor.playbook import (
    LegNode,
    Pack,
    Playbook,
    PlaybookError,
    disabled_leg_macros,
    fill_refs,
    load_pack,
    scan_playbooks,
)
from physiclaw.agent.engine.dto import (
    AssistantMessage,
    FinishReason,
    Message,
    TextBlock,
    ToolCall,
    ToolResultMessage,
)
from physiclaw.agent.macros import inputs as macro_inputs
from physiclaw.agent.macros.model import Macro, MacroError
from physiclaw.common import gesture_vocab, paths
from physiclaw.common.listing import Screen
from physiclaw.common.logger import write_json_atomic
from physiclaw.common.text import read_text

log = logging.getLogger(__name__)

ARMED_FILENAME = "armed.json"
_ARMED_SCHEMA = 1


# ---------- the arm file ----------


def _armed_path():
    return paths.playbooks_dir() / ARMED_FILENAME


def arm(app: str, name: str, inputs: dict[str, str]) -> Playbook:
    """Validate and write the arm file; returns the armed spec for the CLI
    to describe. Raises PlaybookError naming what blocks arming — the
    same live-readiness rules `playbooks check` warns about (disabled
    playbook, disabled leg macros) are hard errors here, because an armed
    playbook is about to drive the phone."""
    spec, _ = _armed_spec(app, name)
    _resolve_inputs(spec, inputs)  # fail at arm time, not first wake
    write_json_atomic(
        _armed_path(),
        {"schema": _ARMED_SCHEMA, "app": app, "playbook": name, "inputs": inputs},
    )
    return spec


def disarm() -> bool:
    """Remove the arm file; False when nothing was armed."""
    p = _armed_path()
    if not p.exists():
        return False
    p.unlink()
    return True


def armed_ref() -> tuple[str, str] | None:
    """The armed ``(app, playbook)`` per the file, without validating the
    pack — the CLI's list marker. None when nothing is armed / unreadable."""
    p = _armed_path()
    if not p.exists():
        return None
    try:
        data = json.loads(read_text(p))
        return str(data["app"]), str(data["playbook"])
    except Exception:
        return None


def _armed_spec(app: str, name: str) -> tuple[Playbook, Pack]:
    """The parsed playbook an arm names, holding it to live-readiness:
    valid, enabled, and every leg macro enabled."""
    pack = load_pack(app)
    entry = next((e for e in scan_playbooks(app, pack) if e.name == name), None)
    if entry is None:
        raise PlaybookError(f"no playbook {app}/{name} on disk")
    if entry.spec is None:
        raise PlaybookError(f"{app}/{name} is invalid: {entry.error}")
    if not entry.spec.enabled:
        raise PlaybookError(
            f"{app}/{name} is disabled — set `enabled: true` once rehearsed"
        )
    disabled = disabled_leg_macros(entry.spec, pack)
    if disabled:
        raise PlaybookError(
            f"{app}/{name} references disabled pack macro(s): "
            f"{', '.join(disabled)} — rehearse, then enable"
        )
    return entry.spec, pack


def _resolve_inputs(spec: Playbook, provided: dict[str, str]) -> dict[str, str]:
    """Provided values against the declared inputs — the macro layer's
    resolution contract verbatim (unknown keys, missing required, defaults,
    strings only), translated to this spec's error class at the one seam."""
    try:
        return macro_inputs.resolve_inputs(spec, provided)
    except MacroError as e:
        raise PlaybookError(str(e)) from e


def load_armed() -> "Program | None":
    """The armed playbook as a ready `Program`, or None. Fail-open on
    everything: no arm file, a pack edited out from under it, bad inputs —
    the session runs as if nothing were armed, with one warning saying
    why."""
    p = _armed_path()
    if not p.exists():
        return None
    try:
        data = json.loads(read_text(p))
        if data.get("schema") != _ARMED_SCHEMA:
            raise PlaybookError(f"unknown armed.json schema {data.get('schema')!r}")
        app = str(data["app"])
        name = str(data["playbook"])
        raw_inputs = data.get("inputs") or {}
        spec, pack = _armed_spec(app, name)
        values = _resolve_inputs(spec, raw_inputs)
    except Exception as e:
        log.warning(
            "armed playbook could not load (%s) — session runs without it: %s",
            p,
            e,
        )
        return None
    log.info("conductor: armed %s/%s (%d nodes)", app, name, len(spec.nodes))
    return Program(
        app=app,
        spec=spec,
        values=values,
        # Qualified names: pack macros are the conductor's hands. A user
        # macro name can never contain "/" (check_name), so the namespace
        # cannot collide — and the run_macro handler only resolves these
        # on a synthesized turn.
        pack_macros={f"{app}/{n}": m for n, m in pack.macros.items()},
        prints=prints_for_app(app),
    )


# ---------- the walk ----------


@dataclass(frozen=True)
class _Pending:
    """The synthesized action whose result the next advance() must read.
    The cursor never moves while an action is pending, so a pending leg
    is always ``nodes[self._idx]``."""

    kind: str  # "peek" | "leg"
    call_id: str


class Program:
    """One armed playbook mid-walk. Constructed per session (the engine
    loads it at wake), so the cursor state lives for exactly one attempt;
    persistence across wakes is the locate peek, not saved state."""

    def __init__(
        self,
        *,
        app: str,
        spec: Playbook,
        values: dict[str, str],
        pack_macros: dict[str, Macro],
        prints: list[PagePrint],
    ):
        self.app = app
        self.spec = spec
        self.values = values
        # Read by the engine when wiring the run_macro handler.
        self.pack_macros = pack_macros
        self._prints = prints
        self._idx = 0
        self._pending: _Pending | None = None
        self._seq = 0

    def advance(self, history: list[Message]) -> AssistantMessage | None:
        """The next synthesized turn, or None — hand over to the model.
        Never raises: a program bug degrades to a handover, not a crashed
        session (the model finishes the task either way)."""
        try:
            return self._advance(history)
        except Exception:
            log.exception("conductor: program crashed — handing over to the model")
            return None

    # ---- one advance ----

    def _advance(self, history: list[Message]) -> AssistantMessage | None:
        if self._pending is None:
            # Observe before acting. The peek is also how a killed
            # session resumes: _locate fast-forwards past legs whose
            # verify page already matches, so completed gestures are
            # never replayed.
            return self._synth(
                "peek",
                "conductor: observing the screen to locate "
                f"{self.app}/{self.spec.name}",
                gesture_vocab.PEEK,
                {},
            )
        result = _result_for(history, self._pending.call_id)
        if result is None:
            return self._handover(
                f"the result of {self._pending.kind} never arrived in history"
            )
        if result.is_error:
            return self._handover(
                f"{self._pending.kind} was blocked or failed: {_text(result)[:200]}"
            )
        # One reading, one verdict: the pending action's check, locate,
        # and the next leg's enter check all compare against this match.
        verdict = match_screen(_screen_of(result), self._prints)
        if self._pending.kind == "peek":
            self._idx = self._locate(verdict)
        else:
            node = self.spec.nodes[self._idx]
            assert isinstance(node, LegNode)
            wrong = self._mismatch(verdict, node.verify)
            if wrong is not None:
                return self._handover(
                    f"leg {node.id!r} did not land on {node.verify!r} ({wrong})"
                )
            self._idx += 1
        self._pending = None
        return self._next(verdict)

    def _next(self, verdict: Verdict) -> AssistantMessage | None:
        """Synthesize the node at the cursor, holding this phase's line:
        legs only, own-pack pages only, enter verified when declared."""
        nodes = self.spec.nodes
        if self._idx >= len(nodes):
            log.info(
                "conductor: playbook %s/%s complete — handing over",
                self.app,
                self.spec.name,
            )
            return None
        node = nodes[self._idx]
        if not isinstance(node, LegNode):
            return self._handover(
                f"node {node.id!r} is a {type(node).__name__} — this phase "
                "executes legs only"
            )
        if "." in node.verify or (node.enter is not None and "." in node.enter):
            return self._handover(
                f"leg {node.id!r} references a reserved built-in page — "
                "not supported in this phase"
            )
        if node.enter is not None:
            wrong = self._mismatch(verdict, node.enter)
            if wrong is not None:
                return self._handover(
                    f"leg {node.id!r} expects page {node.enter!r} ({wrong})"
                )
        try:
            inputs = {
                k: fill_refs(v, self.values, where=f"leg {node.id!r} `with.{k}`")
                for k, v in node.args.items()
            }
        except PlaybookError as e:
            return self._handover(str(e))
        args: dict = {"name": f"{self.app}/{node.macro}"}
        if inputs:
            args["inputs"] = inputs
        return self._synth(
            "leg",
            f"conductor: leg {node.id} ({self._idx + 1}/{len(nodes)}) — "
            f"macro {node.macro}, verify {node.verify}",
            gesture_vocab.RUN_MACRO,
            args,
        )

    # ---- page identity ----

    def _mismatch(self, verdict: Verdict, page: str) -> str | None:
        """None when the verdict is a match on `page`; else a short reason."""
        if verdict.kind == "match" and verdict.page_id == f"{self.app}.{page}":
            return None
        seen = verdict.page_id or "no known page"
        return f"screen reads as {verdict.kind}: {seen} — {verdict.detail}"

    def _locate(self, verdict: Verdict) -> int:
        """Cursor for the current screen: just past the LAST leg whose
        verify page matches it (that page proves the leg's outcome holds),
        else the top."""
        if verdict.kind != "match":
            log.info("conductor: screen is %s — starting from the top", verdict.kind)
            return 0
        resume = 0
        for i, node in enumerate(self.spec.nodes):
            if (
                isinstance(node, LegNode)
                and f"{self.app}.{node.verify}" == verdict.page_id
            ):
                resume = i + 1
        if resume:
            log.info(
                "conductor: screen already on %s — resuming at node %d/%d",
                verdict.page_id,
                resume + 1,
                len(self.spec.nodes),
            )
        return resume

    # ---- synthesis ----

    def _synth(
        self, kind: str, summary: str, tool: str, args: dict
    ) -> AssistantMessage:
        """One synthesized [note, one-other] turn — exactly the shape the
        loop enforces on model turns, so dispatch, guards, compaction, and
        the wire log see an ordinary turn — with its action registered as
        the pending one."""
        self._seq += 1
        cid = f"conductor-{self._seq}"
        self._pending = _Pending(kind=kind, call_id=f"{cid}-act")
        return AssistantMessage(
            content="",
            tool_calls=[
                ToolCall(id=f"{cid}-note", name="note", arguments={"summary": summary}),
                ToolCall(id=f"{cid}-act", name=tool, arguments=args),
            ],
            finish_reason=FinishReason.TOOL_CALLS,
            synthesized=True,
        )

    def _handover(self, reason: str) -> AssistantMessage | None:
        """Always None — typed as the advance result so call sites read
        `return self._handover(...)`."""
        log.warning(
            "conductor: handing %s/%s over to the model — %s",
            self.app,
            self.spec.name,
            reason,
        )
        return None


# ---------- history readers ----------


def _result_for(history: list[Message], call_id: str) -> ToolResultMessage | None:
    for msg in reversed(history):
        if isinstance(msg, ToolResultMessage) and msg.tool_call_id == call_id:
            return msg
    return None


def _screen_of(result: ToolResultMessage) -> Screen:
    """The screen a tool result carries — its text blocks parsed as a
    listing (macro results and peeks both end with the current view)."""
    return Screen.read(_text(result))


def _text(result: ToolResultMessage) -> str:
    """All text of a tool result, joined."""
    if isinstance(result.content, str):
        return result.content
    return "\n".join(b.text for b in result.content if isinstance(b, TextBlock))
