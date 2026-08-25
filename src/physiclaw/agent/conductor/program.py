"""Armed playbook → synthesized turns — the conductor's first interception.

One `Program` is one explicitly armed playbook mid-walk. The conductor
asks it for each turn; it answers with a synthesized ``[note, one-other]``
assistant turn (a LEG as ``run_macro``; the opening ``peek``; a decision's
own tap/swipe primitive), with a ``DecisionRequest`` the conductor brokers
through the micro-caller and feeds back via ``resolve()``, or with
``None`` — "I hand over". ``None`` is permanent for the session: the
conductor goes quiet and the model takes over with the transcript as the
handoff, because every synthesized turn and every tool result is ordinary
history the model can read.

This phase executes LEG and DECIDE nodes, strictly:

  - Before a leg: its ``enter:`` page (when declared) must match the
    current screen. After a leg: its ``verify:`` page must match the
    screen the macro result carries. Anything else — wrong page,
    occluded, unknown, a blocked or errored call, a node type this phase
    cannot drive (CONFIRM/HUMAN_GATE), a reserved ``ios.*`` /
    ``channel.*`` page — hands over. No retries, no recovery legs yet.
  - A DECIDE becomes one micro-call over the decide-time screen. A pick
    is acted on by a conductor tap primitive at the chosen row (ruled:
    macros are the hands for rehearsed stretches; a decision's single
    act — whose bbox only the decide-time screen knows — is the
    conductor's, and every dispatch guard still applies). A ``scroll``
    routed back to the same node swipes and re-asks, bounded by
    ``max_visits``. A failed or under-confident call hands over.
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

from physiclaw.agent.conductor.calls import CALLS, ESCALATE
from physiclaw.agent.conductor.match import Verdict, match_screen
from physiclaw.agent.conductor.micro import (
    DecisionRequest,
    MicroOutcome,
    build_request,
)
from physiclaw.agent.conductor.pages import PagePrint, prints_for_app
from physiclaw.agent.conductor.playbook import (
    DecideNode,
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

# Where the scroll-for-more swipe originates: a mid-content band, clear
# of top chrome and the tab bar, so the drag scrolls the list rather
# than dismissing or paging anything. Stylus up → page scrolls down.
SCROLL_BBOX = (0.2, 0.35, 0.8, 0.65)


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
    (or the decide a "swipe" re-asks) is always ``nodes[self._idx]``; a
    pending "tap" already had its cursor routed at resolve time."""

    kind: str  # "peek" | "leg" | "tap" | "swipe"
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
        self._ids = {n.id: i for i, n in enumerate(spec.nodes)}
        # Read by the engine: no DECIDE nodes → no micro-caller wired.
        self.needs_micro = any(isinstance(n, DecideNode) for n in spec.nodes)
        self._idx = 0
        self._pending: _Pending | None = None
        self._seq = 0
        # Decision state: recorded outputs (`{node.field}` refs read
        # them), per-node ask counts (max_visits bound), the journal line
        # the next synthesized note carries (record-don't-replay), the
        # decide-time screen/verdict the current step works from, and the
        # once-read memory.md snapshot (it cannot change mid-walk — the
        # model isn't running while the program synthesizes turns).
        self._outputs: dict[str, str] = {}
        self._visits: dict[str, int] = {}
        self._journal: str | None = None
        self._screen: Screen | None = None
        self._verdict: Verdict | None = None
        self._memory: str | None = None

    def advance(
        self, history: list[Message]
    ) -> "AssistantMessage | DecisionRequest | None":
        """The next synthesized turn; a DecisionRequest for the conductor
        to broker (feed the outcome back via ``resolve``); or None — hand
        over to the model. Never raises: a program bug degrades to a
        handover, not a crashed session (the model finishes the task
        either way)."""
        try:
            return self._advance(history)
        except Exception:
            log.exception("conductor: program crashed — handing over to the model")
            return None

    def resolve(
        self, outcome: MicroOutcome | None
    ) -> "AssistantMessage | DecisionRequest | None":
        """Continue the walk with a micro-call's outcome (None = the call
        failed or was under-confident — hand over). Same return contract
        and same never-raises guarantee as ``advance``."""
        try:
            return self._resolve(outcome)
        except Exception:
            log.exception("conductor: program crashed — handing over to the model")
            return None

    # ---- one advance ----

    def _advance(
        self, history: list[Message]
    ) -> "AssistantMessage | DecisionRequest | None":
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
        # the next leg's enter check, and a decision's candidates all
        # work from this screen.
        self._screen = _screen_of(result)
        self._verdict = match_screen(self._screen, self._prints)
        kind = self._pending.kind
        self._pending = None
        if kind == "peek":
            self._idx = self._locate(self._verdict)
        elif kind == "leg":
            node = self.spec.nodes[self._idx]
            assert isinstance(node, LegNode)
            wrong = self._mismatch(self._verdict, node.verify)
            if wrong is not None:
                return self._handover(
                    f"leg {node.id!r} did not land on {node.verify!r} ({wrong})"
                )
            self._idx += 1
        elif kind == "swipe":
            # The scroll-for-more between re-asks: same decide, fresh
            # screen, bounded by max_visits in _request.
            node = self.spec.nodes[self._idx]
            assert isinstance(node, DecideNode)
            return self._request(node)
        # kind == "tap": the pick's act — the cursor was routed at
        # resolve time; the target node's own enter check judges the
        # landing in _next.
        return self._next()

    def _next(self) -> "AssistantMessage | DecisionRequest | None":
        """Walk the node at the cursor, holding this phase's line: legs
        and decisions only, own-pack pages only, enter verified when
        declared. Works from the stored screen/verdict — every path here
        observed one first."""
        verdict = self._verdict
        if verdict is None:
            return self._handover("no screen observed yet")
        nodes = self.spec.nodes
        if self._idx >= len(nodes):
            log.info(
                "conductor: playbook %s/%s complete — handing over",
                self.app,
                self.spec.name,
            )
            return None
        node = nodes[self._idx]
        if isinstance(node, DecideNode):
            return self._request(node)
        if not isinstance(node, LegNode):
            return self._handover(
                f"node {node.id!r} is a {type(node).__name__} — this phase "
                "executes legs and decisions only"
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
                k: fill_refs(v, self._values(), where=f"leg {node.id!r} `with.{k}`")
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

    # ---- decisions ----

    def _values(self) -> dict[str, str]:
        """Ref-resolution values: armed inputs plus every recorded
        decision output (keyed `node.field`, exactly the dotted ref)."""
        return {**self.values, **self._outputs}

    def _request(self, node: DecideNode) -> "AssistantMessage | DecisionRequest | None":
        """One micro-call request off the decide-time screen — or a
        handover (None, typed as the shared step result) when the ask
        budget is spent or the walk has nothing to decide over."""
        self._visits[node.id] = self._visits.get(node.id, 0) + 1
        if self._visits[node.id] > node.max_visits:
            return self._handover(
                f"decide {node.id!r} exceeded max_visits ({node.max_visits})"
            )
        if self._screen is None:
            return self._handover(f"decide {node.id!r} has no screen to decide over")
        try:
            args = {
                k: str(fill_refs(v, self._values(), where=f"decide {node.id!r}"))
                for k, v in node.args.items()
            }
        except PlaybookError as e:
            return self._handover(str(e))
        return build_request(
            node.call, node.id, node.outs, args, self._screen, self._context(node)
        )

    def _context(self, node: DecideNode) -> str:
        """The declared context slices, assembled: `inputs.*` from the
        armed values; any `memory.*` request pulls memory.md whole,
        labeled with the slices asked for (finer slicing needs a memory
        format that has sections — not this phase's fight)."""
        parts: list[str] = []
        memory_slices: list[str] = []
        for entry in node.context:
            root, _, member = entry.partition(".")
            if root == "inputs":
                parts.append(f"{member}: {self.values.get(member, '')}")
            else:
                memory_slices.append(member)
        if memory_slices:
            if self._memory is None:
                # Read off the shared path constant, not engine.memory:
                # the conductor may not import engine behavior
                # (architecture rule — engine.dto only). Cached for the
                # walk: nothing can rewrite the file mid-program.
                f = paths.memory_file()
                self._memory = read_text(f).strip() if f.exists() else ""
            if self._memory:
                parts.append(
                    f"memory (for {', '.join(memory_slices)}):\n{self._memory}"
                )
        return "\n".join(parts)

    def _resolve(
        self, outcome: MicroOutcome | None
    ) -> "AssistantMessage | DecisionRequest | None":
        node = self.spec.nodes[self._idx]
        assert isinstance(node, DecideNode)
        if outcome is None:
            return self._handover(
                f"decide {node.id!r}: micro-call failed or under-confident"
            )
        self._journal = (
            f"decided {node.id}: {outcome.out} — {outcome.reason} "
            f"({outcome.confidence:.2f})"
        )
        target = node.on[outcome.out]
        if target == ESCALATE:
            return self._handover(
                f"decide {node.id!r} routed {outcome.out!r} → escalate "
                f"({outcome.reason})"
            )
        decl = CALLS[node.call]
        if outcome.picked is not None:
            # Act on the pick: record the declared payload, then the
            # conductor's own tap at the chosen row — the one gesture a
            # rehearsed macro cannot know (its bbox exists only on the
            # decide-time screen). The routed node's enter check judges
            # the landing.
            for field in decl.payload:
                self._outputs[f"{node.id}.{field}"] = outcome.picked.key
            self._idx = self._ids[target]
            return self._synth(
                "tap",
                f"conductor: tapping picked {outcome.picked.key!r}",
                "tap",
                {"bbox": list(outcome.picked.bbox)},
            )
        if outcome.out == decl.reask_arm and target == node.id:
            # The sanctioned self-loop (the parser lints self-routes to
            # exactly this arm): swipe up (page scrolls down), then
            # re-ask over whatever the fresh screen shows.
            return self._synth(
                "swipe",
                "conductor: scrolling for more candidates",
                gesture_vocab.SWIPE,
                {"bbox": list(SCROLL_BBOX), "direction": "up"},
            )
        self._idx = self._ids[target]
        return self._next()

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
        the pending one. A fresh decision outcome journals into this
        note's summary (record-don't-replay: the transcript carries the
        decision, never a re-ask)."""
        if self._journal is not None:
            summary = f"{summary} | {self._journal}"
            self._journal = None
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
