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

The walk executes every node type, strictly:

  - Before a leg: its ``enter:`` page (when declared) must match the
    current screen. After a leg: its ``verify:`` page must match the
    screen the macro result carries. Anything else — wrong page,
    occluded, unknown, a blocked or errored call, a reserved ``ios.*``
    page — hands over. No retries, no recovery legs yet.
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

  - CONFIRM and HUMAN_GATE message the user over the channel pack
    (authored ``message:`` body or a generic fallback, plus the code
    envelope: money disclosure and reply instructions), then park
    (CONFIRM) or hold for the tiered reply check (the gate). Parks write
    ``playbooks/parked.json``; ANY next wake resumes at the stored node.
  - Money runs in code: the parser guarantees a payment leg directly
    follows its ``gate: payment`` HUMAN_GATE; consent binds the quoted
    total; the fire-time predicates re-read the sheet.

Arming is a manual testing surface (``physiclaw playbooks arm``): one
``playbooks/armed.json`` names the playbook and its input values.
Activation (parse_task on a channel-thread match) arms mid-session.
All loading is fail-open — a missing, stale, or invalid file degrades
to a normal session, never takes one down.
"""

import json
import logging
from dataclasses import dataclass, field

from physiclaw.agent.conductor import reply
from physiclaw.agent.conductor.calls import CALLS, ESCALATE
from physiclaw.agent.conductor.match import PRICE_RE, Verdict, match_screen
from physiclaw.agent.conductor.micro import (
    CONFIRM_REPLY,
    NOT_A_TASK,
    PARSE_TASK,
    DecisionRequest,
    MicroOutcome,
    build_request,
)
from physiclaw.agent.conductor.pages import (
    CHANNEL_APP,
    OPEN_MACRO,
    SEND_MACRO,
    THREAD_ID,
    THREAD_PAGE,
    PagePrint,
    page_id,
    prints_for_app,
)
from physiclaw.agent.conductor.playbook import (
    GATE_MAX_CHECKS,
    ConfirmNode,
    DecideNode,
    HumanGateNode,
    LegNode,
    Pack,
    Playbook,
    PlaybookError,
    disabled_leg_macros,
    fill_refs,
    list_apps,
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

# ---------- the user channel ----------
# `playbooks/channel/` is the ONE infrastructure pack: the IM thread's
# page fingerprint plus rehearsed macros, recorded on-device. The three
# convention names live in pages.py (the scaffold interpolates them).

PARKED_FILENAME = "parked.json"
_PARKED_SCHEMA = 1

# HUMAN_GATE ask-and-hold bounds: in-session reply polling cadence, and
# how many silent rounds before the session parks for the next wake.
# (Unclear-reply rounds are bounded separately by GATE_MAX_CHECKS.)
GATE_WAIT_SECONDS = 45
SILENCE_ROUNDS = 3

# The one deny disposition — both reply tiers (word list, LLM) hand over
# with the same instruction, no re-asks.
_DENY_HANDOVER = "user declined the ask — acknowledge them and wrap up"

# runtime.sentinel.WAIT, spelled literally: the conductor may not import
# engine runtimes; a test pins the two equal.
PARK_STATUS = "WAIT"


def _amounts(screen: Screen) -> list[float]:
    """Every ¥/￥ amount visible on the screen — `match.PRICE_RE`, the
    one spelling of a currency amount, run over raw row labels; never a
    model."""
    out: list[float] = []
    for row in screen.rows:
        out.extend(float(m) for m in PRICE_RE.findall(row.label))
    return out


def qualified_macro(app: str, name: str) -> str:
    """The qualified `app/name` dispatch key — the ONE spelling of the
    convention the run_macro handler resolves (user macro names can
    never contain "/", so no collision)."""
    return f"{app}/{name}"


def _qualified(app: str, pack: Pack) -> dict[str, Macro]:
    """A pack's macros under their qualified dispatch keys."""
    return {qualified_macro(app, n): m for n, m in pack.macros.items()}


@dataclass(frozen=True)
class Channel:
    """The loaded user-channel infrastructure: thread fingerprints plus
    the qualified macros. `send`/`open` resolve only when the macro
    exists AND is enabled — unavailable members degrade to hand-over at
    the moment they are needed, never earlier."""

    prints: list[PagePrint]
    macros: dict[str, Macro]  # qualified channel/<name>, enabled or not

    def _live(self, name: str) -> str | None:
        key = qualified_macro(CHANNEL_APP, name)
        m = self.macros.get(key)
        return key if m is not None and m.enabled else None

    @property
    def send(self) -> str | None:
        return self._live(SEND_MACRO)

    @property
    def open(self) -> str | None:
        return self._live(OPEN_MACRO)


def load_channel() -> Channel | None:
    """The channel pack, fail-open: absent or broken → None (asks and
    activation degrade; legs run unaffected)."""
    try:
        pack = load_pack(CHANNEL_APP)
        prints = prints_for_app(CHANNEL_APP)
    except Exception as e:
        log.warning("channel pack unusable (%s) — asks will hand over", e)
        return None
    if not any(p.decl.name == THREAD_PAGE for p in prints):
        return None
    return Channel(prints=prints, macros=_qualified(CHANNEL_APP, pack))


# ---------- the arm file ----------


def _armed_path():
    return paths.playbooks_dir() / ARMED_FILENAME


def arm(app: str, name: str, inputs: dict[str, str]) -> "tuple[Playbook, list[str]]":
    """Validate and write the arm file; returns (spec, warnings) for the
    CLI to describe. Raises PlaybookError naming what blocks arming — the
    same live-readiness rules `playbooks check` warns about (disabled
    playbook, disabled leg macros) are hard errors here, because an armed
    playbook is about to drive the phone. Warnings are the actionable
    non-blockers: a declared `memory.<slug>` slice with no matching
    `## <slug>` section on THIS device runs empty (fail-closed), and a
    gate ask quoting no word-tier reply word rides the LLM tier."""
    spec, _ = _armed_spec(app, name)
    _resolve_inputs(spec, inputs)  # fail at arm time, not first wake
    write_json_atomic(
        _armed_path(),
        {"schema": _ARMED_SCHEMA, "app": app, "playbook": name, "inputs": inputs},
    )
    return spec, _memory_slice_warnings(spec) + _gate_word_warnings(spec)


def _gate_word_warnings(spec: Playbook) -> list[str]:
    """Advisory, never blocking: a gate ask that quotes no word the
    deterministic reply tier matches still works — every reply just
    rides the LLM tier (bounded by GATE_MAX_CHECKS). An ask in a
    language our word lists don't cover is legal; the author is told
    the cost, not refused."""
    out = []
    for node in spec.nodes:
        if not isinstance(node, HumanGateNode):
            continue
        for key, text in (
            ("message", node.message),
            ("over_message", node.over_message),
        ):
            if text is None:
                continue
            norm = reply.normalize(text)
            if not any(w in norm for w in reply.CONFIRM_WORDS) or not any(
                w in norm for w in reply.DENY_WORDS
            ):
                out.append(
                    f"gate {node.id!r} `{key}` quotes no reply word the word "
                    "tier matches (好的/ok…, 不用/no…) — every reply will "
                    "spend an LLM check"
                )
    return out


def _memory_slice_warnings(spec: Playbook) -> list[str]:
    slugs = sorted(
        {
            entry.partition(".")[2]
            for node in spec.nodes
            if isinstance(node, DecideNode)
            for entry in node.context
            if entry.startswith("memory.")
        }
    )
    if not slugs:
        return []
    sections = _split_sections(_read_memory())
    have = frozenset().union(*(tokens for tokens, _ in sections)) if sections else ()
    missing = [slug for slug in slugs if slug.casefold() not in have]
    if not missing:
        return []
    return [
        f"memory slice(s) {', '.join(missing)}: no `## <slug>` section in "
        "memory.md on this device — those decisions run without memory "
        "context (fail-closed)"
    ]


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


def load_armed(channel: "Channel | None" = None) -> "Program | None":
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
    return _build_program(app, spec, pack, values, channel)


def _build_program(
    app: str,
    spec: Playbook,
    pack: Pack,
    values: dict[str, str],
    channel: "Channel | None",
) -> "Program":
    """The one Program constructor call — armed, parked, and activation
    builds all come through here, channel included: a program is whole at
    construction, never patched up afterwards."""
    return Program(
        app=app,
        spec=spec,
        values=values,
        pack_macros=_qualified(app, pack),
        prints=prints_for_app(app),
        channel=channel,
    )


# ---------- parked state ----------


def _parked_path():
    return paths.playbooks_dir() / PARKED_FILENAME


def clear_parked() -> bool:
    p = _parked_path()
    if not p.exists():
        return False
    p.unlink()
    return True


def load_parked(channel: "Channel | None" = None) -> "Program | None":
    """A parked walk restored at its node, or None. One-shot: the file is
    consumed on load (a crash mid-resume loses the park, and the next
    wake runs as a plain session — fail-open, never a loop). The WAIT
    job that may also fire is just the alarm clock; ANY wake resumes.
    `channel` avoids a second channel load when the caller (session_setup)
    already holds one."""
    p = _parked_path()
    if not p.exists():
        return None
    try:
        data = json.loads(read_text(p))
        if data.get("schema") != _PARKED_SCHEMA:
            raise PlaybookError(f"unknown parked schema {data.get('schema')!r}")
        app = str(data["app"])
        name = str(data["playbook"])
        spec, pack = _armed_spec(app, name)
        program = _build_program(
            app,
            spec,
            pack,
            {str(k): str(v) for k, v in (data.get("values") or {}).items()},
            channel if channel is not None else load_channel(),
        )
        program._restore_parked(data)
    except Exception as e:
        log.warning("parked playbook could not load (%s) — dropped: %s", p, e)
        return None
    finally:
        clear_parked()  # one-shot: consumed on ANY load outcome
    log.info(
        "conductor: resuming parked %s/%s at node %d (%s)",
        app,
        name,
        program._idx + 1,
        "awaiting reply" if program._gate.awaiting else "walk",
    )
    return program


# ---------- activation ----------


@dataclass
class Activation:
    """The parse_task trigger: armed once per session, the first time a
    screen matches the channel thread page — deterministic, zero cost
    until then. `entries` is the single source: the answer space is its
    keys, the menu a render of it, each value the parsed spec+pack so a
    positive answer activates without re-reading disk."""

    entries: dict[str, tuple[Playbook, Pack]]
    channel: Channel
    attempted: bool = False

    def request(self, history: list[Message]) -> DecisionRequest | None:
        """A parse_task request when the latest screen is the user's
        thread; None otherwise. Fires at most once per session."""
        if self.attempted:
            return None
        result = _last_result(history)
        if result is None or result.is_error:
            return None
        screen = _screen_of(result)
        verdict = match_screen(screen, self.channel.prints)
        if verdict.kind != "match" or verdict.page_id != THREAD_ID:
            return None
        self.attempted = True
        # Playbook refs only — the `not_a_task` escape is the call's own
        # (its _SPECS row appends it; no caller can forget the exit).
        return build_request(
            PARSE_TASK,
            "activation",
            tuple(self.entries),
            {"menu": self._menu()},
            screen,
        )

    def _menu(self) -> str:
        lines = ["Available playbooks:"]
        for ref, (spec, _pack) in self.entries.items():
            inputs = ", ".join(f"{i.name} ({i.description})" for i in spec.inputs)
            lines.append(
                f"- {ref}: {spec.description}"
                + (f" [inputs: {inputs}]" if inputs else "")
            )
        return "\n".join(lines)

    def build(self, outcome: MicroOutcome | None) -> "Program | None":
        """A ready Program from a parse_task outcome, or None (not a
        task, low confidence, or inputs that don't resolve — all stay in
        default mode, fail-open)."""
        if outcome is None or outcome.out == NOT_A_TASK:
            return None
        app = outcome.out.partition("/")[0]
        spec, pack = self.entries[outcome.out]
        try:
            values = _resolve_inputs(spec, outcome.payload or {})
        except PlaybookError as e:
            log.warning("activation %s: inputs did not resolve (%s)", outcome.out, e)
            return None
        log.info("conductor: activated %s (%s)", outcome.out, outcome.reason)
        return _build_program(app, spec, pack, values, self.channel)


def session_setup() -> "tuple[Program | None, Activation | None, dict[str, Macro]]":
    """The engine's one wake-time conductor call, fail-open throughout:
    (parked-or-armed program, activation trigger, the hidden qualified
    macro registry). The registry spans EVERY pack plus the channel —
    mid-session activation may arm any playbook, so all conductor hands
    must be dispatchable from the start."""
    hidden: dict[str, Macro] = {}
    channel = load_channel()
    if channel is not None:
        hidden.update(channel.macros)
    program = load_parked(channel) or load_armed(channel)
    if program is not None:
        # A live program names only its own pack + the channel — the
        # full cross-pack discovery below is activation's need, and
        # activation is off while a program drives.
        hidden.update(program.pack_macros)
        return program, None, hidden
    if channel is None:
        # No channel → no activation trigger; nothing else can consume
        # the discovery, so skip the every-pack parse entirely.
        return None, None, hidden
    entries: dict[str, tuple[Playbook, Pack]] = {}
    for app in list_apps():
        if app == CHANNEL_APP:
            continue
        try:
            pack = load_pack(app)
        except Exception as e:
            log.warning("pack %s unusable at wake (%s) — skipped", app, e)
            continue
        hidden.update(_qualified(app, pack))
        for entry in scan_playbooks(app, pack):
            spec = entry.spec
            if spec is None or not spec.enabled or disabled_leg_macros(spec, pack):
                continue
            entries[f"{app}/{entry.name}"] = (spec, pack)
    activation = Activation(entries=entries, channel=channel) if entries else None
    return None, activation, hidden


# ---------- the walk ----------


@dataclass(frozen=True)
class _Pending:
    """The synthesized action whose result the next advance() must read.
    The cursor never moves while an action is pending, so a pending leg
    (or the decide a "swipe" re-asks) is always ``nodes[self._idx]``; a
    pending "tap" already had its cursor routed at resolve time.
    `channel` marks actions that land on the user thread — the synth
    site declares it, so the reader never re-derives it from kind
    names."""

    kind: str  # "peek"|"leg"|"tap"|"swipe"|"gate-*"|"confirm-sent"|"park"
    call_id: str
    channel: bool = False


@dataclass
class _Gate:
    """The ask-and-hold state — one object, one park projection. The
    walk owns the cursor; this owns everything between "ask sent" and
    "consent bound": the ask text, the thread snapshot it is diffed
    against, the money numbers, the two bounded counters, and the
    LLM-tier handshake (`llm_reply` non-None = a confirm_reply request
    is outstanding and owns the next resolve())."""

    ask: str = ""
    baseline: set[str] = field(default_factory=set)
    quoted: float | None = None
    cap: float | None = None
    consented: float | None = None
    silence: int = 0
    checks: int = 0
    awaiting: bool = False  # ask sent, polling for the reply
    llm_reply: str | None = None
    tried_open: bool = False

    def to_park(self) -> dict:
        """The persisted projection — the one field list, beside the
        fields. Counters and the in-flight handshake deliberately reset
        on resume; `consented` persists so a post-consent park can never
        resume into a refused payment."""
        return {
            "ask_text": self.ask,
            "baseline": sorted(self.baseline),
            "quoted": self.quoted,
            "cap": self.cap,
            "consented": self.consented,
            "awaiting": self.awaiting,
        }

    @classmethod
    def from_park(cls, data: dict) -> "_Gate":
        return cls(
            ask=str(data.get("ask_text") or ""),
            baseline=set(data.get("baseline") or []),
            quoted=data.get("quoted"),
            cap=data.get("cap"),
            consented=data.get("consented"),
            awaiting=bool(data.get("awaiting")),
        )


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
        channel: Channel | None = None,
    ):
        self.app = app
        self.spec = spec
        self.values = values
        # The program's own qualified dispatch contribution — merged into
        # the wake registry by session_setup (armed/parked path).
        self.pack_macros = pack_macros
        self._prints = prints
        self._ids = {n.id: i for i, n in enumerate(spec.nodes)}
        # Read by the engine: decisions AND gate reply checks ride the
        # micro channel — either kind of node means wiring one up.
        self.needs_micro = any(
            isinstance(n, (DecideNode, HumanGateNode)) for n in spec.nodes
        )
        # The user-channel infrastructure (constructor-injected via
        # _build_program). None degrades to hand-over at the first ask.
        self.channel = channel
        self._idx = 0
        self._pending: _Pending | None = None
        self._seq = 0
        self._gate = _Gate()
        self._resumed = False
        # Decision state: recorded outputs (`{node.field}` refs read
        # them), per-node ask counts (max_visits bound), the journal line
        # the next synthesized note carries (record-don't-replay), the
        # decide-time screen/verdict the current step works from, and the
        # once-parsed memory.md sections (the file cannot change mid-walk
        # — the model isn't running while the program synthesizes turns).
        self._outputs: dict[str, str] = {}
        self._visits: dict[str, int] = {}
        self._journal: str | None = None
        self._screen: Screen | None = None
        self._verdict: Verdict | None = None
        self._memory: list[tuple[frozenset[str], str]] | None = None

    def _parked_dict(self, resume_idx: int) -> dict:
        """The park projection: walk state here, gate state via
        `_Gate.to_park` — each field list lives beside its fields."""
        return {
            "schema": _PARKED_SCHEMA,
            "app": self.app,
            "playbook": self.spec.name,
            "idx": resume_idx,
            "values": self.values,
            "outputs": self._outputs,
            "visits": self._visits,
            **self._gate.to_park(),
        }

    def _restore_parked(self, data: dict) -> None:
        self._idx = int(data["idx"])
        self._outputs = {str(k): str(v) for k, v in (data.get("outputs") or {}).items()}
        self._visits = {str(k): int(v) for k, v in (data.get("visits") or {}).items()}
        self._gate = _Gate.from_park(data)
        self._resumed = True

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
            if self._gate.awaiting:
                # Parked-at-gate resume: go straight to reading the
                # thread — the ask was sent before the park.
                return self._synth_gate_peek()
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
        kind = self._pending.kind
        # One reading, one verdict: channel-facing actions (declared at
        # the synth site) match against the thread; everything else
        # against the pack's own pages.
        in_channel = self._pending.channel
        self._pending = None
        self._screen = _screen_of(result)
        self._verdict = match_screen(
            self._screen,
            self.channel.prints if in_channel and self.channel else self._prints,
        )
        if kind == "peek":
            if self._resumed:
                # A parked walk trusts its stored cursor — the next
                # node's own checks judge whether the world still fits.
                self._resumed = False
            else:
                self._idx = self._locate(self._verdict)
        elif kind == "leg":
            node = self.spec.nodes[self._idx]
            assert isinstance(node, LegNode)
            wrong = self._mismatch(self._verdict, page_id(self.app, node.verify))
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
        elif kind in ("gate-sent", "confirm-sent"):
            wrong = self._thread_mismatch()
            if wrong is not None:
                return self._handover(
                    f"channel send did not land on the thread ({wrong})"
                )
            self._gate.baseline = {
                r.label.strip() for r in self._screen.rows if r.label.strip()
            }
            if kind == "confirm-sent":
                # CONFIRM: message away → park; the walk continues past
                # this node on the resuming wake.
                return self._park(resume_idx=self._idx + 1, awaiting=False)
            self._gate.awaiting = True
            self._gate.silence = 0
            self._gate.checks = 0
            return self._synth_wait()
        elif kind == "gate-wait":
            return self._synth_gate_peek()
        elif kind in ("gate-peek", "gate-open"):
            return self._gate_check()
        elif kind not in ("tap", "gate-return"):
            # A typo'd kind at a _synth site must fail loudly, never
            # silently walk the next node.
            return self._handover(f"unknown pending action kind {kind!r}")
        # "tap" / "gate-return": the cursor was routed at resolve (or
        # confirm) time; the target node's own checks judge the landing
        # in _next.
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
        if isinstance(node, (ConfirmNode, HumanGateNode)):
            # Both ask nodes share the one precondition: a live send.
            if self.channel is None or self.channel.send is None:
                return self._handover(
                    "no channel send macro — record playbooks/channel to enable asks"
                )
            if isinstance(node, ConfirmNode):
                return self._start_confirm(node)
            return self._start_gate(node)
        if not isinstance(node, LegNode):
            return self._handover(
                f"node {node.id!r} is a {type(node).__name__} — not executable"
            )
        if "." in node.verify or (node.enter is not None and "." in node.enter):
            return self._handover(
                f"leg {node.id!r} references a reserved built-in page — "
                "not supported in this phase"
            )
        if node.enter is not None:
            wrong = self._mismatch(verdict, page_id(self.app, node.enter))
            if wrong is not None:
                return self._handover(
                    f"leg {node.id!r} expects page {node.enter!r} ({wrong})"
                )
        if node.irreversible == "payment":
            # The fire-time money predicates — in code, on the current
            # screen, after the human's consent: staleness (the sheet
            # must still show the consented total) and the bound.
            blocked = self._money_block(node)
            if blocked is not None:
                return self._handover(blocked)
        try:
            inputs = {
                k: fill_refs(v, self._values(), where=f"leg {node.id!r} `with.{k}`")
                for k, v in node.args.items()
            }
        except PlaybookError as e:
            return self._handover(str(e))
        args: dict = {"name": qualified_macro(self.app, node.macro)}
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
        """The declared context slices, assembled least-privilege:
        `inputs.*` from the armed values; `memory.<slug>` pulls ONLY the
        matching `## <slug>` section of memory.md — micro-calls may run
        on a different vendor's cheap tier, so the whole memory file
        must never ride along for one playbook's shopping preferences.
        Fail closed: no matching section → no memory context (logged
        here, warned about at arm time) — a degraded pick escalates
        safely; a widened privacy boundary does not."""
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
                # Parsed once for the walk: nothing can rewrite the file
                # mid-program (the model isn't running).
                self._memory = _split_sections(_read_memory())
            sliced = _match_sections(self._memory, memory_slices)
            if sliced:
                parts.append(f"memory (for {', '.join(memory_slices)}):\n{sliced}")
            else:
                log.info(
                    "memory slice(s) %s: no matching `## <slug>` section in "
                    "memory.md — decision runs without memory context",
                    memory_slices,
                )
        return "\n".join(parts)

    def _resolve(
        self, outcome: MicroOutcome | None
    ) -> "AssistantMessage | DecisionRequest | None":
        if self._gate.llm_reply is not None:
            # The gate's LLM tier came back. None/unclear → keep waiting
            # (this round's check was already counted in _gate_check).
            judged = self._gate.llm_reply
            self._gate.llm_reply = None
            if outcome is not None and outcome.out == "deny":
                return self._handover(_DENY_HANDOVER)
            if outcome is not None and outcome.out == "revise":
                # A change request ("ok, but make it two boxes" / remove
                # an item) or a question is a TASK change, not a gate
                # verdict — the model takes over with the thread in the
                # transcript and acts on the user's words.
                return self._handover(
                    f"user asked for changes ({judged!r}) — read the "
                    "thread and adjust the order before any payment"
                )
            if outcome is not None and outcome.out == "confirm":
                return self._gate_confirmed()
            return self._synth_wait()
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

    # ---- asks, the gate, parking ----

    def _start_confirm(self, node: ConfirmNode) -> "AssistantMessage | None":
        # `message:` verbatim, refs filled — the playbook owns every
        # word the user reads (only its author knows their language),
        # and a CONFIRM has no consent contract to protect (any wake
        # resumes the walk).
        try:
            text = str(
                fill_refs(
                    node.message,
                    self._values(),
                    where=f"confirm {node.id!r} `message`",
                )
            )
        except PlaybookError as e:
            return self._handover(str(e))
        self._gate.ask = text
        return self._synth_send("confirm-sent", text)

    def _start_gate(self, node: HumanGateNode) -> "AssistantMessage | None":
        # All prose is the playbook's; code fills the consent slots the
        # parser required. For a payment gate: read the sheet NOW — the
        # ask quotes it via {gate.total}, and consent binds to exactly
        # this number (the ask IS the consent record).
        template, values = node.message, self._values()
        if node.gate == "payment":
            cap = self._mandate_cap()
            if cap is None:
                return self._handover(
                    f"gate {node.id!r}: mandate cap could not be resolved"
                )
            amts = _amounts(self._screen) if self._screen else []
            if not amts:
                return self._handover(
                    f"gate {node.id!r}: no total readable on the sheet"
                )
            quoted = max(amts)
            self._gate.quoted, self._gate.cap = quoted, cap
            values = {**values, "gate.total": f"{quoted:g}", "gate.cap": f"{cap:g}"}
            if quoted > cap:
                # Over-cap (ruled): the SAME plain-consent reply opens
                # the gate, but the breach must be disclosed — the
                # parser guarantees this template quotes total AND cap.
                assert node.over_message is not None
                template = node.over_message
        try:
            text = str(fill_refs(template, values, where=f"gate {node.id!r} `message`"))
        except PlaybookError as e:
            return self._handover(str(e))
        self._gate.ask = text
        self._gate.tried_open = False
        return self._synth_send("gate-sent", text)

    def _mandate_cap(self) -> float | None:
        m = self.spec.mandate
        if m is None:
            return None
        if isinstance(m.max_amount, float):
            return m.max_amount
        try:
            return float(self.values[m.max_amount.name])
        except (KeyError, ValueError):
            return None

    def _money_block(self, node: LegNode) -> str | None:
        """The two fire-time predicates. None = pay; else the logged
        reason to block and hand over."""
        if self._gate.consented is None:
            return f"payment leg {node.id!r} reached without a confirmed total"
        assert self._screen is not None
        consented = self._gate.consented
        amts = _amounts(self._screen)
        if not any(abs(a - consented) < 0.01 for a in amts):
            return (
                f"sheet changed after consent: confirmed ¥{consented:g}, "
                f"now sees {amts or 'no amounts'}"
            )
        bound = max(self._gate.cap or 0.0, consented)
        over = [a for a in amts if a > bound + 0.005]
        if over:
            return f"amount(s) {over} exceed the consented bound ¥{bound:g}"
        return None

    def _gate_check(self) -> "AssistantMessage | DecisionRequest | None":
        """One reply-evaluation round over the freshly peeked thread —
        the ruled tiers: new-message precondition, word match, LLM."""
        node = self.spec.nodes[self._idx]
        assert isinstance(node, HumanGateNode)
        wrong = self._thread_mismatch()
        if wrong is not None:
            if self.channel and self.channel.open and not self._gate.tried_open:
                self._gate.tried_open = True
                return self._synth(
                    "gate-open",
                    "conductor: reopening the user thread",
                    gesture_vocab.RUN_MACRO,
                    {"name": self.channel.open},
                    channel=True,
                )
            return self._handover(f"cannot reach the user thread ({wrong})")
        assert self._screen is not None
        new = reply.new_incoming(self._screen.rows, self._gate.baseline, self._gate.ask)
        if not new:
            self._gate.silence += 1
            if self._gate.silence >= SILENCE_ROUNDS:
                return self._park(resume_idx=self._idx, awaiting=True)
            return self._synth_wait()
        verdict = reply.classify_all(new)
        if verdict == "deny":
            return self._handover(_DENY_HANDOVER)
        if verdict == "confirm":
            return self._gate_confirmed()
        # Unclear → the LLM tier, bounded; these messages are judged once.
        self._gate.baseline |= set(new)
        joined = " / ".join(new)
        self._gate.checks += 1
        if self._gate.checks > GATE_MAX_CHECKS:
            return self._park(resume_idx=self._idx, awaiting=True)
        self._gate.llm_reply = joined
        # Empty outs: confirm_reply's answer space is fixed whole in its
        # _SPECS row, never caller-supplied.
        return build_request(
            CONFIRM_REPLY,
            node.id,
            (),
            {"ask": self._gate.ask, "reply": joined},
            self._screen,
        )

    def _gate_confirmed(self) -> "AssistantMessage | DecisionRequest | None":
        node = self.spec.nodes[self._idx]
        assert isinstance(node, HumanGateNode)
        # Informed consent binds the money predicates: the quoted total
        # becomes the consented one (over-cap included — ruled).
        self._gate.consented = self._gate.quoted
        self._gate.awaiting = False
        self._journal = f"user confirmed {node.gate}"
        self._idx += 1
        if node.return_macro is not None:
            return self._synth(
                "gate-return",
                f"conductor: back to the app via {node.return_macro}",
                gesture_vocab.RUN_MACRO,
                {"name": qualified_macro(self.app, node.return_macro)},
            )
        return self._next()

    def _thread_mismatch(self) -> str | None:
        if self.channel is None:
            return "no channel pack"
        assert self._verdict is not None
        return self._mismatch(self._verdict, THREAD_ID)

    def _synth_send(self, kind: str, text: str) -> AssistantMessage:
        assert self.channel is not None and self.channel.send is not None
        return self._synth(
            kind,
            "conductor: messaging the user",
            gesture_vocab.RUN_MACRO,
            {"name": self.channel.send, "inputs": {"message": text}},
            channel=True,
        )

    def _synth_wait(self) -> AssistantMessage:
        return self._synth(
            "gate-wait",
            "conductor: waiting for the user's reply",
            "wait",
            {"seconds": GATE_WAIT_SECONDS},
        )

    def _synth_gate_peek(self) -> AssistantMessage:
        return self._synth(
            "gate-peek",
            "conductor: checking for a reply",
            gesture_vocab.PEEK,
            {},
            channel=True,
        )

    def _park(self, *, resume_idx: int, awaiting: bool) -> AssistantMessage:
        """Write the parked state and close the session WAIT. No job is
        synthesized: a WAIT without a session-created job auto-schedules
        the follow-up alarm (`contract.drive`) — the file, not the job,
        is what resumes the walk on ANY next wake."""
        self._gate.awaiting = awaiting
        write_json_atomic(_parked_path(), self._parked_dict(resume_idx))
        recap = f"waiting for the user's reply on {self.app}/{self.spec.name}"
        return self._synth(
            "park",
            f"conductor: parking — {recap}",
            "end_session",
            {"status": PARK_STATUS, "recap": recap},
        )

    # ---- page identity ----

    def _mismatch(self, verdict: Verdict, expected_id: str) -> str | None:
        """None when the verdict is a match on the full `expected_id`
        (`app.page`); else a short reason — pack pages and the channel
        thread judged by one spelling."""
        if verdict.kind == "match" and verdict.page_id == expected_id:
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
                and page_id(self.app, node.verify) == verdict.page_id
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
        self, kind: str, summary: str, tool: str, args: dict, *, channel: bool = False
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
        self._pending = _Pending(kind=kind, call_id=f"{cid}-act", channel=channel)
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


def _read_memory() -> str:
    """memory.md off the shared path constant, NOT engine.memory: the
    conductor may import only engine.dto (architecture rule)."""
    f = paths.memory_file()
    return read_text(f).strip() if f.exists() else ""


def _split_sections(text: str) -> list[tuple[frozenset[str], str]]:
    """memory.md carved at its `## ` headings: (heading tokens, whole
    section text) per section — parsed once, matched many times."""
    sections: list[tuple[frozenset[str], str]] = []
    tokens: frozenset[str] | None = None
    body: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if tokens is not None:
                sections.append((tokens, "\n".join(body).strip()))
            tokens = frozenset(line[3:].casefold().split())
            body = [line]
        elif tokens is not None:
            body.append(line)
    if tokens is not None:
        sections.append((tokens, "\n".join(body).strip()))
    return sections


def _match_sections(
    sections: list[tuple[frozenset[str], str]], slugs: list[str]
) -> str:
    wanted = {slug.casefold() for slug in slugs}
    return "\n".join(body for tokens, body in sections if wanted & tokens)


def memory_sections(text: str, slugs: list[str]) -> str:
    """The `## <heading>` sections of memory.md matching the requested
    slugs — the least-privilege slice, FAIL CLOSED: no match means no
    memory context (a degraded pick escalates safely; a privacy boundary
    must never silently widen to the whole file). A slug matches a
    heading only as a whole whitespace-separated token — `shopping`
    never bleeds into `## shopping_blacklist`; bilingual headings work
    as `## shopping_prefs 购物偏好`."""
    return _match_sections(_split_sections(text), slugs)


# ---------- history readers ----------


def _last_result(history: list[Message]) -> ToolResultMessage | None:
    """The most recent tool result of ANY call — what activation reads
    (model turns included; the conductor is not driving yet)."""
    for msg in reversed(history):
        if isinstance(msg, ToolResultMessage):
            return msg
    return None


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
