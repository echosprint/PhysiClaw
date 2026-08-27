"""Wake-time setup — how a `Program` comes to exist.

Three doors into a walk, all fail-open (a missing, stale, or invalid
file degrades to a normal session, never takes one down):

  - ``load_suspended`` — a suspended walk resumes at its stored node
    (one-shot; ANY wake resumes, the WAIT job is just the alarm clock).
  - ``load_armed`` — the standing order from `physiclaw playbooks arm`.
  - ``Activation`` — mid-session: the first screen that matches the
    channel thread page fires ONE parse_task micro-call over the
    playbook menu; a positive answer builds the program on the spot.

``session_setup`` is the engine's single wake-time conductor call: it
resolves those doors in priority order and assembles the hidden
qualified-macro registry — every pack plus the channel on the
activation path (mid-session activation may arm any playbook, so all
conductor hands must be dispatchable); a live program narrows it to its
own pack + channel, and no channel means a channel-less registry.
``_build_program`` is the one Program constructor call — a program is
whole at construction: channel, origin, and any suspended state included.
"""

import json
import logging
from dataclasses import dataclass

from physiclaw.agent.conductor import arming
from physiclaw.agent.conductor.channel import Channel, load_channel
from physiclaw.agent.conductor.ledger import check_ledger_value
from physiclaw.agent.conductor.match import match_screen
from physiclaw.agent.conductor.micro import (
    LIST_INPUT_MARK,
    NOT_A_TASK,
    PARSE_TASK,
    DecisionRequest,
    MicroOutcome,
    build_request,
)
from physiclaw.agent.conductor.pages import CHANNEL_APP, THREAD_ID, prints_for_app
from physiclaw.agent.conductor.playbook import (
    Pack,
    Playbook,
    PlaybookError,
    disabled_leg_macros,
    list_apps,
    load_pack,
    qualified_pack,
    scan_playbooks,
)
from physiclaw.agent.conductor.program import Program
from physiclaw.agent.conductor.views import last_result, screen_of
from physiclaw.agent.engine.dto import Message
from physiclaw.agent.macros.model import Macro
from physiclaw.common.text import read_text

log = logging.getLogger(__name__)


def load_armed(channel: "Channel | None" = None) -> "Program | None":
    """The armed playbook as a ready `Program`, or None. Fail-open on
    everything: no arm file, a pack edited out from under it, bad inputs —
    the session runs as if nothing were armed, with one warning saying
    why."""
    try:
        data = arming.read_armed()
        if data is None:
            return None
        if data.get("schema") != arming.ARMED_SCHEMA:
            raise PlaybookError(f"unknown armed.json schema {data.get('schema')!r}")
        app = str(data["app"])
        name = str(data["playbook"])
        raw_inputs = data.get("inputs") or {}
        spec, pack = arming.armed_spec(app, name)
        values = arming.resolve_inputs(spec, raw_inputs)
        # `arm` validated the value it wrote, but the file is on disk —
        # re-hold a hand-edited ledger to the same caps.
        check_ledger_value(spec, values)
    except Exception as e:
        log.warning(
            "armed playbook could not load (%s) — session runs without it: %s",
            arming.armed_path(),
            e,
        )
        return None
    log.info("conductor: armed %s/%s (%d nodes)", app, name, len(spec.nodes))
    return _build_program(app, spec, pack, values, channel, origin="armed")


def load_suspended(channel: "Channel | None" = None) -> "Program | None":
    """A suspended walk restored at its node, or None. One-shot: the file is
    consumed on load (a crash mid-resume loses the suspension, and the next
    wake runs as a plain session — fail-open, never a loop). The WAIT
    job that may also fire is just the alarm clock; ANY wake resumes.
    `channel` avoids a second channel load when the caller (session_setup)
    already holds one."""
    p = arming.suspended_path()
    if not p.exists():
        return None
    try:
        data = json.loads(read_text(p))
        if data.get("schema") != arming.SUSPENDED_SCHEMA:
            raise PlaybookError(f"unknown suspended schema {data.get('schema')!r}")
        app = str(data["app"])
        name = str(data["playbook"])
        spec, pack = arming.armed_spec(app, name)
        program = _build_program(
            app,
            spec,
            pack,
            {str(k): str(v) for k, v in (data.get("values") or {}).items()},
            channel if channel is not None else load_channel(),
            # The suspension carries its lineage: an activation-built walk's
            # terminal outcome must never consume an arm file it never
            # owned, even when a same-named arm exists by coincidence.
            # (Absent in older suspends — those were armed-lineage only.)
            origin="activation" if data.get("origin") == "activation" else "suspended",
            suspended=data,
        )
    except Exception as e:
        log.warning("suspended playbook could not load (%s) — dropped: %s", p, e)
        return None
    finally:
        arming.clear_suspended()  # one-shot: consumed on ANY load outcome
    return program


def _build_program(
    app: str,
    spec: Playbook,
    pack: Pack,
    values: dict[str, str],
    channel: "Channel | None",
    *,
    origin: str,
    suspended: dict | None = None,
) -> "Program":
    """The one Program constructor call — armed, suspended, and activation
    builds all come through here: a program is whole at construction
    (channel, origin, and any suspended state included), never patched up
    afterwards. `origin` decides whether a terminal outcome consumes the
    arm file (see `Program.retire`)."""
    return Program(
        app=app,
        spec=spec,
        values=values,
        pack_macros=qualified_pack(app, pack),
        prints=prints_for_app(app),
        channel=channel,
        origin=origin,
        suspended=suspended,
    )


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
        result = last_result(history)
        if result is None or result.is_error:
            return None
        screen = screen_of(result)
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
            inputs = ", ".join(
                # The mark keys the parse_task prompt's JSON-array rule.
                f"{i.name} {LIST_INPUT_MARK} ({i.description})"
                if i.kind == "list"
                else f"{i.name} ({i.description})"
                for i in spec.inputs
            )
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
            values = arming.resolve_inputs(spec, outcome.payload or {})
            check_ledger_value(spec, values)
        except PlaybookError as e:
            log.warning("activation %s: inputs did not resolve (%s)", outcome.out, e)
            return None
        log.info("conductor: activated %s (%s)", outcome.out, outcome.reason)
        return _build_program(
            app, spec, pack, values, self.channel, origin="activation"
        )


def session_setup() -> "tuple[Program | None, Activation | None, dict[str, Macro]]":
    """The engine's one wake-time conductor call, fail-open throughout:
    (suspended-or-armed program, activation trigger, the hidden qualified
    macro registry). The registry spans every pack plus the channel ON
    THE ACTIVATION PATH — mid-session activation may arm any playbook,
    so all conductor hands must be dispatchable; a live program narrows
    it to its own pack + channel, and no channel means the channel-less
    registry only."""
    hidden: dict[str, Macro] = {}
    channel = load_channel()
    if channel is not None:
        hidden.update(channel.macros)
    program = load_suspended(channel) or load_armed(channel)
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
        hidden.update(qualified_pack(app, pack))
        for entry in scan_playbooks(app, pack):
            spec = entry.spec
            if spec is None or not spec.enabled or disabled_leg_macros(spec, pack):
                continue
            entries[f"{app}/{entry.name}"] = (spec, pack)
    activation = Activation(entries=entries, channel=channel) if entries else None
    return None, activation, hidden
