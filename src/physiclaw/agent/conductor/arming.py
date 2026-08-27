"""The arm and suspension files — what survives between wakes.

Two files under `playbooks/`, each with one lifecycle:

  - ``armed.json`` — the standing order (`physiclaw playbooks arm`):
    which playbook drives the next wakes, with which inputs. Sticky
    across incidental handovers; consumed by a TERMINAL walk outcome
    (`Program.retire`) or by `disarm`.
  - ``suspended.json`` — one suspended walk (a CONFIRM sent, or a gate
    still waiting). One-shot: consumed on load, whatever the outcome —
    a crash mid-resume loses the suspension and the wake runs plain,
    never a loop.

`arm` is the strict validation seam (an armed playbook is about to
drive the phone, so live-readiness problems are hard errors and the
actionable non-blockers come back as warnings); the loaders in
`setup.py` re-read these files fail-open at wake.
"""

import json
import logging
from pathlib import Path

from physiclaw.agent.conductor import memory, reply
from physiclaw.agent.conductor.ledger import check_ledger_value
from physiclaw.agent.conductor.playbook import (
    ConfirmNode,
    DecideNode,
    HumanGateNode,
    LegNode,
    Pack,
    Playbook,
    PlaybookError,
    disabled_leg_macros,
    load_pack,
    scan_playbooks,
)
from physiclaw.agent.macros import inputs as macro_inputs
from physiclaw.agent.macros.model import MacroError
from physiclaw.common import paths
from physiclaw.common.logger import write_json_atomic
from physiclaw.common.text import read_text

log = logging.getLogger(__name__)

ARMED_FILENAME = "armed.json"
ARMED_SCHEMA = 1

SUSPENDED_FILENAME = "suspended.json"
SUSPENDED_SCHEMA = 1


def armed_path() -> Path:
    return paths.playbooks_dir() / ARMED_FILENAME


def suspended_path() -> Path:
    return paths.playbooks_dir() / SUSPENDED_FILENAME


def read_armed() -> dict | None:
    """The raw armed.json mapping, or None when absent — the ONE reader
    of the file this module owns. Parse errors propagate: callers own
    their failure story (`armed_ref` goes quiet, `setup.load_armed` logs
    and stands down)."""
    p = armed_path()
    if not p.exists():
        return None
    data = json.loads(read_text(p))
    if not isinstance(data, dict):
        raise PlaybookError("armed.json is not a mapping")
    return data


def arm(app: str, name: str, inputs: dict[str, str]) -> "tuple[Playbook, list[str]]":
    """Validate and write the arm file; returns (spec, warnings) for the
    CLI to describe. Raises PlaybookError naming what blocks arming — the
    same live-readiness rules `playbooks check` warns about (disabled
    playbook, disabled leg macros) are hard errors here, because an armed
    playbook is about to drive the phone. Warnings are the actionable
    non-blockers: a declared `memory.<slug>` slice with no matching
    `## <slug>` section on THIS device runs empty (fail-closed), and a
    gate ask quoting no word-tier reply word rides the LLM tier."""
    spec, _ = armed_spec(app, name)
    values = resolve_inputs(spec, inputs)  # fail at arm time, not first wake
    check_ledger_value(spec, values)
    warnings = []
    suspended = suspended_ref()
    if suspended == (app, name):
        # A suspended walk of this same playbook would resume FIRST on
        # the next wake, with its old values — the new order must win.
        clear_suspended()
        warnings.append(
            "dropped a suspended walk of this playbook — the new arm starts fresh"
        )
    elif suspended is not None:
        warnings.append(
            f"a suspended walk for {suspended[0]}/{suspended[1]} resumes first on "
            "the next wake — this arm runs after it settles"
        )
    write_json_atomic(
        armed_path(),
        {"schema": ARMED_SCHEMA, "app": app, "playbook": name, "inputs": inputs},
    )
    return spec, warnings + _memory_slice_warnings(spec) + _gate_word_warnings(
        spec
    ) + _gate_reentry_warnings(spec)


def disarm() -> bool:
    """Remove the arm file; False when nothing was armed. The arm file
    ONLY — `Program.retire` consumes done deals through here, and a
    suspension present at that moment belongs to some other walk."""
    p = armed_path()
    if not p.exists():
        return False
    p.unlink()
    return True


def stand_down() -> "tuple[bool, tuple[str, str] | None]":
    """The CLI's full stop: disarm AND drop any suspended walk — a user
    disarming means "stop driving", and a surviving suspension would resume
    (and keep acting) on the very next wake. Returns (was_armed, the
    dropped suspension's (app, playbook) or None)."""
    dropped = suspended_ref()
    if not clear_suspended():
        dropped = None
    return disarm(), dropped


def armed_ref() -> tuple[str, str] | None:
    """The armed ``(app, playbook)`` per the file, without validating the
    pack — the CLI's list marker. None when nothing is armed / unreadable."""
    try:
        data = read_armed()
        if data is None:
            return None
        return str(data["app"]), str(data["playbook"])
    except Exception:
        return None


def armed_spec(app: str, name: str) -> tuple[Playbook, Pack]:
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


def resolve_inputs(spec: Playbook, provided: dict[str, str]) -> dict[str, str]:
    """Provided values against the declared inputs — the macro layer's
    resolution contract verbatim (unknown keys, missing required, defaults,
    strings only), translated to this spec's error class at the one seam."""
    try:
        return macro_inputs.resolve_inputs(spec, provided)
    except MacroError as e:
        raise PlaybookError(str(e)) from e


def clear_suspended() -> bool:
    p = suspended_path()
    if not p.exists():
        return False
    p.unlink()
    return True


def suspended_ref() -> tuple[str, str] | None:
    """The suspended walk's ``(app, playbook)`` per the file, without
    validating anything else. None when nothing is suspended / unreadable."""
    p = suspended_path()
    try:
        if not p.exists():
            return None
        data = json.loads(read_text(p))
        return str(data["app"]), str(data["playbook"])
    except Exception:
        return None


def _memory_slice_warnings(spec: Playbook) -> list[str]:
    """Arm-time advisory: a declared `memory.<slug>` slice with no
    matching `## <slug>` section on THIS device runs empty (fail-closed
    — memory.py owns the contract; this is its arm-time projection)."""
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
    sections = memory.read_sections()
    have = frozenset().union(*(tokens for tokens, _ in sections)) if sections else ()
    missing = [slug for slug in slugs if slug.casefold() not in have]
    if not missing:
        return []
    return [
        f"memory slice(s) {', '.join(missing)}: no `## <slug>` section in "
        "memory.md on this device — those decisions run without memory "
        "context (fail-closed)"
    ]


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


def _gate_reentry_warnings(spec: Playbook) -> list[str]:
    """Advisory: after any gate's confirmed reply the phone shows the IM
    thread. The payment case is a parse ERROR; for other gates a
    fall-through leg with no `enter:` (and no gate `return:`) runs its
    macro blind off the thread — the verify check catches the landing,
    but the gestures already happened on the messenger."""
    out = []
    nodes = spec.nodes
    for i, node in enumerate(nodes):
        if not isinstance(node, HumanGateNode) or node.return_macro is not None:
            continue
        nxt = nodes[i + 1] if i + 1 < len(nodes) else None
        if nxt is None or isinstance(nxt, (ConfirmNode, HumanGateNode)):
            continue
        if not (isinstance(nxt, LegNode) and nxt.enter is not None):
            out.append(
                f"gate {node.id!r}: no `return:` and {nxt.id!r} declares no "
                "`enter:` — after a confirmed reply the first action runs "
                "off the IM thread"
            )
    return out
