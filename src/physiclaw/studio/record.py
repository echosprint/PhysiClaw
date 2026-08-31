"""Macro recording brains — draft steps from browser gestures, the
step editor's mutations, house-style MACRO.yml emission, and save.

The editor is the product, not the recorder: recorded output is a
draft the user refines (rename, comment, delete, insert an assertion,
mark a dismissal `skip_when`), and every mutation validates through
the REAL macro parser (`parse_inline_macro` while drafting — the
inline body grammar is MACRO.yml minus identity — and `parse_macro`
at save), so a draft that saves is a macro that will replay.
"""

from pathlib import Path

from physiclaw.common import gesture_vocab, paths
from physiclaw.common.text import write_text
from physiclaw.macros.model import (
    MACRO_FILENAME,
    MAX_STEPS,
    WAIT,
    Macro,
    MacroError,
    check_name,
    label_readings,
)
from physiclaw.macros.parse import parse_inline_macro, parse_macro
from physiclaw.studio.curate import quote_yaml
from physiclaw.studio.draft import DraftError, delete_snap

# What a browser gesture records as a step. `peek` is a view, not a
# gesture; `unlock_phone`/`sequence`/`screenshot` are outside the step
# grammar (`ALLOWED_STEP_TOOLS`) on purpose.
RECORDABLE_TOOLS = frozenset(
    gesture_vocab.PRESS_TOOLS
    | gesture_vocab.NAV_TOOLS
    | {gesture_vocab.SWIPE, gesture_vocab.SEND_TO_CLIPBOARD}
)

# Draft-only step keys, stripped before the grammar sees the step
# (the snapshot ref, the emitted-as-YAML-comment note, the page guess;
# the snapshot's listing lives in the draft's shot registry).
_DRAFT_STEP_KEYS = frozenset({"snap", "comment", "page"})

# The assertion step's settle time — long enough for a screen change to
# land, far under MAX_WAIT_SECONDS.
EXPECT_WAIT_SECONDS = 2
# Padding around the proving row's bbox for the `within` region.
EXPECT_PAD = 0.05


def _macro(draft: dict, name: str) -> dict:
    m = draft.setdefault("macros", {}).get(name)
    if m is None:
        raise DraftError(f"no drafted macro {name!r}")
    return m


def add_macro(draft: dict, name: str) -> None:
    macros = draft.setdefault("macros", {})
    if name in macros:
        raise DraftError(f"macro {name!r} already drafted")
    try:
        check_name(name)
    except MacroError as e:
        raise DraftError(str(e)) from e
    macros[name] = {
        "description": "",
        "inputs": {},
        "steps": [],
        "target": "pack",
        "next_step": 1,
    }


def delete_macro(draft: dict, name: str) -> None:
    """Remove the macro draft and its step snapshots."""
    m = _macro(draft, name)
    if draft.get("recording", {}).get("macro") == name:
        draft["recording"] = None
    del draft["macros"][name]
    for s in m["steps"]:
        delete_snap(draft, s.get("snap"))


def update_macro(
    draft: dict,
    name: str,
    *,
    description: str | None = None,
    inputs: dict | None = None,
    target: str | None = None,
) -> None:
    m = _macro(draft, name)
    before = {k: m[k] for k in ("description", "inputs", "target")}
    if description is not None:
        m["description"] = description
    if inputs is not None:
        m["inputs"] = inputs
    if target is not None:
        if target not in ("pack", "global"):
            raise DraftError("`target` is pack or global")
        m["target"] = target
    try:
        _validate(name, m)
    except MacroError:
        m.update(before)
        raise


# ---------- recording ----------


def start_recording(draft: dict, name: str, replace: int | None = None) -> None:
    m = _macro(draft, name)
    if replace is not None and not (0 <= replace < len(m["steps"])):
        raise DraftError(f"no step {replace} to re-record")
    draft["recording"] = {"macro": name, "replace": replace}


def stop_recording(draft: dict) -> None:
    draft["recording"] = None


def check_recordable(draft: dict, tool: str, args: dict) -> bool:
    """Whether this act will be recorded — called BEFORE the gesture
    fires, so a refusal (a press without its label) stops the arm too."""
    rec = draft.get("recording")
    if not rec or tool not in RECORDABLE_TOOLS:
        return False
    m = _macro(draft, rec["macro"])
    if rec["replace"] is None and len(m["steps"]) >= MAX_STEPS:
        raise DraftError(f"macro {rec['macro']!r} is at the {MAX_STEPS}-step cap")
    if tool in gesture_vocab.PRESS_TOOLS and not label_readings(args):
        raise DraftError(
            "recording: a press step needs `label` beside its bbox — "
            "what the coordinates are (on-screen text, or a description)"
        )
    return True


def record_step(draft: dict, tool: str, args: dict, snap: str) -> int:
    """Append (or replace, when re-recording one step) the acted gesture
    as a draft step referencing its after-screen snapshot (already in
    the shot registry). Returns the step index. Recording a replacement
    disarms (and drops the replaced snapshot); appending stays armed."""
    rec = draft["recording"]
    m = _macro(draft, rec["macro"])
    step = {
        "name": f"step-{m['next_step']}",
        "tool": tool,
        **({"with": args} if args else {}),
        "snap": snap,
    }
    m["next_step"] += 1
    if rec["replace"] is not None:
        index = rec["replace"]
        step["name"] = m["steps"][index]["name"]  # keep the address stable
        delete_snap(draft, m["steps"][index].get("snap"))
        m["steps"][index] = step
        draft["recording"] = None
    else:
        m["steps"].append(step)
        index = len(m["steps"]) - 1
    _validate(rec["macro"], m)
    return index


# ---------- the step editor ----------


def update_step(draft: dict, name: str, index: int, fields: dict) -> None:
    """Merge editable fields into one step: name, comment, with, guard,
    skip_when, expect, hint. `null` deletes the key."""
    m = _macro(draft, name)
    steps = m["steps"]
    if not (0 <= index < len(steps)):
        raise DraftError(f"no step {index} in {name!r}")
    allowed = {"name", "comment", "with", "guard", "skip_when", "expect", "hint"}
    unknown = sorted(set(fields) - allowed)
    if unknown:
        raise DraftError(f"step field(s) not editable: {', '.join(unknown)}")
    before = dict(steps[index])
    for k, v in fields.items():
        if v is None:
            steps[index].pop(k, None)
        else:
            steps[index][k] = v
    try:
        _validate(name, m)
    except MacroError:
        steps[index] = before
        raise


def insert_step(draft: dict, name: str, index: "int | None", step: dict) -> None:
    """Insert an authored (not recorded) step — the assertion path.
    `index=None` appends."""
    m = _macro(draft, name)
    if index is None:
        index = len(m["steps"])
    if not (0 <= index <= len(m["steps"])):
        raise DraftError(f"cannot insert at {index} in {name!r}")
    step = {"name": f"step-{m['next_step']}", **step}
    m["next_step"] += 1
    m["steps"].insert(index, step)
    try:
        _validate(name, m)
    except MacroError:
        del m["steps"][index]
        raise


def delete_step(draft: dict, name: str, index: int) -> None:
    """Remove one step and its snapshot."""
    m = _macro(draft, name)
    if not (0 <= index < len(m["steps"])):
        raise DraftError(f"no step {index} in {name!r}")
    delete_snap(draft, m["steps"].pop(index).get("snap"))


def expect_step(text: str, bbox: list) -> dict:
    """The assertion-mode step: wait, then require the proving text
    where the user clicked it (`within` padded from the row's bbox)."""
    within = [
        max(0.0, bbox[0] - EXPECT_PAD),
        max(0.0, bbox[1] - EXPECT_PAD),
        min(1.0, bbox[2] + EXPECT_PAD),
        min(1.0, bbox[3] + EXPECT_PAD),
    ]
    return {
        "tool": WAIT,
        "with": {"seconds": EXPECT_WAIT_SECONDS},
        "expect": {"text": text, "within": [round(v, 3) for v in within]},
    }


def mark_dismissal(draft: dict, name: str, index: int) -> None:
    """A recording accident becomes a guard: a tap that dismissed a
    popup is skipped when the popup's label is already gone."""
    m = _macro(draft, name)
    if not (0 <= index < len(m["steps"])):
        raise DraftError(f"no step {index} in {name!r}")
    step = m["steps"][index]
    readings = label_readings(step.get("with", {}))
    if step.get("tool") not in gesture_vocab.PRESS_TOOLS or not readings:
        raise DraftError("only a labeled press step can be a dismissal")
    update_step(draft, name, index, {"skip_when": {"not": readings[0]}})


# ---------- validation / emission / save ----------


def macro_data(m: dict) -> dict:
    """The draft macro as inline-grammar data (draft-only keys stripped)."""
    steps = [
        {k: v for k, v in s.items() if k not in _DRAFT_STEP_KEYS} for s in m["steps"]
    ]
    return {**({"inputs": m["inputs"]} if m["inputs"] else {}), "steps": steps}


def _validate(name: str, m: dict) -> None:
    if not m["steps"]:
        return  # an empty macro is mid-recording, not an error
    parse_inline_macro(macro_data(m), name)


def emit_macro_yaml(name: str, m: dict) -> str:
    """House-style MACRO.yml text. Strings quote via JSON (valid YAML,
    CJK readable); step comments render as YAML comments."""
    lines = [f"name: {name}"]
    lines.append(f"description: {quote_yaml(m['description'] or ('recorded ' + name))}")
    lines.append("enabled: false  # rehearse first, then enable")
    if m["inputs"]:
        lines.append("inputs:")
        for iname, spec in m["inputs"].items():
            lines.append(f"  {iname}:")
            for k in ("description", "default", "example"):
                if spec.get(k) is not None:
                    lines.append(f"    {k}: {quote_yaml(spec[k])}")
    lines.append("steps:")
    for s in m["steps"]:
        if s.get("comment"):
            lines.append(f"  # {s['comment']}")
        lines.append(f"  - name: {s['name']}")
        lines.append(f"    tool: {s['tool']}")
        for key in ("with", "guard", "skip_when", "expect"):
            if key in s:
                lines.append(f"    {key}: {quote_yaml(s[key])}")
        if s.get("hint"):
            lines.append(f"    hint: {quote_yaml(s['hint'])}")
    return "\n".join(lines) + "\n"


def macro_spec(name: str, m: dict) -> Macro:
    """The draft as a parsed `Macro` — emission validated through the
    save-door grammar, whatever the caller does with it next."""
    try:
        return parse_macro(emit_macro_yaml(name, m), name)
    except MacroError as e:
        raise DraftError(f"macro {name!r}: {e}") from e


def save_macro(app: str, name: str, m: dict) -> Path:
    """Write MACRO.yml where the target says: the pack's private
    `macros/` or the global `~/.physiclaw/macros/`. Emission is
    parse-validated first; `enabled: false` always — rehearse-then-
    enable stays a deliberate human act."""
    macro_spec(name, m)
    if m["target"] == "global":
        root = paths.macros_dir() / name
    else:
        root = paths.playbooks_dir() / app / "macros" / name
    root.mkdir(parents=True, exist_ok=True)
    path = root / MACRO_FILENAME
    write_text(path, emit_macro_yaml(name, m))
    return path
