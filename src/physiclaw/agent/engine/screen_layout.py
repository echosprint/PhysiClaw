"""First-run screen layout: validate the boxes the agent measured, accumulate.

The agent captures three pages one at a time — Spotlight search, the IM chat
with the keyboard down, and the same chat with it up. `screenshot` already
returns every on-screen element with its bbox, so the AGENT reads the input /
keyboard / Send boxes off that list (tapping and `peek`-ing to confirm the
keyboard is up), then calls `report_screen_layout(page, field, bbox, app)` once
per box with the coordinates it measured. This module does NOT detect anything
— it only sanity-checks each box (valid geometry, in the rough region that
field belongs to) and, if it passes, merges it into the running layout and
writes
``~/.physiclaw/screen-layout/``:

  layout.json  the structured bboxes (source of truth, built up per page),
               plus a `layout_learned` bool so out-of-module readers (the core
               orchestrator's /api/status) can tell setup is done without
               importing this module or knowing its page/field schema
  layout.md    the rendered card that `prompt._render_screen_layout` injects

`is_learned()` (all pages captured) / `missing_pages()` / `load_layout_md()`
are the read side the prompt builder uses to decide first-run setup vs.
inject-the-layout.
"""

import json
import logging
from dataclasses import dataclass, replace
from typing import NotRequired, TypedDict

from physiclaw.agent.engine.dto import Message, UserMessage
from physiclaw.agent.engine.geometry import center_of, inside
from physiclaw.common import paths
from physiclaw.common.gesture_vocab import (
    NAV_TOOLS,
    PRESS_TOOLS,
    SEQUENCE,
    STEP_ACTIONS,
    STEP_ARG,
    STEP_TOOL,
    SWIPE,
)
from physiclaw.common.text import read_text, write_text

log = logging.getLogger(__name__)

# The built-in skill that drives first-run capture — dropped from the prompt
# once the layout is learned (its body is only useful during setup).
SKILL_NAME = "screen-layout"

# Any chat app is supported — the agent passes whichever it's in. This just
# nice-cases a few common ones for the layout card; unknown apps are used
# verbatim (the agent may pass proper casing itself).
_APP_LABELS = {
    "wechat": "WeChat",
    "whatsapp": "WhatsApp",
    "telegram": "Telegram",
    "signal": "Signal",
    "messenger": "Messenger",
    "imessage": "iMessage",
    "line": "LINE",
    "wcom": "WeCom",
    "qq": "QQ",
    "kakaotalk": "KakaoTalk",
}


def _app_label(app: str) -> str:
    """Display name for the IM app — nice-cased if known, else as passed."""
    return _APP_LABELS.get(app.strip().lower(), app.strip())


# The three capture pages, and the layout fields each one contributes. Used to
# report what's still missing and to decide when setup is complete. Pages whose
# name starts with "chat" are IM-app-specific and require `app`.
PAGES = ("spotlight", "chat-no-keyboard", "chat-keyboard")
_PAGE_FIELDS = {
    # Spotlight owns the shared keyboard landmarks (space, backspace, and the
    # search/return key) — its keyboard is up and the search flow needs them.
    "spotlight": ("spotlight_input", "spotlight_paste", "backspace", "return", "space"),
    "chat-no-keyboard": ("chat_input_kb_hidden",),
    "chat-keyboard": ("chat_input_kb_visible", "send", "chat_paste"),
}
# Every field name, in capture order — the tool records one per call.
ALL_FIELDS = tuple(f for fields in _PAGE_FIELDS.values() for f in fields)


def _is_chat_page(page: str) -> bool:
    """Chat pages are IM-app-specific (need `app`); Spotlight is system-wide."""
    return page.startswith("chat")


def _load() -> dict:
    """Read the accumulated layout, or {} if none/unreadable yet."""
    p = paths.screen_layout_json()
    if not p.exists():
        return {}
    try:
        data = json.loads(read_text(p))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def missing_pages() -> list[str]:
    """Pages whose fields aren't all captured yet, in capture order."""
    d = _load()
    return [p for p in PAGES if not all(f in d for f in _PAGE_FIELDS[p])]


def is_learned() -> bool:
    """True once every page has been captured — keys the prompt's first-run
    branch off. ``~/.physiclaw/screen-layout/`` is otherwise incomplete."""
    return not missing_pages()


# Keys the user legitimately presses many times in a row (backspace-clearing
# a field, spacing through text). The stuck guard exempts these targets from
# same-target counting — repeated presses there are typing, not looping.
REPEATABLE_KEY_FIELDS = ("backspace", "space", "return")


def repeatable_key_boxes() -> list[list[float]]:
    """Bboxes of the learned keyboard keys that are safe to press
    repeatedly. Empty while the layout is unlearned."""
    d = _load()
    return [d[f] for f in REPEATABLE_KEY_FIELDS if isinstance(d.get(f), list)]


def _step_center(step, tool: str) -> tuple[float, float] | None:
    """Center of a sequence step's bbox if it's the given press tool."""
    if not isinstance(step, dict) or step.get(STEP_TOOL) != tool:
        return None
    return center_of(step.get(STEP_ARG) or [])


def _taps_box(steps, box) -> bool:
    """True if any step is a tap landing inside `box` (None/garbage → False)."""
    if not isinstance(box, list):
        return False
    for s in steps:
        c = _step_center(s, "tap")
        if c is not None and inside(c, box):
            return True
    return False


# Boxes that are only ever pressed while typing/pasting — a press there
# neither raises nor dismisses the keyboard.
_KEYBOARD_REGION_FIELDS = (
    "chat_input_kb_visible",
    "send",
    "chat_paste",
    "backspace",
    "return",
    "space",
    "spotlight_input",
    "spotlight_paste",
)


@dataclass
class KeyboardTracker:
    """Conservative cross-call belief about the on-screen keyboard.

    "up" is claimed only when the camera verified the raising press (a
    changed-verdict press on the chat input's keyboard-hidden box) and
    every gesture since provably preserves it (typing/pasting boxes).
    Nav gestures mean "down"; swipes, batches with presses, and presses
    outside the keyboard region decay to "unknown". Views, local tools,
    and clipboard syncs never touch the screen. Consumers act only on
    "up" — "down"/"unknown" fail open.
    """

    state: str = "unknown"  # "up" | "down" | "unknown"

    def observe(self, name: str, arguments: dict, changed: bool | None) -> None:
        if name in NAV_TOOLS:
            self.state = "down"
            return
        if name == SWIPE:
            self.state = "unknown"
            return
        if name == SEQUENCE:
            actions = arguments.get(STEP_ACTIONS)
            steps = actions if isinstance(actions, list) else []
            if any(
                isinstance(x, dict) and x.get(STEP_TOOL) in (*PRESS_TOOLS, SWIPE)
                for x in steps
            ):
                # A batch verdict can't be attributed per step — any press
                # or swipe inside may have moved the keyboard.
                self.state = "unknown"
            return
        if name not in PRESS_TOOLS:
            return  # views / local tools / clipboard — screen untouched
        c = center_of(arguments.get("bbox") or [])
        if c is None:
            self.state = "unknown"
            return
        d = _load()
        hidden = d.get("chat_input_kb_hidden")
        if isinstance(hidden, list) and inside(c, hidden):
            # The raising press — but only the camera proves the keyboard
            # actually rose (a dead press must not claim "up").
            self.state = "up" if changed is True else "unknown"
            return
        for f in _KEYBOARD_REGION_FIELDS:
            box = d.get(f)
            if isinstance(box, list) and inside(c, box):
                return  # typing/pasting — keyboard state preserved
        self.state = "unknown"  # a press elsewhere may have dismissed it


def lint_gesture(name: str, arguments: dict, *, keyboard_up: bool) -> str | None:
    """Pre-dispatch layout lint — blocking message, or None.

    Covers both shapes of the wrong-box failure: a `sequence` (checked
    always — in-batch evidence plus the caller's keyboard belief) and a
    STANDALONE `long_press` (checked only when the keyboard is believed
    up: the raising tap happened in an earlier call, so no in-batch
    evidence can exist).
    """
    if name == SEQUENCE:
        return lint_sequence(arguments.get(STEP_ACTIONS), keyboard_up=keyboard_up)
    if name != "long_press" or not keyboard_up:
        return None
    d = _load()
    hidden = d.get("chat_input_kb_hidden")
    visible = d.get("chat_input_kb_visible")
    if not (isinstance(hidden, list) and isinstance(visible, list)):
        return None
    c = center_of(arguments.get("bbox") or [])
    if c is None or not inside(c, hidden):
        return None
    return (
        f"BLOCKED — not executed: this long-press targets the chat input's "
        f"KEYBOARD-HIDDEN box {hidden}, but the keyboard is up (your earlier "
        "press on the input raised it), so that region is now the keyboard "
        "itself and no Paste popover can appear there. Long-press the "
        f"chat input's KEYBOARD-VISIBLE box {visible} instead "
        "(SYSTEM § Screen layout)."
    )


def lint_sequence(actions, *, keyboard_up: bool = False) -> str | None:
    """Pre-dispatch layout lint for a `sequence` batch — blocking message,
    or None.

    The one signature it refuses: a `long_press` on the chat input's
    KEYBOARD-HIDDEN box at a point in the batch where the keyboard must
    be up — because an earlier step tapped that box (which raises the
    keyboard), because the caller already believes the keyboard is up
    (`keyboard_up`, raised in an earlier call), or because a later step
    taps the Paste box (which only ever appears above the RISEN input).
    With the keyboard up, the kb-hidden region IS the keyboard's bottom
    rows, so the popover never opens and the batch "succeeds" into a
    no-op the stuck guard can't see (every press changes the screen).
    The im template's correct box is `chat_input_kb_visible`.

    Fail-open by design: unlearned layout, malformed steps, or missing
    fields return None — this lint must never invent a blocker.
    """
    if not isinstance(actions, list):
        return None
    d = _load()
    hidden = d.get("chat_input_kb_hidden")
    visible = d.get("chat_input_kb_visible")
    paste = d.get("chat_paste")
    if not (isinstance(hidden, list) and isinstance(visible, list)):
        return None

    input_tapped_at: int | None = None
    for i, step in enumerate(actions, 1):
        tap_c = _step_center(step, "tap")
        if tap_c is not None and inside(tap_c, hidden):
            if input_tapped_at is None:
                input_tapped_at = i
            continue
        lp_c = _step_center(step, "long_press")
        if lp_c is None or not inside(lp_c, hidden):
            continue
        pastes_after = _taps_box(actions[i:], paste)
        if input_tapped_at is None and not keyboard_up and not pastes_after:
            continue  # bare long-press near the bottom — not the paste flow
        if input_tapped_at is not None:
            why = f"step {input_tapped_at} taps that box, which raises the keyboard"
        elif keyboard_up:
            why = "the keyboard is already up (raised by an earlier press on the input)"
        else:
            why = "the batch then taps the Paste box, which only appears above the RISEN input"
        return (
            f"BLOCKED — no step ran: step {i} long-presses the chat input's "
            f"KEYBOARD-HIDDEN box {hidden}, but {why} — that region is now "
            "the keyboard, so no Paste popover can appear. Re-issue the "
            f"batch with the long-press on the KEYBOARD-VISIBLE box "
            f"{visible} (SYSTEM § Screen layout)."
        )

    # Reused-paste-box guard: the batch taps the learned chat Paste box but
    # long-presses a box OTHER than the chat input it belongs to. chat_paste
    # only exists above chat_input_kb_visible after long-pressing IT; copying
    # the `im` bundled sequence into another app (a search field in Meituan /
    # JD / …) long-presses THAT app's box, so the reused chat_paste coord lands
    # on empty screen and the paste silently no-ops — the batch still "ok"s and
    # the agent loops. The correct im template long-presses chat_input_kb_visible,
    # so it never trips this.
    if isinstance(paste, list) and _taps_box(actions, paste):
        for step in actions:
            c = _step_center(step, "long_press")
            if c is not None and not inside(c, visible):
                return (
                    f"BLOCKED — no step ran (clipboard unchanged): reuses IM "
                    f"chat Paste box {paste} after long-pressing a different "
                    "field — its Paste popover is elsewhere. Redo every "
                    "step: long-press the field ALONE (own turn), read Paste "
                    "from that view (`search-in-app`). Don't reuse "
                    "chat_paste."
                )
    return None


def prune_builtin_skills(skills: dict) -> dict:
    """Drop the first-run `screen-layout` skill once the layout is learned —
    after setup its body is dead weight in `## Built-in Skills`. No-op (and a
    fresh dict) while still incomplete, so the skill stays available for
    capture."""
    if is_learned():
        return {k: v for k, v in skills.items() if k != SKILL_NAME}
    return skills


# Built-in skill markdown placeholder -> learned layout field. Keyed by skill:
# the same token means different fields in different skills (im's <paste-button>
# is the chat paste; open-app's is the Spotlight paste). Only `im` is filled —
# it has a copy-paste `sequence` template the agent runs verbatim. Just the
# tokens that appear in that fenced template (filling is code-block-only, so a
# prose-only token like <backspace> would never substitute).
# Skill placeholders use the same `{{token}}` syntax as doctrine's
# config tokens (common.doctrine.DOCTRINE_TOKENS) — one placeholder
# style everywhere. Two domains though: doctrine tokens are ALL substituted
# (leftovers warn), while skill element tokens are substituted only
# here, only inside code blocks, only once the layout is learned —
# elsewhere they stay symbolic and the agent resolves them from
# SYSTEM § Screen layout.
_SKILL_BOX_TOKENS = {
    "im": {
        "{{input-hidden}}": "chat_input_kb_hidden",
        "{{input-visible}}": "chat_input_kb_visible",
        "{{paste-button}}": "chat_paste",
        "{{send}}": "send",
    },
    # Same token name, different field per skill: {{paste-button}} is the
    # CHAT paste box in `im` but the SPOTLIGHT paste box here.
    "open-app": {
        "{{search-field}}": "spotlight_input",
        "{{backspace}}": "backspace",
        "{{paste-button}}": "spotlight_paste",
    },
}


def _sub_in_code_blocks(body: str, subs: dict) -> str:
    """Apply `subs` (token -> replacement) only INSIDE fenced ``` code blocks.
    Splitting on the fence, the odd-indexed segments are the code bodies."""
    parts = body.split("```")
    for i in range(1, len(parts), 2):
        for token, val in subs.items():
            parts[i] = parts[i].replace(token, val)
    return "```".join(parts)


def fill_builtin_boxes(skills: dict) -> dict:
    """Once the layout is learned, swap the bbox placeholders
    (`{{input-hidden}}`, `{{send}}`, ...) for concrete coordinates INSIDE the
    skill's fenced code template (im's send `sequence`) so the agent runs it
    verbatim. Prose keeps the readable placeholder names — they reference
    § Screen layout. No-op (a fresh dict) while incomplete. Inputs aren't
    mutated; filled skills are copies."""
    if not is_learned():
        return dict(skills)
    layout = _load()
    out = {}
    for name, s in skills.items():
        subs = {
            token: _fmt(layout[field])
            for token, field in _SKILL_BOX_TOKENS.get(name, {}).items()
            if layout.get(field)
        }
        body = _sub_in_code_blocks(s.body, subs) if subs else s.body
        out[name] = replace(s, body=body) if body != s.body else s
    return out


def load_layout_md() -> str:
    """The rendered layout card, or '' if nothing captured yet."""
    p = paths.screen_layout_md()
    return read_text(p).strip() if p.exists() else ""


# ─── First-run tail reminder ──────────────────────────────────


def tail_reminder() -> str:
    """A first-run nudge listing, per page, which fields are still missing and
    what to do — pinned at the request tail so it's fresh and reflects capture
    progress. '' once every field is captured (nothing to append)."""
    d = _load()
    missing = {p: [f for f in _PAGE_FIELDS[p] if f not in d] for p in PAGES}
    missing = {p: fs for p, fs in missing.items() if fs}
    if not missing:
        return ""
    done = [p for p in PAGES if p not in missing]
    lines = [
        "[First-run setup needed]",
        "You can't reliably open apps by Spotlight search or send IM messages "
        "until the screen layout is learned.",
    ]
    if done:
        lines.append(f"Pages fully captured: {', '.join(done)}.")
    lines.append("Still to capture (page → missing fields):")
    lines += [f"  • {p}: {', '.join(fs)}" for p, fs in missing.items()]
    lines.append(
        "Follow the `screen-layout` skill (in `## Built-in Skills`) to capture "
        "each field above."
    )
    return "\n".join(lines)


def inject_tail(messages: list[Message]) -> list[Message]:
    """Append the first-run setup reminder as the LAST message while the
    screen layout isn't fully learned; return `messages` unchanged once it
    is. Wake/turn-volatile (reflects capture progress), so it rides the tail
    rather than the cached SYSTEM prompt. Original list is not mutated."""
    reminder = tail_reminder()
    return messages + [UserMessage(content=reminder)] if reminder else messages


# ─── Validation ───────────────────────────────────────────────
#
# The agent measures the boxes; we only reject ones that are clearly wrong.
# Each field has a rough region (center within a band, minimum width) — coarse
# on purpose, just enough to catch a mis-picked element before it's persisted.


class _Region(TypedDict):
    """Rough region for one field — any key may be absent."""

    cx: NotRequired[tuple[float, float]]  # (lo, hi) band for the center x
    cy: NotRequired[tuple[float, float]]  # (lo, hi) band for the center y
    wmin: NotRequired[float]  # minimum width


_REGION: dict[str, _Region] = {
    # Bands are calibrated from real WeChat/Spotlight pages, with margin for
    # other apps/devices. Spotlight's field sits LOW — just above the keyboard
    # (~y 0.60); the chat input bar is at the very bottom with the keyboard down
    # (~y 0.94) and just above the keyboard with it up (~y 0.61).
    "spotlight_input": {"cy": (0.50, 0.70), "wmin": 0.70},
    "chat_input_kb_hidden": {"cy": (0.80, 0.99), "wmin": 0.40},
    "chat_input_kb_visible": {"cy": (0.50, 0.70), "wmin": 0.20},
    # Paste is the edit-menu button raised by long-pressing the input box — a
    # small bubble that floats just ABOVE the field (~y 0.53). No width floor
    # (it's a button, not a bar); the floor allows for the callout riding a
    # little higher above a tall input box.
    "spotlight_paste": {"cy": (0.40, 0.72)},
    "chat_paste": {"cy": (0.40, 0.72)},
    # Keyboard: space bar and bottom-right key (~y 0.90), backspace one row up
    # (~y 0.84), all on the right for the two corner keys.
    "space": {"cy": (0.80, 1.00), "wmin": 0.20},
    "backspace": {"cy": (0.70, 1.00), "cx": (0.80, 1.00)},
    "return": {"cy": (0.80, 1.00), "cx": (0.80, 1.00)},
    # Send sits in one of two places, app-depending: a keyboard key at the
    # bottom (WeChat-style) OR a button on the input bar's right (WhatsApp /
    # Telegram / most apps). Either way it's on the right and in the lower part
    # of the screen, so one broad region covers every app.
    "send": {"cy": (0.50, 1.00), "cx": (0.80, 1.00)},
}


def _validate_box(field: str, raw) -> str | None:
    """Return an error string if `raw` is not a plausible bbox for `field`,
    else None. Checks geometry then the field's rough region."""
    try:
        bbox = [float(v) for v in raw]
    except (TypeError, ValueError):
        return f"{field}: bbox must be four numbers [left, top, right, bottom]."
    if len(bbox) != 4:
        return f"{field}: expected 4 numbers, got {len(bbox)}."
    left, top, right, bottom = bbox
    if not all(0.0 <= v <= 1.0 for v in bbox):
        return f"{field}: coordinates must be normalized 0–1 ({bbox})."
    if left >= right or top >= bottom:
        return f"{field}: need left<right and top<bottom ({bbox})."

    region: _Region = _REGION.get(field, {})
    cx, cy, width = (left + right) / 2, (top + bottom) / 2, right - left
    if "cx" in region and not region["cx"][0] <= cx <= region["cx"][1]:
        return f"{field}: looks off — expected it toward x∈{region['cx']} (got center x={cx:.2f})."
    if "cy" in region and not region["cy"][0] <= cy <= region["cy"][1]:
        return f"{field}: looks off — expected it toward y∈{region['cy']} (got center y={cy:.2f})."
    if "wmin" in region and width < region["wmin"]:
        return f"{field}: too narrow for an input bar (width {width:.2f} < {region['wmin']})."
    return None


# ─── Rendering ────────────────────────────────────────────────


def _fmt(bbox) -> str:
    if not bbox:
        return "(not captured yet)"
    return "[" + ", ".join(f"{v:.3f}" for v in bbox) + "]"


def _render_md(layout: dict) -> str:
    """Render the accumulated layout as the markdown injected into SYSTEM.
    Partial-aware — uncaptured boxes show `(not captured yet)`."""
    im_app = layout.get("im_app")
    app = f"{im_app}'s" if im_app else "per-IM-app"
    return "\n".join(
        [
            "Key input boxes and keyboard landmarks, measured from your phone — "
            "`[left, top, right, bottom]`, 0–1. Trust these over generic priors; "
            "re-ground from the current view if a tap misses.",
            "",
            f"**Chat input + Send below are {app}.** For any other IM app, ignore "
            "them and ground live. Spotlight + keyboard keys are system-wide.",
            "",
            "### Input boxes",
            "",
            f"- Spotlight search {_fmt(layout.get('spotlight_input'))} — paste an app name to open it",
            f"- Chat input, keyboard hidden {_fmt(layout.get('chat_input_kb_hidden'))} — tap to raise the keyboard",
            f"- Chat input, keyboard visible {_fmt(layout.get('chat_input_kb_visible'))} — long-press to Paste",
            f"- Send {_fmt(layout.get('send'))} — tap to send an IM",
            "",
            "### Paste buttons",
            "",
            "Only exist AFTER long-pressing the input box (above):",
            "",
            f"- Spotlight search → Paste at {_fmt(layout.get('spotlight_paste'))}",
            f"- Chat input (keyboard visible) → Paste at {_fmt(layout.get('chat_paste'))}",
            "",
            "### Keyboard keys",
            "",
            f"- backspace ⌫ {_fmt(layout.get('backspace'))}",
            f"- return / search / send {_fmt(layout.get('return'))}",
            f"- space {_fmt(layout.get('space'))}",
            "",
            "Bottom-right key: label varies (`Send` / `Return` / `Search` / `搜索` / "
            "`前往`), bbox stable. Tap not registering → check the current view: is "
            "the keyboard actually visible?",
        ]
    )


# ─── Record (the tool entry point) ────────────────────────────


def record(page: str, field: str, bbox, app: str | None = None) -> str:
    """Validate ONE box the agent measured — `field` (e.g. 'send') with its
    `bbox` on `page` — and, if it passes, merge it into layout.json and
    re-render layout.md. `app` (any chat app) is required for the chat pages:
    it labels the chat boxes as belonging to that app. Returns a message for
    the agent — the layout so far plus what's still to capture, or a re-measure
    instruction on failure."""
    if page not in PAGES:
        return f"unknown page {page!r}; expected one of: {', '.join(PAGES)}."
    allowed = _PAGE_FIELDS[page]
    if field not in allowed:
        return (
            f"{field!r} is not a {page} field; expected one of: {', '.join(allowed)}."
        )
    if _is_chat_page(page) and not (app and app.strip()):
        return (
            f"{page} needs the IM app: pass app='<the chat app you're in>' "
            "(e.g. 'wechat', 'whatsapp', 'telegram', ...) so the chat boxes are "
            "labelled with their app."
        )
    err = _validate_box(field, bbox)
    if err:
        return (
            f"Layout not saved — {err} Re-read the box off the screenshot "
            "(copy the element's bbox verbatim) and call again."
        )

    layout = _load()
    # Reporting the same field again is fine — it OVERWRITES (the re-measure /
    # correction path). `was_complete` distinguishes the call that finishes
    # setup (→ restart) from a later correction to an already-complete layout.
    was_complete = not any(f not in layout for f in ALL_FIELDS)
    layout[field] = [round(float(v), 3) for v in bbox]
    if _is_chat_page(page) and app and app.strip():
        layout["im_app"] = _app_label(app)
    still = [f for f in ALL_FIELDS if f not in layout]
    # Persist a plain `learned` flag alongside the boxes so readers outside
    # this module (e.g. the core orchestrator's /api/status) can tell setup is
    # done by reading the file — without importing this module or knowing its
    # page/field schema. `is_learned()` derives the same truth from the schema.
    layout["layout_learned"] = not still
    md = _render_md(layout)
    paths.screen_layout_dir().mkdir(parents=True, exist_ok=True)
    write_text(paths.screen_layout_json(), json.dumps(layout, indent=2))
    write_text(paths.screen_layout_md(), md + "\n")

    if still:
        # The per-turn first-run notice (tail_reminder) already lists which
        # fields remain and points at the skill — don't repeat it here.
        note = f"{len(still)} box(es) still to capture — see the first-run notice."
    elif not was_complete:
        note = (
            "All boxes captured — setup is done. If you woke for a task, the "
            "session restarts with the layout loaded to finish it; otherwise it "
            "just ends (the layout is saved and loads on the next wake)."
        )
    else:
        note = (
            "The layout was already complete; this updated one box. No restart "
            "needed — carry on with the request."
        )
    return f"Saved `{field}` = {layout[field]}. {note}"
