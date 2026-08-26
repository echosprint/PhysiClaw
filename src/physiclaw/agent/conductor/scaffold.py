"""Authoring-surface texts for app packs: the `init` scaffold + README.

The scaffold IS the format reference a new user edits in place
(the macro-scaffold doctrine), and the README is kept current at
``playbooks/README.md`` via ``ensure_readme``. Caps and vocabularies
interpolate from `playbook`/`calls` constants, never hand-copied."""

from pathlib import Path

from physiclaw.agent.conductor.calls import CALLS
from physiclaw.agent.conductor.pages import (
    CHANNEL_APP,
    OPEN_MACRO,
    PAGES_FILENAME,
    SEND_MACRO,
    THREAD_PAGE,
    reserved_unloadable,
)
from physiclaw.agent.conductor.playbook import (
    GATE_MAX_CHECKS,
    IRREVERSIBLE_CLASSES,
    MAX_INPUTS,
    MAX_NODES,
    PACK_MACROS_DIRNAME,
    PlaybookError,
    check_name,
)
from physiclaw.agent.macros import scaffold as macro_scaffold
from physiclaw.agent.macros.model import MACRO_FILENAME

EXAMPLE_MACRO = "example-leg"
EXAMPLE_PLAYBOOK = "example"


def render_pages_stub() -> str:
    return """\
# Page DECLARATIONS for this app — semantics only, never geometry.
# Positions, tolerances, and thresholds are captured on YOUR device
# (`physiclaw conductor calibrate`); this file says which label texts
# identify each page. Use `physiclaw conductor propose --live` on each
# page for anchor candidates.
home:
  anchors:
    - "EDIT ME"                    # a label text that identifies this page
    # - {text: "tab", region: bottom}   # coarse band: top/bottom/left/right
  # forbid: ["popup text"]         # veto terms — kills look-alike takeovers
  # scrollable: true               # content may scroll under fixed chrome
"""


def render_playbook_stub(app: str) -> str:
    return PLAYBOOK_TEMPLATE.format(
        app=app,
        playbook=EXAMPLE_PLAYBOOK,
        macro=EXAMPLE_MACRO,
        max_inputs=MAX_INPUTS,
        max_nodes=MAX_NODES,
        gate_checks=GATE_MAX_CHECKS,
        calls=", ".join(sorted(CALLS)),
        classes="|".join(IRREVERSIBLE_CLASSES),
    )


PLAYBOOK_TEMPLATE = """\
name: {playbook}          # must equal the filename stem
description: EDIT ME — one line saying what task this playbook does

# A valid playbook is enabled by default; this scaffold starts off.
enabled: false

# Inputs the conductor fills at activation; reference them as {{name}}
# in `with:` values. ≤ {max_inputs} inputs; `default` makes one optional.
inputs:
  message:
    description: EDIT ME — what this value means
    example: "hello"

# Node types (≤ {max_nodes} nodes, forward-only; a DECIDE may self-loop
# only on its call's re-ask arm — choose_item's `scroll`, the conductor
# swipes between re-asks — bounded by max_visits):
#   LEG        — run one of THIS pack's macros ({app}/macros/<name>/),
#                verified against a declared page (`verify:`)
#   DECIDE     — one scoped decision call ({calls});
#                `on:` must route EVERY out; outputs wire forward as
#                {{node.field}}. `context: [memory.<slug>]` shares ONLY
#                the `## <slug>` section of memory.md (fail-closed)
#   CONFIRM    — send `message:` to the user (the EXACT text sent —
#                write it in the user's language), then park
#   HUMAN_GATE — send `message:`, then hold: continue only on a
#                confirmed reply; after {gate_checks} unconfirmed checks
#                the session parks for the next wake-up. `gate: payment`
#                fills {{gate.total}}/{{gate.cap}} at runtime; its
#                `message:` must quote {{gate.total}} and an
#                `over_message:` (sent when the total exceeds the cap)
#                must quote both. Optional `return: <macro>` runs after
#                the confirm to get back into the app (the ask left it
#                for the IM thread)
# A LEG doing real damage carries `irreversible: <{classes}>`;
# `payment` must sit behind a HUMAN_GATE and requires a `mandate:`.
nodes:
  - id: open
    type: LEG
    macro: {macro}
    with: {{message: "{{message}}"}}
    verify: home
  - id: confirm
    type: CONFIRM
    compose: status-update
    # The EXACT text sent to the user — write it in THEIR language.
    message: "EDIT ME — done with {{message}}, reply to continue"
"""


README_CONTENT = """\
# App packs

One directory per app, self-contained — everything its playbooks use:

    playbooks/<app>/
      pages.yml            page declarations (semantics; geometry is
                           captured on-device into learned/pages/)
      <playbook>.yml       one playbook per file, name = filename stem
      macros/<n>/MACRO.yml pack-private macros (same format as
                           ~/.physiclaw/macros/, but never shown to the
                           model's macro list — the conductor's hands)

Validate everything: `physiclaw playbooks check`. Scaffold a pack:
`physiclaw playbooks init <app>`.

Rules the checker enforces: a playbook references only its own pack's
macros and pages (plus the reserved `ios.*` / `channel.*` built-ins);
every DECIDE out is routed; `{{name}}` refs name declared inputs and
`{{node.field}}` refs name EARLIER decide outputs; graphs are
forward-only except a DECIDE's bounded re-ask self-loop
(choose_item's `scroll`); `irreversible:
payment` nodes sit behind a HUMAN_GATE and require a `mandate:`.

Limits: ≤ {max_nodes} nodes, ≤ {max_inputs} inputs per playbook.
Macro format: see ~/.physiclaw/macros/README.md (identical grammar).
""".format(max_nodes=MAX_NODES, max_inputs=MAX_INPUTS)


# ---------- the channel pack (conductor infrastructure) ----------

CHANNEL_PAGES_STUB = f"""\
# The user-channel thread page — the ONE page the conductor must
# recognize: your own chat thread in your IM app. Anchor on the chat
# header (your name / the contact name) + stable chrome.
{THREAD_PAGE}:
  anchors:
    - "EDIT ME"                    # the thread header text
"""

# Rehearsable skeletons for the two channel macros. Steps are
# placeholders that PARSE clean; record the real gesture path on your
# device (spotlight/dock → your thread), then rehearse and enable.
CHANNEL_SEND_STUB = f"""\
name: {SEND_MACRO}
description: open the user's IM thread and send {{{{message}}}}
enabled: false
inputs:
  message:
    description: the text to send to the user
steps:
  - name: go-home
    tool: home_screen
  - name: open-im
    tool: tap
    with: {{bbox: [0.1, 0.9, 0.2, 0.98]}}    # EDIT ME: the IM app's dock icon
  - name: open-thread
    tool: tap
    with: {{bbox: [0.1, 0.15, 0.9, 0.22]}}   # EDIT ME: your thread's row
  - name: raise-keyboard
    tool: tap
    with: {{bbox: [0.1, 0.9, 0.7, 0.96]}}    # EDIT ME: the input box (hidden)
  - name: clip
    tool: send_to_clipboard
    with: {{text: "{{message}}"}}
  - name: hold-input
    tool: long_press
    with: {{bbox: [0.1, 0.5, 0.7, 0.56]}}    # EDIT ME: the input box (visible)
  - name: paste
    tool: tap
    with: {{bbox: [0.1, 0.44, 0.3, 0.5]}}    # EDIT ME: the Paste button
  - name: send
    tool: tap
    with: {{bbox: [0.8, 0.5, 0.98, 0.56]}}   # EDIT ME: the Send key
"""

CHANNEL_OPEN_STUB = f"""\
name: {OPEN_MACRO}
description: open the user's IM thread (read only, no send)
enabled: false
steps:
  - name: go-home
    tool: home_screen
  - name: open-im
    tool: tap
    with: {{bbox: [0.1, 0.9, 0.2, 0.98]}}    # EDIT ME: the IM app's dock icon
  - name: open-thread
    tool: tap
    with: {{bbox: [0.1, 0.15, 0.9, 0.22]}}   # EDIT ME: your thread's row
"""


def _init_channel_pack(root: Path) -> Path:
    """`playbooks/channel/` — the infrastructure pack: the thread page +
    the send/open macros the conductor's asks run. No playbooks live
    here; packs never name it."""
    from physiclaw.common.text import write_text

    for name, stub in (
        (SEND_MACRO, CHANNEL_SEND_STUB),
        (OPEN_MACRO, CHANNEL_OPEN_STUB),
    ):
        d = root / PACK_MACROS_DIRNAME / name
        d.mkdir(parents=True)
        write_text(d / MACRO_FILENAME, stub)
    write_text(root / PAGES_FILENAME, CHANNEL_PAGES_STUB)
    ensure_format_readme()
    return root


def render_example_macro() -> str:
    """The pack's example macro — the macro scaffold verbatim (it parses
    clean and is the macro-format documentation)."""
    return macro_scaffold.render_init(EXAMPLE_MACRO)


def ensure_format_readme() -> None:
    """Keep ``playbooks/README.md`` current — the macro-store pattern:
    rewritten whenever the shipped constant changed, fail-open, called
    from every user-facing CLI moment."""
    from physiclaw.common import paths
    from physiclaw.common.logger import ensure_readme

    ensure_readme(paths.playbooks_dir(), README_CONTENT)


def init_pack(app: str) -> Path:
    """Scaffold ``playbooks/<app>/`` — pages stub, disabled example
    playbook, example pack macro — and return the pack root. Raises
    PlaybookError on a bad/reserved name or an existing directory."""
    from physiclaw.common import paths
    from physiclaw.common.text import write_text

    check_name(app, "app name")
    if reserved_unloadable(app):
        raise PlaybookError(f"{app!r} is a reserved namespace (built-in pages)")
    root = paths.playbooks_dir() / app
    if root.exists():
        raise PlaybookError(f"pack directory already exists: {root}")
    if app == CHANNEL_APP:
        return _init_channel_pack(root)
    macro_dir = root / PACK_MACROS_DIRNAME / EXAMPLE_MACRO
    macro_dir.mkdir(parents=True)
    write_text(root / PAGES_FILENAME, render_pages_stub())
    write_text(root / f"{EXAMPLE_PLAYBOOK}.yml", render_playbook_stub(app))
    write_text(macro_dir / MACRO_FILENAME, render_example_macro())
    ensure_format_readme()
    return root
