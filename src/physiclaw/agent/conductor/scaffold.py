"""Authoring-surface texts for app packs: the `init` scaffold + README.

The scaffold IS the format reference a new user edits in place
(the macro-scaffold doctrine), and the README is kept current at
``playbooks/README.md`` via ``ensure_readme``. Caps and vocabularies
interpolate from `playbook`/`calls` constants, never hand-copied."""

from pathlib import Path

from physiclaw.agent.conductor.calls import CALLS
from physiclaw.agent.conductor.pages import PAGES_FILENAME, RESERVED_APPS
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
# bounded by max_visits):
#   LEG        — run one of THIS pack's macros ({app}/macros/<name>/),
#                verified against a declared page (`verify:`)
#   DECIDE     — one scoped decision call ({calls});
#                `on:` must route EVERY out; outputs wire forward as
#                {{node.field}}
#   CONFIRM    — compose + send a message to the user, then park
#   HUMAN_GATE — compose + send the full context, then hold: continue
#                only on a confirmed reply; after {gate_checks} unconfirmed
#                checks the session parks for the next wake-up
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
forward-only except a DECIDE's bounded self-loop; `irreversible:
payment` nodes sit behind a HUMAN_GATE and require a `mandate:`.

Limits: ≤ {max_nodes} nodes, ≤ {max_inputs} inputs per playbook.
Macro format: see ~/.physiclaw/macros/README.md (identical grammar).
""".format(max_nodes=MAX_NODES, max_inputs=MAX_INPUTS)


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
    if app in RESERVED_APPS:
        raise PlaybookError(f"{app!r} is a reserved namespace (built-in pages)")
    root = paths.playbooks_dir() / app
    if root.exists():
        raise PlaybookError(f"pack directory already exists: {root}")
    macro_dir = root / PACK_MACROS_DIRNAME / EXAMPLE_MACRO
    macro_dir.mkdir(parents=True)
    write_text(root / PAGES_FILENAME, render_pages_stub())
    write_text(root / f"{EXAMPLE_PLAYBOOK}.yml", render_playbook_stub(app))
    write_text(macro_dir / MACRO_FILENAME, render_example_macro())
    ensure_format_readme()
    return root
