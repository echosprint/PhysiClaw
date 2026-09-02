"""Authoring-surface texts for app packs: the `init` scaffold + README.

The scaffold IS the format reference a new user edits in place
(the macro-scaffold doctrine), and the README is kept current at
``playbooks/README.md`` via ``ensure_readme``. Caps and vocabularies
interpolate from `playbook`/`calls` constants, never hand-copied."""

import logging
from pathlib import Path

from physiclaw.common.paths import PACK_FILENAME
from physiclaw.conductor.calls import AGENT_TOOLS
from physiclaw.conductor.pages import (
    CHANNEL_APP,
    IOS_APP,
    LOCKED_PAGE,
    OPEN_MACRO,
    SEND_MACRO,
    THREAD_PAGE,
)
from physiclaw.conductor.playbook import (
    GATE_MAX_CHECKS,
    IRREVERSIBLE_CLASSES,
    MAX_NODES,
    PACK_MACROS_DIRNAME,
    PlaybookError,
    check_name,
)
from physiclaw.conductor.route import (
    DEFAULT_AGENT_CALLS,
    DEFAULT_AGENT_SCROLLS,
    MAX_AGENT_CALLS,
    RECOVER_TOOLS,
)
from physiclaw.macros import scaffold as macro_scaffold
from physiclaw.macros.model import MACRO_FILENAME, MAX_INPUTS

log = logging.getLogger(__name__)


def read_template_manifest(src: Path) -> dict:
    """A template pack's PLAYBOOK.yml, loaded RAW (its `<<TOKEN>>`s still
    in place) for `playbooks install` — name/description plus the
    `placeholders:` spec map that drives the prompts. {} when absent;
    loud PlaybookError on bad YAML or a malformed `placeholders:`
    section, so install fails BEFORE it mutates anything."""
    from physiclaw.common.text import read_text
    from physiclaw.conductor import _spec

    path = src / PACK_FILENAME
    if not path.exists():
        return {}
    try:
        data = _spec.yaml_loader.load(read_text(path))
    except Exception as e:  # loader errors are not confined to YAMLError
        raise PlaybookError(f"{path}: invalid YAML: {e or type(e).__name__}") from e
    if not isinstance(data, dict):
        raise PlaybookError(f"{path} must be a YAML mapping")
    spec = data.get("placeholders") or {}
    if not isinstance(spec, dict) or not all(
        isinstance(v, dict) for v in spec.values()
    ):
        raise PlaybookError(
            f"{path}: `placeholders` must map TOKEN -> {{description, example}}"
        )
    return data


EXAMPLE_MACRO = "example-move"
EXAMPLE_PLAYBOOK = "example"


def render_pack_stub(app: str) -> str:
    return PACK_TEMPLATE.format(
        app=app,
        playbook=EXAMPLE_PLAYBOOK,
        macro=EXAMPLE_MACRO,
        max_inputs=MAX_INPUTS,
        max_nodes=MAX_NODES,
        gate_checks=GATE_MAX_CHECKS,
        classes="|".join(IRREVERSIBLE_CLASSES),
        agent_tools=", ".join(AGENT_TOOLS),
        max_agent_calls=MAX_AGENT_CALLS,
        agent_calls=DEFAULT_AGENT_CALLS,
        agent_scrolls=DEFAULT_AGENT_SCROLLS,
        recover_tools=" / ".join(RECOVER_TOOLS),
    )


PACK_TEMPLATE = """\
# The pack spec — ONE file, read top-down: meta, then each walk as a
# ROUTE with two kinds of steps — moves the walker makes itself
# (start/do) and briefs it hands the model (agent) — alternating with
# `page:` waypoints, then `pages:` for anything off the route.
# Directory macros (macros/<name>/MACRO.yml) hold hands shared across
# playbooks; single-use hands embed in their moves.
app: {app}           # which app this pack automates = the directory name
description: EDIT ME — what this pack automates, and when to adopt it

# Per-installation constants — mark them in any file below as the
# token name in doubled angle brackets; `physiclaw playbooks install`
# prompts adopters and records values in playbooks/placeholders.yml
# (files keep their tokens; parsers fill them at load):
# placeholders:
#   CONTACT:
#     description: EDIT ME — what to fill in
#     example: SomeOne

# Named fixed spots the author KNOWS — recover hands tap them, and
# agent episodes are granted them by name (`give: [landmarks.back]`).
# Open vocabulary; each is {{label, bbox}}, label-healed live when the
# label is readable on screen.
# landmarks:
#   back:
#     label: "back chevron (top-left)"
#     bbox: [0.015, 0.045, 0.095, 0.095]

# The walks. Key = playbook name (referenced as {app}/<name>). A route
# may open with pure-text agent steps and one `start`, then alternates
# WHERE (page — checked every time) with WHAT (a move); ≤ {max_nodes}
# moves, forward-only — whatever needs judgment is an `agent` step,
# whatever needs a human is an `ask`. Entry kinds:
#   page   — where the walk must BE. Declare it in place (anchors
#            beside the waypoint — semantics only; geometry is captured
#            on YOUR device via `physiclaw conductor calibrate`), or
#            reference a page declared elsewhere bare. `recover:`
#            declares ITS recovery hand — one gesture
#            ({recover_tools}; tap takes with: landmarks.<name>) or an
#            argument-less `macro:`; after it runs the walk re-locates
#            on the route. What you declare is what runs (after the
#            built-in unlock and one settle re-peek): a page declaring
#            none hands over.
#   start  — the unconditional cold-launch (at most one, immediately
#            before the first page — the landing it must reach);
#            `macro:` is its hand. It re-runs whenever recovery walks
#            from the route top.
#   do     — run a gesture macro. The name IS the macro (the directory
#            macro of that name, or the `macro:` body beside it — the
#            MACRO.yml grammar minus name/description/enabled); the
#            page that FOLLOWS is its landing check. A move doing real
#            damage carries `irreversible: <{classes}>` — `payment`
#            must directly follow an `ask` that approves it.
#   agent  — hand the step to the model with YOUR prompt (refs fill
#            once when it opens). No `tools` = a pure-text call: prompt
#            in, `returns:` fields out (read downstream as
#            {{name.field}}), legal before the first page. With
#            `tools: [{agent_tools}]` it is an EPISODE framed by the
#            adjacent pages: each turn the model answers with a screen
#            row, a landmark granted via `give:`, a scroll verb, done,
#            or escalate — never coordinates; `limit:
#            {{calls, scrolls}}` bounds it (≤ {max_agent_calls} calls),
#            and `done` counts only on the page that follows, judged by
#            the matcher.
#   ask    — message the user and WAIT for approval; after
#            {gate_checks} unconfirmed checks the session suspends.
#            `approve: payment` fills {{ask.total}} — the `message:`
#            must quote it. A `budget:` adds {{ask.cap}} and requires
#            `over_budget_message:` quoting both; without one the
#            consented total is the fire-time bound. `resume:`
#            re-enters the app after the reply.
#   tell   — message the user, then pause until any reply.
playbooks:
  {playbook}:
    description: EDIT ME — one line saying what task this playbook does
    # A valid playbook is enabled by default; this scaffold starts off.
    enabled: false
    # Values filled at activation; reference them as {{inputs.name}} in
    # `with:` values and agent prompts. ≤ {max_inputs}; `default`
    # present = optional.
    inputs:
      message:
        description: EDIT ME — what this value means
        example: "hello"
    route:
      # An agent step with no tools may open the route — derive values
      # from the user's words before the phone is touched:
      # - agent: parse
      #   prompt: |
      #     EDIT ME — say exactly what to derive and the rules.
      #     The user said: "{{inputs.message}}"
      #   returns:
      #     keyword: EDIT ME — what this field holds
      - start: app              # ── the cold-launch; `home` below is the
        macro:                  #    landing it must reach
          steps:
            - name: go-home
              tool: home_screen
            - name: open-app
              tool: tap
              with: {{label: "the app icon", bbox: [0.1, 0.1, 0.3, 0.2]}}
            - name: settle
              tool: wait
              with: {{seconds: 3}}
      - page: home
        # `anchors:` is ONE clause: a bare string, {{text|or, region}},
        # or {{and: [...]}} for several anchors at once. `or:` lists
        # alternate READINGS of one anchor (any hit scores it once) —
        # never write alternates as separate anchors: each declared
        # anchor is a share of the page's score.
        anchors: "EDIT ME"      # a label text that identifies this page
        # anchors: {{or: ["Search", "搜索"], region: top}}
        # anchors: {{and: ["EDIT ME", {{text: "tab", region: bottom}}]}}
        # forbid: ["popup text"]  # veto terms — kills look-alike takeovers
        # scrollable: true        # content may scroll under fixed chrome
        recover:                # not this page → force_quit, then the
          tool: force_quit      # walk (and `start`) re-runs from the top
      - do: {macro}             # the recorded gesture (macros/{macro}/)
        with: {{message: "{{inputs.message}}"}}
      - page: home              # the landing check — hand over if not reached
      # An acting agent episode — the judgment stretch, delegated whole:
      # - agent: choose
      #   prompt: |
      #     EDIT ME — the goal, the rules, and where to finish.
      #   tools: [{agent_tools}]
      #   give: [landmarks.back]
      #   returns:
      #     summary: EDIT ME — what to report back
      #   limit: {{calls: {agent_calls}, scrolls: {agent_scrolls}}}
      # - page: home
      - tell: done
        # The EXACT text sent to the user — write it in THEIR language.
        message: "EDIT ME — done with {{inputs.message}}, reply to continue"
"""


README_CONTENT = """\
# App packs

One directory per app, self-contained — everything its playbooks use:

    playbooks/<app>/
      PLAYBOOK.yml         the whole pack in one file: meta (app,
                           description, placeholders), `playbooks:`
                           (each walk a ROUTE: page → move → page →
                           move, top-down), and an optional `pages:`
                           appendix for pages off any route. Anchors
                           are semantics; geometry is captured
                           on-device into learned/pages/.
      macros/<n>/MACRO.yml pack-private macros shared across playbooks
                           (same format as ~/.physiclaw/macros/, never
                           shown to the model's macro list); single-use
                           hands embed in their moves instead.

Validate everything: `physiclaw playbooks check`. Scaffold a pack:
`physiclaw playbooks init <app>` — or start from a shared template
(`physiclaw playbooks install <dir>` records its `<<PLACEHOLDER>>`
values in `playbooks/placeholders.yml`; the repo's `playbooks/`
directory ships some, and when physiclaw runs from a source checkout
those load directly — home packs shadow same-named tree packs).

The route's shape IS the contract the checker enforces: an optional
prefix of pure-text `agent` steps and one `start` (the unconditional
cold-launch) opens the route, the first page is the start contract,
every `do` — and every acting `agent` episode — is followed by the
page it lands on, `{{inputs.name}}` refs name declared inputs and
`{{move.field}}` refs name EARLIER agent outputs, moves run forward
only, and `irreversible: payment` moves (do and agent alike) directly
follow an `ask` with `approve: payment` — a `budget:` is optional;
without one the consented total is the fire-time bound.

An `agent` step is the model's, inside the author's fence: `prompt:`
is the whole brief (refs fill once when the step opens), `tools:`
the closed gesture allowlist, `give:` the landmarks it may name
blind, `returns:` the fields it must fill, `limit:` its call/scroll
budget; each episode turn the model answers with a screen row, a
granted landmark, a scroll verb, done, or escalate — never
coordinates — and `done` counts only on the following page, judged
by the matcher.

A pack may declare `landmarks:` — named fixed spots ({{label, bbox}},
open vocabulary) that recover hands tap and agent episodes are
granted. A page's `recover:` declares its recovery hand (one gesture
or an argument-less macro) — what you declare is what runs, after the
built-in unlock and one settle re-peek; a page declaring none hands
over.

A page is declared ONCE — beside its waypoint, in the `pages:`
appendix, or on another route — and referenced bare everywhere else.
Macros embed as `macro: {{steps: [...]}}` (the MACRO.yml grammar minus
name/description/enabled, enabled with the playbook); an ask's
`resume:` and a page's `recover:` take the same form and dispatch
argument-less.

Limits: ≤ {max_nodes} moves, ≤ {max_inputs} inputs per playbook.
Macro format: see ~/.physiclaw/macros/README.md (identical grammar).
""".format(max_nodes=MAX_NODES, max_inputs=MAX_INPUTS)


# ---------- the channel pack (conductor infrastructure) ----------

CHANNEL_PACK_STUB = f"""\
app: {CHANNEL_APP}
description: >-
  The user-channel pack — the conductor's route to YOUR user's IM
  thread. Holds the thread page + the send/open macros its asks run;
  no playbooks live here, and packs never name it.

# The ONE page the conductor must recognize: your own chat thread in
# your IM app. Anchor on the chat header (your name / the contact
# name) + stable chrome.
pages:
  {THREAD_PAGE}:
    anchors:
      - "EDIT ME"                  # the thread header text
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
    with: {{label: "the IM app's dock icon", bbox: [0.1, 0.9, 0.2, 0.98]}}    # EDIT ME
  - name: open-thread
    tool: tap
    with: {{label: "your user's chat-row name", bbox: [0.1, 0.15, 0.9, 0.22]}}   # EDIT ME
  - name: raise-keyboard
    tool: tap
    with: {{label: "the input box (hidden)", bbox: [0.1, 0.9, 0.7, 0.96]}}   # EDIT ME
  - name: clip
    tool: send_to_clipboard
    with: {{text: "{{message}}"}}
  - name: hold-input
    tool: long_press
    with: {{label: "the input box (visible)", bbox: [0.1, 0.5, 0.7, 0.56]}}  # EDIT ME
  - name: paste
    tool: tap
    with: {{label: "Paste", bbox: [0.1, 0.44, 0.3, 0.5]}}    # EDIT ME: the Paste button
  - name: send
    tool: tap
    with: {{label: "Send", bbox: [0.8, 0.5, 0.98, 0.56]}}    # EDIT ME: the Send key
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
    with: {{label: "the IM app's dock icon", bbox: [0.1, 0.9, 0.2, 0.98]}}    # EDIT ME
  - name: open-thread
    tool: tap
    with: {{label: "your user's chat-row name", bbox: [0.1, 0.15, 0.9, 0.22]}}   # EDIT ME
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
    write_text(root / PACK_FILENAME, CHANNEL_PACK_STUB)
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
    PlaybookError on a bad name or an existing directory; `ios` is the
    exception, being idempotent (see below)."""
    from physiclaw.common import paths
    from physiclaw.common.text import write_text

    check_name(app, "app name")
    root = paths.playbooks_dir() / app
    if app == IOS_APP:
        # Idempotent on purpose: `session_setup` materializes this pack on
        # the first qualifying wake, so by the time anyone runs
        # `playbooks init ios` it usually exists — and erroring there would
        # withhold the next-steps text, which is the command's whole value.
        ensure_ios_pack()
        ensure_format_readme()
        return root
    if root.exists():
        raise PlaybookError(f"pack directory already exists: {root}")
    if app == CHANNEL_APP:
        return _init_channel_pack(root)
    macro_dir = root / PACK_MACROS_DIRNAME / EXAMPLE_MACRO
    macro_dir.mkdir(parents=True)
    write_text(root / PACK_FILENAME, render_pack_stub(app))
    write_text(macro_dir / MACRO_FILENAME, render_example_macro())
    ensure_format_readme()
    return root


# ---------- the ios pack (OS states the conductor must name) ----------

IOS_PACK_STUB = f"""\
app: {IOS_APP}
description: >-
  iOS system states the conductor must name — no playbooks, no macros.
  Geometry is captured on YOUR device (conductor calibrate ios --guided).

pages:
  # The lock screen. Telling "locked" apart from "a screen I don't
  # recognize" matters because they demand opposite actions —
  # `unlock_phone` costs 20-40s and does nothing on an unlocked phone,
  # while a recovery macro's taps land uselessly on a lock screen. Every
  # other unrecognized screen collapses into one arm, which is why no
  # other system page is declared yet.
  #
  # THIS DECLARATION IS A BONUS, NOT THE MECHANISM. Measured across
  # every state a real iPhone's cover can be put in — resting Always-On
  # Display, woken and fully lit, and after a swipe — iOS printed no
  # hint text at all, so the anchor below has nothing to match and the
  # page scores 0. The conductor therefore recognizes the cover by its
  # SHAPE instead (a clock and nothing else — `match.reads_as_cover`),
  # which needs no declaration and no calibration. Keep this page: on a
  # device or version that DOES print a hint it is the sharper signal,
  # and it costs nothing when it never matches.
  {LOCKED_PAGE}:
    anchors:
      # ONE anchor, several acceptable readings — never separate
      # anchors: each declared anchor is a share of the page's score,
      # so a second spelling of the same label would halve it.
      #
      # Verify this reading against YOUR phone before trusting it —
      # lock it and run `physiclaw conductor propose --live` to see
      # what it actually prints, then put that beside this one:
      #
      #     - text: ["Swipe up for Face ID or Enter Passcode", "<yours>"]
      - text: ["Swipe up for Face ID or Enter Passcode"]
        region: bottom
"""


def ensure_ios_pack() -> None:
    """Materialize `playbooks/ios/` if it does not exist — the
    `ensure_format_readme` pattern, called the first time the conductor
    goes looking for OS pages.

    It is a template the user OWNS and edits (the lock-screen reading
    depends on their phone's language), so it must exist on disk rather
    than be read out of the wheel — but it must not depend on them having
    run `playbooks init ios` either, because the conductor needs those
    pages on the very first wake. Written once, then never touched:
    a user file, including a deliberately emptied one, is theirs.

    Fail-open: a read-only home degrades to "no ios pages", which the
    conductor already handles (every OS state reads as unknown)."""
    from physiclaw.common import paths
    from physiclaw.common.text import write_text

    root = paths.playbooks_dir() / IOS_APP
    # Keyed on the FILE, not the dir: an old-layout ios pack (pages.yml)
    # self-migrates by gaining the unified spec, while an existing
    # PLAYBOOK.yml — including a deliberately emptied one — is theirs.
    if (root / PACK_FILENAME).exists():
        return
    try:
        root.mkdir(parents=True, exist_ok=True)
        write_text(root / PACK_FILENAME, IOS_PACK_STUB)
    except OSError:
        log.warning("could not scaffold %s — OS pages unavailable", root, exc_info=True)
