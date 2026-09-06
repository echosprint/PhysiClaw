"""Authoring-surface texts for app packs: the `init` scaffold + README.

The scaffold IS the format reference a new user edits in place
(the macro-scaffold doctrine), and the README is kept current at
``playbooks/README.md`` via ``ensure_readme``. Caps and vocabularies
interpolate from `playbook`/`calls` constants, never hand-copied."""

import logging
from pathlib import Path

from physiclaw.common.bbox import BANDS
from physiclaw.common.paths import (
    PACK_FILENAME,
    PACK_MACROS_DIRNAME,
    PLAYBOOK_FILENAME,
)
from physiclaw.conductor.spec import specfile
from physiclaw.conductor.spec.calls import AGENT_TOOLS
from physiclaw.conductor.spec.conventions import (
    BOOT_PLAYBOOK,
    CHANNEL_APP,
    IOS_APP,
    LOCKED_PAGE,
    OPEN_MACRO,
    SEND_MACRO,
    THREAD_PAGE,
)
from physiclaw.conductor.spec.limits import (
    DEFAULT_AGENT_CALLS,
    DEFAULT_AGENT_SCROLLS,
    DEFAULT_ASK_ROUNDS,
    DEFAULT_ASK_WAIT_SECONDS,
    DEFAULT_RECOVER_LIMIT,
    MAX_AGENT_CALLS,
    MAX_NODES,
)
from physiclaw.conductor.spec.model import (
    IRREVERSIBLE_CLASSES,
    PlaybookError,
    check_name,
)
from physiclaw.conductor.spec.route import (
    BARE_HANDS,
)
from physiclaw.macros import scaffold as macro_scaffold
from physiclaw.macros import store as macro_store
from physiclaw.macros.model import MAX_INPUTS

log = logging.getLogger(__name__)


def read_template_manifest(src: Path) -> dict:
    """A template pack's PLAYBOOK.yml, loaded RAW (its `<<TOKEN>>`s still
    in place) for `playbooks install` — name/description plus the
    `placeholders:` spec map that drives the prompts. {} when absent;
    loud PlaybookError on bad YAML or a malformed `placeholders:`
    section, so install fails BEFORE it mutates anything."""
    from physiclaw.common.text import read_text

    path = src / PACK_FILENAME
    if not path.exists():
        return {}
    try:
        data = specfile.yaml_loader.load(read_text(path))
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


def render_manifest_stub(app: str) -> str:
    return MANIFEST_TEMPLATE.format(app=app)


def render_playbook_stub(app: str) -> str:
    """The scaffold's example playbook file (`<EXAMPLE_PLAYBOOK>.yml`)."""
    return PLAYBOOK_TEMPLATE.format(
        app=app,
        playbook=EXAMPLE_PLAYBOOK,
        macro=EXAMPLE_MACRO,
        max_inputs=MAX_INPUTS,
        max_nodes=MAX_NODES,
        classes="|".join(IRREVERSIBLE_CLASSES),
        agent_tools=", ".join(AGENT_TOOLS),
        max_agent_calls=MAX_AGENT_CALLS,
        agent_calls=DEFAULT_AGENT_CALLS,
        agent_scrolls=DEFAULT_AGENT_SCROLLS,
        recover_tools=" / ".join(BARE_HANDS),
        recover_limit=DEFAULT_RECOVER_LIMIT,
        bands=" / ".join(BANDS),
        ask_wait=DEFAULT_ASK_WAIT_SECONDS,
        ask_rounds=DEFAULT_ASK_ROUNDS,
    )


MANIFEST_TEMPLATE = """\
# The pack MANIFEST — what the app is and what its playbooks share.
# Never a route: each playbook is its own folder beside this file
# (<name>/PLAYBOOK.yml, referenced as {app}/<name>), and the recorded
# hands routes share live in macros/<name>.yml. Every section here is
# optional; an empty manifest is a valid pack.
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
# Open vocabulary; each is {{label, at, [page]}}, label-healed live
# when the label is readable on screen; `page:` scopes it — an episode
# is offered it only while that page is the verified reading.
# landmarks:
#   back:
#     label: "back chevron (top-left)"
#     at: [0.015, 0.045, 0.095, 0.095]
#     page: detail

# Pages more than one playbook lands on are declared here once and
# referenced bare from every route; a page only one route uses may be
# declared beside its waypoint instead. Anchors are semantics; geometry
# is captured on YOUR device via `physiclaw playbooks pages calibrate`.
# A `recover:` here is the hand every route inherits for the page (a
# gesture, a tap on a landmark, or a pack macro BY NAME — the manifest
# carries no bodies); a route may declare its own to override it.
# pages:
#   home:
#     anchors:
#       - {{text: ["Search", "搜索"], within: top}}
#     recover: force_quit
"""


PLAYBOOK_TEMPLATE = """\
# One playbook — this folder is its name ({app}/{playbook}); the pack's
# APP.yml holds what every playbook shares, and this folder holds what
# is this route's alone: macros/<name>.yml it recorded, prompts/<name>.md
# its agent steps read (`prompt: prompts.<name>`), README.md for people.
# A route alternates
# WHERE (a page, checked every time) with WHAT (a move), forward-only,
# ≤ {max_nodes} moves. What you declare is what runs: nothing retries,
# unlocks, or waits unless a line here says so. The entries:
#   page   — where the walk must BE; `recover:` is the hand that runs when
#            it is not ({recover_tools}, `{{tap: landmarks.<name>}}`,
#            `{{macro: <name>}}`, or one per reading: covered / elsewhere /
#            locked), `tries:` beside it (default {recover_limit}).
#   start  — the cold launch, once, right before the first page.
#   do     — a recorded macro (`macro:` a pack macro or an inline body,
#            `with:` its inputs); the page after it is the landing check.
#            Money: `irreversible: {classes}` right after an `approve:
#            payment` ask.
#   agent  — the model drives inside YOUR prompt (inline, or
#            `prompt: prompts.<name>` for prompts/<name>.md) with the
#            `tools:` and `give:` grants you list; `returns:` fields read downstream as
#            {{name.field}}. No tools = a pure-text call, legal before the
#            first page. `limit: {{calls, scrolls}}` bounds an episode
#            (≤ {max_agent_calls} calls).
#   ask    — message the user and wait for `yes:` / `no:` (whole-message);
#            `wait:` × `rounds:` is its patience (default {ask_wait}s ×
#            {ask_rounds}); `approve: payment` reads the amount beside
#            `total_label:` into {{ask.total}}; `resume:` re-enters the app.
#   tell   — message the user and move on (a trailing tell ends the walk).
# Values: the manifest's placeholders fill at install, {{inputs.x}} /
# {{node.field}} / {{ask.total}} when the walk reaches the move, {{x}}
# inside a macro from its `with:`. Reference: ~/.physiclaw/playbooks/README.md.
name: {playbook}         # = this folder's name; referenced as {app}/{playbook}
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
        - home_screen
        - tap: "the app icon"
          at: [0.1, 0.1, 0.3, 0.2]
        - wait: 3
  - page: home
    # `anchors:` is a list; all should show. Alternate readings of ONE
    # anchor go inside it as a list — never as separate anchors, since
    # each declared anchor is a share of the page's score. `within:`
    # pins an anchor to a band ({bands}) or a box.
    anchors:
      - "EDIT ME"           # a label text that identifies this page
      # - {{text: ["Search", "搜索"], within: top}}
    # forbid: ["popup text"]  # veto terms — kills look-alike takeovers
    # scrollable: true        # content may scroll under fixed chrome
    recover: force_quit     # not this page → force_quit, then the walk
                            # (and `start`) re-runs from the top
    # recover:              # or one hand per reading:
    #   covered: {{tap: landmarks.dismiss}}
    #   elsewhere: go_back
    #   locked: unlock_phone
    # tries: 2              # this page's tries per walk (default {recover_limit})
  - do: {macro}             # the recorded gesture
    macro: {macro}          # the pack macro (macros/{macro}/)
    with: {{message: "{{inputs.message}}"}}
  - page: home              # the landing check — hand over if not reached
  # An acting agent episode — the judgment stretch, delegated whole:
  # - agent: choose
  #   prompt: |
  #     EDIT ME — the goal, the rules, and where to finish.
  #   tools: [{agent_tools}]
  #   give: [landmarks.back, macros.{macro}]
  #   context: [memory.shopping, daylog]
  #   returns:
  #     summary: EDIT ME — what to report back
  #   limit: {{calls: {agent_calls}, scrolls: {agent_scrolls}}}
  # - page: home
  # A human gate before money moves — the payment move follows it:
  # - ask: confirm-pay
  #   approve: payment
  #   total_label: "合计"     # the label the sheet total sits beside
  #   message: "EDIT ME — total ¥{{ask.total}}, reply ok to pay or no to cancel"
  #   yes: ["ok"]
  #   no: ["no"]
  #   wait: {ask_wait}              # seconds between reply polls
  #   rounds: {ask_rounds}          # silent polls before the session suspends
  #   resume: {macro}         # re-enter the app after the reply
  - tell: done
    # The EXACT text sent to the user — write it in THEIR language.
    message: "EDIT ME — done with {{inputs.message}}"
"""


README_CONTENT = """\
# App packs

One directory per app, self-contained — everything its playbooks use:

    playbooks/<app>/
      APP.yml              the MANIFEST: what the app is and what its
                           playbooks share — app, description,
                           placeholders, landmarks, and a `pages:`
                           appendix for pages more than one route
                           lands on. Every section optional; an empty
                           file is a valid pack. Never a route.
      README.md            for people: what the pack does, the device
                           it was recorded on, the traps. Never loaded.
      macros/<n>.yml       the pack's hands, shared by every route
                           (same format as ~/.physiclaw/macros/, never
                           shown to the model's macro list).
      <name>/              one playbook per folder; the folder is the
        PLAYBOOK.yml       name (referenced as <app>/<name>, and `name:`
                           inside must agree) and the body is the
                           playbook: name, description, enabled,
                           inputs, route (page → move → page → move).
        macros/<n>.yml     this route's own recorded hands (dispatch
                           <app>/<name>.<n>); a name may not also be a
                           pack macro's — there is no lookup order.
        prompts/<n>.md     this route's prose for the model, verbatim
                           (`prompt: prompts.<n>` in an agent step).
        README.md          the route's notes for people. Never loaded.

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
follow an `ask` with `approve: payment` — the total the user consented
to is the fire-time bound.

What the playbook declares is what runs — no more, no less. An
`agent` step is the model's, inside the author's fence: `prompt:` is
the whole brief (refs fill once when the step opens; the conductor
adds only the output contract), `tools:` the closed gesture allowlist,
`give:` the landmarks it may name blind (`landmarks.<name>`) and the
pack macros it may run whole (`macros.<name>`), `context:` what to
load beside the prompt (`memory`, `memory.<slug>`, `daylog` — nothing
else travels), `returns:` the fields it must fill, `limit:` its
call/scroll budget; each episode turn the model answers with a screen
row, a granted landmark or macro, a scroll verb, done, or escalate —
never coordinates — and `done` counts only on the following page,
judged by the matcher. An `ask` reads the reply against its own
`yes:`/`no:` words (any other reply hands over), waits by its own
`wait:` seconds for `rounds:` silent polls, and a payment ask reads
the amount beside the label its `total_label:` names.

A pack may declare `landmarks:` — named fixed spots ({{label, at,
[page]}}, open vocabulary) that recover hands tap and agent episodes
are granted; a `page:` scope offers the spot only on that page. A
page's `recover:` declares its recovery hand — a bare gesture
(`go_back`, `force_quit`, `home_screen`, `unlock_phone`), `{{tap:
landmarks.<name>}}`, or `{{macro: <name>}}` — or one hand per reading
(`covered:` for a sheet over the page itself, `locked:` for the
phone's lock screen, `elsewhere:` for any other screen), with `tries:`
beside it as its own bound — nothing recovers in the background; a page
declaring none hands over. A page's `anchors:` is a list of texts the page shows;
alternate readings of one text go inside it (`text: [..]`), and
`within:` pins it to a band or a box.

The channel pack (`playbooks/channel/`) is the conductor's own: the
thread page, the send/open macros an `ask` runs, and `boot/` — the
walk every wake plays before any playbook (reach the thread, read the
request, hand the matching playbook the baton), a route like any
other whose `select` step is the one entry only that file may carry.

Replay a walk offline against a recorded session's screens —
`physiclaw playbooks replay <app>/<name> --session <id>` — to see
where it would hand over before touching the phone.

A page is declared ONCE per pack — beside its waypoint in one file,
or in the manifest's `pages:` appendix when routes share it — and
referenced bare everywhere else; the same page declared in two files
is a pack error, never a merge.
Macros embed as `macro: {{steps: [...]}}` (the macro-file grammar minus
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
  thread: the thread page, the send/open macros its asks run, and the
  boot ({BOOT_PLAYBOOK}/PLAYBOOK.yml) every wake walks first.

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
  - home_screen
  - tap: "the IM app's dock icon"
    at: [0.1, 0.9, 0.2, 0.98]        # EDIT ME
  - tap: "your user's chat-row name"
    at: [0.1, 0.15, 0.9, 0.22]       # EDIT ME
  - tap: "the input box (hidden)"
    at: [0.1, 0.9, 0.7, 0.96]        # EDIT ME
  - send_to_clipboard: "{{message}}"
  - long_press: "the input box (visible)"
    at: [0.1, 0.5, 0.7, 0.56]        # EDIT ME
  - tap: "Paste"
    at: [0.1, 0.44, 0.3, 0.5]        # EDIT ME: the Paste button
  - tap: "Send"
    at: [0.8, 0.5, 0.98, 0.56]       # EDIT ME: the Send key
"""

CHANNEL_OPEN_STUB = f"""\
name: {OPEN_MACRO}
description: open the user's IM thread (read only, no send)
enabled: false
steps:
  - home_screen
  - tap: "the IM app's dock icon"
    at: [0.1, 0.9, 0.2, 0.98]        # EDIT ME
  - tap: "your user's chat-row name"
    at: [0.1, 0.15, 0.9, 0.22]       # EDIT ME
"""


# The boot: what every wake does before any playbook, declared. A route
# like any other — the thread page with its hands, then the one step
# only this file may carry.
CHANNEL_BOOT_STUB = f"""\
# {CHANNEL_APP}/{BOOT_PLAYBOOK} — the walk every wake plays before any playbook:
# reach YOUR user's thread, read the request there, hand the matching
# playbook the baton. A route like any other: edit the hands and the
# limits, step it (physiclaw playbooks step {CHANNEL_APP}/{BOOT_PLAYBOOK}),
# replay it over a recorded wake. Live once `{OPEN_MACRO}` is enabled.
name: {BOOT_PLAYBOOK}
description: reach the user's thread and read the request there
enabled: true
route:
  - page: {THREAD_PAGE}                  # where the walk must BE (the manifest declares it)
    recover:                       # …and what to do when it is not
      locked: unlock_phone           # a sleeping phone gets no taps: wake it first
      covered: {{macro: {OPEN_MACRO}}}          # the thread under a sheet or keyboard
      elsewhere: {{macro: {OPEN_MACRO}}}        # any other screen: the rehearsed hand
    tries: 4                         # unlocks + opens together (an unlock races the keypad)
  - select: parse                  # the boot's own step: read the thread and select
    limit: {{scrolls: 2}}            # the playbook it asks for (scrolls up for an
                                   # older request)
"""


def _init_channel_pack(root: Path) -> Path:
    """`playbooks/channel/` — the infrastructure pack: the thread page,
    the send/open macros the conductor's asks run, and the boot
    playbook. App packs never name it."""
    from physiclaw.common.text import write_text

    macros_root = root / PACK_MACROS_DIRNAME
    macros_root.mkdir(parents=True)
    for name, stub in (
        (SEND_MACRO, CHANNEL_SEND_STUB),
        (OPEN_MACRO, CHANNEL_OPEN_STUB),
    ):
        write_text(macro_store.macro_path(macros_root, name), stub)
    write_text(root / PACK_FILENAME, CHANNEL_PACK_STUB)
    write_text(root / "README.md", render_pack_readme(CHANNEL_APP))
    (root / BOOT_PLAYBOOK).mkdir()
    write_text(root / BOOT_PLAYBOOK / PLAYBOOK_FILENAME, CHANNEL_BOOT_STUB)
    ensure_format_readme()
    return root


def ensure_channel_boot(root: Path) -> None:
    """Materialize `boot/PLAYBOOK.yml` in the channel pack at `root` when
    it has none — the `ensure_ios_pack` pattern, called as the pack loads: the
    boot is a template the user owns and edits (its hands, its limits),
    so it lives on disk, but a channel pack recorded before the boot
    was a file must not lose its wake. Written once, then never
    touched. Fail-open."""
    _ensure_file(
        root / BOOT_PLAYBOOK / PLAYBOOK_FILENAME, CHANNEL_BOOT_STUB, "no boot at wake"
    )


def render_pack_readme(app: str) -> str:
    """The pack's `README.md` seed — the author's notes, never loaded.
    Written once by `init`; the loader reads no `.md` outside `prompts/`."""
    return PACK_README_TEMPLATE.format(app=app)


def render_playbook_readme(app: str, playbook: str) -> str:
    """A playbook folder's `README.md` seed, same rule."""
    return PLAYBOOK_README_TEMPLATE.format(app=app, playbook=playbook)


PACK_README_TEMPLATE = """\
# {app}

What this pack automates, in one paragraph, and when the boot should
offer it.

## Device

The phone, OS version, app version, and system language the hands were
recorded on. Coordinates replay as-is on the same model; note what
differs elsewhere.

## Traps

What broke a run and how the pack now avoids it — one line each.
"""

PLAYBOOK_README_TEMPLATE = """\
# {app}/{playbook}

What this route does, end to end, and what the user says to start it.

## Recorded facts

Where each hand's coordinates came from, and what the screens looked
like when they were recorded.

## Rehearsal

    physiclaw playbooks replay {app}/{playbook} --session <id>
    physiclaw playbooks run {app}/{playbook} -i <input>=<value>
"""


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
    """Scaffold ``playbooks/<app>/`` — the manifest, its README, a
    disabled example playbook folder (route, README), an example pack
    macro — and return the pack root. Raises
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
    (root / PACK_MACROS_DIRNAME).mkdir(parents=True)
    (root / EXAMPLE_PLAYBOOK).mkdir()
    write_text(root / PACK_FILENAME, render_manifest_stub(app))
    write_text(root / "README.md", render_pack_readme(app))
    write_text(root / EXAMPLE_PLAYBOOK / PLAYBOOK_FILENAME, render_playbook_stub(app))
    write_text(
        root / EXAMPLE_PLAYBOOK / "README.md",
        render_playbook_readme(app, EXAMPLE_PLAYBOOK),
    )
    write_text(
        macro_store.macro_path(root / PACK_MACROS_DIRNAME, EXAMPLE_MACRO),
        render_example_macro(),
    )
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
  # page scores 0. The matcher therefore recognizes the cover by its
  # SHAPE first (a clock and nothing else — `match.reads_as_locked`),
  # which needs no declaration and no calibration, and every page's
  # `locked:` recover hand fires on it. Keep this page: on a device or
  # version that DOES print a hint it is the sharper signal, and it
  # costs nothing when it never matches.
  {LOCKED_PAGE}:
    anchors:
      # ONE anchor, several acceptable readings — never separate
      # anchors: each declared anchor is a share of the page's score,
      # so a second spelling of the same label would halve it.
      #
      # Verify this reading against YOUR phone before trusting it —
      # lock it and run `physiclaw playbooks pages propose --live` to see
      # what it actually prints, then put that beside this one:
      #
      #     - text: ["Swipe up for Face ID or Enter Passcode", "<yours>"]
      - text: ["Swipe up for Face ID or Enter Passcode"]
        within: bottom
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

    root = paths.playbooks_dir() / IOS_APP
    # Keyed on the FILE, not the dir: an existing APP.yml — including a
    # deliberately emptied one — is theirs.
    _ensure_file(root / PACK_FILENAME, IOS_PACK_STUB, "OS pages unavailable")


def _ensure_file(path: Path, stub: str, cost: str) -> None:
    """Write `stub` at `path` unless a file is there — a user-owned
    template materialized once, then never touched. Fail-open: an
    unwritable home logs `cost` and the conductor runs without it."""
    from physiclaw.common.text import write_text

    if path.exists():
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_text(path, stub)
    except OSError:
        log.warning("could not write %s — %s", path, cost, exc_info=True)
