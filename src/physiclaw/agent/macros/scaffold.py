"""The authoring-surface texts: the `init` scaffold and the README.

Both are documentation-as-artifact — the scaffold IS the format reference
a new user edits in place, and the README is the one kept current at
``~/.physiclaw/macros/README.md`` via ``ensure_readme`` (rewritten
whenever these constants change, so installs never document a retired
format). Tool lists and caps interpolate from `model`'s constants, never
hand-copied, so a whitelist or limit change updates every surface in the
same edit.
"""

from physiclaw.agent.macros.model import (
    ALLOWED_STEP_TOOLS,
    MACRO_FILENAME,
    MAX_INPUTS,
    MAX_PROSE_LEN,
    MAX_RUN_SECONDS,
    MAX_STEPS,
    MAX_WAIT_SECONDS,
)


def render_init(name: str) -> str:
    """The scaffold body for ``macros/<name>/MACRO.yml`` — parses clean
    (disabled) so `check` passes before any editing."""
    return INIT_TEMPLATE.format(name=name, tools="  ".join(sorted(ALLOWED_STEP_TOOLS)))


# Scaffold written by `physiclaw macros init <name>` — a real, rehearsable
# WeChat-message flow (coordinates from a reference rig), not lorem ipsum:
# the template IS the format documentation AND the best-practice example.
# It must parse clean so a fresh scaffold passes `macros check` before any
# editing (`tests/agent/macros/test_scaffold.py` pins that, including that
# it deliberately stops before Send).
INIT_TEMPLATE = """\
name: {name}                   # must equal the directory name
description: Open WeChat, paste a message into your user's chat, stop before Send

# A valid macro is enabled by default; this scaffold starts off. Set it to
# true (or delete the line) after `physiclaw macros run {name}` succeeds.
enabled: false

# Inputs the agent fills at call time; "{{message}}" in `with:` strings below.
# An input with `default` is optional; without one it is required.
inputs:
  message:
    description: The message text to paste into the chat
    example: "Task done, total 30"

# A real flow from a reference rig — REHEARSE BEFORE ENABLING:
#     physiclaw macros run {name} -i message=test
# and fix each bbox until every step lands on YOUR rig, phone, and layout.
# Allowed tools:
#   {tools}
# QUOTE text values: YAML turns unquoted true/false into booleans and bare
# numbers into numbers — `macros check` rejects those with a pointer, but
# quoting avoids the round-trip.
steps:
  - name: start-home           # unique, no spaces: `start_at` uses it
    tool: home_screen

  - name: open-wechat
    tool: tap
    with:
      bbox: [0.317, 0.893, 0.488, 0.98]   # ← the dock icon; move to YOURS
                               # No guard here. A guard runs BEFORE its own
                               # step, so "WeChat is open" is a claim about
                               # the NEXT step, never about this one.

  - name: await-wechat         # settle, then confirm — one step, one camera
    tool: wait                 # read. `seconds` is exactly that many
    with:                      # seconds; start small, since the arm's own
      seconds: 2               # gesture latency already covers most cold
                               # starts. Raise it only if rehearsal shows
                               # the check landing early.
    expect:                    # WeChat reopens WHERE IT WAS LEFT, so the
                               # title bar reads either the app name (chat
                               # list) or your user's name (straight back
                               # inside the chat). Accept both — the next
                               # step sorts out which one we got.
      {{or: ["WeChat", "Weixin", "your-user-name"],       # ← EDIT the name
        within: [0.15, 0.02, 0.85, 0.1]}}
    hint: "WeChat did not open — check the dock position, or log in first"

  - name: open-chat
    tool: tap
    with:
      bbox: [0.066, 0.086, 0.249, 0.186]   # ← the top chat row; move to YOURS
    skip_when:                 # WeChat often reopens INSIDE the last chat —
                               # when the title bar already shows your user,
                               # tapping the list position would hit messages
      {{text: "your-user-name", within: [0.15, 0.02, 0.85, 0.1]}}   # ← EDIT
    guard:                     # about to tap a FIXED row position, so check
                               # the name is really sitting there: a message
                               # in another chat reorders the list and this
                               # bbox would then open the wrong person
      require:
        {{text: "your-user-name", within: [0.05, 0.08, 0.95, 0.19]}}  # ← EDIT
      # forbid: "Account protection"   # tripwire: abort if that text is up
      hint: "that chat is not the top row — open it by hand"

  - name: stage-text           # names this step in logs and abort reports
    tool: send_to_clipboard
    with:
      text: "{{message}}"

  - name: focus-input-box      # the keyboard slides up after this tap
    tool: tap
    with:
      bbox: [0.1, 0.915, 0.69, 0.955]
    skip_when:                 # idempotence: this step's POSTCONDITION is
                               # "keyboard up" — if it already is, the box
                               # has moved up and this bbox hits the keys.
                               # Anchor on the space bar, which exists only
                               # while the keyboard is up; one `within`
                               # scopes both spellings, so the word cannot
                               # match inside a chat bubble. Either way the
                               # NEXT step runs on a keyboard-up screen.
      {{or: ["space", "空格"], within: [0.25, 0.85, 0.80, 0.95]}}

  - name: paste-menu           # box sits higher with keyboard up
    tool: long_press
    with:
      bbox: [0.118, 0.578, 0.893, 0.637]
    guard:                     # last check before the paste chain: still in
      require:                 # the right chat, whichever way we got here
        {{text: "your-user-name", within: [0.15, 0.02, 0.85, 0.1]}}   # ← EDIT
      hint: "not in the right chat — paste by hand"

  - name: paste
    tool: tap
    with:
      bbox: [0.099, 0.544, 0.197, 0.564]
    guard:
      require:                 # region-scoped: THE paste bubble, not the
                               # word Paste appearing anywhere else on screen
        {{text: "Paste", within: [0.03, 0.5, 0.35, 0.62]}}
      hint: "Paste menu did not appear — long-press the input box again"

# Deliberately NO send step. Tapping Send commits the message, and commit
# actions stay with the agent: it reads the pasted text on the returned
# screen, verifies it, and taps Send itself. End your macros just before
# the irreversible tap.
"""

# Format reference kept at ~/.physiclaw/macros/README.md via ensure_readme —
# rewritten whenever this constant changes, so installs never document a
# retired format.
README_CONTENT = f"""\
# Macros

One directory per macro: `~/.physiclaw/macros/<name>/{MACRO_FILENAME}`.
Scaffold one with `physiclaw macros init <name>`, edit it, then:

    physiclaw macros list           # every macro: enabled / disabled / invalid
    physiclaw macros check          # lint every {MACRO_FILENAME}
    physiclaw macros run <name> -i key=value   # rehearse on the live rig
    # then flip the scaffold's `enabled: false` to true (or delete the line)
    physiclaw macros stats          # success/abort counters per macro
    physiclaw macros runs [<hex6>]  # per-step logs; every run has an id
                                    # `macro-run-<hex6>` (printed after
                                    # rehearsals, stamped in results/stats)
                                    # and its own log/macros/<id>/ dir with
                                    # events.jsonl + per-step screenshots

## Format (GitHub-Actions-shaped YAML)

- `name` (required) — must equal the directory name; lowercase/digits/hyphens.
- `description` (required) — tells the agent when to use it. It, and each
  input's `description` / `example`, must be ONE line of at most
  {MAX_PROSE_LEN} characters: they render verbatim into the cached system
  prompt on every wake.
- `enabled` — absent defaults to true; set `enabled: false` to hide a macro
  from the agent (the `init` scaffold starts false until you rehearse).
- `inputs.<name>` — `description` required; `default` makes it optional;
  `example` helps the agent fill it. String values only; max {MAX_INPUTS}.
- `steps` — a list; each step: `tool` (one of:
  {", ".join(sorted(ALLOWED_STEP_TOOLS))}); `with:` mapping of arguments,
  `{{input}}` placeholders in strings (`{{{{`/`}}}}` for a literal brace);
  and `name` (required, unique per macro, lowercase/digits/hyphens with no
  spaces — the identifier `start_at` uses and the label in logs).
  Max {MAX_STEPS} steps.
- `steps[].guard` — checked BEFORE the step fires, so it describes the
  screen the step NEEDS, never the screen the step produces. To wait on an
  app you just launched, use a `wait` step carrying an `expect`.
  `require` must hold; `forbid` must not.
  EVERY check — `require`, `forbid`, `expect`, `skip_when` — is exactly ONE
  clause, so there is no per-field shape to remember. A clause is one of
  four forms, with the combinators spelled out rather than implied by
  nesting:
      "WeChat"                              substring of the whole screen
      {{text: "WeChat", within: [l,t,r,b]}}   element-granular, region-scoped
      {{or: [c, c]}}   {{and: [c, c]}}         any-of / all-of
      {{not: c}}                             absent
  They nest, so `{{or: [{{not: "x"}}, {{and: ["y", "zz"]}}]}}` is legal. Any of
  them may carry `within`, which scopes the whole subtree — write
  `{{or: ["space", "空格"], within: [...]}}` rather than repeating the bbox
  on each alternative; an inner `within` overrides the one it sits in.
  For two conditions write `{{and: [...]}}`: these fields took LISTS once,
  which made bracket shape carry meaning (a list was AND at the top level
  and an error one level down) — the exact trap the explicit combinators
  exist to kill. `forbid: X` is still exactly `require: {{not: X}}`, kept
  because the popup-tripwire reading earns a name.
  A guard checks ONCE: it is a predicate, not a wait. It reads the screen
  text already held, free, and peeks when none is held (step 1, or after a
  step that returned no view; `send_to_clipboard` keeps the held screen).
  An unreadable screen is never a satisfied guard: a failed peek is retried
  once, then aborts with `could not read the screen` — retry the read,
  don't go fixing the screen. `hint` rides into the abort report to steer
  the agent's recovery.
- `tool: wait` with `{{seconds: N}}` (1–{MAX_WAIT_SECONDS}, or 0 with an
  `expect` to check immediately) sleeps exactly N seconds and calls
  nothing. Waiting used to live inside the guard as
  `wait_seconds`; it was really a poll count wearing a duration's name, so
  it moved out here where a second means a second. The arm's own gesture
  latency (several seconds per tap) already absorbs most cold starts —
  rehearse before reaching for a long wait.
- `steps[].expect` — ONE clause, checked after a `wait` sleeps, with that
  step's `hint` shown if it fails. `wait` + `expect` is THE settle idiom:
  pause, then confirm the screen you waited for arrived. Costs exactly one
  camera read.
  It is valid ONLY on a `wait` step. A gesture's own view is captured ~2s
  after the touch and is the same frame the next step's `guard` reads for
  free, so an `expect` there would assert nothing new — just a second name
  for one check on one frame. To confirm what a gesture produced, add a
  `wait` step after it; that also buys the settle time ~2s often isn't.
- A whole run is capped at {MAX_RUN_SECONDS}s, checked between steps, and
  aborts with `timeout` — one macro must never own half a session.
- `steps[].skip_when` — idempotence, not branching: ONE clause (same
  grammar as `require`) describing the step's POSTCONDITION. When
  already true the step is skipped — use ONLY when skipping leaves the
  same screen state as executing (e.g. don't tap the input box when the
  keyboard is already up). Checked before the guard; free when the screen
  text is already held, otherwise one peek; never polls.
- Single-character texts match a WHOLE element label (keyboard keys OCR as
  standalone letters; a substring single char would match anything) and
  are only allowed in the `{{text, within}}` region form — `macros check`
  rejects them elsewhere.

## YAML gotcha — quote your text

Guards and inputs match LITERAL screen text. YAML silently turns unquoted
`true`/`false` into booleans and bare numbers (`007`) into numbers;
`macros check` rejects those with the quoted fix, and unquoted
`{{placeholder}}` parses as a mapping (also caught). Anchors and aliases
(`&name` / `*name`) are rejected outright — write the value out in full.
Habit: quote every text value.

## Best practices

- Rehearse before enabling; bboxes are only trustworthy because you tested
  them on this rig.
- Put a guard on the step that NEEDS the state, not on the step that
  creates it. A launch or navigation step is followed by a `wait` carrying
  an `expect`, so a slow start costs a little time instead of aborting a
  good run — and the abort, if it comes, names the launch rather than
  whatever step tripped three lines later.
- Guard the steps where drift would hurt: taps at a FIXED position in a
  list that can reorder, and anything just short of a commit.
- Reach for `skip_when` whenever a step's effect may already be in place —
  the app reopening inside the last screen, the keyboard already up. It is
  what makes one macro survive several starting states.
- Keep macros short and single-purpose. Anything needing a decision mid-way
  belongs to the agent, not a macro.
- A rising `consecutive_aborts` streak in stats means the app layout
  changed; re-rehearse.

`stats.json` here is machine-written; delete a macro's directory and its
stats go on the next run. When sharing a macro, share only its directory.

Fleet-wide switch: `[macros] enabled = false` in `~/.physiclaw/config.toml`
hides ALL macros from the agent; these CLI commands keep working so you can
still author and rehearse.
"""
