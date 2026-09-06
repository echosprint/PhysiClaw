"""The authoring-surface texts: the `init` scaffold and the README.

Both are documentation-as-artifact — the scaffold IS the format reference
a new user edits in place, and the README is the one kept current at
``~/.physiclaw/macros/README.md`` via ``ensure_readme`` (rewritten
whenever these constants change, so installs never document a retired
format). Tool lists and caps interpolate from `model`'s constants, never
hand-copied, so a whitelist or limit change updates every surface in the
same edit.
"""

from physiclaw.common import gesture_vocab
from physiclaw.common.bbox import BANDS
from physiclaw.macros.model import (
    ALLOWED_STEP_TOOLS,
    ARGLESS_TOOLS,
    HANDLE_OBJECT_CHARS,
    MAX_CLAUSE_DEPTH,
    MAX_INPUTS,
    MAX_LABEL_READINGS,
    MAX_PROSE_LEN,
    MAX_RUN_SECONDS,
    MAX_STEPS,
    MAX_WAIT_SECONDS,
)


def render_init(name: str) -> str:
    """The scaffold body for ``macros/<name>.yml`` — parses clean
    (disabled) so `check` passes before any editing."""
    return INIT_TEMPLATE.format(
        name=name, tools="verbs: " + "  ".join(sorted(ALLOWED_STEP_TOOLS))
    )


# Scaffold written by `physiclaw macros init <name>` — a real, rehearsable
# WeChat-message flow (coordinates from a reference rig), not lorem ipsum:
# the template IS the format documentation AND the best-practice example.
# It must parse clean so a fresh scaffold passes `macros check` before any
# editing (`tests/macros/test_scaffold.py` pins that, including that
# it deliberately stops before Send).
INIT_TEMPLATE = """\
name: {name}                   # must equal the file name
description: Open WeChat, paste a message into your user's chat, stop before Send

# A valid macro is enabled by default; this scaffold starts off. Set it to
# true (or delete the line) after `physiclaw macros run {name}` succeeds.
enabled: false

# Inputs the agent fills at call time; "{{message}}" in the steps below.
# An input with `default` is optional; without one it is required.
inputs:
  message:
    description: The message text to paste into the chat
    example: "Task done, total 30"

# A real flow from a reference rig — REHEARSE BEFORE ENABLING:
#     physiclaw macros run {name} -i message=test
# and fix each `at` box until every step lands on YOUR rig, phone, and
# layout. A step is one verb and its object, plus what it needs beside it:
#     {tools}
#   tap / double_tap / long_press: what is tapped     at: [l, t, r, b]
#   swipe: up | down | left | right                   at: [l, t, r, b]  [size:] [speed:]
#   send_to_clipboard: the text        wait: seconds  (expect / hint go here)
#   home_screen  go_back  force_quit  peek            bare words, no object
#   when: / skip_when:  run the step only while / skip it while a check holds
#   require: / forbid:  checked BEFORE the step fires   hint: steers recovery
# Each step's handle is derived for you (idx8-tap-paste): the step log,
# the agent's list, and --start-at / --stop-after all use it.
# QUOTE text values: YAML turns unquoted true/false into booleans and bare
# numbers into numbers — `macros check` rejects those with a pointer, but
# quoting avoids the round-trip.
steps:
  - home_screen

  # The object says what the box IS: the element's own on-screen text when
  # it has any (the runner then heals the tap to where that text sits
  # today), or a plain description for icons and blank areas.
  - tap: "WeChat dock icon"
    at: [0.317, 0.893, 0.488, 0.98]   # ← the dock icon; move to YOURS
    # No check here. A check runs BEFORE its own step, so "WeChat is
    # open" is a claim about the NEXT step, never about this one.

  # Settle, then confirm — one step, one camera read. `wait` is exactly
  # that many seconds; start small, since the arm's own gesture latency
  # already covers most cold starts. WeChat reopens WHERE IT WAS LEFT, so
  # the title bar reads either the app name (chat list) or your user's
  # name (straight back inside the chat) — accept both.
  - wait: 2
    expect: {{text: ["WeChat", "Weixin", "your-user-name"], within: top}}   # ← EDIT the name
    hint: "WeChat did not open — check the dock position, or log in first"

  # `skip_when`: WeChat often reopens INSIDE the last chat — when the title
  # bar already shows your user, tapping the list position would hit
  # messages. `require`: about to tap a FIXED row position, so check the
  # name is really sitting there — a message in another chat reorders the
  # list and this box would then open the wrong person.
  - tap: "your-user-name"                    # ← EDIT — the row's own text, so
    at: [0.066, 0.086, 0.249, 0.186]         #   the tap follows it if rows shift
    skip_when: {{text: "your-user-name", within: [0.15, 0.02, 0.85, 0.1]}}    # ← EDIT
    require: {{text: "your-user-name", within: [0.05, 0.08, 0.95, 0.19]}}  # ← EDIT
    # forbid: "Account protection"   # tripwire: abort if that text is up
    hint: "that chat is not the top row — open it by hand"

  - send_to_clipboard: "{{message}}"

  # The keyboard slides up after this tap. `skip_when` is idempotence: this
  # step's POSTCONDITION is "keyboard up" — if it already is, the box has
  # moved up and this tap hits the keys. Anchor on the space bar, which
  # exists only while the keyboard is up; the list means either spelling.
  - tap: "message input box"
    at: [0.1, 0.915, 0.69, 0.955]
    skip_when: {{text: ["space", "空格"], within: [0.25, 0.85, 0.80, 0.95]}}

  # The box sits higher with the keyboard up. Last check before the paste
  # chain: still in the right chat, whichever way we got here.
  - long_press: "message input box (keyboard up)"
    at: [0.118, 0.578, 0.893, 0.637]
    require: {{text: "your-user-name", within: [0.15, 0.02, 0.85, 0.1]}}   # ← EDIT
    hint: "not in the right chat — paste by hand"

  # "Paste" is the menu bubble's own text — a healing target. The check
  # is region-scoped: THE paste bubble, not the word anywhere on screen.
  - tap: "Paste"               # rehearse just this one: --start-at idx8-tap-paste
    at: [0.099, 0.544, 0.197, 0.564]
    require: {{text: "Paste", within: [0.03, 0.5, 0.35, 0.62]}}
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

One file per macro: `~/.physiclaw/macros/<name>.yml`.
Scaffold one with `physiclaw macros init <name>`, edit it, then:

    physiclaw macros list           # every macro: enabled / disabled / invalid
    physiclaw macros check          # lint every macro file
    physiclaw mcp                   # the server `run` drives — NOT `server`,
                                    # which also wakes the agent and would
                                    # move the phone mid-rehearsal
    physiclaw macros run <name> -i key=value   # rehearse on the live rig
    # then flip the scaffold's `enabled: false` to true (or delete the line)
    physiclaw macros stats          # success/abort counters per macro
    physiclaw macros runs [<hex6>]  # per-step logs; every run has an id
                                    # `macro-run-<hex6>` (printed after
                                    # rehearsals, stamped in results/stats)
                                    # and its own log/macros/<id>/ dir with
                                    # events.jsonl + per-step screenshots.
                                    # Each step prints the arguments it
                                    # fired with — the bbox you edit when a
                                    # tap lands wrong

## Format

- `name` (required) — must equal the file name; lowercase/digits/hyphens.
- `description` (required) — tells the agent when to use it. It, and each
  input's `description` / `example`, must be ONE line of at most
  {MAX_PROSE_LEN} characters: they render verbatim into the cached system
  prompt on every wake.
- `enabled` — absent defaults to true; set `enabled: false` to hide a macro
  from the agent (the `init` scaffold starts false until you rehearse).
- `inputs.<name>` — `description` required; `default` makes it optional;
  `example` helps the agent fill it. String values only; max {MAX_INPUTS}.
  `{{input}}` placeholders fill strings in the steps (`{{{{`/`}}}}` for a
  literal brace); `<<UPPERCASE>>` is reserved for install-time pack
  placeholders (filled from playbooks/placeholders.yml at load, never
  typed literally).
- `steps` — a list, max {MAX_STEPS}; each step is ONE verb with its object,
  plus the qualifiers it needs beside it:

      - tap: "Paste"                     what is pressed — the element's own
        at: [left, top, right, bottom]   on-screen text or a description; the
                                         box as 0–1 screen fractions
      - swipe: up                        up / down / left / right, with `at`;
        at: [0.1, 0.2, 0.9, 0.8]         optional `size` ({"/".join(gesture_vocab.SWIPE_DISTANCES)})
                                         and `speed` ({"/".join(gesture_vocab.SWIPE_SPEEDS)})
      - send_to_clipboard: "{{message}}"   the text
      - wait: 2                          seconds (0–{MAX_WAIT_SECONDS}); `expect` lives here
      - home_screen                      bare words: {", ".join(sorted(ARGLESS_TOOLS))}

  The verbs: {", ".join(sorted(ALLOWED_STEP_TOOLS))}. A press
  ({" / ".join(sorted(gesture_vocab.PRESS_TOOLS))}) whose object IS
  on-screen text HEALS: it taps where that text sits today if a match lies
  near the recorded spot, falling back to `at` otherwise — never asserting
  (presence checks stay `require`'s job). The object may list up to
  {MAX_LABEL_READINGS} alternate readings of ONE target
  (`tap: ["免密支付", "立即支付"]`). A step that heals on every run wants
  re-recording — the step log names each heal, and the run log records the
  coordinates that actually fired.
- Step handles are derived, never written: `idx<N>-<verb>-<object>`
  (`idx1-home_screen`, `idx3-wait-2`, `idx8-tap-paste`; a long object is
  cut to {HANDLE_OBJECT_CHARS} characters at a word boundary, a readings
  list takes its first reading). The step log, the run log, the agent's
  `steps:` list and `--start-at` / `--stop-after` all use them.
- `when` / `skip_when` — idempotence, not branching: ONE check describing the
  step's POSTCONDITION. `skip_when: X` skips the step while X shows; `when: X`
  runs it only while X shows. Use ONLY when skipping leaves the same screen
  state as executing (don't tap the input box when the keyboard is already
  up; tap 跳过 only when the ad is up). Checked before `require`; free when
  the screen text is already held, otherwise one peek; never polls.
- `require` / `forbid` — checked BEFORE the step fires, so they describe
  the screen the step NEEDS, never the screen the step produces (to wait
  on an app you just launched, use a `wait` carrying an `expect`).
  `require` must hold; `forbid` must not (`forbid: X` is
  `require: {{not: X}}`, kept because the popup-tripwire reading earns a
  name). A check runs ONCE: it is a predicate, not a wait. It reads the
  screen text already held, free, and peeks when none is held (step 1, or
  after a step that returned no view; `send_to_clipboard` keeps the held
  screen). An unreadable screen is never a satisfied check: a failed peek
  is retried once, then aborts with `could not read the screen` — retry
  the read, don't go fixing the screen.
- `wait: N` sleeps exactly N seconds and calls nothing (0 only with an
  `expect`, to check immediately). The arm's own gesture latency (several
  seconds per tap) already absorbs most cold starts — rehearse before
  reaching for a long wait. A whole run is capped at {MAX_RUN_SECONDS}s,
  checked between steps, and aborts with `timeout`.
- `expect` — ONE check, run after a `wait` sleeps. `wait` + `expect` is
  THE settle idiom: pause, then confirm the screen you waited for arrived,
  for exactly one camera read. It is valid ONLY on a `wait`: a gesture's
  own view is captured ~2s after the touch and is the same frame the next
  step's checks read for free, so an `expect` there would assert nothing
  new. To confirm what a gesture produced, add a `wait` after it.
- `hint` — rides into the abort report when this step's checks fail, to
  steer the agent's recovery.
- EVERY check — `require`, `forbid`, `expect`, `when`, `skip_when` — is ONE
  clause, in one of these forms:
      "WeChat"                                substring of the whole screen
      ["WeChat", "Weixin"]                    any of them
      {{text: "WeChat", within: top}}           element-granular, in a band
                                              ({" / ".join(BANDS)})
                                              or a [l,t,r,b] box; `text`
                                              may list alternates
      {{or: [c, c]}}   {{and: [c, c]}}           any-of / all-of
      {{not: c}}                               absent
  They nest up to {MAX_CLAUSE_DEPTH} levels — the cap is on NESTING, not
  breadth, so flattening siblings into one operator is the fix when you hit
  it. Any of them may carry `within`, which scopes the whole subtree; an
  inner `within` overrides the one it sits in. Single-character texts match
  a WHOLE element label (keyboard keys OCR as standalone letters) and are
  only allowed with `within` — `macros check` rejects them elsewhere.

## YAML gotcha — quote your text

Checks and inputs match LITERAL screen text. YAML silently turns unquoted
`true`/`false` into booleans and bare numbers (`007`) into numbers;
`macros check` rejects those with the quoted fix, and unquoted
`{{placeholder}}` parses as a mapping (also caught). Anchors and aliases
(`&name` / `*name`) are rejected outright — write the value out in full.
Habit: quote every text value.

## Best practices

- Rehearse before enabling; bboxes are only trustworthy because you tested
  them on this rig.
- Put `require` on the step that NEEDS the state, not on the step that
  creates it. A launch or navigation step is followed by a `wait` carrying
  an `expect`, so a slow start costs a little time instead of aborting a
  good run — and the abort, if it comes, names the launch rather than
  whatever step tripped three lines later.
- Check the steps where drift would hurt: taps at a FIXED position in a
  list that can reorder, and anything just short of a commit.
- Give text targets their REAL on-screen text as the object — that is
  what lets the tap follow a button that moved. Volatile text (prices,
  times, item titles) makes a bad object for the same reason it makes a
  bad anchor.
- Point a check's `within` at the SAME region the step is about to tap.
  Checking that the text is somewhere in a wider band, then tapping a
  fixed box elsewhere, passes the check and still hits the wrong row.
- Reach for `skip_when` whenever a step's effect may already be in place —
  the app reopening inside the last screen, the keyboard already up. It is
  what makes one macro survive several starting states.
- `home_screen` opens the App SWITCHER when you are already on the home
  screen, so a macro starting there wants a `skip_when` anchored on
  something only the home screen shows.
- Keep macros short and single-purpose. Anything needing a decision mid-way
  belongs to the agent, not a macro.
- A rising `consecutive_aborts` streak in stats means the app layout
  changed; re-rehearse.

`stats.json` here is machine-written; delete a macro's file and its
stats go on the next run. When sharing a macro, share the one file.

Fleet-wide switch: `[macros] enabled = false` in `~/.physiclaw/config.toml`
hides ALL macros from the agent; these CLI commands keep working so you can
still author and rehearse.
"""
