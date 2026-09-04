# Shared playbook packs

Template packs for the conductor — the deterministic walks it runs
against a phone. Like `skills/`, these ship with the repo so a
task one user has rehearsed can be adopted by another with minimal
adaptation.

## Pack layout — a manifest and one file per playbook

A pack is a folder. The manifest names the app and holds what every
playbook of that app shares; each playbook is its own file beside it,
the way a Helm chart is one `Chart.yaml` beside one template per
resource, and a Maestro workspace one `config.yaml` beside one flow
per journey:

    <app>/
    ├── PLAYBOOK.yml         # the MANIFEST: meta + placeholders + landmarks + shared pages
    ├── <name>.yml           # one playbook per file: name (= the stem), description, enabled, inputs, route
    └── macros/<n>/MACRO.yml # the pack's hands (recorded gestures) routes share

`PLAYBOOK.yml` never carries a route. Every section is optional, so an
empty file is a valid pack (the file is the pack marker):

- `app` (must equal the folder when present) and `description` (what
  the pack automates and when to adopt it — `install` prints it)
- `placeholders:` — per-installation constants (the `inputs:` of
  `action.yml`): prompt prose + example per token
- `landmarks:` — named fixed spots (`{label, at, [page]}`) the
  pack's recover hands tap and its agent episodes are granted by name;
  a `page:` scope offers the spot only while that page reads
- `pages:` — fingerprints more than one route lands on (a page only
  one route uses may be declared beside its waypoint instead). Anchors
  are a list of texts the page shows — alternate readings of one text
  go inside it (`text: [..]`), `within:` pins it to a band or a box —
  semantics only; geometry is learned on-device via
  `physiclaw playbooks pages calibrate`. A page here may carry its
  `recover:` hand too — a gesture, a landmark tap, or a pack macro by
  name — which every route inherits unless it declares its own.

`<name>.yml` is the playbook, headed like a macro or a skill: `name`
(must equal the file's), `description` (the line the activation menu
shows — name the app the way users say it, 淘宝 not taobao, so two
packs offering the same task read apart), `enabled`, `inputs`, and
`route` — a ROUTE of `page:` waypoints (checked every time, each
optionally declaring its own `recover:` — one hand (`go_back`,
`{tap: landmarks.<name>}`, `{macro: <name>}`), or
`covered:`/`elsewhere:`/`locked:` hands per reading, with `tries:`
beside it)
alternating with moves — `start` (the cold launch, usually a pack
macro every route shares), `do` (a recorded macro), `agent` (the
model drives inside your prompt's fence, with the tools, the `give:`
grants — landmarks to tap, pack macros to run — and the `context:`
you list), `ask` (human gate; `yes:`/`no:` are the replies it reads,
`wait:` and `rounds:` its patience, `total_label:` the label a payment total sits beside,
`resume:` re-enters the app), `tell`. A move's enter/verify checks
derive from the adjacent waypoints; there is no branching and no loop
— judgment is an `agent` step, approval is an `ask`. What the playbook
declares is what runs: a page without `recover:` hands over, and
nothing retries or unlocks in the background.

## Values, and the one check shape

Three spellings, filled at three times:

| Spelling                                        | Filled                                        | Lives in                                             |
|-------------------------------------------------|-----------------------------------------------|------------------------------------------------------|
| `<<TOKEN>>`                                     | at install, from `placeholders.yml`           | any string in the pack, macros included              |
| `{inputs.x}`, `{node.field}`, `{ask.total}`     | when the walk reaches the move                | a move's `with:`, an `ask`/`tell` `message:`, a prompt |
| `{x}`                                           | when the macro runs, from the move's `with:`  | a macro's steps and checks                           |

A check reads the same everywhere it appears — a macro step's `require`
/ `forbid` / `expect` / `when` / `skip_when`, a page's anchors:

    "text"                              the text shows
    ["text", "alt"]                     any of them (alternate readings)
    {text: "t" | [alts], within: top}   in a band (top / bottom / left /
                                        right) or a [l, t, r, b] box
    {and: [..]}  {or: [..]}  {not: ..}  combinators (macro checks only)

Adaptation notes and the rehearsal checklist ride as comments in the
files.

`channel/` is the conductor's own pack: the thread page, the
send/open macros an `ask` runs, and `boot.yml` — the walk every wake
plays before any playbook (reach the thread, read the request, hand
the matching playbook the baton). Its `select` step is the one
entry only that file may carry; the hands and limits around it are
yours to edit, step, and replay like any route.

## Install

    physiclaw playbooks install playbooks/taobao
    physiclaw playbooks install playbooks/channel --set CONTACT=QiaoQian

`install` copies a pack into `~/.physiclaw/playbooks/<app>` VERBATIM —
tokens stay in the files, diffable against this template forever — and
records their values in `~/.physiclaw/playbooks/placeholders.yml`,
prompting (with the manifest's prose) for any token that has no value
yet. `PLAYBOOK.yml` installs with the pack, so the docs travel with it.

When physiclaw runs from a source checkout, these template packs load
directly (repo edits are live, no install step); a same-named pack
installed at home shadows the tree's.

## Placeholders

`<<CONTACT>>`-style tokens mark per-installation constants (the IM
contact name of your user thread, in these packs). Values live ONLY in
the local `playbooks/placeholders.yml`; every parser fills tokens from
it at load and rejects a token with no value, so a template adopted
without values fails loudly instead of tapping around looking for the
literal string.

Placeholders are deliberately not playbook inputs: inputs are per-run
values the activation extracts from the user's message, while a
placeholder is a constant of your installation that also lives where
inputs cannot reach (page anchors, macro checks).

## After installing

Everything lands disabled — the rehearse-then-enable rule. Recorded
gesture coordinates come from the authoring device; on the same phone
model they usually replay as-is, otherwise re-record the steps that
miss. Each manifest ends with its own checklist; the generic one:

1. replay offline: `physiclaw playbooks replay <app>/<playbook> --session <id>`
   walks recorded screens through the real route and shows where it
   would hand over — no phone, nothing written
2. rehearse: `physiclaw playbooks run <app>/<playbook> --input k=v`
3. capture page geometry: `physiclaw playbooks pages calibrate <app>`
4. set `enabled: true` in the pack files
