# Shared playbook packs

Template packs for the conductor — the deterministic walks described in
the docs' Custom section. Like `skills/`, these ship with the repo so a
task one user has rehearsed can be adopted by another with minimal
adaptation.

## Pack layout — the `action.yml` shape

A pack is a folder with ONE spec file — the whole pack in a single
self-describing YAML document, the way a GitHub workflow is one file
(and a composite action is one `action.yml`):

    <app>/
    ├── PLAYBOOK.yml         # the whole pack: meta + pages + playbooks
    └── macros/<n>/MACRO.yml # the pack's hands (recorded gestures)

`PLAYBOOK.yml` sections, top to bottom:

- `app` (must equal the folder) and `description` (what the pack
  automates and when to adopt it — `install` prints it, so that one
  line is the whole index)
- `placeholders:` — per-installation constants (the `inputs:` of
  `action.yml`): prompt prose + example per token
- `landmarks:` — named fixed spots (`{label, bbox}`) the pack's
  recover hands tap and its agent episodes are granted by name
- `playbooks:` — the walks, keyed by name (referenced as
  `<app>/<name>`). Each is a ROUTE: an optional pure-text `agent` and
  the `start` cold-launch, then `page:` waypoints (checked every
  time, each optionally declaring its own `recover:` hand)
  alternating with moves — `do` (a recorded macro), `agent` (the
  model drives inside your prompt's fence), `ask` (human gate;
  `resume:` re-enters the app), `tell`. A move's enter/verify checks
  derive from the adjacent waypoints; there is no branching and no
  loop — judgment is an `agent` step, approval is an `ask`.
- `pages:` — optional appendix for fingerprints no move lands on.
  Anchors are one clause (`"text"`, `{or: [...], region}`,
  `{and: [...]}`) — semantics only; geometry is learned on-device via
  `conductor calibrate`.

Adaptation notes and the rehearsal checklist ride as comments in the
same file.

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
inputs cannot reach (page anchors, macro guards).

## After installing

Everything lands disabled — the rehearse-then-enable rule. Recorded
gesture coordinates come from the authoring device; on the same phone
model they usually replay as-is, otherwise re-record the steps that
miss. Each manifest ends with its own checklist; the generic one:

1. rehearse: `physiclaw playbooks run <app>/<playbook> --input k=v`
2. capture page geometry: `physiclaw conductor calibrate <app>`
3. set `enabled: true` in the pack files
