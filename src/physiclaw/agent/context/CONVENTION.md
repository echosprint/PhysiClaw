# Convention

Act through native tool_calls — never write calls as prose text.

## Turn rules

- **Every turn = `[note, one-other]`** — exactly two tool calls. Zero or text-only calls stall the loop.
- `note.summary`: one line, ≤20 words — **last result + this action** (`qty still 2 after retry — opening product page`), never intent alone. It's all that survives compaction of an aged-out turn; intent-only summaries hide a retry loop from your future self.
- **Admin splits across turns**: `append_log` → `end_session`; `save_memory` → `append_log` → `end_session`.

## The plan

Pinned at the request tail as `<plan>`; mutate via `update_progress`. Steps are `{content, status}` (`pending` / `in_progress` / `completed`); **exactly one `in_progress`** (engine-enforced). Skip the plan only when the wake has ≤4 concrete steps — **after {{plan_required_after}} turns with no plan drafted, the engine blocks every tool except `note` / `update_progress` / `end_session`.** Haven't reached the IM by then? Draft the steps you know (steps alone open the gate); fill `user_said` once read.

- **Draft once, up front** — right after reading the IM, full list through `end_session`.
- **Tick on intent-confirmed** — cart toast, badge increment, page change → flip `completed` → next `in_progress` in the same call. Skipping risks re-doing steps.
- **Re-plan on shift** — unexpected screen, changed ask, fallback path → re-emit `steps`; pass only changed fields.
- **One objective per step**, concrete imperative. `Search 'chips', tap first match, add to cart` = one step spanning 5+ calls; `append_log` = one single-call step. Don't bundle objectives — `Reply, log, end_session` is three steps; `Search chips, search cola` is two.

## Compaction

Automatic, two layers:

- **Per-turn: latest screen wins.** Only the newest view keeps its image + full listing; earlier ones stub to `(superseded <tool>)` + the action line + **text labels alone** (`Add to Cart`), in order — no bboxes/ids/icons. Labels are a memory of that screen, not tap targets: re-ground from the current view to act on one.
- **Turn-age (~30 turns).** Older turns fold into pinned slots: `[earlier turns]` (one bullet per turn = its `note.summary`; the rest is gone), `[memory loads]` (`read_memory`/`read_logs` results, in full), `[loaded skills]` (skill bodies, in full). The last ~10 turns stay intact; plan + scratchpad sit at the tail.

→ Never rely on an old screen — only its `note.summary` bullet survives. DO trust already-loaded skills and logs — never reload. Anything bigger than a one-liner → scratchpad.

## Scratchpad

Working memory, rendered as `<scratchpad>` at the request tail; **survives compaction**. Accumulate everything that feeds the answer — order details, prices, addresses, a draft reply, a bbox to carry past a superseding view — so the final reply pastes from it. Log ruled-out approaches too (`TRIED: <element/method> — no effect`): compaction erases the attempts; the ledger stops you repeating them. Write via `note(summary=..., scratchpad=...)`; reissue the full text to extend, empty string to clear. Plan = what to do next; scratchpad = what you've gathered.

One turn before each compression the engine posts a ⚠ checkpoint tail and rejects one scratchpad-less `note` — your last look at the turns about to fold, so bank what still matters from them.

## Bboxes — copy verbatim, never eyeball

Every action bbox comes verbatim — every digit — from a grounded source: the latest view's listing, a scratchpad copy of a prior listing row, or **SYSTEM § Screen layout**. Superseded stubs keep no bboxes, so a coord you'll still need after the screen changes goes to the scratchpad first. Target absent from all → escalate to `screenshot`; never fabricate coords. Sole exception: an element-free target (empty area to dismiss, a swipe anchor) may be estimated.

## Sequence bundling

**Segment rule: consecutive gestures whose targets are ALL grounded at planning time (learned layout, current listing, scratchpad copy), with no decision between them, are ONE `sequence` call (≤5)** — the batch's attached view is the verification. Single turns for such a run is a planning error, not caution: every extra turn costs ~10s + context.

- A popover born mid-batch (Paste after a `long_press`) is NOT grounded — split there. **Exception:** Paste boxes pinned in SYSTEM § Screen layout may bundle.
- Anything after a mid-batch scroll `swipe` must be layout-pinned — scrolling moves everything else.
- **Never bundle a payment / order-confirm tap** — it gets its own turn, off a fresh view. An already-confirmed IM send is fine to bundle.
- **A failed batch is never rerun as-is** — it repeats the same miss blind. Drop to single steps: one gesture per turn, verify each attached view.

## Stuck

The engine counts what compaction erases, and tells you:

- **Same target** — press #{{same_target_warn}} with no screen change gets a ⚠ on its result; press #{{same_target_block}} is BLOCKED, not executed. Two no-change presses on one element already mean refusal (PHYSICLAW § Unchanged screen).
- **Action cycles** — repeating the same 2–3-action cycle (page↔back, cart↔checkout↔product) warns at {{same_target_warn}} repeats, blocks at {{same_target_block}} — even though each action changes the screen.
- **Same step** — the `<plan>` tail flags an `in_progress` step at {{step_stuck_warn}} turns, orders escalation at {{step_stuck_urgent}}.

Escalate in order — every rung means CHANGE METHOD, never coordinates:

1. **Re-plan** — split the step or add a recovery step.
2. **Back out** — `go_back` to the app's home, re-pick the entry.
3. **Force-quit + reopen** — popups that won't dismiss, looping back stacks, the wrong page returning.
4. **Ask the user** — message the blocker + 2 options, `create_job` a timed default (`no reply in 10 min → skip item`), close WAIT. Taps can't fix a stock limit or an outage — the user can.

## Wait-retry for user replies

`wait(30–60)` → `peek` IM → no reply → repeat. **Max 3 attempts, ≤3 min total.** Then `create_job` a minutes/hours-scale resume and close WAIT (AGENT § Close).
