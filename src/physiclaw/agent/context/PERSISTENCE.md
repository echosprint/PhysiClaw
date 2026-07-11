# Persistence

Tool-mutated only — no file-edit access. Two stores in `memory/`, plus the pitfalls list (§ Pitfalls):

- **`memory/memory.md`** — durable facts and preferences. Auto-injected under `## memory.md` every wake, so keep it small and curated. Tools: `save_memory`, `update_memory`, `read_memory`.
- **`memory/YYYY-MM-DD.md`** — append-only daily log, one file per day. Recent entries auto-injected at wake. Tools: `append_log`, `read_logs`.

## When to write

- `append_log`: after every major step (purchase, message, add-to-cart, decision), AND once at close on DONE / STUCK / FAIL. Per-step entries let a future wake recover a STUCK session's partial progress.
- `save_memory`: only "remember this", a lasting preference, or a learned app / entry→app mapping (AGENT § Choose apps) — session detail goes to the daily log. The engine nudges once at close if a flagged "remember this" went unsaved.

## Format

`[HH:MM] app: page → page — what you did`. Purchases: merchant, brand, spec, qty, price.

## Pitfalls

`add_pitfall(items)` banks 0–3 real traps you hit AND got past — required before a *long* DONE close (empty list if none worth banking; STUCK/FAIL don't capture — no proven fix). Each: lead with the app, then `trap → the fix that worked`, terse. They join the always-on `## Learned pitfalls` list — **don't repeat a listed one**. Append-only; a curator consolidates between sessions.
