# Persistence

Two stores, tool-mutated only — no file-edit access to `memory/`:

- **`memory/memory.md`** — durable facts and preferences. Auto-injected under `## memory.md` every wake, so keep it small and curated. Tools: `save_memory`, `update_memory`, `read_memory`.
- **`memory/YYYY-MM-DD.md`** — append-only daily log, one file per day. Recent entries auto-injected at wake. Tools: `append_log`, `read_logs`.

## When to write

- `append_log`: after every major step (purchase, message, add-to-cart, decision), AND once at close on DONE / STUCK / FAIL. Per-step entries let a future wake recover a STUCK session's partial progress.
- `save_memory`: only on "remember this", a lasting preference, or a learned app / entry→app mapping (AGENT § Choose apps) — session detail belongs in the daily log.

## Format

`[HH:MM] app: page → page — what you did`. Purchases: merchant, brand, spec, qty, price.
