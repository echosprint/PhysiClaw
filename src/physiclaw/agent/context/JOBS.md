# Jobs

Scheduled work lives in `jobs/jobs.md`: id + 5-field cron + a context blob injected at wake under `## Scheduled jobs firing now`.

## Lifecycle

```text
[pend] ──(cron fires)──▶ [fired] ──(finish_job)──▶ [done | fail | cancel]
```

- **one-time** (default) — terminal; auto-purged after the retention window. For follow-ups, reminders, deferred actions — **including reply watchers**: schedule ONE check at a specific minute; if still unresolved, create the next check then. Never a `*/N` periodic watcher.
- **periodic** — for genuinely recurring routines (min interval 30 min; `create_job` rejects faster). `finish_job(id, "done"|"fail")` closes THIS occurrence — the job resets to `pend` and WILL fire again; `cancel` is the only permanent stop.

## Outcome marking

The engine never auto-marks. Every fired job needs its own `finish_job(id, status, recap)` this wake — a wake may fire several. Left in `fired` → auto-failed after 24 h. Next wake: `list_jobs("fired")` to find orphans.

## Ids & immutability

Id: `<user>-<topic>-<YYYY-MM-DD>`, lowercase letters/digits/hyphens (e.g. `alice-water-plants-2026-05-01`) — the date keeps recurring topics unique. No `update_job`; duplicate ids are rejected even against terminal entries. To reschedule / edit / revive:

1. `finish_job(old_id, "cancel", "rescheduling — new id <new_id>")`
2. `create_job(new_id, ...)`
