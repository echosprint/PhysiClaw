---
name: cron
description: Manage scheduled jobs in jobs/jobs.md — list, add, or cancel cron entries. Verifies the file parses before and after every change.
allowed-tools: Bash, Read, Edit, Write
---

# Cron Job Management

The runtime checks `jobs/jobs.md` every minute. Each `## <id>` section
is a scheduled job. When a job's `Schedule:` matches the current minute
and its `Status:` is `pend`, the hook fires a `Trigger` and the
runtime spawns Claude Code with the job's context.

**Jobs are never deleted.** They transition through statuses:
`pend` → `fired` (hook) → `pend` (periodic done/fail) or `done`/`fail` (one-time).
`pend` → `cancel` (user).

## CLI tools

```bash
uv run python -m physiclaw.hooks.cron verify          # parse + list every job
uv run python -m physiclaw.hooks.cron jobs-to-do      # list fired jobs awaiting agent execution
uv run python -m physiclaw.hooks.cron done <id> <result>  # agent: mark complete with result
uv run python -m physiclaw.hooks.cron fail <id> <reason>  # agent: mark failed with reason
uv run python -m physiclaw.hooks.cron cancel <id>     # cancel a job
```

Always run `verify` first, and again after any change.

### Layout

1. `## <id>` heading — lowercase kebab-case
2. **Description** — one plain line below the heading (not a list item).
   For the user: "what is this job?"
3. **Fields** — markdown list (`- Key: value`).

**Description vs Context**: Description is for the user scanning the
file. Context is for the agent executing the job — include everything
Claude needs: which app to open, what to do, edge cases.

### Fields

| Field              | Set by   | Notes                                            |
|--------------------|----------|--------------------------------------------------|
| description line   | skill    | One line under heading (required)                |
| `Type:`            | skill    | `periodic` or `one-time` (required)              |
| `Schedule:`        | skill    | 5-field cron in backticks (required)             |
| `Create time:`     | skill    | ISO timestamp when the job was added             |
| `Next fire time:`  | skill    | Computed after each fire                         |
| `Last fire time:`  | hook     | When the hook last fired this job                |
| `Execution time:`  | agent    | When Claude finished — set via `done`/`fail` CLI |
| `Execution result:`| agent    | One-line result summary — set via `done`/`fail`  |
| `Status:`          | mixed    | `pend`, `fired`, `cancel`, `done`, or `fail`     |
| `Context:`         | skill    | Full agent instructions (required, last field)   |

**Status lifecycle:**

- `pend` — waiting for next fire time to arrive
- `fired` — hook fired the job, awaiting agent execution
- `cancel` — user cancelled; never fires
- `done` — agent finished a one-time job (`done <id>`)
- `fail` — agent failed a job (`fail <id>`)

**Transitions:**
`pend` → `fired` (hook fires) → `pend` (periodic done/fail) or
`done`/`fail` (one-time). `pend` → `cancel` (user).

**Time format:** ISO 8601 truncated to minute, naive local time.
Example: `2026-04-09T09:00`. Unfired fields use `(never)`.

## Workflows

### List jobs

1. Run verify
2. Show each job's id, description, type, schedule, status, and times

### Add a job

1. Run verify
2. Based on user request, determine: **id**, **description**, **type**,
   **schedule**, **context**
3. Create `jobs/jobs.md` if it doesn't exist:

   ```markdown
   # Cron Jobs
   ```

4. Append using this template:

   ```markdown
   ## <id (e.g. `weather-check`, `mom-birthday`)>

   <one-line description>

   - Type: <periodic|one-time>
   - Schedule: `<cron expression>`
   - Create time: <now as ISO>
   - Next fire time: <compute from schedule and current time, fill as ISO>
   - Last fire time: (never)
   - Execution time: (never)
   - Execution result: (never)
   - Status: pend
   - Context: <full instructions for the agent>
   ```

5. Run verify (this also validates Next fire time matches the schedule)
6. Tell the user when the job will next fire

### Cancel a job

1. Run verify
2. Read `jobs/jobs.md`, confirm which job to cancel with the user
3. Edit the `Status:` field to `cancel`
4. Run verify
