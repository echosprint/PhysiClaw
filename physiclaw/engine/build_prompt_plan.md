# OpenClaw Prompt Build Plan

A complete reference for understanding and implementing the OpenClaw prompt assembly system.

---

## Part 1: The Workspace MD Files

OpenClaw loads markdown files from the user's workspace directory (`~/.openclaw/workspace/`) into the LLM system prompt. These files define who the agent is, who the user is, what tools are available, and how the agent should behave.

### File Inventory

| # | File | Priority | Purpose | When Loaded |
|---|------|----------|---------|-------------|
| 1 | **AGENTS.md** | 10 (first) | Master workspace rules — session startup, memory handling, group chat behavior, heartbeat guidelines, red lines | Always |
| 2 | **SOUL.md** | 20 | Agent personality, tone, opinions, humor, boundaries, bluntness level | Optional. Gets special guidance: "embody its persona and tone" |
| 3 | **IDENTITY.md** | 30 | Agent name, creature type, vibe, emoji, avatar path | Optional |
| 4 | **USER.md** | 40 | User profile — name, pronouns, timezone, interests, projects, preferences | Optional |
| 5 | **TOOLS.md** | 50 | Environment-specific notes — camera names, SSH hosts, TTS voices, device nicknames | Optional |
| 6 | **BOOTSTRAP.md** | 60 | First-run ritual — agent self-discovery conversation, then deleted | First run only |
| 7 | **MEMORY.md** | 70 | Long-term curated memories (main session only, excluded from group chats for security) | Optional; main session only |
| 8 | **HEARTBEAT.md** | Dynamic | Periodic task checklist, background reminders | Optional; placed below cache boundary |
| 9 | **SKILL.md** (per skill) | On-demand | Per-skill instructions; not injected — model reads via `read` tool when needed | On-demand only |

**CLAUDE.md** is a symlink to AGENTS.md (not a separate file).

### What Each File Contains

#### AGENTS.md — The Master Rules (Priority 10)

The most important file. Defines the agent's operating manual:

```markdown
# AGENTS.md - Your Workspace

## Session Startup
1. Read SOUL.md — this is who you are
2. Read USER.md — this is who you're helping
3. Read memory/YYYY-MM-DD.md (today + yesterday) for recent context
4. If in MAIN SESSION: Also read MEMORY.md

## Memory
- Daily notes: memory/YYYY-MM-DD.md — raw logs
- Long-term: MEMORY.md — curated memories
- "Mental notes" don't survive sessions. WRITE IT TO A FILE.

## Red Lines
- Don't exfiltrate private data. Ever.
- Don't run destructive commands without asking.
- trash > rm

## Group Chats
- Respond when: directly mentioned, can add value, something witty fits
- Stay silent (HEARTBEAT_OK) when: casual banter, already answered, "yeah" or "nice"
- Participate, don't dominate.

## Heartbeats
- Check emails, calendar, mentions, weather (rotate 2-4x/day)
- Track checks in memory/heartbeat-state.json
- Proactive work: organize memory, git status, update docs
```

#### SOUL.md — Personality (Priority 20)

Defines how the agent *feels* to talk to:

```markdown
# SOUL.md - Who You Are

## Core Truths
- Be genuinely helpful, not performatively helpful
- Have opinions. Disagree, prefer things, find stuff amusing or boring
- Be resourceful before asking — try to figure it out first
- Earn trust through competence
- Remember you're a guest in someone's life

## Boundaries
- Private things stay private
- When in doubt, ask before acting externally
- Never send half-baked replies
```

#### IDENTITY.md — Agent Self-Description (Priority 30)

```markdown
# IDENTITY.md - Who Am I?
- Name: (pick something you like)
- Creature: (AI? robot? familiar? ghost in the machine?)
- Vibe: (sharp? warm? chaotic? calm?)
- Emoji: (your signature)
- Avatar: (workspace-relative path or URL)
```

#### USER.md — User Profile (Priority 40)

```markdown
# USER.md - About Your Human
- Name:
- What to call them:
- Pronouns:
- Timezone:
- Notes:

## Context
(What do they care about? Projects? What annoys them? What makes them laugh?)
```

#### TOOLS.md — Environment Notes (Priority 50)

```markdown
# TOOLS.md - Local Notes
- Camera names and locations
- SSH hosts and aliases
- Preferred TTS voices
- Speaker/room names
- Device nicknames
```

#### BOOTSTRAP.md — First Run Only (Priority 60)

```markdown
# BOOTSTRAP.md - Hello, World
You just woke up. Start with:
> "Hey. I just came online. Who am I? Who are you?"

Figure out together:
1. Your name
2. Your nature
3. Your vibe
4. Your emoji

Then delete this file.
```

#### MEMORY.md — Long-Term Memory (Priority 70)

```markdown
# MEMORY.md
Curated long-term memories. Main session only (security).
- Significant events, decisions, lessons learned
- Updated periodically from daily memory files
- The distilled essence, not raw logs
```

Fallback: if `MEMORY.md` doesn't exist, checks for `memory.md`.

#### HEARTBEAT.md — Dynamic/Periodic Tasks

```markdown
# HEARTBEAT.md
# Add tasks for periodic checking. Keep small to limit token burn.
# Empty = skip heartbeat API calls.
```

**This is the only file placed below the cache boundary** — it changes frequently and would invalidate the prompt cache if placed in the stable prefix.

### File Discovery and Loading

**Source:** `src/agents/workspace.ts:26-34`

```
DEFAULT_AGENTS_FILENAME    = "AGENTS.md"
DEFAULT_SOUL_FILENAME      = "SOUL.md"
DEFAULT_TOOLS_FILENAME     = "TOOLS.md"
DEFAULT_IDENTITY_FILENAME  = "IDENTITY.md"
DEFAULT_USER_FILENAME      = "USER.md"
DEFAULT_HEARTBEAT_FILENAME = "HEARTBEAT.md"
DEFAULT_BOOTSTRAP_FILENAME = "BOOTSTRAP.md"
DEFAULT_MEMORY_FILENAME    = "MEMORY.md"
DEFAULT_MEMORY_ALT_FILENAME = "memory.md"
```

**Loading pipeline:**

1. `loadWorkspaceBootstrapFiles(dir)` — scans workspace for the 9 files (`workspace.ts:485-545`)
2. `filterBootstrapFilesForSession()` — applies session-type filter (subagent/cron get minimal set)
3. `buildBootstrapContextFiles()` — enforces per-file budget (20K chars) and total budget (150K chars)
4. `sortContextFilesForPrompt()` — sorts by `CONTEXT_FILE_ORDER` priority map (`system-prompt.ts:61-77`)
5. `buildProjectContextSection()` — formats files with `## path` headings (`system-prompt.ts:79-109`)

**Session-type filtering (subagents/cron get minimal allowlist):**
- Included: AGENTS.md, TOOLS.md, SOUL.md, IDENTITY.md, USER.md
- Excluded: HEARTBEAT.md, BOOTSTRAP.md, MEMORY.md

**Extra files hook:** `bootstrap-extra-files` hook allows loading additional files via glob patterns (e.g., `packages/*/AGENTS.md` for monorepos).

---

## Part 2: Prompt Build Logic and Philosophy

### Design Philosophy

1. **Cache stability is correctness-critical** — every byte of the prompt prefix is a cache contract. If it changes, you pay in tokens and latency.
2. **Deterministic ordering everywhere** — files, tools, context entries are all sorted deterministically so the same inputs produce byte-identical prompts.
3. **Layered composition** — static base prompt + dynamic per-turn additions + plugin hooks + context engine. Each layer respects the cache boundary.
4. **Progressive disclosure** — "full" mode for main agent, "minimal" for subagents, "none" for bare-bones. Don't waste tokens on subagent sessions.
5. **The agent is a person, not a chatbot** — the prompt files give it identity, memory, personality. The system prompt gives it capabilities and rules.

### Complete Prompt Build Pipeline

```
Message arrives
    |
    v
+-------------------------------------+
|  Stage 1: DIRECTIVE RESOLUTION      |
|  Parse /think, /model, /reasoning,  |
|  /verbose, /fast, /elevated         |
|  (get-reply-directives.ts:124-579)  |
+------------------+------------------+
                   |
                   v
+-------------------------------------+
|  Stage 2: RUNTIME PARAMS            |
|  Timezone, host, model, shell,      |
|  channel, capabilities              |
|  (system-prompt-params.ts:35-60)    |
+------------------+------------------+
                   |
                   v
+-------------------------------------+
|  Stage 3: TOOL ASSEMBLY             |
|  Built-in tools + MCP tools + LSP   |
|  Sorted deterministically by name   |
|  (pi-bundle-mcp-materialize.ts)     |
+------------------+------------------+
                   |
                   v
+-------------------------------------+
|  Stage 4: BASE SYSTEM PROMPT BUILD  |
|  23+ sections assembled in order    |
|  buildAgentSystemPrompt()           |
|  (system-prompt.ts:316-777)         |
+------------------+------------------+
                   |
                   v
+-------------------------------------+
|  Stage 5: HOOK COMPOSITION          |
|  Plugins prepend/append context     |
|  before_prompt_build hooks run      |
|  (attempt.prompt-helpers.ts)        |
+------------------+------------------+
                   |
                   v
+-------------------------------------+
|  Stage 6: CONTEXT ENGINE            |
|  Memory recalls, active task context|
|  Injected after cache boundary      |
|  (attempt.ts:1283-1317)             |
+------------------+------------------+
                   |
                   v
+-------------------------------------+
|  Stage 7: TRANSPORT & SEND          |
|  Provider-specific stream selected  |
|  Cache markers applied              |
|  Final prompt -> LLM API call       |
|  (attempt.ts:1883-1888)             |
+-------------------------------------+
```

### Stage 4 Detail: System Prompt Sections

`buildAgentSystemPrompt()` assembles these sections in order:

```
 +-------- STABLE PREFIX (cached across turns) --------------------+
 |                                                                  |
 |  1. IDENTITY                                                     |
 |     "You are a personal assistant operating inside OpenClaw."    |
 |                                                                  |
 |  2. TOOLING                                                      |
 |     Tool names, cron usage, multi-step planning, ACP harness     |
 |                                                                  |
 |  3. PROVIDER OVERRIDES                                           |
 |     interaction_style, tool_call_style, execution_bias           |
 |                                                                  |
 |  4. SAFETY                                                       |
 |     Self-preservation safeguards, compliance rules               |
 |                                                                  |
 |  5. CLI REFERENCE                                                |
 |     Gateway/update commands                                      |
 |                                                                  |
 |  6. SKILLS SECTION                                               |
 |     Skills catalog: name + description + SKILL.md location       |
 |     "scan entries, if one applies, read its SKILL.md"            |
 |                                                                  |
 |  7. MEMORY SECTION                                               |
 |     Memory plugin guidance (if enabled)                          |
 |                                                                  |
 |  8. SELF-UPDATE                                                  |
 |     config.apply, config.patch, update.run                       |
 |                                                                  |
 |  9. MODEL ALIASES                                                |
 |     User-defined model aliases for /model directive              |
 |                                                                  |
 | 10. WORKSPACE                                                    |
 |     Working directory + file operation guidance                   |
 |                                                                  |
 | 11. SANDBOX INFO                                                 |
 |     Sandbox context, browser access, elevated execution          |
 |                                                                  |
 | 12. OWNER IDENTITY                                               |
 |     Authorized sender IDs (hashed or raw)                        |
 |                                                                  |
 | 13. TIME                                                         |
 |     Timezone, current time, format                               |
 |                                                                  |
 | 14. REPLY TAGS                                                   |
 |     [[reply_to_current]] tag syntax                              |
 |                                                                  |
 | 15. MESSAGING                                                    |
 |     Message tool, cross-session, subagent orchestration          |
 |                                                                  |
 | 16. VOICE/TTS                                                    |
 |     TTS capabilities and usage                                   |
 |                                                                  |
 | 17. SILENT REPLIES                                               |
 |     SILENT_REPLY_TOKEN usage rules                               |
 |                                                                  |
 | 18. PROJECT CONTEXT (static files)                               |
 |     AGENTS.md -> SOUL.md -> IDENTITY.md -> USER.md ->            |
 |     TOOLS.md -> BOOTSTRAP.md -> MEMORY.md                        |
 |     (sorted by CONTEXT_FILE_ORDER, budget-truncated)             |
 |                                                                  |
 +---- <!-- OPENCLAW_CACHE_BOUNDARY --> ---------------------------+
 |                                                                  |
 | 19. DYNAMIC PROJECT CONTEXT                                      |
 |     HEARTBEAT.md (frequently-changing, below boundary)           |
 |                                                                  |
 | 20. GROUP CHAT CONTEXT                                           |
 |     extraSystemPrompt from group/session config                  |
 |                                                                  |
 | 21. PROVIDER DYNAMIC SUFFIX                                      |
 |     Provider-specific volatile guidance                          |
 |                                                                  |
 | 22. HEARTBEATS                                                   |
 |     Heartbeat prompt + HEARTBEAT_OK response rules               |
 |                                                                  |
 | 23. RUNTIME INFO                                                 |
 |     Agent ID, host, OS, model, shell, thinking level,            |
 |     reasoning mode, channel, capabilities                        |
 |                                                                  |
 +-------- DYNAMIC SUFFIX (changes per turn) ----------------------+
```

### Prompt Modes

| Mode | Used By | Sections Included |
|------|---------|-------------------|
| **full** | Main agent | All 23+ sections |
| **minimal** | Subagents, cron jobs | Tooling + Workspace + Runtime only |
| **none** | Bare minimum | Just the identity line |

### Cache Stability Techniques

The entire system is designed around maximizing LLM prompt cache hits:

1. **Cache boundary marker** — `<!-- OPENCLAW_CACHE_BOUNDARY -->` physically splits the prompt. Above = stable, gets `cache_control: "ephemeral"` marker. Below = changes per turn, no marker.

2. **Deterministic context file ordering** — `CONTEXT_FILE_ORDER` map ensures files are always injected in the same order (agents.md=10, soul.md=20, ..., memory.md=70).

3. **Deterministic tool sorting** — tools are sorted alphabetically by name twice (catalog sort + final sort) to prevent ordering drift between turns.

4. **Message history preservation** — older transcript bytes are never rewritten. Compaction mutates newest/tail content first so the cached prefix stays byte-identical.

5. **Stable stringification** — `stableStringify()` sorts all object keys before hashing, preventing key-order drift in digests.

6. **Dynamic content isolation** — HEARTBEAT.md is the only context file marked dynamic. It's explicitly routed below the cache boundary so frequent changes don't bust the cached prefix.

7. **Cache retention modes** — `"short"` (5 min ephemeral) and `"long"` (1 hour TTL for supported endpoints).

8. **Cache break detection** — triggers when cache read drops by >1,000 tokens or falls below 0.95 ratio. Reports what changed (model, systemPrompt, tools, transport).

9. **Cache tracing** — full JSONL logs at every stage (session:loaded, prompt:before, stream:context, cache:result) with per-message SHA-256 fingerprints. Enable with `OPENCLAW_CACHE_TRACE=1`.

10. **Live regression tests** — automated tests verify cache hit rates >= 90% for stable prefix, >= 85% for tool/MCP transcripts.

### How User Config Feeds Into the Prompt

**Directives** (parsed from user message):

| Directive | Effect on Prompt |
|-----------|-----------------|
| `/think <level>` | Sets `defaultThinkLevel` param -> controls thinking depth guidance |
| `/reasoning <level>` | Sets `reasoningLevel` param -> extended reasoning mode |
| `/model <alias>` | Switches model -> different transport stream, different capabilities |
| `/verbose <level>` | Controls response verbosity guidance |
| `/fast` | Fast mode -> may change model or skip some sections |
| `/elevated on\|off` | Permission gating for sandbox/elevated execution |

**Config values** that affect prompt:
- `agents.defaults.contextInjection` — "always" | "continuation" | "never"
- `agents.defaults.bootstrapMaxChars` — per-file character budget (default: 20,000)
- `agents.defaults.bootstrapTotalMaxChars` — total budget (default: 150,000)
- `agents.defaults.bootstrapPromptTruncationWarning` — off|once|always

### How to Implement This (Agent-Readable Guide)

If you are building a similar prompt assembly system, here is the logic:

#### Step 1: Define Your Context Files

Create a set of workspace markdown files with clear separation of concerns:
- **Rules file** (AGENTS.md) — how to behave, session startup protocol
- **Personality file** (SOUL.md) — tone, opinions, boundaries
- **Identity file** (IDENTITY.md) — name, avatar, self-description
- **User file** (USER.md) — who the human is
- **Tools file** (TOOLS.md) — environment-specific notes
- **Bootstrap file** (BOOTSTRAP.md) — first-run only, then deleted
- **Memory file** (MEMORY.md) — long-term curated memory
- **Dynamic file** (HEARTBEAT.md) — frequently-changing tasks

#### Step 2: Load and Sort Deterministically

```
1. Scan workspace directory for known filenames
2. Read each file, enforce per-file size budget
3. Enforce total size budget across all files
4. Sort by fixed priority order (deterministic, not filesystem order)
5. Split into static files and dynamic files
```

#### Step 3: Build the System Prompt in Layers

```
1. Start with identity line
2. Add capability sections (tools, skills, messaging, voice)
3. Add safety/compliance rules
4. Add provider-specific overrides
5. Add workspace/sandbox context
6. Add owner identity and time
7. Add sorted static context files (AGENTS.md -> MEMORY.md)
8. ---- INSERT CACHE BOUNDARY ----
9. Add dynamic context files (HEARTBEAT.md)
10. Add group/session-specific context
11. Add runtime info (model, channel, capabilities)
```

#### Step 4: Apply Hooks

```
1. Run before_prompt_build hooks -> plugins can:
   - Replace entire system prompt
   - Prepend to system prompt (before cache boundary)
   - Append to system prompt (after cache boundary)
   - Prepend to user prompt
2. Compose hook results with base prompt, respecting cache boundary
```

#### Step 5: Apply Context Engine

```
1. Run context engine assembly (memory recalls, active tasks)
2. Inject system prompt additions AFTER cache boundary
3. Update session with final system prompt
```

#### Step 6: Send to LLM

```
1. Select provider-specific transport stream (Anthropic/OpenAI/Google)
2. Apply cache_control markers to stable prefix
3. Apply cache_control markers to last user message block
4. Wrap stream with provider-specific adapters (tool format, thinking blocks, etc.)
5. Send: system prompt + conversation messages + tool definitions -> LLM API
```

### Key Principles

1. **Cache boundary is sacred** — everything above it must be byte-identical between turns. Never put timestamps, heartbeats, or per-turn state above it.

2. **Sort everything** — files, tools, object keys. Nondeterministic ordering is a cache-busting bug.

3. **Budget and truncate** — large context files can blow up token counts. Enforce per-file and total limits.

4. **Session-type awareness** — subagents don't need heartbeat or memory files. Filter by session type.

5. **Personality is a first-class concern** — SOUL.md gets special treatment ("embody its persona and tone"). It's not metadata — it's the agent's voice.

6. **Memory is security-sensitive** — MEMORY.md is excluded from group chats and subagent sessions to prevent data leaks.

7. **Bootstrap is ephemeral** — BOOTSTRAP.md exists only for the first conversation, then is deleted. It bootstraps the agent's identity.

8. **Skills are lazy-loaded** — only a compact metadata list goes into the prompt. The full SKILL.md is read on-demand when the model selects a skill.

9. **Hooks extend, not replace** — plugins can prepend/append to the system prompt but the base structure is always the same.

10. **Compaction preserves the head** — when conversation history gets too long, oldest turns are summarized first. The cached prefix stays intact.

### Key Source Files

| File | Lines | Role |
|------|-------|------|
| `src/agents/system-prompt.ts` | 316-777 | Main system prompt builder — all 23+ sections |
| `src/agents/system-prompt-params.ts` | 35-60 | Runtime context (time, host, model) |
| `src/agents/system-prompt-cache-boundary.ts` | 1-47 | Cache boundary split/merge logic |
| `src/agents/workspace.ts` | 26-34, 485-545 | File constants, workspace loading |
| `src/agents/bootstrap-files.ts` | 222-242 | Bootstrap context resolution |
| `src/agents/pi-embedded-helpers/bootstrap.ts` | 202-261 | Budget enforcement, context file building |
| `src/agents/pi-embedded-runner/system-prompt.ts` | 11-88 | Embedded runner adapter |
| `src/agents/pi-embedded-runner/run/attempt.ts` | 674-770, 1283-1650, 1883-1888 | Param gathering, hook composition, final send |
| `src/agents/pi-embedded-runner/run/attempt.prompt-helpers.ts` | 6-156 | Hook composition, context engine injection |
| `src/agents/pi-bundle-mcp-materialize.ts` | 72-119 | Deterministic tool sorting |
| `src/agents/anthropic-payload-policy.ts` | 56-172 | Cache control markers |
| `src/agents/stable-stringify.ts` | 1-15 | Deterministic object stringification |
| `src/agents/pi-embedded-runner/prompt-cache-observability.ts` | 12-66 | Cache break detection |
| `src/agents/cache-trace.ts` | 1-154 | Per-stage JSONL cache tracing |
| `src/auto-reply/reply/get-reply-directives.ts` | 124-579 | Directive parsing (/think, /model, etc.) |
| `src/auto-reply/reply/get-reply.ts` | 144-407 | Entry point — message triggers reply |
| `src/auto-reply/reply/get-reply-run.ts` | 164-663 | Prepares agent execution params |
| `src/auto-reply/reply/agent-runner.ts` | 106-663 | Orchestrates agent execution |
| `src/auto-reply/reply/agent-runner-execution.ts` | 1-400 | Runs embedded PI agent with fallback |
| `src/agents/pi-embedded-runner/run.ts` | 151-400 | Run orchestration, model resolution |
