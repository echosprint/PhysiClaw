# Engine — Prompt Concatenation & LLM API

The engine replaces `claude -p` subprocess spawning with direct LLM API
calls. This gives us structured prompt assembly, tool definitions, and
streaming — without shelling out to a CLI binary.

## Why

The current `runtime/claude.py` spawns `claude -p <prompt>` as a
subprocess. This works but has limitations:

- No programmatic tool use (MCP tools must be pre-configured in the CLI)
- No structured output or streaming
- No retry/fallback logic
- The prompt is a flat string — no system/user message separation
- No model selection or parameter tuning per job

## Architecture

```
cron hook fires → Trigger(description, source)
                      ↓
              engine.build_prompt(trigger, job)
                      ↓
              engine.call_llm(messages, tools)
                      ↓
              LLM response → execute tool calls → mark done/fail
```

### Prompt concatenation

The engine assembles messages from multiple sources:

1. **System prompt** — from CLAUDE.md (project instructions, targeting
   workflow, tool descriptions)
2. **Job context** — from the job's `Context:` field in jobs.md
3. **Trigger info** — what fired, when, source tag
4. **Completion instruction** — how to mark done/fail via CLI

Each source maps to a message role:

```python
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"{trigger_info}\n\n{job_context}\n\n{completion_instruction}"},
]
```

### LLM API call

No SDK. Plain HTTP via `httpx`, using the OpenAI-compatible
`/v1/chat/completions` endpoint. This keeps the engine provider-
agnostic — works with OpenAI, Anthropic (via proxy), DeepSeek, Kimi,
MiniMax, local models (Ollama, vLLM), or any OpenAI-compatible API.

```python
import httpx

response = httpx.post(
    f"{base_url}/v1/chat/completions",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "model": model,
        "messages": messages,
        "tools": tool_definitions,
        "max_tokens": 4096,
    },
)
result = response.json()
```

Configuration via environment variables:

- `LLM_BASE_URL` — API endpoint (default: `https://api.openai.com`)
- `LLM_API_KEY` — bearer token
- `LLM_MODEL` — model name (default: `gpt-4o`)

### Tool definitions

The engine registers PhysiClaw MCP tools as OpenAI function-calling
schema so the LLM can invoke them directly in the API response.

```json
{
  "type": "function",
  "function": {
    "name": "camera_view",
    "description": "See the phone screen via camera",
    "parameters": { ... }
  }
}
```

## TODO

- [ ] `engine/prompt.py` — prompt builder (concatenate system + context + trigger)
- [ ] `engine/llm.py` — HTTP client, OpenAI-compatible `/v1/chat/completions`
- [ ] `engine/tools.py` — convert MCP tool definitions to OpenAI function-calling schema
- [ ] `engine/runner.py` — orchestrator (build prompt → call LLM → execute tool calls → loop until done → mark done/fail)
- [ ] Replace `runtime/claude.py` subprocess with engine calls
- [ ] Per-job model/endpoint override in jobs.md
