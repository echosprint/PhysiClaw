"""Concrete provider implementations.

One file per vendor. The trivial vendors (qwen, moonshot, openai,
deepseek) are pure declarations — `PROVIDER_ID + BASE_URL` plus the
declared knobs the bases consume (`API_KEY_ENV_VARS`,
`SYSTEM_PROMPT_FRAGMENT`, `CACHE_MARKERS`) — with no method
overrides. Vendors with real behavior document their quirks in place:

  - `google.py`    — `GoogleProvider` (Gemini via the OpenAI shim):
                     tool-result image splits, thought signatures,
                     cache markers disabled.
  - `anthropic.py` — `AnthropicProvider` (Claude): Anthropic-compat
                     wire shape via the official SDK.

`BASE_URL` is overridable per-instance via `~/.physiclaw/config.toml`'s
`[providers.<id>] base_url = "..."`.

Registry assembly (id → class map, lookup helpers) lives in
`provider/registry.py`.
"""
