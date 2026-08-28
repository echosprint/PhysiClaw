"""Engine — provider-agnostic tool-use loop (low-level replacement for `claude -p`).

Submodules import lazily — there's no eager re-export of `engine.run`
because an eager one once triggered a cycle with the provider package
(before the message shapes moved to `physiclaw.contract.dto`). Callers
should `from physiclaw.agent.engine.engine import run` directly.
"""
