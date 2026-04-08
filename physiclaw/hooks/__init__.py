"""Built-in PhysiClaw runtime hooks.

Every module in this package is auto-imported by `Runtime.start()` via
`physiclaw.runtime.hook.load_hooks()`. To add a new hook, create a new
`.py` file here that uses `@register` from `physiclaw.runtime.hook` —
no other wiring required.
"""
