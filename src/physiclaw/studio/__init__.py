"""PhysiClaw Studio — the browser skin for a person: drive the phone by
hand, step a playbook node by node, run a macro gesture by gesture.

A frontend only, in its own process, over a running `physiclaw mcp`
it never starts. Manual gestures are published MCP tool calls with
verbatim arguments; stepping goes through the same drivers the CLI
wraps (`debug/stepping.py`), on the same position file, so what a
person does here is exactly what an agent does at the terminal.
"""
