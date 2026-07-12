"""Shared, domain-agnostic infrastructure for PhysiClaw.

Everything in this package is free of hardware, vision, and agent
knowledge, so both the ``physiclaw.core`` (phone-control) and
``physiclaw.agent`` (LLM loop) layers can depend on it *downward*
without depending on each other. That one-way dependency is what keeps
core and agent decoupled — see ``tests/test_architecture.py``.

Members:
    text           — file read/write helpers
    paths          — data-root (``~/.physiclaw``) resolver
    config         — ``config.toml`` load / access
    proxy          — proxy env-var normalization for httpx
    runtime_state  — live-server state file (host/port)
    verdict        — screen-change markers shared by core and agent
    platform       — OS-specific branching (single source of truth)
    logger         — log formatting, tagging, and daily-file retention
    dumps          — image/artifact dumps (screenshots, tool calls)

This package stays import-light: nothing here pulls a heavy dependency
at import time. ``dumps`` needs ``cv2`` for two of its four helpers, but
imports it lazily inside those functions, so even ``import
physiclaw.common.dumps`` stays cheap. Members are imported by full path
where needed rather than re-exported here.
"""
