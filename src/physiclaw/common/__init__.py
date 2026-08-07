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
    bbox           — bbox validation shared by core and agent
    gesture_vocab  — gesture tool names + sequence step keys
    listing        — element-listing grammar (core composes, agent parses)
    doctrine       — `{{token}}` rendering for doctrine delivery paths
    platform       — OS-specific branching (single source of truth)
    logger         — log formatting, tagging, and daily-file retention
    dumps          — image/artifact dumps (screenshots, tool calls)
    image          — LLM-bound image ingress cap (provider ingress; the
                     server encodes views to the same edge cap)

This package stays import-light: nothing here pulls a heavy dependency
at import time. ``dumps`` and ``image`` need ``cv2``, but import it
lazily inside the functions that use it, so importing them stays
cheap. Members are imported by full path
where needed rather than re-exported here.
"""
