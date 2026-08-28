"""The contract — shapes shared across the agent's package seams.

Everything here is data or protocol, never behavior: the packages on
either side of a seam depend on this one so they need not depend on
each other (the agent ↔ conductor split relies on it — see
`tests/test_architecture.py` for the enforced graph).

    dto.py     — the message/tool-call shapes the engine loop, the
                 providers, and every turn plugin all speak
    wire.py    — the wire.jsonl text codec: the scrubber the RawLog
                 writes through and the reader the corpus extracts
                 with, kept beside each other so a provider shape
                 added to one is added to the other in the same edit
    plugin.py  — the turn-plugin protocol: how a peer package takes
                 the turn before the LLM sees it

Imports: `common` (and, for macro types in the plugin protocol, the
macros package). Never `agent`, never `conductor` — both of those
import *this*.
"""
