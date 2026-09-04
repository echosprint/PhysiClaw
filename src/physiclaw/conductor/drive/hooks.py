"""The callable seams a driver takes — typed once, so every skin and
every test passes the same shapes.

A rehearsal, a stepped node, or a macro run is driven by four hooks:
where progress lines go, who sees every real result, who may rewrite
a result before the walk reads it (the virtual channel), and who
receives each model round-trip. None of them return anything the
driver reads back, except `Transform`, whose non-None answer replaces
the result. The phone side is `macros.runner.McpCaller`, the one MCP
shape the whole tree drives through.
"""

from typing import Protocol

from physiclaw.contract.dto import ToolCall
from physiclaw.macros.steps import McpCaller

__all__ = ["Emit", "Observe", "OnExchange", "Transform", "McpCaller"]


class Emit(Protocol):
    """One progress line for the eye."""

    def __call__(self, line: str, /) -> None: ...


class Observe(Protocol):
    """Every real result, before any rewrite — the studio renders the
    phone off it."""

    def __call__(self, call: ToolCall, blocks: list[dict], /) -> None: ...


class Transform(Protocol):
    """A result rewrite over the dispatch seam: None keeps the real
    blocks, a list replaces them (the debug fake-channel)."""

    def __call__(self, call: ToolCall, blocks: list[dict], /) -> list[dict] | None: ...


class OnExchange(Protocol):
    """One model round-trip record (`rehearsal.exchanges`)."""

    def __call__(self, record: dict, /) -> None: ...
