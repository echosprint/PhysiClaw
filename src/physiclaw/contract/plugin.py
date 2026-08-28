"""The turn-plugin protocol — a peer package takes the turn before the LLM.

Each turn, the engine loop offers the request to its plugins in
registration order; the first to return an `AssistantMessage` has
SPOKEN the turn and the provider is never called. This is arbitration,
not a middleware chain: turns do not flow through the other plugins,
because the question being answered is "who produces this turn", not
"how is this request decorated". A plugin that returns None is passing,
and the engine falls through — with no plugins at all (config
`[agent] plugins = "none"`), every turn is a plain provider call, which
is the rollback switch.

The engine knows a turn is plugin-minted because it received it from
`advance()` — provenance is structural, never inferred from call-id
strings — and dispatches it under that provenance: the qualified
macros a plugin contributed at setup (`SessionSetup.gated_macros`)
resolve only on such turns, so the model can never borrow them.

Plugins are loaded from dotted `module:factory` paths in
`[agent] plugins` (comma-separated). Config registration is deliberate:
we own every plugin today, so a config string the engine loads blindly
is the whole discovery story. The upgrade trigger to entry-point
discovery is a plugin shipping as a separately installed distribution —
do not build it before then.

Everything here is in-process for now, but the shapes are
serialization-friendly on purpose (data in, one message out): a future
remote worker is a second transport for this same protocol, with the
two sink protocols becoming events returned in the advance response.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from physiclaw.contract.dto import AssistantMessage, Message
from physiclaw.macros.model import Macro


class ChatProvider(Protocol):
    """The slice of the session's provider a plugin may hold — enough to
    fall back to the session model (e.g. a cheap-tier decision call whose
    configured tier failed to build) and to wire-log its own scoped
    round-trips verbatim. Deliberately NOT the full provider: session
    management (collapse cadence, lifecycle close) stays the engine's.
    The full interface lives in `physiclaw.provider`; this protocol
    exists so the contract need not import it (provider imports
    contract, never the reverse)."""

    async def chat(
        self, history: list[Message], tools: list[dict]
    ) -> AssistantMessage: ...

    def serialize_history(self, messages: list[Message]) -> list[dict]: ...


class EventSink(Protocol):
    """`Trace.write`-shaped: one structured event into events.jsonl."""

    def write(self, event: dict) -> None: ...


class WireSink(Protocol):
    """`RawLog.write_micro`-shaped: one scoped LLM round-trip, verbatim."""

    def write_micro(
        self, call: str, request: list[dict], raw: dict[str, Any]
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class SetupContext:
    """What the engine offers a plugin at wake — exactly what the one
    real plugin reads, nothing speculative (grow it when a plugin reads
    more). The sinks are the one non-serializable crossing — acceptable
    in-process; a remote transport replaces them with events in the
    advance response."""

    session_provider: ChatProvider
    events: EventSink | None = None
    wire: WireSink | None = None


@dataclass(frozen=True, slots=True)
class SessionSetup:
    """A plugin's wake-time contribution. `gated_macros` are qualified
    `app/name` macros dispatchable ONLY while a plugin-minted turn is
    executing — they join the prompt bundle's macro surface but never
    its model-visible list."""

    gated_macros: dict[str, Macro] = field(default_factory=dict)


@runtime_checkable
class TurnPlugin(Protocol):
    """One turn arbiter. Implementations never let an exception escape
    `advance` — a plugin failure must degrade to "the LLM speaks", not
    kill the session — and the engine wraps the call fail-open anyway.
    """

    async def session_setup(self, ctx: SetupContext) -> SessionSetup | None: ...

    # No tool-schema parameter: a plugin mints turns from its own
    # vocabulary; the schemas ride `chat` to the provider. Add the
    # parameter when a plugin actually arbitrates on the tool surface.
    async def advance(self, history: list[Message]) -> AssistantMessage | None: ...

    async def aclose(self) -> None: ...
