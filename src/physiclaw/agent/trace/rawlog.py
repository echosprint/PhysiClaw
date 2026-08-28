"""`RawLog` — per-session capture of the provider round-trips."""

import json
import logging
from typing import Any

from physiclaw.agent.trace import store
from physiclaw.common.logger import iso_now, save_image
from physiclaw.contract import wire

log = logging.getLogger(__name__)


class RawLog:
    """Per-session JSONL sink for later analysis.

    Emits `session_start` once, then one line per provider round-trip
    (request OR response). Open inside the engine's try/finally — call
    `close()` on session end.
    """

    def __init__(self, session_id: str):
        d = store._session_dir(session_id)
        self._image_dir = d / "images"
        self._image_dir.mkdir(parents=True, exist_ok=True)
        store._purge_old()
        self.session_id = session_id
        self.path = d / "wire.jsonl"
        self._f = open(self.path, "a", encoding="utf-8", newline="\n")
        # The turn currently being scrubbed — set by write_request before
        # _scrub_images so extracted images carry their turn in the name.
        self._turn = -1

    def write_session_start(
        self,
        *,
        provider: str,
        model: str,
        prompt_hash: str,
        tools: list[dict],
    ) -> None:
        # Tools don't change mid-session (engine builds the registry once
        # at bootstrap), so logging them once at start is sufficient and
        # keeps per-turn records lean.
        self._emit(
            "session_start",
            provider=provider,
            model=model,
            prompt_hash=prompt_hash,
            tools=tools,
        )

    def write_request(self, turn: int, messages: list[dict]) -> None:
        self._turn = turn
        scrubbed = wire.scrub_messages(messages, self._persist_image)
        self._emit("request", turn=turn, messages=scrubbed)

    def write_response(
        self,
        turn: int,
        raw: dict[str, Any],
        *,
        elapsed_ms: int,
        synthesized: bool = False,
    ) -> None:
        # Wire fidelity: a synthesized response was composed by the
        # conductor — nothing was sent to the provider, and no request
        # record precedes it (the loop skips the request write entirely).
        extra = {"synthesized": True} if synthesized else {}
        self._emit("response", turn=turn, elapsed_ms=elapsed_ms, **extra, raw=raw)

    def write_micro(self, call: str, request: list[dict], raw: dict[str, Any]) -> None:
        """One conductor micro-call round-trip — the exact prompt and the
        raw reply in a single record (kind "micro"). Small fixed-shape
        contexts, no images, so no scrubbing pass is needed."""
        self._emit("micro", call=call, request=request, raw=raw)

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()

    def _emit(self, kind: str, **data: Any) -> None:
        """Fail-open like the sibling sinks (`Trace._write_event`,
        `DailyLogWriter.line`): a full disk or an unserializable payload
        costs the wire record, never the session."""
        obj = {"t": iso_now(), "kind": kind, **data}
        try:
            self._f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self._f.flush()
        except (OSError, TypeError, ValueError):
            log.warning("wire.jsonl write failed", exc_info=True)

    def _persist_image(self, mime: str, b64_data: str) -> str:
        """Decode `b64_data`, write to
        `sessions/<sid>/images/<HHMMSS>_<mmm>_t<turn><ext>`, return the path
        relative to the session dir (so wire.jsonl + images move together
        when the dir is copied). The `image_filename` stamp sorts the frames
        chronologically and its `_t<turn>` tag links each straight to its
        turn. Returns "" on decode failure so the caller can fall back to a
        byte-count stub."""
        name = save_image(self._image_dir, self._turn, mime, b64_data)
        return f"images/{name}" if name else ""
