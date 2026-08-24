"""`RawLog` — per-session capture of the provider round-trips."""

import json
import logging
from typing import Any

from physiclaw.agent.trace import store
from physiclaw.common.logger import iso_now, save_image

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
        self._emit("request", turn=turn, messages=self._scrub_images(messages))

    def write_response(
        self,
        turn: int,
        raw: dict[str, Any],
        *,
        elapsed_ms: int,
    ) -> None:
        self._emit("response", turn=turn, elapsed_ms=elapsed_ms, raw=raw)

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

    def _scrub_images(self, messages: list[dict]) -> list[dict]:
        """Copy of `messages` with inline base64 image data replaced by
        a reference to an on-disk file under `images/<HHMMSS>_<mmm>_t<turn>.ext`
        in the session dir. No cross-request dedup, by design: a screen still
        in context is re-persisted each request it appears in (a fresh stamp
        each time), so the frames sort chronologically on disk for debugging.

        Handles two wire shapes (recognized at the block level, not the
        provider level):

          - OpenAI: `{"type": "image_url", "image_url": {"url": "data:..."}}`
          - Anthropic: `{"type": "image", "source": {"type": "base64",
            "media_type": "...", "data": "..."}}`

        On decode failure, falls back to a byte-count stub so the raw
        log still distinguishes an image from a tap result."""
        out: list[dict] = []
        for m in messages:
            c = m.get("content")
            if not isinstance(c, list):
                out.append(m)
                continue
            new_c: list[dict] = []
            for b in c:
                if isinstance(b, dict):
                    new_c.append(self._scrub_block(b))
                else:
                    new_c.append(b)
            out.append({**m, "content": new_c})
        return out

    def _scrub_block(self, b: dict) -> dict:
        """Scrub one content block; handles OpenAI `image_url`, Anthropic
        `image`, and Anthropic `tool_result` (whose nested `content` may
        itself contain image blocks). Pass-through for everything else
        (text, tool_use, …)."""
        bt = b.get("type")
        if bt == "image_url":
            url = (b.get("image_url") or {}).get("url", "")
            if not url.startswith("data:"):
                return b
            head, _, data = url.partition(",")
            mime = head[5:].partition(";")[0]
            rel = self._persist_image(mime, data) if data else ""
            scrubbed = rel or f"{head},<{len(data)}b unreadable>"
            return {"type": "image_url", "image_url": {"url": scrubbed}}
        if bt == "image":
            src = b.get("source") or {}
            if src.get("type") != "base64":
                return b
            data = src.get("data") or ""
            mime = src.get("media_type") or "image/jpeg"
            rel = self._persist_image(mime, data) if data else ""
            scrubbed_src = (
                {"type": "ref", "ref": rel}
                if rel
                else {"type": "base64", "byte_count": len(data)}
            )
            return {"type": "image", "source": scrubbed_src}
        if bt == "tool_result":
            inner = b.get("content")
            if isinstance(inner, list):
                scrubbed_inner = [
                    self._scrub_block(x) if isinstance(x, dict) else x for x in inner
                ]
                return {**b, "content": scrubbed_inner}
            return b
        return b


# ---------- reading wire.jsonl back ----------


def iter_request_texts(path):
    """Every ``(message_role, text)`` in a wire.jsonl's request records —
    the reader beside the writer: `_scrub_block` above encodes the same
    two wire shapes on the way out (top-level text blocks vs Anthropic
    `tool_result` blocks whose `content` nests its own block list), so a
    provider shape added there must be added here in the same edit.
    Streams the file; skips unparseable lines."""
    import json as _json

    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = _json.loads(line)
            except ValueError:
                continue
            if rec.get("kind") != "request":
                continue
            for msg in rec.get("messages", ()):
                role = msg.get("role", "")
                for text in _iter_texts(msg.get("content")):
                    yield role, text


def _iter_texts(content):
    if isinstance(content, str):
        yield content
        return
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            yield block.get("text", "")
        elif block.get("type") == "tool_result":
            yield from _iter_texts(block.get("content"))
