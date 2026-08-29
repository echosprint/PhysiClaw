"""Tests for `physiclaw.debug.interceptor` — the fake-channel result
transformer: which real results are rewritten to the virtual thread,
which pass through, and how the on-thread state moves. Provenance is
the session's synthesized-turn bit, passed by dispatch."""

from __future__ import annotations

from conftest import write_channel_pages

from physiclaw.common.verdict import SCREEN_CHANGED
from physiclaw.contract.dto import ToolCall
from physiclaw.debug import thread as vthread
from physiclaw.debug.interceptor import build

# What the REAL dispatch produced — the transformer receives it and
# (for channel actions) discards it for the virtual observation.
REAL_BLOCKS = [
    {"type": "text", "text": "real action | screen: changed"},
    {"type": "image", "data": "", "mimeType": "image/jpeg"},
    {"type": "text", "text": 'id [kind] "real row" [0.1,0.1,0.2,0.2] 0.9'},
]


def _call(name: str, args: dict | None = None) -> ToolCall:
    return ToolCall(id="conductor-walk-1-act", name=name, arguments=args or {})


def _open() -> ToolCall:
    return _call("run_macro", {"name": "channel/open"})


def _send(message: str = "Total is $4.50 — reply ok to confirm") -> ToolCall:
    return _call("run_macro", {"name": "channel/send", "inputs": {"message": message}})


def _listing_of(blocks):
    from physiclaw.common.listing import Screen

    return Screen.read(blocks[-1]["text"])


def test_channel_send_observation_is_rewritten_as_a_gesture_reply() -> None:
    write_channel_pages()
    fc = build()

    blocks = fc.intercept(
        _send("Total is $4.50 — reply ok to confirm"), True, REAL_BLOCKS
    )

    assert blocks is not None and blocks is not REAL_BLOCKS
    assert SCREEN_CHANGED in blocks[0]["text"]  # action text carries the verdict
    assert blocks[1]["type"] == "image"
    labels = [r.label for r in _listing_of(blocks).rows]
    assert "Total is $4.50 — reply ok to confirm" in labels
    assert "real row" not in labels  # the real observation is fully replaced


def test_channel_send_persists_the_agent_bubble() -> None:
    write_channel_pages()
    build().intercept(_send("ask text"), True, REAL_BLOCKS)

    assert vthread.load().bubbles == [
        vthread.Bubble(sender=vthread.AGENT, text="ask text")
    ]


def test_peek_after_channel_open_is_rewritten_as_a_view_reply() -> None:
    write_channel_pages()
    vthread.seed("buy milk", [])
    fc = build()
    fc.intercept(_open(), True, REAL_BLOCKS)

    blocks = fc.intercept(_call("peek"), True, REAL_BLOCKS)

    assert blocks is not None
    assert blocks[0]["type"] == "image"  # view shape: no action text block
    assert "buy milk" in [r.label for r in _listing_of(blocks).rows]


def test_gate_peek_carries_the_released_staged_reply() -> None:
    # The whole point of the seam: ask → peek → the staged confirm is
    # there, below the ask, fresh against the baseline.
    write_channel_pages()
    vthread.seed("buy milk", ["ok"])
    fc = build()
    fc.intercept(_send("reply ok to confirm"), True, REAL_BLOCKS)

    blocks = fc.intercept(_call("peek"), True, REAL_BLOCKS)

    assert "ok" in [r.label for r in _listing_of(blocks).rows]


def test_peek_before_any_channel_action_keeps_the_real_result() -> None:
    assert build().intercept(_call("peek"), True, REAL_BLOCKS) is None


def test_an_app_macro_passes_through_and_leaves_the_thread() -> None:
    write_channel_pages()
    fc = build()
    fc.intercept(_open(), True, REAL_BLOCKS)

    leg = fc.intercept(
        _call("run_macro", {"name": "taobao/open-app"}), True, REAL_BLOCKS
    )

    assert leg is None
    assert fc.intercept(_call("peek"), True, REAL_BLOCKS) is None  # off-thread again


def test_wait_keeps_the_thread_state() -> None:
    # The gate's ask → wait → peek cycle: the wait between polls must
    # not drop the conductor off the virtual thread.
    write_channel_pages()
    fc = build()
    fc.intercept(_send(), True, REAL_BLOCKS)

    assert fc.intercept(_call("wait", {"seconds": 45}), True, REAL_BLOCKS) is None
    assert fc.intercept(_call("peek"), True, REAL_BLOCKS) is not None


def test_model_turns_are_never_rewritten() -> None:
    write_channel_pages()

    blocks = build().intercept(_send(), False, REAL_BLOCKS)

    assert blocks is None


def test_a_harness_crash_fails_open_to_the_real_result(monkeypatch) -> None:
    write_channel_pages()
    fc = build()
    monkeypatch.setattr(
        vthread, "record_send", lambda message: (_ for _ in ()).throw(RuntimeError())
    )

    assert fc.intercept(_send(), True, REAL_BLOCKS) is None
