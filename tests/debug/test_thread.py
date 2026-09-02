"""Tests for `physiclaw.debug.thread` — the virtual thread file, the
staged-reply timing rule, and the listing renderer. The render must
satisfy the REAL consumers: the listing grammar (`Screen.read`), the
page matcher (`match_screen` against the channel pack's own
fingerprint), and the reply reader's bubble geometry
(`reply.new_incoming`)."""

from __future__ import annotations

import json

from conftest import write_channel_pages

from physiclaw.common import paths
from physiclaw.common.listing import Screen
from physiclaw.conductor import reply
from physiclaw.conductor.match import match_screen
from physiclaw.conductor.pages import THREAD_ID, prints_for_app
from physiclaw.debug import thread as vthread


def _render(bubbles) -> Screen:
    return Screen.read(vthread.render_listing(bubbles, vthread.thread_print()))


# ---------- the file ----------


def test_load_is_empty_when_missing() -> None:
    assert vthread.load() == vthread.Thread()


def test_seed_and_stage_round_trip_through_the_file() -> None:
    vthread.seed("buy milk", ["ok"])
    vthread.stage(["no thanks"])

    loaded = vthread.load()

    assert loaded.bubbles == [vthread.Bubble(sender=vthread.USER, text="buy milk")]
    assert loaded.staged == ["ok", "no thanks"]


def test_load_ignores_an_unknown_schema() -> None:
    paths.debug_dir().mkdir(parents=True, exist_ok=True)
    vthread.thread_path().write_text(
        json.dumps({"schema": 99, "bubbles": [{"from": "user", "text": "x"}]}),
        encoding="utf-8",
    )

    assert vthread.load() == vthread.Thread()


def test_load_ignores_unreadable_json() -> None:
    paths.debug_dir().mkdir(parents=True, exist_ok=True)
    vthread.thread_path().write_text("not json", encoding="utf-8")

    assert vthread.load() == vthread.Thread()


# ---------- staged-reply timing ----------


def test_staged_reply_waits_while_no_ask_is_outstanding() -> None:
    # Newest bubble is the user's own task — a released reply here would
    # answer a question nobody asked.
    vthread.seed("buy milk", ["ok"])

    bubbles = vthread.peek_bubbles()

    assert bubbles == [vthread.Bubble(sender=vthread.USER, text="buy milk")]
    assert vthread.load().staged == ["ok"]


def test_staged_reply_releases_on_the_peek_after_an_ask() -> None:
    vthread.seed("buy milk", ["ok"])
    vthread.record_send("Total is $4.50 — reply ok to confirm")

    bubbles = vthread.peek_bubbles()

    assert bubbles[-1] == vthread.Bubble(sender=vthread.USER, text="ok")
    assert vthread.load().staged == []


def test_record_send_itself_releases_nothing() -> None:
    # The send's result is the gate's baseline snapshot — a reply
    # released into it would be baselined away.
    vthread.seed("buy milk", ["ok"])

    bubbles = vthread.record_send("reply ok to confirm")

    assert bubbles[-1].sender == vthread.AGENT
    assert vthread.load().staged == ["ok"]


def test_one_release_per_ask_in_script_order() -> None:
    vthread.seed("buy milk", ["ok, but make it two boxes", "ok"])
    vthread.record_send("ask one")
    first = vthread.peek_bubbles()[-1]
    second = vthread.peek_bubbles()[-1]
    vthread.record_send("ask two")

    third = vthread.peek_bubbles()[-1]

    assert first == vthread.Bubble(
        sender=vthread.USER, text="ok, but make it two boxes"
    )
    assert second == first  # spent ask releases nothing more
    assert third == vthread.Bubble(sender=vthread.USER, text="ok")


# ---------- the render vs its real consumers ----------


def test_render_parses_as_listing_rows() -> None:
    write_channel_pages(("MyChat", "InputBox"))

    screen = _render([vthread.Bubble(vthread.USER, "buy milk")])

    labels = [r.label for r in screen.rows]
    assert "buy milk" in labels and "MyChat" in labels and "InputBox" in labels


def test_render_matches_the_channel_thread_fingerprint() -> None:
    write_channel_pages(("MyChat", "InputBox"))
    screen = _render([vthread.Bubble(vthread.USER, "buy milk")])

    verdict = match_screen(screen, prints_for_app("channel"))

    assert verdict.matches(THREAD_ID)


def test_user_bubble_reads_as_incoming_and_agent_ask_is_excluded() -> None:
    # The gate's exact read: baseline snapshot taken at ask time, then a
    # fresh user reply below the ask — new_incoming must return exactly it.
    write_channel_pages()
    ask = "Total is $4.50. Reply ok to confirm the payment."
    at_ask = _render(
        [
            vthread.Bubble(vthread.USER, "buy milk"),
            vthread.Bubble(vthread.AGENT, ask),
        ]
    )
    baseline = {r.label.strip() for r in at_ask.rows if r.label.strip()}
    after_reply = _render(
        [
            vthread.Bubble(vthread.USER, "buy milk"),
            vthread.Bubble(vthread.AGENT, ask),
            vthread.Bubble(vthread.USER, "ok"),
        ]
    )

    yes, no = frozenset({"ok", "好的"}), frozenset({"不用"})
    new = reply.new_incoming(after_reply.rows, baseline, ask)

    assert new == ["ok"]
    assert reply.classify_all(new, yes, no) == "confirm"


def test_multiline_bubble_renders_as_wrapped_rows() -> None:
    write_channel_pages()

    screen = _render([vthread.Bubble(vthread.USER, "buy milk\ntwo boxes")])

    labels = [r.label for r in screen.rows]
    assert "buy milk" in labels and "two boxes" in labels


def test_render_survives_a_missing_channel_pack() -> None:
    screen = _render([vthread.Bubble(vthread.USER, "buy milk")])

    assert [r.label for r in screen.rows] == ["buy milk"]


# ---------- wire shapes ----------


def test_view_blocks_lead_with_an_image() -> None:
    # Block 0 must NOT be text, or `verdict.action_text` would misread
    # the listing as composed action text (and `screen_text` drop it).
    blocks = vthread.view_blocks("listing")

    assert blocks[0]["type"] == "image"
    assert blocks[1] == {"type": "text", "text": "listing"}


def test_gesture_blocks_lead_with_the_action_text() -> None:
    blocks = vthread.gesture_blocks("did it | screen: changed", "listing")

    assert blocks[0] == {"type": "text", "text": "did it | screen: changed"}
    assert blocks[1]["type"] == "image"
    assert blocks[2] == {"type": "text", "text": "listing"}
