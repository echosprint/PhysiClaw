"""Tests for hardware/assembly/mark/server.py — the /save handler.

No real socket: a ``Handler`` is built bare (``__new__`` plus the
attributes ``BaseHTTPRequestHandler``'s send path needs) and ``do_POST``
is invoked directly, with the patch persistence monkeypatched to an
in-memory store so the read→modify→write window is observable.
"""

from __future__ import annotations

import io
import json
import threading
from types import SimpleNamespace

import pytest

from hardware.assembly.mark import server

SAVE_BODY = json.dumps({"shapes": [], "viewBox": "0 0 10 10", "preop": "orig"}).encode()


def make_handler(body: bytes) -> server.Handler:
    """A Handler wired to in-memory streams instead of a socket."""
    h = server.Handler.__new__(server.Handler)
    h.command = "POST"
    h.path = "/save"
    h.request_version = "HTTP/1.1"
    h.requestline = "POST /save HTTP/1.1"
    h.client_address = ("127.0.0.1", 0)
    h.rfile = io.BytesIO(body)
    h.wfile = io.BytesIO()
    h.headers = {"Content-Length": str(len(body))}
    return h


@pytest.fixture
def store(tmp_path, monkeypatch):
    """In-memory patch persistence for /save.

    ``load_patch`` / ``write_patch`` / ``upsert_entry`` and the replay
    chain are replaced with deterministic fakes over one shared entry
    list; ``in_load`` is a hook a test can use to dwell inside the
    read→modify→write window."""
    src = tmp_path / "belt_20_clamp_exploded_cam0.svg"
    src.write_bytes(b"<svg/>")
    monkeypatch.setattr(server.Handler, "src_path", src, raising=False)

    state = SimpleNamespace(entries=[], in_load=lambda: None)
    ids = iter(["aaaa", "bbbb", "cccc", "dddd"])
    id_lock = threading.Lock()

    def fake_load(src_path):
        entries = [dict(e) for e in state.entries]
        state.in_load()
        return entries

    def fake_write(src_path, entries):
        state.entries = entries
        return tmp_path / "belt_20_clamp_exploded_cam0.json"

    def fake_upsert(src_path, entries, edit_id, preop, shapes, viewbox):
        with id_lock:
            op_id = next(ids)
        entry = {"id": op_id, "preop": preop, "shapes": shapes, "viewBox": viewbox}
        return op_id, [*entries, entry]

    monkeypatch.setattr(server, "load_patch", fake_load)
    monkeypatch.setattr(server, "write_patch", fake_write)
    monkeypatch.setattr(server, "upsert_entry", fake_upsert)
    monkeypatch.setattr(server, "chain_to", lambda entries, op_id: [])
    monkeypatch.setattr(server, "apply_chain", lambda src_bytes, chain: b"<svg/>")
    monkeypatch.setattr(
        server, "snapshot_path", lambda src_path, op_id: tmp_path / f"snap_{op_id}.svg"
    )
    return state


def test_save_appends_the_op_and_replies_with_its_id(store):
    handler = make_handler(SAVE_BODY)

    handler.do_POST()

    response = handler.wfile.getvalue()
    assert b" 200 " in response.splitlines()[0]
    assert b"X-Op-Id: aaaa" in response
    assert [e["id"] for e in store.entries] == ["aaaa"]


def test_concurrent_saves_do_not_lose_an_op(store):
    # ThreadingHTTPServer runs each /save on its own thread. Two saves that
    # both read the patch before either writes would each upsert their own
    # op and the last write would clobber the other — _SAVE_LOCK must
    # serialize the whole read→modify→write.
    #
    # The barrier releases the two threads together ONLY if both can sit
    # inside load_patch at once (the lost-update interleaving). With the
    # lock held, the second thread can't reach it: the first times out,
    # proceeds alone, and both ops land.
    barrier = threading.Barrier(2)

    def dwell():
        try:
            # Short timeout: with a correct lock the second thread can
            # NEVER arrive, so this wait always times out — it prices
            # the happy path, not the detection (a broken lock releases
            # the barrier instantly).
            barrier.wait(timeout=0.1)
        except threading.BrokenBarrierError:
            pass

    store.in_load = dwell
    errors: list[Exception] = []

    def save():
        try:
            make_handler(SAVE_BODY).do_POST()
        except Exception as exc:  # surface thread failures to the test
            errors.append(exc)

    threads = [threading.Thread(target=save) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert sorted(e["id"] for e in store.entries) == ["aaaa", "bbbb"]
