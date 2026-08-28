"""Tests for `physiclaw.macros.runlog` — the per-step forensic
log: run ids, the per-run dir layout (events.jsonl + images/), event
shapes, and the fail-open posture."""

from __future__ import annotations

import json
import re

from physiclaw.common import paths
from physiclaw.macros import runlog

IMAGE = {"type": "image", "mime_type": "image/jpeg", "data": "aGk="}


def _lines(rl: runlog.RunLogger) -> list[dict]:
    return [
        json.loads(line)
        for line in (rl.dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]


# ---------- run ids ----------


def _events(rl):
    import json

    lines = (rl.dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(x) for x in lines]


def test_new_run_id_shape() -> None:
    rid = runlog.new_run_id()

    assert re.fullmatch(r"macro-run-[0-9a-f]{6}", rid)


def test_run_ids_are_unique() -> None:
    assert runlog.new_run_id() != runlog.new_run_id()


# ---------- dir layout ----------


def test_logger_creates_the_run_dir_with_images() -> None:
    rl = runlog.RunLogger("demo", "cli")

    assert rl.dir == paths.macros_log_dir() / rl.run_id
    assert (rl.dir / "images").is_dir()


# ---------- event emission ----------


def test_logger_emits_start_step_end_under_one_id() -> None:
    rl = runlog.RunLogger("demo", "cli")

    rl.start({"message": "hi"})
    rl.step(1, "tap", "open", "ok", args={"bbox": [0.1, 0.2, 0.3, 0.4]}, ms=42)
    rl.end(ok=True)

    events = _lines(rl)
    assert [e["event"] for e in events] == ["start", "step", "end"]
    assert {e["run"] for e in events} == {rl.run_id}
    assert {e["macro"] for e in events} == {"demo"}


def test_start_records_caller_and_null_session_for_cli() -> None:
    rl = runlog.RunLogger("demo", "cli")

    rl.start({"message": "hi"})

    start = _lines(rl)[0]
    assert start["caller"] == "cli"
    assert start["session"] is None
    assert start["inputs"] == {"message": "hi"}


def test_start_picks_up_live_session_marker() -> None:
    marker = paths.active_session_marker()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("20260803-101112-b7c465\n", encoding="utf-8")
    rl = runlog.RunLogger("demo", "engine")

    rl.start({})

    assert _lines(rl)[0]["session"] == "20260803-101112-b7c465"


def test_step_records_outcome_verdict_and_guard_fields() -> None:
    rl = runlog.RunLogger("demo", "engine")

    rl.step(
        3,
        "tap",
        "open chat",
        "guard_failed",
        verdict=None,
        guard_polls=2,
        detail="require '微信' not on screen",
        screen_text="Settings only",
        ms=4100,
    )

    step = _lines(rl)[0]
    assert step["outcome"] == "guard_failed"
    assert step["guard_polls"] == 2
    assert step["screen_text"] == "Settings only"
    assert step["verdict"] is None


def test_step_saves_the_view_image_and_references_it() -> None:
    rl = runlog.RunLogger("demo", "cli")

    rl.step(2, "tap", "open", "ok", view=[{"type": "text", "text": "t"}, IMAGE])

    step = _lines(rl)[0]
    assert step["image"].endswith(".jpg")
    assert "_t2" in step["image"]  # named by step number
    assert (rl.dir / "images" / step["image"]).read_bytes() == b"hi"


def test_step_omits_image_when_view_has_none() -> None:
    rl = runlog.RunLogger("demo", "cli")

    rl.step(1, "send_to_clipboard", "stage", "ok", view=[{"type": "text", "text": "t"}])

    assert "image" not in _lines(rl)[0]


def test_step_omits_screen_text_when_empty() -> None:
    rl = runlog.RunLogger("demo", "engine")

    rl.step(1, "tap", "open", "ok", verdict=True)

    step = _lines(rl)[0]
    assert "screen_text" not in step
    assert step["verdict"] == "changed"


def test_long_values_are_clipped() -> None:
    rl = runlog.RunLogger("demo", "cli")

    rl.start({"message": "x" * 500})
    rl.step(1, "tap", "n", "guard_failed", screen_text="y" * 2000)

    events = _lines(rl)
    assert len(events[0]["inputs"]["message"]) <= 120
    assert len(events[1]["screen_text"]) <= 500


def test_end_reports_outcome_and_duration() -> None:
    rl = runlog.RunLogger("demo", "cli")

    rl.end(ok=False, aborted_step=5, reason="guard_failed", detail="d")

    end = _lines(rl)[0]
    assert end["ok"] is False
    assert end["aborted_step"] == 5
    assert end["ms"] >= 0


# ---------- fail-open ----------


def test_emit_survives_unwritable_log_dir(monkeypatch) -> None:
    # A file where the dir should be — every write fails, nothing raises.
    blocker = paths.HOME / "log-blocked"
    blocker.parent.mkdir(parents=True, exist_ok=True)
    blocker.write_text("not a dir", encoding="utf-8")
    monkeypatch.setattr(paths, "macros_log_dir", lambda: blocker / "macros")

    rl = runlog.RunLogger("demo", "cli")
    rl.start({})
    rl.step(1, "tap", "n", "ok", view=[IMAGE])
    rl.end(ok=True)  # must not raise


# ---------- start_at is fully traceable ----------


def test_start_event_records_start_at() -> None:
    rl = runlog.RunLogger("demo", "engine")

    rl.start({}, start_at="focus-input-box")

    assert _events(rl)[0]["start_at"] == "focus-input-box"


def test_start_at_is_absent_on_a_whole_run() -> None:
    rl = runlog.RunLogger("demo", "engine")

    rl.start({"msg": "hi"})

    assert _events(rl)[0]["start_at"] == ""


def test_every_step_has_a_fate_recorded(tmp_path) -> None:
    # The trail must account for EVERY step, skipped ones included — a log
    # that jumps from step 1 to step 3 leaves the reader guessing whether
    # step 2 ran and failed silently.
    rl = runlog.RunLogger("demo", "engine")
    rl.start({}, start_at="third")
    rl.step(1, "home_screen", "first", "skipped", detail="start_at 'third'")
    rl.step(2, "tap", "second", "skipped", detail="start_at 'third'")
    rl.step(3, "tap", "third", "ok", guard_polls=1)

    steps = [e for e in _events(rl) if e["event"] == "step"]

    assert [s["i"] for s in steps] == [1, 2, 3]
    assert [s["outcome"] for s in steps] == ["skipped", "skipped", "ok"]
    # The peek that verified the entry state is recorded on the OK path too,
    # not just on failures — it is the evidence the resume was checked.
    assert steps[2]["guard_polls"] == 1
