"""Tests for `physiclaw.studio.record` — recording brains, the step
editor, emission through the real macro parser, and save."""

from __future__ import annotations

import pytest

from physiclaw.common import paths
from physiclaw.macros.model import MACRO_FILENAME, MacroError
from physiclaw.macros.parse import parse_macro
from physiclaw.studio import draft as ds
from physiclaw.studio import record
from physiclaw.studio.draft import DraftError

LISTING = '1 [text] "搜索" [0.100,0.050,0.200,0.090] 0.95'
TAP = {"label": "搜索", "bbox": [0.1, 0.05, 0.2, 0.09]}


def _armed(app: str = "shopdemo") -> dict:
    d = ds.load_draft(app)
    record.add_macro(d, "search")
    record.start_recording(d, "search")
    return d


def _record(d: dict, tool: str = "tap", args: dict | None = None) -> int:
    snap = ds.save_snap(d, "aGk=", LISTING)
    return record.record_step(d, tool, dict(TAP) if args is None else args, snap)


# ---------- recording ----------


def test_add_macro_refuses_duplicates_and_bad_names() -> None:
    d = ds.load_draft("shopdemo")
    record.add_macro(d, "search")

    with pytest.raises(DraftError, match="already drafted"):
        record.add_macro(d, "search")
    with pytest.raises(DraftError):
        record.add_macro(d, "Bad Name")


@pytest.mark.parametrize(
    ("tool", "args", "expected"),
    [
        ("tap", TAP, True),
        ("go_back", {}, True),
        ("swipe", {"bbox": [0.4, 0.4, 0.6, 0.6], "direction": "up"}, True),
        ("send_to_clipboard", {"text": "hi"}, True),
        ("peek", {}, False),  # a view, not a gesture
        ("unlock_phone", {}, False),  # outside ALLOWED_STEP_TOOLS
    ],
)
def test_check_recordable_vocabulary(tool, args, expected) -> None:
    d = _armed()

    assert record.check_recordable(d, tool, args) is expected


def test_check_recordable_is_false_when_disarmed() -> None:
    d = _armed()
    record.stop_recording(d)

    assert record.check_recordable(d, "tap", TAP) is False


def test_a_recorded_press_requires_its_label_before_the_arm_moves() -> None:
    d = _armed()

    with pytest.raises(DraftError, match="label"):
        record.check_recordable(d, "tap", {"bbox": [0.1, 0.1, 0.2, 0.2]})


def test_record_step_appends_and_stays_armed() -> None:
    d = _armed()

    assert _record(d) == 0
    assert _record(d, "go_back", {}) == 1

    steps = d["macros"]["search"]["steps"]
    assert [s["name"] for s in steps] == ["step-1", "step-2"]
    assert steps[0]["with"] == TAP and "with" not in steps[1]
    assert d["recording"] is not None


def test_rerecord_replaces_in_place_keeps_the_address_and_disarms() -> None:
    d = _armed()
    _record(d)
    old_snap = d["macros"]["search"]["steps"][0]["snap"]
    record.start_recording(d, "search", replace=0)

    snap = ds.save_snap(d, "aGk=", LISTING)
    index = record.record_step(d, "double_tap", dict(TAP), snap)

    step = d["macros"]["search"]["steps"][0]
    assert index == 0
    assert step["tool"] == "double_tap" and step["name"] == "step-1"
    assert d["recording"] is None
    # The replaced snapshot left the registry and the disk.
    assert old_snap not in d["shots"]


# ---------- the step editor ----------


def test_update_step_validates_and_rolls_back() -> None:
    d = _armed()
    _record(d)

    with pytest.raises(MacroError):
        record.update_step(d, "search", 0, {"name": "Bad Name"})

    assert d["macros"]["search"]["steps"][0]["name"] == "step-1"
    record.update_step(d, "search", 0, {"name": "open-search", "comment": "the box"})
    assert d["macros"]["search"]["steps"][0]["name"] == "open-search"


def test_expect_step_is_the_assertion_grammar() -> None:
    d = _armed()
    _record(d)

    record.insert_step(
        d, "search", None, record.expect_step("综合", [0.1, 0.1, 0.3, 0.15])
    )

    step = d["macros"]["search"]["steps"][1]
    assert step["tool"] == "wait"
    assert step["with"] == {"seconds": record.EXPECT_WAIT_SECONDS}
    assert step["expect"] == {"text": "综合", "within": [0.05, 0.05, 0.35, 0.2]}


def test_mark_dismissal_turns_the_tap_into_skip_when() -> None:
    d = _armed()
    _record(d)
    _record(d, "go_back", {})

    record.mark_dismissal(d, "search", 0)

    assert d["macros"]["search"]["steps"][0]["skip_when"] == {"not": "搜索"}
    with pytest.raises(DraftError, match="labeled press"):
        record.mark_dismissal(d, "search", 1)


def test_delete_step_drops_its_snap() -> None:
    d = _armed()
    _record(d)
    snap = d["macros"]["search"]["steps"][0]["snap"]

    record.delete_step(d, "search", 0)

    assert d["macros"]["search"]["steps"] == []
    assert snap not in d["shots"]


# ---------- emission / save ----------


def test_emitted_yaml_parses_with_inputs_and_placeholders() -> None:
    d = _armed()
    # Declare the input FIRST — a `{query}` placeholder in a step is
    # refused (loudly, by the real parser) until it exists.
    record.update_macro(
        d,
        "search",
        description="search taobao",
        inputs={"query": {"description": "what to search", "example": "iphone case"}},
    )
    _record(d, "send_to_clipboard", {"text": "{query}"})
    _record(d)
    record.update_step(d, "search", 1, {"comment": "the search box"})

    spec = record.macro_spec("search", d["macros"]["search"])

    assert spec.name == "search" and spec.enabled is False
    assert [i.name for i in spec.inputs] == ["query"]
    assert len(spec.steps) == 2
    yaml_text = record.emit_macro_yaml("search", d["macros"]["search"])
    assert "# the search box" in yaml_text


@pytest.mark.parametrize(
    ("target", "root"),
    [
        ("pack", lambda: paths.playbooks_dir() / "shopdemo" / "macros"),
        ("global", lambda: paths.macros_dir()),
    ],
)
def test_save_macro_writes_to_the_target(target, root) -> None:
    d = _armed()
    _record(d)
    record.update_macro(d, "search", target=target)

    path = record.save_macro("shopdemo", "search", d["macros"]["search"])

    assert path == root() / "search" / MACRO_FILENAME
    assert parse_macro(path.read_text(), "search").enabled is False


def test_update_macro_rejects_a_bad_target_and_bad_inputs() -> None:
    d = _armed()
    _record(d)

    with pytest.raises(DraftError, match="pack or global"):
        record.update_macro(d, "search", target="repo")
    with pytest.raises(MacroError):
        record.update_macro(d, "search", inputs={"Bad Name": {"description": "x"}})

    assert d["macros"]["search"]["inputs"] == {}


def test_delete_macro_disarms_and_drops_its_snaps() -> None:
    d = _armed()
    _record(d)
    snap = d["macros"]["search"]["steps"][0]["snap"]

    record.delete_macro(d, "search")

    assert d["recording"] is None and d["macros"] == {}
    assert snap not in d["shots"]
