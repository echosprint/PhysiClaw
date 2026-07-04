"""Tests for `physiclaw.agent.engine.screen_layout` — validate/accumulate.

The agent measures the boxes off a screenshot and reports them; this module
only sanity-checks and merges. No bridge fetch, no vision model.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from physiclaw.agent.engine import screen_layout as sl
from physiclaw.agent.engine.skill import Skill
from physiclaw import paths


@pytest.fixture(autouse=True)
def _isolate_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the layout paths at a per-test dir."""
    d = tmp_path / "screen-layout"
    monkeypatch.setattr(paths, "screen_layout_dir", lambda: d)
    monkeypatch.setattr(paths, "screen_layout_json", lambda: d / "layout.json")
    monkeypatch.setattr(paths, "screen_layout_md", lambda: d / "layout.md")
    return d


def _write_layout(fields: dict) -> None:
    d = paths.screen_layout_dir()
    d.mkdir(parents=True, exist_ok=True)
    paths.screen_layout_json().write_text(json.dumps(fields))


# Real-ish boxes that pass each field's region check.
_SPOTLIGHT = {
    "spotlight_input": [0.020, 0.582, 0.958, 0.660],  # near the bottom, above the keyboard
    "spotlight_paste": [0.090, 0.512, 0.205, 0.560],
    "space": [0.251, 0.868, 0.747, 0.920],
    "backspace": [0.867, 0.809, 0.995, 0.861],
    "return": [0.752, 0.868, 0.990, 0.918],
}
_CHAT_HIDDEN = {"chat_input_kb_hidden": [0.098, 0.900, 0.716, 0.950]}
_CHAT_VISIBLE_WECHAT = {
    "chat_input_kb_visible": [0.098, 0.586, 0.716, 0.634],
    "send": [0.752, 0.868, 0.990, 0.918],  # keyboard bottom-right key
    "chat_paste": [0.120, 0.520, 0.300, 0.566],
}

# Placeholder-value seeds derived from the field tables, so they track field
# additions automatically (is_learned/missing_pages check presence, not region).
_COMPLETE = {f: [0, 0, 1, 1] for f in sl.ALL_FIELDS}
_SPOTLIGHT_DONE = {f: [0, 0, 1, 1] for f in sl._PAGE_FIELDS["spotlight"]}


# ---------- missing_pages / is_learned ----------


def test_missing_pages_all_when_empty() -> None:
    assert sl.missing_pages() == list(sl.PAGES)
    assert sl.is_learned() is False


def test_missing_pages_shrinks_as_fields_captured() -> None:
    _write_layout(_SPOTLIGHT_DONE)
    assert "spotlight" not in sl.missing_pages()
    assert "chat-no-keyboard" in sl.missing_pages()
    assert sl.is_learned() is False


def test_is_learned_true_when_all_fields_present() -> None:
    _write_layout(_COMPLETE)
    assert sl.missing_pages() == []
    assert sl.is_learned() is True


def test_load_layout_md_returns_content_or_empty() -> None:
    assert sl.load_layout_md() == ""
    d = paths.screen_layout_dir()
    d.mkdir(parents=True, exist_ok=True)
    paths.screen_layout_md().write_text("LAYOUT CARD\n")
    assert sl.load_layout_md() == "LAYOUT CARD"


# ---------- prune_builtin_skills ----------


def test_prune_builtin_skills_keeps_screen_layout_while_incomplete() -> None:
    skills = {"im": object(), sl.SKILL_NAME: object()}
    assert sl.SKILL_NAME in sl.prune_builtin_skills(skills)


def test_prune_builtin_skills_drops_screen_layout_when_learned() -> None:
    _write_layout(_COMPLETE)
    skills = {"im": object(), sl.SKILL_NAME: object()}
    out = sl.prune_builtin_skills(skills)

    assert sl.SKILL_NAME not in out
    assert "im" in out  # other built-in skills untouched


# ---------- fill_builtin_boxes ----------


def _im_skill() -> Skill:
    # Prose uses the placeholders too; only the fenced sequence should fill.
    body = (
        "Prose: tap `{{input-hidden}}`, then `{{send}}`.\n"
        "```python\n"
        'sequence(step1={"arg": {{input-hidden}}}, step5={"arg": {{send}}})\n'
        "```\n"
    )
    return Skill(name="im", description="d", body=body, dir=Path("/x"), flat=True)


def test_fill_builtin_boxes_noop_while_incomplete() -> None:
    skills = {"im": _im_skill()}
    out = sl.fill_builtin_boxes(skills)

    assert out["im"].body == skills["im"].body       # unchanged
    assert "{{input-hidden}}" in out["im"].body        # placeholders stay


def test_fill_builtin_boxes_fills_code_block_only_leaves_prose() -> None:
    _write_layout({
        **{f: [0, 0, 1, 1] for f in sl.ALL_FIELDS},
        "chat_input_kb_hidden": [0.098, 0.900, 0.716, 0.950],
        "send": [0.752, 0.868, 0.990, 0.918],
    })
    out = sl.fill_builtin_boxes({"im": _im_skill()})
    prose, code = out["im"].body.split("```python")

    # Prose keeps the readable placeholder names.
    assert "{{input-hidden}}" in prose
    assert "{{send}}" in prose
    # The fenced template gets concrete bboxes, no placeholders left.
    assert "[0.098, 0.900, 0.716, 0.950]" in code
    assert "[0.752, 0.868, 0.990, 0.918]" in code
    assert "{{input-hidden}}" not in code
    assert "{{send}}" not in code


def test_fill_builtin_boxes_leaves_unmapped_skills_untouched() -> None:
    _write_layout(_COMPLETE)
    other = Skill(
        name="search-in-app", description="d",
        body="```python\n{{paste-button}}\n```", dir=Path("/x"),
    )
    out = sl.fill_builtin_boxes({"search-in-app": other})

    assert out["search-in-app"] is other              # same object, unfilled


def test_fill_builtin_boxes_open_app_uses_spotlight_fields() -> None:
    # {{paste-button}} is a PER-SKILL mapping: chat_paste in `im`,
    # spotlight_paste in `open-app`.
    _write_layout({
        **_COMPLETE,
        "spotlight_input": [0.005, 0.599, 0.982, 0.657],
        "spotlight_paste": [0.082, 0.561, 0.188, 0.583],
        "backspace": [0.864, 0.804, 0.986, 0.856],
    })
    skill = Skill(
        name="open-app", description="d", dir=Path("/x"),
        body=(
            "Prose {{search-field}} stays.\n"
            "```python\n"
            "{{search-field}} {{backspace}} {{paste-button}}\n"
            "```\n"
        ),
    )
    out = sl.fill_builtin_boxes({"open-app": skill})
    prose, code = out["open-app"].body.split("```python")

    assert "{{search-field}}" in prose
    assert "[0.005, 0.599, 0.982, 0.657]" in code   # spotlight_input
    assert "[0.864, 0.804, 0.986, 0.856]" in code   # backspace
    assert "[0.082, 0.561, 0.188, 0.583]" in code   # spotlight_paste
    assert "{{" not in code


# ---------- tail_reminder / inject_tail ----------


def test_tail_reminder_lists_missing_fields_per_page_when_nothing_captured() -> None:
    out = sl.tail_reminder()

    assert "First-run setup needed" in out
    # every page appears with its full field list, in _PAGE_FIELDS order
    for page, fields in sl._PAGE_FIELDS.items():
        assert f"{page}: {', '.join(fields)}" in out
    assert "`screen-layout` skill" in out  # defers the how-to to the skill


def test_tail_reminder_reports_done_and_missing_when_partial() -> None:
    # spotlight done; a chat-keyboard field partially captured → list the
    # specific fields still missing on each incomplete page.
    _write_layout({**_SPOTLIGHT_DONE, "chat_input_kb_visible": [0, 0, 1, 1]})

    out = sl.tail_reminder()

    assert "Pages fully captured: spotlight" in out
    assert "chat-no-keyboard: chat_input_kb_hidden" in out
    # chat-keyboard still needs send + chat_paste (input already captured)
    assert "chat-keyboard: send, chat_paste" in out


def test_tail_reminder_empty_when_all_captured() -> None:
    _write_layout(_COMPLETE)
    assert sl.tail_reminder() == ""


def test_inject_tail_appends_reminder_when_incomplete() -> None:
    from physiclaw.agent.engine.dto import UserMessage

    out = sl.inject_tail([])

    assert len(out) == 1
    assert isinstance(out[0], UserMessage)
    assert "First-run setup needed" in out[0].content


def test_inject_tail_noop_when_complete() -> None:
    _write_layout(_COMPLETE)
    original = ["a", "b"]
    assert sl.inject_tail(original) == original


# All fields except `send`, so a single chat-keyboard `send` call completes it.
_ALL_BUT_SEND = {**_SPOTLIGHT, **_CHAT_HIDDEN, **{k: v for k, v in _CHAT_VISIBLE_WECHAT.items() if k != "send"}}


# ---------- record: guards ----------


def test_record_rejects_unknown_page() -> None:
    assert "unknown page" in sl.record("bogus", "spotlight_input", [0.03, 0.08, 0.88, 0.13])


def test_record_chat_page_needs_app() -> None:
    out = sl.record("chat-no-keyboard", "chat_input_kb_hidden", [0.098, 0.9, 0.716, 0.95])  # no app
    assert "needs the IM app" in out
    assert not paths.screen_layout_json().exists()  # nothing saved


def test_record_rejects_field_not_on_page() -> None:
    out = sl.record("spotlight", "send", [0.75, 0.87, 0.99, 0.92])
    assert "not a spotlight field" in out
    assert not paths.screen_layout_json().exists()


def test_record_rejects_empty_bbox() -> None:
    out = sl.record("spotlight", "spotlight_input", [])
    assert "4 numbers" in out
    assert not paths.screen_layout_json().exists()


def test_record_rejects_bad_geometry() -> None:
    # left >= right.
    out = sl.record("spotlight", "spotlight_input", [0.9, 0.08, 0.3, 0.13])
    assert "left<right" in out
    assert "Layout not saved" in out


def test_record_rejects_out_of_region_box() -> None:
    # A spotlight search field can't sit at the very bottom of the screen.
    out = sl.record("spotlight", "spotlight_input", [0.03, 0.90, 0.88, 0.95])
    assert "looks off" in out
    assert not paths.screen_layout_json().exists()


# ---------- record: happy path ----------


def test_record_saves_one_box_and_confirms() -> None:
    out = sl.record("spotlight", "spotlight_input", [0.02, 0.582, 0.958, 0.66])

    saved = json.loads(paths.screen_layout_json().read_text())
    assert saved["spotlight_input"] == [0.02, 0.582, 0.958, 0.66]  # rounded, stored
    assert "Saved `spotlight_input`" in out
    assert "still to capture" in out.lower()  # incomplete; count only, no field list
    assert "space" not in out  # remaining fields NOT re-listed (tail_reminder does that)
    assert paths.screen_layout_md().exists()  # card persisted to disk


def test_record_saves_paste_buttons() -> None:
    sl.record("spotlight", "spotlight_paste", [0.090, 0.512, 0.205, 0.560])
    sl.record("chat-keyboard", "chat_paste", [0.120, 0.520, 0.300, 0.566], "wechat")

    saved = json.loads(paths.screen_layout_json().read_text())
    assert saved["spotlight_paste"] == [0.09, 0.512, 0.205, 0.56]
    assert saved["chat_paste"] == [0.12, 0.52, 0.3, 0.566]


def test_record_rejects_paste_out_of_region() -> None:
    # A spotlight Paste callout can't sit at the very bottom (keyboard area).
    out = sl.record("spotlight", "spotlight_paste", [0.37, 0.90, 0.55, 0.95])
    assert "looks off" in out


def test_render_md_includes_paste_section() -> None:
    md = sl._render_md(_COMPLETE)
    assert "### Paste buttons" in md


def test_record_merges_without_clobber_and_labels_app() -> None:
    _write_layout({"spotlight_input": [0.02, 0.582, 0.958, 0.66]})
    sl.record("chat-no-keyboard", "chat_input_kb_hidden", [0.098, 0.9, 0.716, 0.95], "wechat")

    saved = json.loads(paths.screen_layout_json().read_text())
    assert saved["spotlight_input"] == [0.02, 0.582, 0.958, 0.66]  # preserved
    assert saved["chat_input_kb_hidden"] == [0.098, 0.9, 0.716, 0.95]
    assert saved["im_app"] == "WeChat"  # app labelled


def test_record_rounds_coordinates() -> None:
    sl.record("spotlight", "spotlight_input", [0.0201, 0.5822, 0.9578, 0.6601])
    saved = json.loads(paths.screen_layout_json().read_text())
    assert saved["spotlight_input"] == [0.02, 0.582, 0.958, 0.66]


def test_record_ignores_app_on_spotlight() -> None:
    sl.record("spotlight", "spotlight_input", [0.02, 0.582, 0.958, 0.66], "wechat")
    saved = json.loads(paths.screen_layout_json().read_text())
    assert "im_app" not in saved  # spotlight isn't app-specific


def test_record_completes_setup_wechat() -> None:
    _write_layout(_ALL_BUT_SEND)
    out = sl.record("chat-keyboard", "send", [0.752, 0.868, 0.990, 0.918], "wechat")

    assert "All boxes captured" in out
    assert sl.is_learned() is True
    saved = json.loads(paths.screen_layout_json().read_text())
    assert saved["send"] == [0.752, 0.868, 0.99, 0.918]  # keyboard key
    assert saved["im_app"] == "WeChat"
    # The `layout_learned` marker is persisted for out-of-module readers (the
    # core orchestrator reads it without importing this module).
    assert saved["layout_learned"] is True


def test_record_writes_learned_false_while_incomplete() -> None:
    sl.record("spotlight", "spotlight_input", [0.02, 0.582, 0.958, 0.66])

    saved = json.loads(paths.screen_layout_json().read_text())
    assert saved["layout_learned"] is False


def test_record_completes_setup_whatsapp_send_from_mic() -> None:
    _write_layout(_ALL_BUT_SEND)
    out = sl.record("chat-keyboard", "send", [0.850, 0.586, 0.950, 0.634], "whatsapp")

    assert "All boxes captured" in out
    saved = json.loads(paths.screen_layout_json().read_text())
    assert saved["send"] == [0.85, 0.586, 0.95, 0.634]  # mic position
    assert saved["im_app"] == "WhatsApp"


def test_record_accepts_any_chat_app() -> None:
    # Send is validated app-independently (right side, lower half), so an
    # arbitrary app — here Telegram, input-bar send button — works and is
    # labelled with the app's name.
    _write_layout(_ALL_BUT_SEND)
    out = sl.record("chat-keyboard", "send", [0.90, 0.586, 0.98, 0.634], "telegram")

    assert "All boxes captured" in out
    saved = json.loads(paths.screen_layout_json().read_text())
    assert saved["im_app"] == "Telegram"


def test_record_labels_unknown_app_verbatim() -> None:
    sl.record("chat-no-keyboard", "chat_input_kb_hidden", [0.098, 0.9, 0.716, 0.95], "SomeNewChatApp")
    saved = json.loads(paths.screen_layout_json().read_text())
    assert saved["im_app"] == "SomeNewChatApp"  # not in the nice-case map → as passed


def test_record_rejects_send_far_from_right() -> None:
    # A Send box on the LEFT half is clearly a mis-pick, whatever the app.
    out = sl.record("chat-keyboard", "send", [0.10, 0.586, 0.20, 0.634], "signal")
    assert "send: looks off" in out


def test_record_same_field_twice_overwrites() -> None:
    # Re-reporting a field corrects it — the second value wins, no duplication.
    sl.record("spotlight", "spotlight_input", [0.02, 0.582, 0.958, 0.66])
    sl.record("spotlight", "spotlight_input", [0.03, 0.585, 0.955, 0.655])

    saved = json.loads(paths.screen_layout_json().read_text())
    assert saved["spotlight_input"] == [0.03, 0.585, 0.955, 0.655]  # overwritten, not duplicated


def test_record_completing_call_announces_restart() -> None:
    _write_layout(_ALL_BUT_SEND)
    out = sl.record("chat-keyboard", "send", [0.752, 0.868, 0.990, 0.918], "wechat")
    # The completing call tells the agent setup is done and the layout is now
    # loaded (restart if there's a task to finish, else the session just ends).
    assert "setup is done" in out
    assert "restarts with the layout loaded" in out


def test_record_after_complete_says_no_restart() -> None:
    # Layout already complete → re-reporting a box updates it but does NOT claim
    # a restart (the handler only restarts on the call that finishes setup).
    _write_layout({**_ALL_BUT_SEND, "send": [0.752, 0.868, 0.990, 0.918], "im_app": "WeChat"})
    out = sl.record("chat-keyboard", "send", [0.760, 0.868, 0.995, 0.918], "wechat")

    assert "No restart needed" in out
    assert "will now restart" not in out
    saved = json.loads(paths.screen_layout_json().read_text())
    assert saved["send"] == [0.76, 0.868, 0.995, 0.918]  # correction saved
