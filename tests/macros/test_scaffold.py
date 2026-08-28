"""Tests for `physiclaw.macros.scaffold` — the authoring texts:
the rendered `init` template and the README, both documentation-as-
artifact that must stay true to the live format."""

from __future__ import annotations

from physiclaw.macros.model import (
    ALLOWED_STEP_TOOLS,
    MAX_INPUTS,
    MAX_STEPS,
    MAX_WAIT_SECONDS,
    Macro,
    OrClause,
)
from physiclaw.macros.parse import parse_macro
from physiclaw.macros.scaffold import README_CONTENT, render_init
from physiclaw.macros.steps import Step


def _step(spec: Macro, name: str) -> Step:
    """Address a scaffold step by name — the assertions below are about
    each step's shape, and should not break when one is inserted."""
    return next(s for s in spec.steps if s.name == name)


# ---------- render_init ----------


def test_render_init_parses_clean_and_disabled() -> None:
    # The template IS the format documentation and must pass `check`
    # unedited — with `enabled: false` so it can't go live by accident.
    text = render_init("my-macro")

    spec = parse_macro(text, "my-macro")

    assert spec.name == "my-macro"
    assert spec.enabled is False


def test_render_init_declares_the_example_input() -> None:
    spec = parse_macro(render_init("my-macro"), "my-macro")

    assert [i.name for i in spec.inputs] == ["message"]
    assert spec.inputs[0].required is True


def test_render_init_documents_all_guard_shapes() -> None:
    # The check forms (any-of list, region-scoped mapping, forbid, the
    # wait+expect settle, hint) must all appear — teaching by example.
    text = render_init("my-macro")

    assert 'or: ["WeChat", "Weixin"' in text  # any-of alternatives
    assert 'text: "Paste", within: [0.03, 0.5, 0.35, 0.62]' in text  # region-scoped
    assert "forbid:" in text
    assert "tool: wait" in text
    assert "seconds:" in text
    assert "expect:" in text
    assert "hint:" in text


def test_render_init_is_the_real_wechat_flow_stopping_before_send() -> None:
    # Best practice demonstrated, not described: dock icon → settle →
    # thread → clipboard → focus → long-press → paste, and NO send step —
    # the commit action stays with the agent. The settle is one `wait`
    # carrying an `expect`: pause, then confirm, in one camera read.
    spec = parse_macro(render_init("my-macro"), "my-macro")

    assert [s.tool for s in spec.steps] == [
        "home_screen",
        "tap",
        "wait",
        "tap",
        "send_to_clipboard",
        "tap",
        "long_press",
        "tap",
    ]
    assert spec.steps[-1].name == "paste"
    assert not any("send" in s.name.lower() for s in spec.steps)
    assert "NO send step" in render_init("my-macro")


def test_render_init_focus_step_skips_when_keyboard_already_up() -> None:
    # The idempotence pattern demonstrated in place: tapping the
    # keyboard-hidden input-box position with the keyboard up hits keys.
    # Anchor: ANY letter key in the letter-rows zone (IME-agnostic) —
    # safe because single-char texts match whole element labels only.
    spec = parse_macro(render_init("my-macro"), "my-macro")

    focus = _step(spec, "focus-input-box")

    assert focus.skip_when is not None
    clause = focus.skip_when
    # The property, not the literal: an any-of over the space-bar label,
    # every alternative region-scoped to the keyboard band so the word
    # cannot match inside a chat bubble.
    assert isinstance(clause, OrClause)
    assert all(c.within is not None for c in clause.children)
    assert {c.text for c in clause.children} == {"space", "空格"}


def test_render_init_thread_step_skips_when_already_in_the_chat() -> None:
    # WeChat often reopens inside the last chat — the thread tap must
    # skip when the title bar already shows the user (placeholder to edit).
    spec = parse_macro(render_init("my-macro"), "my-macro")

    thread = _step(spec, "open-chat")

    assert thread.skip_when is not None
    assert thread.skip_when.text == "your-user-name"
    assert thread.skip_when.within is not None  # scoped to the title bar


def test_render_init_waits_for_the_launch_on_a_settle_step() -> None:
    # THE check-placement rule, demonstrated in place: a guard runs BEFORE
    # its own step, so "WeChat is open" cannot ride on the tap that opens
    # it. It is a POSTcondition, so it belongs in an `expect` — carried by
    # the `wait` that also buys the cold start.
    spec = parse_macro(render_init("my-macro"), "my-macro")

    opener, settle = _step(spec, "open-wechat"), _step(spec, "await-wechat")

    assert opener.guard is None
    assert settle.tool == "wait"
    assert settle.seconds >= 1
    assert settle.guard is None  # the check is a postcondition, not a gate
    assert settle.expect is not None
    assert isinstance(settle.expect, OrClause)
    assert all(c.within is not None for c in settle.expect.children)
    assert settle.hint


def test_render_init_settle_guard_accepts_both_landing_states() -> None:
    # WeChat reopens where it was left, so the title bar reads the app name
    # OR the chat's. A settle `expect` listing only the app name aborts
    # exactly the case the NEXT step's skip_when exists to absorb — the two
    # must agree, so pin them to each other rather than to literals.
    spec = parse_macro(render_init("my-macro"), "my-macro")

    settle, thread = _step(spec, "await-wechat"), _step(spec, "open-chat")
    assert settle.expect is not None

    accepted = {c.text for c in settle.expect.children}

    assert {"WeChat", "Weixin"} <= accepted  # landed on the chat list
    assert thread.skip_when is not None
    assert thread.skip_when.text in accepted  # landed inside the chat


def test_render_init_guards_the_fixed_row_tap_and_the_paste() -> None:
    # Both remaining guards are region-scoped, and each sits on the step
    # that NEEDS the state: the row must be where the bbox points before
    # tapping it, the paste bubble must be up before tapping Paste.
    spec = parse_macro(render_init("my-macro"), "my-macro")

    row_guard = _step(spec, "open-chat").guard
    paste_guard = _step(spec, "paste").guard

    assert row_guard is not None
    assert row_guard.require.within is not None
    assert paste_guard is not None
    assert paste_guard.require.within is not None  # region-scoped bubble


def test_render_init_derives_the_tool_list() -> None:
    # No hand-copied tool names — the whitelist interpolates.
    text = render_init("my-macro")

    assert "  ".join(sorted(ALLOWED_STEP_TOOLS)) in text


# ---------- README_CONTENT ----------


def test_readme_derives_tools_and_caps() -> None:
    assert ", ".join(sorted(ALLOWED_STEP_TOOLS)) in README_CONTENT
    assert f"max {MAX_INPUTS}" in README_CONTENT
    assert f"Max {MAX_STEPS} steps" in README_CONTENT
    assert f"1–{MAX_WAIT_SECONDS}" in README_CONTENT


def test_readme_documents_the_yaml_quoting_gotcha() -> None:
    assert "quote" in README_CONTENT.lower()
    assert "MACRO.yml" in README_CONTENT
