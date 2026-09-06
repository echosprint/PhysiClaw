"""Tests for `drive.setup` — the wake's one setup call, and the two log
lines it owes every wake: the roster (every playbook on disk with its
state) and the decision (the boot drives and what it offers, or the
plain model session and the one reason). A playbook not running is
the first thing to know about a wake, so nothing here stays quiet."""

from __future__ import annotations

import logging

import pytest
from conductor_fakes import CHANNEL_OPEN, FLOW, PACK_MACRO, write_channel, write_pack

from physiclaw.common import paths
from physiclaw.conductor.drive import setup
from physiclaw.conductor.drive.activation import discover, playbook_gap
from physiclaw.conductor.spec.pack import load_pack, scan_playbooks

DISABLED_MACRO = "enabled: false\n" + PACK_MACRO.format(name="open-app")


@pytest.fixture
def wake_log(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    caplog.set_level(logging.INFO, logger="physiclaw.conductor")
    return caplog


def _lines(caplog: pytest.LogCaptureFixture, prefix: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith(prefix)]


# ---------- the roster ----------


def test_roster_names_every_playbook_with_its_state(wake_log) -> None:
    write_channel(CHANNEL_OPEN)
    write_pack(
        playbooks={
            "flow": FLOW,
            "later": "enabled: false\n" + FLOW,
            "broken": "route: 7\n",
        }
    )

    setup.session_setup()

    (roster,) = _lines(wake_log, "conductor: playbooks on disk")
    assert "demo/flow (live)" in roster
    assert "demo/later (disabled)" in roster
    assert "demo/broken (invalid: " in roster


def test_roster_names_the_disabled_macro_that_holds_a_playbook_back(wake_log) -> None:
    write_channel(CHANNEL_OPEN)
    root = write_pack(playbooks={"flow": FLOW})
    (root / "macros" / "open-app.yml").write_text(DISABLED_MACRO, encoding="utf-8")

    setup.session_setup()

    (roster,) = _lines(wake_log, "conductor: playbooks on disk")
    assert "demo/flow (disabled macro: open-app)" in roster


def test_roster_carries_an_unusable_pack_as_a_line(wake_log) -> None:
    write_channel(CHANNEL_OPEN)
    bad = paths.playbooks_dir() / "junk"
    bad.mkdir(parents=True)
    (bad / "APP.yml").write_text("app: [\n", encoding="utf-8")

    program, _ = setup.session_setup()

    (roster,) = _lines(wake_log, "conductor: playbooks on disk")
    assert "junk (pack unusable: " in roster
    assert program is None


def test_playbook_gap_reads_the_one_live_rule() -> None:
    # `pack.live_gap` is the rule `require_live` raises off — the roster
    # and the boot's own gate can never disagree about "live".
    write_pack(
        playbooks={
            "flow": FLOW,
            "later": "enabled: false\n" + FLOW,
            "bad": "route: 7\n",
        }
    )
    pack = load_pack("demo")
    by_name = {e.name: e for e in scan_playbooks("demo", pack)}

    assert playbook_gap(by_name["flow"], pack) is None
    assert playbook_gap(by_name["later"], pack) == "disabled"
    assert playbook_gap(by_name["bad"], pack).startswith("invalid: ")
    assert discover().entries.keys() == {"demo/flow"}


# ---------- the decision ----------


def test_a_live_playbook_and_boot_log_what_the_boot_offers(wake_log) -> None:
    write_channel(CHANNEL_OPEN)
    write_pack(playbooks={"flow": FLOW})

    program, _ = setup.session_setup()

    assert program is not None
    assert _lines(wake_log, "conductor: boot drives") == [
        "conductor: boot drives — offering demo/flow"
    ]
    assert not _lines(wake_log, "conductor: plain model session")


def test_no_channel_pack_is_the_first_reason(wake_log) -> None:
    write_pack(playbooks={"flow": FLOW})

    program, _ = setup.session_setup()

    assert program is None
    (why,) = _lines(wake_log, "conductor: plain model session")
    assert "no usable channel pack" in why
    # The roster still prints: the user sees the pack IS there.
    (roster,) = _lines(wake_log, "conductor: playbooks on disk")
    assert "demo/flow (live)" in roster


def test_a_channel_without_a_live_boot_says_so(wake_log) -> None:
    write_channel(CHANNEL_OPEN)
    write_pack(playbooks={"flow": FLOW})
    (paths.playbooks_dir() / "channel" / "boot" / "PLAYBOOK.yml").write_text(
        "name: boot\ndescription: off\nenabled: false\nroute:\n"
        "  - page: thread\n  - select: parse\n",
        encoding="utf-8",
    )

    program, _ = setup.session_setup()

    assert program is None
    (why,) = _lines(wake_log, "conductor: plain model session")
    assert "no live boot" in why
    # …and the channel logged the specific gap above it.
    assert any("channel/boot: disabled" in m for m in _lines(wake_log, "conductor:"))


def test_no_task_pack_on_disk_says_so(wake_log) -> None:
    write_channel(CHANNEL_OPEN)

    program, _ = setup.session_setup()

    assert program is None
    assert _lines(wake_log, "conductor: playbooks on disk") == [
        "conductor: playbooks on disk — none"
    ]
    (why,) = _lines(wake_log, "conductor: plain model session")
    assert "no task pack on disk" in why


def test_only_disabled_playbooks_point_at_the_roster(wake_log) -> None:
    write_channel(CHANNEL_OPEN)
    write_pack(playbooks={"flow": "enabled: false\n" + FLOW})

    program, _ = setup.session_setup()

    assert program is None
    (why,) = _lines(wake_log, "conductor: plain model session")
    assert "no live playbook" in why
