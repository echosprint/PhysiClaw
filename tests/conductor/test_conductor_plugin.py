"""Tests for the conductor's plugin face — `plugin.build()` behind the
`contract.plugin.TurnPlugin` seam.

The engine never imports this module by name; these tests pin what it
relies on structurally: the factory yields a protocol-satisfying
object, setup degrades to "plain model session" on an empty home, and
advance is a pass (None) whenever no driver is live.
"""

from __future__ import annotations

import pytest

from physiclaw.conductor import plugin as plugin_mod
from physiclaw.contract.plugin import SessionSetup, SetupContext, TurnPlugin


def _ctx(provider=None) -> SetupContext:
    return SetupContext(session_provider=provider)


def test_build_satisfies_the_protocol() -> None:
    assert isinstance(plugin_mod.build(), TurnPlugin)


@pytest.mark.asyncio
async def test_setup_on_an_empty_home_yields_no_macros_and_passes_turns() -> None:
    # No channel, no playbooks: session_setup arms nothing, and every
    # advance is a pass — indistinguishable from no conductor at all.
    plug = plugin_mod.build()

    contribution = await plug.session_setup(_ctx())

    assert isinstance(contribution, SessionSetup)
    assert contribution.gated_macros == {}
    assert await plug.advance([]) is None
    await plug.aclose()  # no micro wired — must be a clean no-op


@pytest.mark.asyncio
async def test_advance_before_setup_is_a_pass() -> None:
    plug = plugin_mod.build()
    assert await plug.advance([]) is None


@pytest.mark.asyncio
async def test_aclose_closes_the_owned_micro_client(mocker) -> None:
    # The micro-caller's lazily built cheap-tier client is the plugin's
    # to close — the engine's finally block reaches it only via aclose.
    plug = plugin_mod.build()
    plug._micro = mocker.AsyncMock()

    await plug.aclose()

    plug._micro.aclose.assert_awaited_once()


# ---------- _wire_micro ----------


def test_wire_micro_none_when_nothing_can_need_one() -> None:
    assert plugin_mod._wire_micro(None, None, _ctx()) is None


def test_wire_micro_builds_the_configured_cheap_tier(mocker, monkeypatch) -> None:
    from physiclaw.common.config import CONFIG

    monkeypatch.setattr(CONFIG.conductor, "micro_model", "moonshot/kimi-test")
    build_spy = mocker.patch.object(plugin_mod, "make_provider")

    caller = plugin_mod._wire_micro(None, mocker.MagicMock(), _ctx())

    assert caller is not None
    # The cheap tier is a lazy factory — nothing is built (or paid for)
    # at wire time; the client exists only after the first decision call.
    assert caller._owned_factory is not None
    build_spy.assert_not_called()


def test_wire_micro_bad_ref_falls_back_to_the_session_model(
    mocker, monkeypatch, caplog
) -> None:
    from physiclaw.common.config import CONFIG

    monkeypatch.setattr(CONFIG.conductor, "micro_model", "not-a-ref")
    provider = mocker.MagicMock()

    caller = plugin_mod._wire_micro(None, mocker.MagicMock(), _ctx(provider))

    # Fail-open at parse time: the caller still exists, with no owned
    # factory — decisions run on the session model.
    assert "unusable" in caplog.text
    assert caller is not None
    assert caller._owned_factory is None
    assert caller._provider is provider
