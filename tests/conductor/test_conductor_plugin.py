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
