"""Tests for `physiclaw.common.gesture_vocab` — the shared tool-name contract.

The contract tests here are the point of the module: they turn a silent
rename-drift (a gesture tool renamed in `core/server/tools.py` while the
engine's guards keep classifying by the old name) into a red test.
Tests may import both layers; the architecture rule binds src/ only.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from physiclaw.common.gesture_vocab import (
    NAV_TOOLS,
    PEEK,
    PRESS_TOOLS,
    RUN_MACRO,
    SCREENSHOT,
    SEND_TO_CLIPBOARD,
    SEQUENCE,
    STEP_ACTIONS,
    STEP_ARG,
    STEP_TOOL,
    SWIPE,
    SWIPE_DISTANCES,
    SWIPE_SPEEDS,
    UNLOCK_PHONE,
)


def test_families_are_disjoint() -> None:
    assert not PRESS_TOOLS & NAV_TOOLS
    assert SWIPE not in PRESS_TOOLS | NAV_TOOLS
    assert SEQUENCE not in PRESS_TOOLS | NAV_TOOLS


def test_step_keys() -> None:
    assert (STEP_ACTIONS, STEP_TOOL, STEP_ARG) == ("actions", "tool_name", "arg")


# ---------- cross-layer contracts ----------


class _RecorderMcp:
    """Records @mcp.tool()'d function names (test_tools.py's FakeMcp shape)."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def tool(self, **kwargs):
        def deco(fn):
            self.names.add(fn.__name__)
            return fn

        return deco


def test_registered_tool_names_cover_the_vocabulary(mocker) -> None:
    """Every vocabulary name must exist as a registered @mcp.tool — a
    rename in tools.py without a vocabulary update fails here, not in
    production where the guards would silently stop classifying it."""
    from physiclaw.core.server import tools as tools_mod

    mcp = _RecorderMcp()
    mocker.patch.object(tools_mod, "save_tool_call")
    tools_mod.register(mcp, MagicMock())

    vocabulary = (
        PRESS_TOOLS
        | NAV_TOOLS
        | {
            SWIPE,
            SEQUENCE,
            PEEK,
            SCREENSHOT,
            SEND_TO_CLIPBOARD,
            UNLOCK_PHONE,
        }
    )
    assert vocabulary <= mcp.names
    # RUN_MACRO lives in this module because the engine's classifiers have to
    # name it, but it is a LOCAL tool — it must never appear in the server
    # registration, or a macro would become a gesture a macro could nest.
    assert RUN_MACRO not in mcp.names


def test_parse_step_accepts_every_press_tool() -> None:
    """The orchestrator's sequence-step dispatch must accept every
    press-family name the engine counts."""
    from physiclaw.core.orchestration.gestures import GestureValidator

    validator = GestureValidator.__new__(GestureValidator)  # no hardware deps
    for tool in PRESS_TOOLS:
        gesture = validator.parse_step(tool, [0.1, 0.2, 0.9, 1.0])
        assert gesture is not None


def test_parse_step_accepts_swipe() -> None:
    from physiclaw.core.orchestration.gestures import GestureValidator

    validator = GestureValidator.__new__(GestureValidator)
    with pytest.raises(ValueError):
        validator.parse_step(SWIPE, "not-a-dict")


def test_swipe_ladder_matches_the_typed_gesture_vocabulary() -> None:
    """The dependency-free ladder (served to the studio page) and the
    Literal types the orchestrator validates against are one vocabulary."""
    from typing import get_args

    from physiclaw.core.orchestration import gestures

    assert tuple(SWIPE_DISTANCES) == get_args(gestures.Size)
    assert SWIPE_SPEEDS == get_args(gestures.Speed)
    assert gestures.SWIPE_DISTANCES is SWIPE_DISTANCES
