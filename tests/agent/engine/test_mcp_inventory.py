"""Tests for `physiclaw.agent.engine.mcp_inventory`.

The module AST-parses `core/server/tools.py` to enumerate
`@mcp.tool()`-decorated functions without importing the file.

Tests use `monkeypatch.setattr` to point `_TOOLS_PY` at synthetic
fixture files, so we don't depend on the actual tools.py contents.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from physiclaw.agent.engine import mcp_inventory


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


# ---------- discover_mcp_tools: file missing / unparseable ----------


def test_discover_returns_empty_when_tools_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mcp_inventory, "_TOOLS_PY", tmp_path / "nope.py")

    assert mcp_inventory.discover_mcp_tools() == []


def test_discover_returns_empty_and_logs_on_syntax_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    p = _write(tmp_path, "tools.py", "def broken(:\n  pass\n")
    monkeypatch.setattr(mcp_inventory, "_TOOLS_PY", p)

    with caplog.at_level(logging.ERROR, logger="physiclaw.agent.engine.mcp_inventory"):
        out = mcp_inventory.discover_mcp_tools()

    assert out == []
    assert any("failed to parse" in r.getMessage() for r in caplog.records)


# ---------- discover_mcp_tools: extraction ----------


def test_discover_returns_decorated_functions_with_docstring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _write(
        tmp_path,
        "tools.py",
        '''
@mcp.tool()
def tap(bbox: list) -> str:
    """Tap a coordinate on the phone screen."""
    return ""

@mcp.tool()
def peek() -> str:
    """Capture an annotated camera frame."""
    return ""
''',
    )
    monkeypatch.setattr(mcp_inventory, "_TOOLS_PY", p)

    out = mcp_inventory.discover_mcp_tools()

    assert out == [
        {"name": "tap", "description": "Tap a coordinate on the phone screen."},
        {"name": "peek", "description": "Capture an annotated camera frame."},
    ]


def test_discover_skips_functions_without_mcp_tool_decorator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _write(
        tmp_path,
        "tools.py",
        '''
def helper() -> None:
    """Internal helper, not a tool."""
    pass

@mcp.tool()
def real_tool() -> str:
    """Public tool."""
    return ""

@some_other.decorator()
def wrong_decorator() -> str:
    """Has a decorator but not @mcp.tool."""
    return ""
''',
    )
    monkeypatch.setattr(mcp_inventory, "_TOOLS_PY", p)

    out = mcp_inventory.discover_mcp_tools()

    assert [t["name"] for t in out] == ["real_tool"]


def test_discover_recognizes_async_def_decorated_with_mcp_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _write(
        tmp_path,
        "tools.py",
        '''
@mcp.tool()
async def async_tool() -> str:
    """An async tool."""
    return ""
''',
    )
    monkeypatch.setattr(mcp_inventory, "_TOOLS_PY", p)

    out = mcp_inventory.discover_mcp_tools()

    assert out == [{"name": "async_tool", "description": "An async tool."}]


def test_discover_returns_empty_string_when_function_has_no_docstring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _write(
        tmp_path,
        "tools.py",
        """
@mcp.tool()
def no_docs() -> str:
    return ""
""",
    )
    monkeypatch.setattr(mcp_inventory, "_TOOLS_PY", p)

    out = mcp_inventory.discover_mcp_tools()

    assert out == [{"name": "no_docs", "description": ""}]


def test_discover_recognizes_decorator_without_call_parens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `@mcp.tool` (no parens) also counts.
    p = _write(
        tmp_path,
        "tools.py",
        '''
@mcp.tool
def bare_decorator() -> str:
    """Decorator without parens."""
    return ""
''',
    )
    monkeypatch.setattr(mcp_inventory, "_TOOLS_PY", p)

    out = mcp_inventory.discover_mcp_tools()

    assert out == [
        {"name": "bare_decorator", "description": "Decorator without parens."}
    ]


def test_discover_ignores_decorator_named_tool_on_other_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _write(
        tmp_path,
        "tools.py",
        '''
@other.tool()
def fake_tool() -> str:
    """Different `tool` attr, not on `mcp`."""
    return ""
''',
    )
    monkeypatch.setattr(mcp_inventory, "_TOOLS_PY", p)

    assert mcp_inventory.discover_mcp_tools() == []


def test_discover_ignores_function_named_tool_called_directly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `@tool()` (just `tool`, no `mcp.` prefix) doesn't count.
    p = _write(
        tmp_path,
        "tools.py",
        '''
@tool()
def looks_like_tool() -> str:
    """Bare `@tool()` without mcp prefix."""
    return ""
''',
    )
    monkeypatch.setattr(mcp_inventory, "_TOOLS_PY", p)

    assert mcp_inventory.discover_mcp_tools() == []


# ---------- _has_mcp_tool_decorator ----------


def test_tools_py_path_resolves_under_core_server() -> None:
    # The captured path lives under core/server/tools.py.
    assert mcp_inventory._TOOLS_PY.name == "tools.py"
    assert mcp_inventory._TOOLS_PY.parent.name == "server"
    assert mcp_inventory._TOOLS_PY.parent.parent.name == "core"
