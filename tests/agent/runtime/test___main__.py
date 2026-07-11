"""Tests for `physiclaw.agent.runtime.__main__` — `python -m` bootstrap.

The module is a 4-line shim that imports `launch` and calls it under
`if __name__ == "__main__"`. This file is excluded from coverage by
pyproject.toml `[tool.coverage.run] omit`, but the bootstrap is still
worth a smoke test so a typo here doesn't silently break
`python -m physiclaw.agent.runtime`.
"""

from __future__ import annotations

import runpy
import sys
from unittest.mock import MagicMock


def test_module_imports_launch_from_launcher() -> None:
    """The shim's only job is to import `launch` from launcher."""
    import physiclaw.agent.runtime.__main__ as main_mod
    from physiclaw.agent.runtime.launcher import launch as launcher_launch

    assert main_mod.launch is launcher_launch


def test_run_as_main_invokes_launch(mocker, monkeypatch) -> None:
    """Invoking `python -m physiclaw.agent.runtime` must call `launch()`.
    `runpy.run_module(..., run_name="__main__")` enters the
    `if __name__ == "__main__"` branch."""
    spy = MagicMock()
    # Patch the symbol on the launcher module so the shim's import gets
    # the spy when runpy reloads it.
    monkeypatch.setattr(
        "physiclaw.agent.runtime.launcher.launch",
        spy,
    )
    # Drop cached __main__ so runpy fully re-executes it.
    sys.modules.pop("physiclaw.agent.runtime.__main__", None)

    runpy.run_module("physiclaw.agent.runtime", run_name="__main__")

    spy.assert_called_once_with()


def test_imported_normally_does_not_invoke_launch(mocker, monkeypatch) -> None:
    """Plain `import physiclaw.agent.runtime.__main__` must NOT call
    launch — the guard exists for exactly this reason."""
    spy = MagicMock()
    monkeypatch.setattr(
        "physiclaw.agent.runtime.launcher.launch",
        spy,
    )
    sys.modules.pop("physiclaw.agent.runtime.__main__", None)

    import physiclaw.agent.runtime.__main__  # noqa: F401

    spy.assert_not_called()
