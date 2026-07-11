"""Tests for `physiclaw.agent.runtime.hook` — hook registry."""

from __future__ import annotations

import pytest

from physiclaw.agent.runtime import hook
from physiclaw.agent.runtime.hook import Trigger, check_hooks, clear, register


@pytest.fixture(autouse=True)
def _reset_hooks() -> None:
    """Tests must not bleed registered hooks across each other."""
    clear()


# ---------- Trigger ----------


def test_trigger_dataclass_is_frozen() -> None:
    t = Trigger(description="x")

    with pytest.raises(Exception):  # FrozenInstanceError
        t.description = "y"  # type: ignore[misc]


def test_trigger_default_source_is_empty_string() -> None:
    assert Trigger(description="x").source == ""


# ---------- register ----------


def test_register_appends_hook() -> None:
    @register
    def fn() -> Trigger | None:
        return None

    assert fn in hook._hooks


def test_register_returns_function_unchanged() -> None:
    def fn() -> Trigger | None:
        return None

    out = register(fn)

    assert out is fn


def test_register_can_be_called_directly() -> None:
    def fn() -> Trigger | None:
        return None

    register(fn)

    assert fn in hook._hooks


# ---------- check_hooks ----------


@pytest.mark.asyncio
async def test_check_hooks_returns_empty_when_nothing_registered() -> None:
    out = await check_hooks()

    assert out == []


@pytest.mark.asyncio
async def test_check_hooks_runs_sync_hooks_and_collects_triggers() -> None:
    @register
    def hook_a() -> Trigger | None:
        return Trigger(description="a", source="src-a")

    @register
    def hook_b() -> Trigger | None:
        return None  # didn't fire

    out = await check_hooks()

    assert out == [Trigger(description="a", source="src-a")]


@pytest.mark.asyncio
async def test_check_hooks_runs_async_hooks() -> None:
    @register
    async def async_hook() -> Trigger | None:
        return Trigger(description="async", source="src")

    out = await check_hooks()

    assert out == [Trigger(description="async", source="src")]


@pytest.mark.asyncio
async def test_check_hooks_preserves_registration_order() -> None:
    @register
    def first() -> Trigger | None:
        return Trigger(description="1")

    @register
    def second() -> Trigger | None:
        return Trigger(description="2")

    out = await check_hooks()

    assert [t.description for t in out] == ["1", "2"]


@pytest.mark.asyncio
async def test_check_hooks_logs_exception_and_continues(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    @register
    def boom() -> Trigger | None:
        raise RuntimeError("hook crashed")

    @register
    def survives() -> Trigger | None:
        return Trigger(description="survived")

    with caplog.at_level(logging.ERROR, logger="physiclaw.agent.runtime.hook"):
        out = await check_hooks()

    assert out == [Trigger(description="survived")]
    # Anchored equality so `XX...XX` literal mutations (mutmut's string
    # wrapping) can't sneak through the substring check.
    assert any(r.getMessage() == "hook failed: boom" for r in caplog.records)


# ---------- clear ----------


def test_clear_removes_all_hooks() -> None:
    @register
    def fn() -> Trigger | None:
        return None

    clear()

    assert hook._hooks == []
    assert hook._hooks_loaded is False


# ---------- load_hooks ----------


def test_load_hooks_is_idempotent(mocker) -> None:
    fake_pkg = mocker.MagicMock()
    fake_pkg.__path__ = ["/fake/path"]
    spy = mocker.patch.object(hook.importlib, "import_module", return_value=fake_pkg)
    mocker.patch.object(hook.pkgutil, "iter_modules", return_value=[])

    hook.load_hooks()
    hook.load_hooks()  # second call no-ops via _hooks_loaded

    pkg_imports = [c for c in spy.call_args_list if c.args[0] == hook.HOOKS_PACKAGE]
    assert len(pkg_imports) == 1


def test_load_hooks_skips_underscore_prefixed_modules(mocker) -> None:
    fake_pkg = mocker.MagicMock()
    fake_pkg.__path__ = ["/fake/path"]
    mocker.patch.object(hook.importlib, "import_module", return_value=fake_pkg)

    fake_modinfos = [
        mocker.MagicMock(name="real-mod"),
        mocker.MagicMock(name="hidden-mod"),
    ]
    fake_modinfos[0].name = f"{hook.HOOKS_PACKAGE}.real_mod"
    fake_modinfos[1].name = f"{hook.HOOKS_PACKAGE}._private"
    mocker.patch.object(hook.pkgutil, "iter_modules", return_value=fake_modinfos)

    import_spy = mocker.spy(hook.importlib, "import_module")

    hook.load_hooks()

    imported_names = [c.args[0] for c in import_spy.call_args_list]
    assert f"{hook.HOOKS_PACKAGE}.real_mod" in imported_names
    assert f"{hook.HOOKS_PACKAGE}._private" not in imported_names


def test_load_hooks_logs_loaded_module_name(
    mocker, caplog: pytest.LogCaptureFixture
) -> None:
    """Asserts the success log line (anchored on full message)."""
    import logging

    fake_pkg = mocker.MagicMock()
    fake_pkg.__path__ = ["/fake/path"]
    mocker.patch.object(hook.importlib, "import_module", return_value=fake_pkg)
    info = mocker.MagicMock()
    info.name = f"{hook.HOOKS_PACKAGE}.cron"
    mocker.patch.object(hook.pkgutil, "iter_modules", return_value=[info])

    with caplog.at_level(logging.INFO, logger="physiclaw.agent.runtime.hook"):
        hook.load_hooks()

    expected = f"loaded hook module: {hook.HOOKS_PACKAGE}.cron"
    assert any(r.getMessage() == expected for r in caplog.records)


def test_load_hooks_continues_past_underscored_module_to_next(mocker) -> None:
    """Underscore-prefixed comes FIRST; if `continue` were `break`, the
    real module after it would never get imported."""
    fake_pkg = mocker.MagicMock()
    fake_pkg.__path__ = ["/fake/path"]
    mocker.patch.object(hook.importlib, "import_module", return_value=fake_pkg)
    fake_modinfos = [mocker.MagicMock(), mocker.MagicMock()]
    fake_modinfos[0].name = f"{hook.HOOKS_PACKAGE}._private"  # comes first
    fake_modinfos[1].name = f"{hook.HOOKS_PACKAGE}.real_mod"
    mocker.patch.object(hook.pkgutil, "iter_modules", return_value=fake_modinfos)
    import_spy = mocker.spy(hook.importlib, "import_module")

    hook.load_hooks()

    imported = [c.args[0] for c in import_spy.call_args_list]
    # `real_mod` must be reached even though `_private` came first —
    # `continue` (not `break`) is required.
    assert f"{hook.HOOKS_PACKAGE}.real_mod" in imported


def test_load_hooks_logs_but_continues_on_module_import_failure(
    mocker, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    fake_pkg = mocker.MagicMock()
    fake_pkg.__path__ = ["/fake/path"]

    def fake_import(name: str, *_a, **_kw):
        if name == hook.HOOKS_PACKAGE:
            return fake_pkg
        raise ImportError(f"can't load {name}")

    mocker.patch.object(hook.importlib, "import_module", side_effect=fake_import)
    fake_modinfos = [mocker.MagicMock()]
    fake_modinfos[0].name = f"{hook.HOOKS_PACKAGE}.broken"
    mocker.patch.object(hook.pkgutil, "iter_modules", return_value=fake_modinfos)

    with caplog.at_level(logging.ERROR, logger="physiclaw.agent.runtime.hook"):
        hook.load_hooks()  # must not raise

    expected = f"failed to load hook module: {hook.HOOKS_PACKAGE}.broken"
    assert any(r.getMessage() == expected for r in caplog.records)
