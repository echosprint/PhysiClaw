"""Tests for `physiclaw.core.logger.logger`."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from physiclaw.core.logger import logger as logger_mod
from physiclaw.core.logger.logger import (
    _colorize,
    _format_args,
    _TaggedFormatter,
    LineLogStream,
    logged,
    make_tagged_logger,
    setup_logging,
)


# ---------- _colorize ----------


def test_colorize_true_when_tty_and_no_color_unset(
    mocker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    mocker.patch.object(logger_mod.sys.stderr, "isatty", return_value=True)

    assert _colorize() is True


def test_colorize_false_when_not_tty(mocker) -> None:
    mocker.patch.object(logger_mod.sys.stderr, "isatty", return_value=False)

    assert _colorize() is False


def test_colorize_false_when_no_color_env_set(
    mocker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mocker.patch.object(logger_mod.sys.stderr, "isatty", return_value=True)
    monkeypatch.setenv("NO_COLOR", "1")

    assert _colorize() is False


# ---------- _TaggedFormatter ----------


def _record(msg: str = "hi", level: int = logging.INFO) -> logging.LogRecord:
    return logging.LogRecord(
        name="x",
        level=level,
        pathname="x.py",
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )


def test_formatter_plain_mode_renders_tag_and_message() -> None:
    fmt = _TaggedFormatter(tag="physiclaw", color=False)

    out = fmt.format(_record("hello"))

    assert "[physiclaw]" in out
    assert "hello" in out
    # No ANSI escape sequences in plain mode.
    assert "\033[" not in out


def test_formatter_color_mode_paints_tag(mocker) -> None:
    fmt = _TaggedFormatter(tag="physiclaw", color=True)

    out = fmt.format(_record("hello"))

    # Cyan ANSI for "physiclaw".
    assert "\033[36m" in out


def test_formatter_paints_warning_yellow() -> None:
    fmt = _TaggedFormatter(tag="physiclaw", color=True)

    out = fmt.format(_record("warn", level=logging.WARNING))

    assert "\033[33m" in out


def test_formatter_paints_error_red() -> None:
    fmt = _TaggedFormatter(tag="physiclaw", color=True)

    out = fmt.format(_record("oh no", level=logging.ERROR))

    assert "\033[31m" in out


def test_formatter_indents_continuation_lines() -> None:
    fmt = _TaggedFormatter(tag="runtime", color=False)

    out = fmt.format(_record("line one\nline two"))

    lines = out.split("\n")
    # Continuation indent matches the prefix length.
    assert "line one" in lines[0]
    assert lines[1].startswith(" " * len("00:00 [runtime] "))


# ---------- setup_logging ----------


def test_setup_logging_force_replaces_handlers(mocker) -> None:
    mocker.patch.object(logger_mod, "_colorize", return_value=False)

    setup_logging("physiclaw", level=logging.WARNING)

    root = logging.getLogger()
    assert root.level == logging.WARNING
    # Last handler is the one we just added with our formatter.
    assert any(isinstance(h.formatter, _TaggedFormatter) for h in root.handlers)


# ---------- make_tagged_logger ----------


def test_make_tagged_logger_is_standalone_and_tagged(mocker) -> None:
    mocker.patch.object(logger_mod, "_colorize", return_value=False)

    lg = make_tagged_logger("physiclaw.test_wizard", "setup wizard")

    # Standalone: won't double-print through the root [physiclaw] handler.
    assert lg.propagate is False
    assert len(lg.handlers) == 1
    fmt = lg.handlers[0].formatter
    assert isinstance(fmt, _TaggedFormatter)
    rec = logging.LogRecord("x", logging.INFO, "f", 1, "hi", None, None)
    assert fmt.format(rec).endswith("[setup wizard] hi")


def test_make_tagged_logger_does_not_stack_handlers(mocker) -> None:
    mocker.patch.object(logger_mod, "_colorize", return_value=False)

    first = make_tagged_logger("physiclaw.test_wizard_2", "setup wizard")
    again = make_tagged_logger("physiclaw.test_wizard_2", "setup wizard")

    assert again is first
    assert len(again.handlers) == 1  # idempotent — no duplicate handler


# ---------- LineLogStream ----------


def test_line_log_stream_emits_full_lines() -> None:
    logger = MagicMock()
    stream = LineLogStream(logger)

    n = stream.write("── 1. Connect phone ──\n")

    logger.info.assert_called_once_with("── 1. Connect phone ──")
    assert n == len("── 1. Connect phone ──\n")


def test_line_log_stream_strips_ansi_and_skips_blank_lines() -> None:
    logger = MagicMock()
    stream = LineLogStream(logger)

    stream.write("\n")  # blank line from print("\n── …") — dropped
    stream.write("  \033[32m✓\033[0m Arm connected\n")  # ANSI stripped

    logger.info.assert_called_once_with("  ✓ Arm connected")


def test_line_log_stream_buffers_partial_lines() -> None:
    logger = MagicMock()
    stream = LineLogStream(logger)

    stream.write("half ")  # no newline yet — buffered, not emitted
    logger.info.assert_not_called()
    stream.write("line\n")
    logger.info.assert_called_once_with("half line")


# ---------- _format_args ----------


def test_format_args_send_to_clipboard_redacts_text() -> None:
    out = _format_args("send_to_clipboard", {"text": "secret password"})

    # Length only, never the value.
    assert "secret" not in out
    assert "<15 chars>" in out


def test_format_args_send_to_clipboard_handles_missing_text() -> None:
    out = _format_args("send_to_clipboard", {})

    assert "<0 chars>" in out


def test_format_args_sequence_summarizes_tool_names() -> None:
    # Real FastMCP shape: ONE kwarg holding the list of step dicts. The
    # old test passed steps as separate kwargs — an encoding the tool
    # never receives — which let every live batch log as "0 steps".
    out = _format_args(
        "sequence",
        {
            "actions": [
                {"tool_name": "tap", "arg": [0, 0, 1, 1]},
                {"tool_name": "send_to_clipboard", "arg": "secret text"},
                "garbage",
            ],
        },
    )

    assert "3 steps:" in out
    assert "tap" in out
    assert "send_to_clipboard" in out
    assert "secret text" not in out  # step args stay redacted


def test_format_args_sequence_handles_missing_actions() -> None:
    assert "0 steps:" in _format_args("sequence", {})


def test_format_args_default_renders_repr() -> None:
    out = _format_args("tap", {"bbox": [0.1, 0.2, 0.3, 0.4]})

    assert "bbox=[0.1, 0.2, 0.3, 0.4]" in out


def test_format_args_truncates_long_arg_strings() -> None:
    long_text = "x" * 200
    out = _format_args("custom", {"value": long_text})

    # Truncated to _MAX_ARG_LOG_LEN with ellipsis.
    assert out.endswith("...")
    assert len(out) == 80  # _MAX_ARG_LOG_LEN


# ---------- logged decorator ----------


@pytest.mark.asyncio
async def test_logged_calls_wrapped_function(
    caplog: pytest.LogCaptureFixture,
) -> None:
    @logged
    async def my_tool(bbox):
        return f"tapped {bbox}"

    with caplog.at_level(logging.INFO, logger="physiclaw.tools"):
        out = await my_tool(bbox=[0, 0, 1, 1])

    assert out == "tapped [0, 0, 1, 1]"
    assert any(
        "my_tool" in r.getMessage() and "—" in r.getMessage() for r in caplog.records
    )


@pytest.mark.asyncio
async def test_logged_skips_logging_when_info_disabled(
    mocker,
    caplog: pytest.LogCaptureFixture,
) -> None:
    @logged
    async def my_tool():
        return "ok"

    # Disable info-level logging.
    mocker.patch.object(
        logger_mod.log,
        "isEnabledFor",
        return_value=False,
    )

    with caplog.at_level(logging.INFO, logger="physiclaw.tools"):
        out = await my_tool()

    assert out == "ok"
    assert not [r for r in caplog.records if "my_tool" in r.getMessage()]


@pytest.mark.asyncio
async def test_logged_marks_failure_when_function_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    @logged
    async def my_tool():
        raise RuntimeError("boom")

    with caplog.at_level(logging.INFO, logger="physiclaw.tools"):
        with pytest.raises(RuntimeError, match="boom"):
            await my_tool()

    # A raise logs a distinct FAILED line carrying the error, so a fast
    # failure isn't mistaken for a fast success in the server stream.
    msgs = [r.getMessage() for r in caplog.records if "my_tool" in r.getMessage()]
    assert msgs
    assert any("FAILED: boom" in m for m in msgs)


@pytest.mark.asyncio
async def test_logged_success_line_has_no_failed_marker(
    caplog: pytest.LogCaptureFixture,
) -> None:
    @logged
    async def my_tool():
        return "ok"

    with caplog.at_level(logging.INFO, logger="physiclaw.tools"):
        await my_tool()

    msgs = [r.getMessage() for r in caplog.records if "my_tool" in r.getMessage()]
    assert msgs
    assert not any("FAILED" in m for m in msgs)


# ---------- _DailyFileHandler ----------


def _file_logger(tmp_path, name: str):
    import logging

    from physiclaw.core.logger.logger import _DailyFileHandler

    handler = _DailyFileHandler(tmp_path, "runtime", "runtime")
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers = [handler]
    return logger, handler


def test_daily_file_handler_writes_uncolored_tagged_lines(tmp_path) -> None:
    from freezegun import freeze_time

    with freeze_time("2026-04-28T10:00:00"):
        logger, handler = _file_logger(tmp_path, "t.daily.plain")
        logger.info("hello \033[31mred\033[0m world")
        handler.close()

    text = (tmp_path / "runtime-2026-04-28.log").read_text()
    assert "[runtime] hello" in text
    assert "\033[" not in text.split("hello")[0]  # prefix uncolored


def test_daily_file_handler_rolls_to_new_day(tmp_path) -> None:
    from freezegun import freeze_time

    with freeze_time("2026-04-28T23:59:00") as ft:
        logger, handler = _file_logger(tmp_path, "t.daily.roll")
        logger.info("before midnight")
        ft.move_to("2026-04-29T00:01:00")
        logger.info("after midnight")
        handler.close()

    assert "before midnight" in (tmp_path / "runtime-2026-04-28.log").read_text()
    assert "after midnight" in (tmp_path / "runtime-2026-04-29.log").read_text()


def test_daily_file_handler_purges_old_dailies_at_construction(tmp_path) -> None:
    import datetime as dt
    import os

    from physiclaw.config import CONFIG

    old = tmp_path / "runtime-2026-01-01.log"
    old.write_text("x")
    ago = (
        dt.datetime.now() - dt.timedelta(days=CONFIG.retention.log_days + 1)
    ).timestamp()
    os.utime(old, (ago, ago))

    _, handler = _file_logger(tmp_path, "t.daily.purge")
    handler.close()

    assert not old.exists()


def test_setup_logging_with_unwritable_dir_falls_back_to_stderr(tmp_path) -> None:
    import logging

    from physiclaw.core.logger.logger import setup_logging

    blocker = tmp_path / "blocked"
    blocker.write_text("a file where a dir must go")

    setup_logging("runtime", logging.INFO, file_dir=blocker / "sub")  # no raise

    root = logging.getLogger()
    assert len(root.handlers) == 1  # stream only
