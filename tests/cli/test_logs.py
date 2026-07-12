"""Tests for `physiclaw logs` — session listing and inspection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from physiclaw.cli import app

runner = CliRunner()


def _summary(sid: str, sentinel: str = "DONE", recap: str = "ok") -> dict:
    return {
        "schema": 1,
        "sid": sid,
        "started_at": "x",
        "ended_at": "y",
        "duration_s": 313.0,
        "model_ref": "moonshot/kimi-k2.6",
        "provider": "moonshot",
        "prompt_hash": "h",
        "triggers": [],
        "outcome": {"sentinel": sentinel, "recap": recap, "crashed": False},
        "turns": 42,
        "provider_calls": 47,
        "provider_time_ms": 1,
        "usage": {
            "input_tokens": 1_200_000,
            "output_tokens": 9_800,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "cache_hit_pct": 87.5,
        },
        "tool_calls": {"note": 42},
        "errors": {},
        "stuck_events": 0,
        "images": 3,
    }


@pytest.fixture
def sessions(physiclaw_home) -> Path:
    from physiclaw.common import paths

    d = paths.engine_sessions_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_session(d: Path, sid: str, **kw) -> Path:
    sd = d / sid
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "summary.json").write_text(json.dumps(_summary(sid, **kw)))
    return sd


def test_logs_lists_recent_sessions(sessions: Path) -> None:
    _write_session(sessions, "20260710_090000")
    _write_session(sessions, "20260710_100000", sentinel="STUCK", recap="lost")

    result = runner.invoke(app, ["logs"])

    assert result.exit_code == 0
    lines = result.output.splitlines()
    # Newest first.
    first = next(x for x in lines if "20260710_100000" in x)
    assert "STUCK" in first and "lost" in first
    assert "1.2M/9.8k" in result.output
    assert "88%" in result.output or "87" in result.output


def test_logs_shows_summaryless_dir_as_unknown(sessions: Path) -> None:
    (sessions / "20260710_110000").mkdir()

    result = runner.invoke(app, ["logs"])

    assert "20260710_110000" in result.output
    assert "?" in result.output


def test_logs_empty_state(sessions: Path) -> None:
    result = runner.invoke(app, ["logs"])

    assert result.exit_code == 0
    assert "no sessions yet" in result.output


def test_logs_json_list_round_trips(sessions: Path) -> None:
    _write_session(sessions, "20260710_090000")

    result = runner.invoke(app, ["logs", "--json"])

    data = json.loads(result.output)
    assert data[0]["sid"] == "20260710_090000"


def test_logs_detail_prints_summary_and_narrative(sessions: Path) -> None:
    sd = _write_session(sessions, "20260710_090000")
    events = [
        {
            "t": "2026-07-10T09:00:01.000",
            "event": "wake",
            "session": "s",
            "model_ref": "m/x",
            "triggers": [{"source": "phone"}],
        },
        {
            "t": "2026-07-10T09:00:05.000",
            "event": "done",
            "sentinel": "DONE",
            "recap": "finished",
        },
    ]
    (sd / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")

    result = runner.invoke(app, ["logs", "20260710_090000"])

    assert result.exit_code == 0
    assert '"sid": "20260710_090000"' in result.output
    assert "OUTCOME: DONE — finished" in result.output
    assert "[09:00:05]" in result.output
    assert "wire.jsonl" in result.output


def test_logs_detail_json_emits_summary(sessions: Path) -> None:
    _write_session(sessions, "20260710_090000")

    result = runner.invoke(app, ["logs", "20260710_090000", "--json"])

    assert json.loads(result.output)["turns"] == 42


def test_logs_detail_unknown_sid_exits_nonzero(sessions: Path) -> None:
    result = runner.invoke(app, ["logs", "nope"])

    assert result.exit_code == 1
    assert "no such session" in result.output


def test_logs_resolves_session_by_hex_suffix(sessions: Path) -> None:
    _write_session(sessions, "20260710_090000_ab12cd")

    result = runner.invoke(app, ["logs", "ab12cd"])

    assert result.exit_code == 0
    assert '"sid": "20260710_090000_ab12cd"' in result.output


def test_logs_ambiguous_suffix_lists_candidates(sessions: Path) -> None:
    _write_session(sessions, "20260710_090000_aaa000")
    _write_session(sessions, "20260710_100000_bbb000")

    result = runner.invoke(app, ["logs", "000"])

    assert result.exit_code == 1
    assert "ambiguous" in result.output
    assert "20260710_090000_aaa000" in result.output
    assert "20260710_100000_bbb000" in result.output


def test_logs_unmatched_suffix_reports_missing(sessions: Path) -> None:
    _write_session(sessions, "20260710_090000_ab12cd")

    result = runner.invoke(app, ["logs", "deadbeef"])

    assert result.exit_code == 1
    assert "no such session" in result.output


# ---------- --save ----------


def _populate(sd: Path) -> None:
    (sd / "events.jsonl").write_text('{"event": "env"}\n')
    (sd / "wire.jsonl").write_text('{"kind": "session_start"}\n')
    img = sd / "images"
    img.mkdir(exist_ok=True)
    (img / "00001_t0.jpg").write_bytes(b"jpeg")


def test_save_packs_session_zip_with_privacy_notice(
    sessions: Path, tmp_path, monkeypatch
) -> None:
    import zipfile

    sd = _write_session(
        sessions,
        "20260710_090000_ab12cd",
        sentinel="STUCK",
        recap="lost on home screen",
    )
    _populate(sd)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["logs", "ab12cd", "--save"])

    assert result.exit_code == 0
    out = tmp_path / "physiclaw-session-20260710_090000_ab12cd.zip"
    assert out.is_file()
    names = set(zipfile.ZipFile(out).namelist())
    assert "20260710_090000_ab12cd/summary.json" in names
    assert "20260710_090000_ab12cd/events.jsonl" in names
    assert "20260710_090000_ab12cd/wire.jsonl" in names
    assert "20260710_090000_ab12cd/images/00001_t0.jpg" in names
    # Privacy notice; no issue-filing prompts.
    assert "PRIVATE" in result.output
    assert "review before sharing" in result.output
    assert "issues" not in result.output


def test_save_without_sid_errors(sessions: Path) -> None:
    result = runner.invoke(app, ["logs", "--save"])

    assert result.exit_code == 1
    assert "needs a session" in result.output


def test_save_unknown_sid_errors(sessions: Path, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["logs", "nope", "--save"])

    assert result.exit_code == 1
    assert "no such session" in result.output


def test_detail_view_hints_at_save(sessions: Path) -> None:
    _write_session(sessions, "20260710_090000_ab12cd")

    result = runner.invoke(app, ["logs", "ab12cd"])

    assert "physiclaw logs ab12cd --save" in result.output


def test_save_accepts_destination_dir(sessions: Path, tmp_path, monkeypatch) -> None:
    sd = _write_session(sessions, "20260710_090000_ab12cd")
    _populate(sd)
    dest = tmp_path / "backups"
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["logs", "ab12cd", "--save", str(dest)])

    assert result.exit_code == 0
    assert (dest / "physiclaw-session-20260710_090000_ab12cd.zip").is_file()


def test_save_accepts_explicit_zip_path(sessions: Path, tmp_path, monkeypatch) -> None:
    sd = _write_session(sessions, "20260710_090000_ab12cd")
    _populate(sd)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["logs", "ab12cd", "--save", str(tmp_path / "bad.zip")])

    assert result.exit_code == 0
    assert (tmp_path / "bad.zip").is_file()


def test_destination_without_save_is_refused(sessions: Path) -> None:
    _write_session(sessions, "20260710_090000_ab12cd")

    result = runner.invoke(app, ["logs", "ab12cd", "out.zip"])

    assert result.exit_code == 1
    assert "needs --save" in result.output


def test_save_zip_embeds_format_readme(sessions: Path, tmp_path, monkeypatch) -> None:
    import zipfile

    sd = _write_session(sessions, "20260710_090000_ab12cd")
    _populate(sd)
    monkeypatch.chdir(tmp_path)

    runner.invoke(app, ["logs", "ab12cd", "--save"])

    z = zipfile.ZipFile(tmp_path / "physiclaw-session-20260710_090000_ab12cd.zip")
    readme = z.read("20260710_090000_ab12cd/README.md").decode("utf-8")
    assert "events.jsonl" in readme
    # arcnames are forward-slashed on every platform
    assert all("\\\\" not in n for n in z.namelist())
