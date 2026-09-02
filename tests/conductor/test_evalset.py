"""Tests for `physiclaw.conductor.evalset` — the case file format,
scoring against MicroResult, and the aggregation `conductor eval` prints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conductor_fakes import make_screen

from physiclaw.conductor import evalset
from physiclaw.conductor.micro import (
    AGENT_FIELDS,
    PARSE_TASK,
    MicroOutcome,
    MicroResult,
)
from physiclaw.contract.dto import Usage


def _write_cases(tmp_path: Path, *cases: dict) -> Path:
    p = tmp_path / "cases.jsonl"
    p.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n",
        encoding="utf-8",
    )
    return p


def _case(**overrides) -> evalset.EvalCase:
    base = dict(
        call=PARSE_TASK,
        expect="taobao/buy",
        args={"menu": "- taobao/buy: buy things"},
        outcomes=("taobao/buy",),
        listing=make_screen(("买牛奶", 0.25, 0.3)).text,
    )
    return evalset.EvalCase(**{**base, **overrides})


def _result(outcome: MicroOutcome | None, detail: str = "") -> MicroResult:
    return MicroResult(
        outcome=outcome,
        detail=detail,
        attempts=1,
        usage=Usage(prompt_tokens=100, completion_tokens=20),
        elapsed_ms=50,
    )


def _answer(out: str, confidence: float = 0.9) -> MicroOutcome:
    return MicroOutcome(out=out, reason="fits", confidence=confidence)


# ---------- read_cases ----------


def test_read_cases_parses_fields(tmp_path: Path) -> None:
    p = _write_cases(
        tmp_path,
        {
            "call": AGENT_FIELDS,
            "expect": "done",
            "args": {"prompt": "the keyword", "fields": "- keyword: k"},
            "listing": "x",
            "note": "smoke",
        },
    )

    (case,) = evalset.read_cases(p)

    assert case.call == AGENT_FIELDS
    assert case.expect == "done"
    assert case.args == {"prompt": "the keyword", "fields": "- keyword: k"}
    assert case.outcomes == ()
    assert case.note == "smoke"


def test_read_cases_rejects_bad_json_with_line_number(tmp_path: Path) -> None:
    p = tmp_path / "cases.jsonl"
    p.write_text("not json\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r":1: bad JSON"):
        evalset.read_cases(p)


def test_read_cases_rejects_unknown_call(tmp_path: Path) -> None:
    p = _write_cases(tmp_path, {"call": "guess_item", "expect": "x"})

    with pytest.raises(ValueError, match="unknown call 'guess_item'"):
        evalset.read_cases(p)


def test_read_cases_rejects_missing_expect(tmp_path: Path) -> None:
    p = _write_cases(tmp_path, {"call": AGENT_FIELDS, "args": {"prompt": "q"}})

    with pytest.raises(ValueError, match="`expect` must be a non-empty string"):
        evalset.read_cases(p)


def test_read_cases_rejects_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "cases.jsonl"
    p.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no cases"):
        evalset.read_cases(p)


def test_build_reads_the_listing_via_runtime_assembler() -> None:
    case = _case()

    req = evalset.build(case, "eval-0")

    assert "买牛奶" in req.listing and req.outcomes == ("taobao/buy",)


# ---------- score ----------


def test_score_matching_answer_is_correct() -> None:
    scored = evalset.score(_case(), _result(_answer("taobao/buy")))

    assert scored.correct
    assert scored.answer == "taobao/buy"


def test_score_escape_arm_matches_out() -> None:
    outcome = _answer("not_a_task", 0.8)

    scored = evalset.score(_case(expect="not_a_task"), _result(outcome))

    assert scored.correct
    assert scored.answer == "not_a_task"


def test_score_wrong_answer_is_answered_but_not_correct() -> None:
    scored = evalset.score(_case(), _result(_answer("not_a_task")))

    assert scored.answer is not None
    assert not scored.correct


def test_score_no_outcome_matches_escalate_expectation() -> None:
    case = _case(expect=evalset.EXPECT_ESCALATE)

    scored = evalset.score(case, _result(None, detail="confidence below floor"))

    assert scored.correct
    assert scored.answer is None
    assert scored.detail == "confidence below floor"


def test_score_no_outcome_against_real_expectation_is_wrong() -> None:
    scored = evalset.score(_case(), _result(None, detail="provider error"))

    assert not scored.correct
    assert scored.answer is None


# ---------- aggregation ----------


def test_summarize_groups_by_call_in_first_seen_order() -> None:
    reply = _case(call=AGENT_FIELDS, expect="confirm", listing="", args={}, outcomes=())
    scored = [
        evalset.score(_case(), _result(_answer("taobao/buy"))),
        evalset.score(reply, _result(None)),
        evalset.score(_case(), _result(_answer("not_a_task"))),
    ]

    reports = evalset.summarize(scored)

    assert [r.call for r in reports] == [PARSE_TASK, AGENT_FIELDS]
    assert (reports[0].n, reports[0].correct, reports[0].wrong) == (2, 1, 1)
    assert (reports[1].n, reports[1].escalated) == (1, 1)


def test_report_rates() -> None:
    scored = [
        evalset.score(_case(), _result(_answer("taobao/buy"))),
        evalset.score(_case(), _result(None)),
    ]

    (report,) = evalset.summarize(scored)

    assert report.accuracy == pytest.approx(0.5)
    assert report.escalation_rate == pytest.approx(0.5)


def test_reliability_buckets_answered_cases_only() -> None:
    scored = [
        evalset.score(_case(), _result(_answer("taobao/buy", 0.95))),
        evalset.score(_case(), _result(_answer("not_a_task", 0.92))),
        evalset.score(_case(), _result(None)),
    ]

    table = evalset.reliability(scored)

    assert table[-1] == (0.9, 1.0, 2, 1)
    assert sum(n for _, _, n, _ in table) == 2


def test_reliability_confidence_one_lands_in_top_bin() -> None:
    scored = [evalset.score(_case(), _result(_answer("taobao/buy", 1.0)))]

    table = evalset.reliability(scored)

    assert table[-1] == (0.9, 1.0, 1, 1)
