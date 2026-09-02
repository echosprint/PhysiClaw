"""Micro-call eval cases — the replay suite behind `physiclaw conductor eval`.

A case file is JSONL, one labeled decision per line — recorded screens
(the `corpus` extraction path) paired with the answer a correct call
gives::

    {"call": "parse_task", "expect": "taobao/buy",
     "args": {"menu": "<menu text>"}, "listing": "<thread text>",
     "outcomes": ["taobao/buy"]}

``expect`` is the raw ANSWER the model should produce, in the call's own
answer space: a playbook ref, "not_a_task" or "scroll_up" for
parse_task, a CONFIRM_OUTS member for confirm_reply. ``outcomes`` names
the caller-supplied arms where the call takes them (parse_task's
playbook refs); calls with fixed answer spaces leave it empty, exactly
like the runtime. ``expect: "(escalate)"`` marks a case whose CORRECT
behavior is no outcome at all (the caller should escalate).

This module owns the file format, scoring, and aggregation — all pure,
so the suite is testable without a provider. The CLI owns the provider
round-trips.
"""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from physiclaw.common.listing import Screen
from physiclaw.common.text import read_text
from physiclaw.conductor.micro import (
    CALL_NAMES,
    DecisionRequest,
    MicroResult,
    build_request,
)

# The expect spelling for "a correct call escalates" — parenthesized so
# it can never collide with a real answer (answer spaces are name/key
# shaped; keys come off screen labels, which the parser strips).
EXPECT_ESCALATE = "(escalate)"

# Reliability buckets: confidence bin edges, half-open [lo, hi) with the
# last bin closed. The 0.5 boundary is where the calibration literature
# puts the overconfidence knee, and the fine bins above it are where the
# floor actually operates.
RELIABILITY_EDGES = (0.0, 0.5, 0.7, 0.8, 0.9, 1.0)


@dataclass(frozen=True)
class EvalCase:
    call: str
    expect: str
    args: dict
    outcomes: tuple[str, ...] = ()
    listing: str = ""
    context: str = ""
    note: str = ""


@dataclass(frozen=True)
class Scored:
    """One case's judged result — everything the reports aggregate.
    `answer is None` means the call escalated (a committed answer is
    always a non-empty string)."""

    case: EvalCase
    answer: str | None
    correct: bool
    confidence: float | None
    prompt_tokens: int
    completion_tokens: int
    elapsed_ms: int
    detail: str


@dataclass(frozen=True)
class CallReport:
    """One call type's aggregate — the eval table row."""

    call: str
    n: int
    correct: int
    wrong: int
    escalated: int
    prompt_tokens: int
    completion_tokens: int
    elapsed_ms: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    @property
    def escalation_rate(self) -> float:
        """No-outcome share — the micro half of the escalation KPI.
        (A wrong answer is worse than an escalation, but it is not one.)"""
        return self.escalated / self.n if self.n else 0.0


def read_cases(path: Path) -> list[EvalCase]:
    """Parse a case file, strictly — a bad line names its number (the
    `corpus.read_corpus` contract). Empty file → error: an eval over
    nothing would report a perfect score."""
    out: list[EvalCase] = []
    known = CALL_NAMES
    for n, line in enumerate(read_text(path).splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError as e:
            raise ValueError(f"{path}:{n}: bad JSON ({e})") from e
        if not isinstance(rec, dict):
            raise ValueError(f"{path}:{n}: case must be a JSON object")
        call = rec.get("call")
        if call not in known:
            raise ValueError(
                f"{path}:{n}: unknown call {call!r} (one of: {', '.join(known)})"
            )
        expect = rec.get("expect")
        if not isinstance(expect, str) or not expect.strip():
            raise ValueError(f"{path}:{n}: `expect` must be a non-empty string")
        args = rec.get("args") or {}
        if not isinstance(args, dict):
            raise ValueError(f"{path}:{n}: `args` must be an object")
        outcomes = rec.get("outcomes") or []
        if not isinstance(outcomes, list):
            raise ValueError(f"{path}:{n}: `outcomes` must be a list")
        out.append(
            EvalCase(
                call=str(call),
                expect=expect.strip(),
                args={str(k): str(v) for k, v in args.items()},
                outcomes=tuple(str(o) for o in outcomes),
                listing=str(rec.get("listing") or ""),
                context=str(rec.get("context") or ""),
                note=str(rec.get("note") or ""),
            )
        )
    if not out:
        raise ValueError(f"{path}: no cases")
    return out


def build(case: EvalCase, node_id: str) -> DecisionRequest:
    """The case's request, through the runtime's one assembler — an eval
    that built requests its own way would measure a different call."""
    return build_request(
        case.call,
        node_id,
        case.outcomes,
        case.args,
        Screen.read(case.listing),
        case.context,
    )


def score(case: EvalCase, result: MicroResult) -> Scored:
    """Judge one result against the case. The judged ANSWER is what the
    model committed in its own answer space — the picked key for a pick
    outcome, the routing arm otherwise — so `expect` reads exactly like
    a line of the call's answer legend."""
    outcome = result.outcome
    answer = None
    if outcome is not None:
        answer = outcome.picked.key if outcome.picked is not None else outcome.out
    correct = (
        answer == case.expect if answer is not None else case.expect == EXPECT_ESCALATE
    )
    return Scored(
        case=case,
        answer=answer,
        correct=correct,
        confidence=outcome.confidence if outcome is not None else None,
        prompt_tokens=result.usage.prompt_tokens,
        completion_tokens=result.usage.completion_tokens,
        elapsed_ms=result.elapsed_ms,
        detail=result.detail,
    )


def summarize(scored: list[Scored]) -> list[CallReport]:
    """Per-call-type aggregate, in first-seen order."""
    order: list[str] = []
    by_call: dict[str, list[Scored]] = {}
    for s in scored:
        if s.case.call not in by_call:
            order.append(s.case.call)
        by_call.setdefault(s.case.call, []).append(s)
    out: list[CallReport] = []
    for call in order:
        rows = by_call[call]
        out.append(
            CallReport(
                call=call,
                n=len(rows),
                correct=sum(1 for s in rows if s.correct),
                wrong=sum(1 for s in rows if s.answer is not None and not s.correct),
                escalated=sum(1 for s in rows if s.answer is None),
                prompt_tokens=sum(s.prompt_tokens for s in rows),
                completion_tokens=sum(s.completion_tokens for s in rows),
                elapsed_ms=sum(s.elapsed_ms for s in rows),
            )
        )
    return out


def reliability(scored: list[Scored]) -> list[tuple[float, float, int, int]]:
    """Confidence-vs-correctness buckets over the ANSWERED cases —
    `(lo, hi, n, correct)` per RELIABILITY_EDGES bin, empty bins
    included so the table shape is stable across runs."""
    counts: Counter = Counter()
    hits: Counter = Counter()
    for s in scored:
        if s.confidence is None:
            continue
        counts[_bin_of(s.confidence)] += 1
        if s.correct:
            hits[_bin_of(s.confidence)] += 1
    return [
        (RELIABILITY_EDGES[i], RELIABILITY_EDGES[i + 1], counts[i], hits[i])
        for i in range(len(RELIABILITY_EDGES) - 1)
    ]


def _bin_of(confidence: float) -> int:
    """Half-open bins, last bin closed (1.0 lands in the top bin)."""
    for i in range(len(RELIABILITY_EDGES) - 2):
        if confidence < RELIABILITY_EDGES[i + 1]:
            return i
    return len(RELIABILITY_EDGES) - 2
