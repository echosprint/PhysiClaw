"""`physiclaw conductor` — page-identity + decision tooling.

Offline-first: `extract` and `match --session` work from recorded
sessions with no hardware; `match --live` and `propose --live` do one
`peek` against a running `physiclaw mcp` server, same connection idiom
as macro rehearsal. `eval` replays labeled model calls (needs provider
credentials, not hardware) — the threshold-tuning surface. Nothing here
touches the engine runtime.
"""

import asyncio
from pathlib import Path

import typer

from physiclaw.cli._format import exit_error
from physiclaw.common.text import read_text

conductor_app = typer.Typer(no_args_is_help=True)


def _input_screens(session: str | None, listing: Path | None, live: bool) -> list:
    """Screens from exactly one input source: a recorded session (suffix
    resolved via the shared `logs <suffix>` convention), a listing file,
    or one live peek."""
    from physiclaw.common.listing import Screen
    from physiclaw.conductor import corpus

    if live:
        listings = [asyncio.run(_live_listing())]
    elif listing is not None:
        listings = [read_text(listing)]
    elif session is not None:
        listings = corpus.session_listings(_resolve_sid(session))
    else:
        exit_error("need one of --session / --listing / --live", code=2)
    return [Screen.read(t) for t in listings]


async def _live_listing() -> str:
    from physiclaw.agent.engine.mcp_tool import McpClient
    from physiclaw.common import verdict

    try:
        async with McpClient() as mcp:
            blocks = await mcp.call_tool("peek", {})
    except ConnectionError as e:
        # Same arm as macro rehearsal: --live needs `physiclaw mcp` running.
        exit_error(f"{e}\nstart the server first: physiclaw mcp")
    return verdict.screen_text(blocks)


def _resolve_sid(suffix: str) -> str:
    from physiclaw.agent.trace.store import find_session_dirs
    from physiclaw.common import paths

    matches = find_session_dirs(paths.engine_sessions_dir(), suffix)
    if len(matches) == 1:
        return matches[0].name
    if not matches:
        exit_error(f"no session matches {suffix!r}")
    exit_error(f"ambiguous session {suffix!r}: {', '.join(m.name for m in matches)}")


@conductor_app.command()
def extract(
    session: str = typer.Argument(help="session id (suffix ok if unique)"),
    out: Path = typer.Option(..., "--out", help="corpus JSONL to write"),
) -> None:
    """Dump a session's listings to a corpus file with '?' labels to edit."""
    from physiclaw.conductor import corpus

    listings = corpus.session_listings(_resolve_sid(session))
    corpus.write_corpus(
        out, [corpus.CorpusItem(label=corpus.UNLABELED, listing=t) for t in listings]
    )
    typer.echo(f"wrote {len(listings)} listings → {out} (edit the '?' labels)")


# Bounded replay fan-out for `eval`.
_MICRO_FANOUT = 4


def _resolve_micro_ref(override: str | None = None) -> str:
    """The model the micro tier actually runs — the engine wiring's own
    resolution (`[conductor] micro_model`, else the session model),
    optionally overridden — so a replay can never silently measure a
    different model than the runtime wires."""
    from physiclaw.common.config import CONFIG, model_ref

    if override:
        return override
    try:
        return CONFIG.conductor.micro_model or model_ref()
    except RuntimeError as e:
        exit_error(str(e))


async def _micro_batch(ref: str, requests: list) -> list:
    """Run requests through one MicroCaller with the shared fan-out."""
    from physiclaw.common.config import CONFIG, parse_model_ref
    from physiclaw.conductor.micro import MicroCaller
    from physiclaw.provider import make_provider

    pid, mid = parse_model_ref(ref)
    provider = make_provider(pid, mid)
    caller = MicroCaller(provider, confidence_floor=CONFIG.conductor.micro_confidence)
    gate = asyncio.Semaphore(_MICRO_FANOUT)

    async def one(req):
        async with gate:
            return await caller.run(req)

    try:
        return await asyncio.gather(*(one(r) for r in requests))
    finally:
        await provider.aclose()


@conductor_app.command()
def match(
    app: str = typer.Argument(help="app pack whose pages to match against"),
    session: str = typer.Option(None, "--session", help="replay a recorded session"),
    listing: Path = typer.Option(None, "--listing", help="a listing text file"),
    live: bool = typer.Option(False, "--live", help="one peek via `physiclaw mcp`"),
) -> None:
    """Print the matcher's verdict for each input screen."""
    from physiclaw.common.paths import PACK_FILENAME
    from physiclaw.conductor import pages
    from physiclaw.conductor.match import match_screen

    prints = pages.prints_for_app(app)
    if not prints:
        exit_error(
            f"no pages declared for app {app!r} (playbooks/{app}/{PACK_FILENAME} `pages:`)"
        )
    for i, screen in enumerate(_input_screens(session, listing, live)):
        v = match_screen(screen, prints)
        typer.echo(
            f"[{i:3d}] {v.kind:8s} {v.page_id or '-':24s} "
            f"score={v.score:.2f} runner={v.runner_up:.2f} dy={v.dy:+.2f}  {v.detail}"
        )


# Guided capture: enough observations per page for a median position and
# a real spread (capture floors the tolerance below 2, and MIN_FREQ needs
# an anchor in most of them), while staying a short chore at the desk.
GUIDED_SHOTS = 4


@conductor_app.command()
def calibrate(
    app: str = typer.Argument(help="app pack to calibrate"),
    corpus_file: Path = typer.Argument(None, help="labeled corpus JSONL"),
    guided: bool = typer.Option(
        False, "--guided", help="capture live, page by page, with prompts"
    ),
    shots: int = typer.Option(
        GUIDED_SHOTS, "--shots", help="live peeks per page (--guided)"
    ),
) -> None:
    """Capture geometry + thresholds for an app.

    From a labeled corpus: lines labeled `<app>.<page>` are that page's
    genuine observations; any other non-'?' label is a hard negative.

    With `--guided`: put the phone on each declared page when asked and
    it peeks live — the way to calibrate pages a recorded session can't
    easily contain (the lock screen, say). Needs `physiclaw mcp` running.

    Either way, writes `learned/pages/<app>.json` and prints the per-page
    report."""
    from physiclaw.conductor import capture, corpus, pages

    decls = pages.scan_app_decls(app)
    if not decls:
        exit_error(f"no pages declared for app {app!r}")
    if guided:
        if corpus_file is not None:
            exit_error("--guided captures live; drop the corpus argument", code=2)
        negatives: list = []
        by_page = _guided_capture(app, sorted(decls), shots)
        if not by_page:
            exit_error("no pages captured")
    else:
        if corpus_file is None:
            exit_error("need a corpus file, or --guided", code=2)
        items = corpus.read_corpus(corpus_file)
        by_page, negatives = corpus.partition(items, app, set(decls))

    learned, reports, warnings = capture.capture_app(app, decls, by_page, negatives)
    for w in warnings:
        typer.echo(f"  ⚠ {w}")
    for r in reports:
        typer.echo(_report_line(r))
    if learned:
        pages.save_learned(app, learned)
        typer.echo(f"saved learned/pages/{app}.json ({len(learned)} pages)")


def _guided_capture(app: str, page_names: list[str], shots: int) -> dict[str, list]:
    """Walk the operator through each declared page, peeking `shots`
    times per page. A page can be skipped (an OS state you can't stage
    right now) — capture treats absent observations as "not calibrated",
    which is exactly right, so a partial run is useful rather than
    broken."""
    by_page: dict[str, list] = {}
    typer.echo(f"Guided capture for {app!r} — {shots} peeks per page.")
    for name in page_names:
        typer.echo(f"\n{app}.{name}")
        answer = typer.prompt(
            "  put the phone on this page, then press Enter ('s' to skip)",
            default="",
            show_default=False,
        )
        if answer.strip().lower() == "s":
            typer.echo("  skipped")
            continue
        screens = []
        for i in range(shots):
            (screen,) = _input_screens(None, None, True)
            if not screen.readable:
                typer.echo(f"  [{i + 1}/{shots}] unreadable view — skipping this peek")
                continue
            screens.append(screen)
            typer.echo(f"  [{i + 1}/{shots}] {len(screen.rows)} rows")
        if screens:
            by_page[name] = screens
    return by_page


def _report_line(r) -> str:
    imax = "-" if r.impostor_max is None else f"{r.impostor_max:.2f}"
    flag = "" if r.separable else "  ⚠ INSEPARABLE"
    return (
        f"{r.page}: obs={r.observations} anchors={r.anchors_learned} "
        f"thr={r.threshold:.2f} genuine_min={r.genuine_min:.2f} "
        f"impostor_max={imax}{flag}"
    )


@conductor_app.command("eval")
def eval_cmd(
    cases: Path = typer.Argument(
        help="eval case file (JSONL — see conductor/evalset.py)"
    ),
    model: str = typer.Option(
        None,
        "--model",
        help="provider/model to run against (default: the micro tier, "
        "else the session model — run twice to compare tiers)",
    ),
) -> None:
    """Replay labeled decision cases and report per-call accuracy, the
    micro-escalation rate, token cost, and a confidence-reliability
    table — the Goal-2 KPI surface. Baseline it before prompt changes,
    re-run after."""
    from physiclaw.conductor import evalset

    try:
        suite = evalset.read_cases(cases)
    except (OSError, ValueError) as e:
        exit_error(str(e))
    ref = _resolve_micro_ref(model)
    requests = [evalset.build(c, f"eval-{i}") for i, c in enumerate(suite)]

    typer.echo(f"model: {ref}   cases: {len(suite)}")
    results = asyncio.run(_micro_batch(ref, requests))
    scored = [evalset.score(case, result) for case, result in zip(suite, results)]
    for i, s in enumerate(scored):
        mark = "ok " if s.correct else ("esc" if s.answer is None else "✗  ")
        got = s.answer if s.answer is not None else f"(escalate: {s.detail})"
        conf = f" conf={s.confidence:.2f}" if s.confidence is not None else ""
        typer.echo(
            f"[{i:3d}] {mark} {s.case.call:13s} expect={s.case.expect!r} "
            f"got={got!r}{conf}"
        )
    typer.echo("")
    for r in evalset.summarize(scored):
        typer.echo(
            f"{r.call:13s} n={r.n:3d} acc={r.accuracy:.0%} wrong={r.wrong} "
            f"escalated={r.escalated} ({r.escalation_rate:.0%})  "
            f"{r.prompt_tokens}+{r.completion_tokens}tok {r.elapsed_ms}ms"
        )
    typer.echo("\nreliability (confidence → correct, answered cases):")
    for lo, hi, n, correct in evalset.reliability(scored):
        rate = f"{correct / n:.0%}" if n else "-"
        typer.echo(f"  [{lo:.1f},{hi:.1f}) n={n:3d} correct={rate}")


@conductor_app.command()
def propose(
    session: str = typer.Option(None, "--session", help="replay a recorded session"),
    listing: Path = typer.Option(None, "--listing", help="a listing text file"),
    live: bool = typer.Option(False, "--live", help="one peek via `physiclaw mcp`"),
) -> None:
    """Suggest anchor declarations from screens — prune by eye into
    the pack's `pages:` section."""
    from physiclaw.conductor.capture import propose_anchors

    for i, screen in enumerate(_input_screens(session, listing, live)):
        candidates = propose_anchors(screen)
        typer.echo(f"[{i:3d}] candidates: {', '.join(repr(s) for s in candidates)}")
