"""`physiclaw conductor` — page-identity + decision tooling.

Offline-first: `extract` and `match --session` work from recorded
sessions with no hardware; `match --live` and `propose --live` do one
`peek` against a running `physiclaw mcp` server, same connection idiom
as macro rehearsal. `micro` replays one decision call over recorded
listings (needs provider credentials, not hardware) — the threshold-
tuning surface. Nothing here touches the engine runtime.
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


@conductor_app.command()
def micro(
    criteria: str = typer.Option("", "--criteria", help="choose_item criteria"),
    question: str = typer.Option("", "--question", help="decide question"),
    outcomes: str = typer.Option(
        "",
        "--outcomes",
        help="decide answers, comma-separated (escalate auto-added)",
    ),
    session: str = typer.Option(None, "--session", help="replay a recorded session"),
    listing: Path = typer.Option(None, "--listing", help="a listing text file"),
    live: bool = typer.Option(False, "--live", help="one peek via `physiclaw mcp`"),
) -> None:
    """Run one decision micro-call over each input screen — the offline
    replay surface for tuning `[conductor] micro_confidence`. Prints the
    outcome, confidence, and token cost per screen."""
    from physiclaw.common.config import CONFIG, model_ref, parse_model_ref
    from physiclaw.conductor.calls import CALLS, ESCALATE
    from physiclaw.conductor.micro import MicroCaller, build_request
    from physiclaw.provider import make_provider

    if bool(criteria) == bool(question):
        exit_error("pass exactly one of --criteria (choose_item) / --question (decide)")
    try:
        # The canonical resolution (env > [agent] model), with the micro
        # tier override on top — the same model the engine's wiring picks,
        # or the tuning replays against the wrong model.
        ref = CONFIG.conductor.micro_model or model_ref()
    except RuntimeError as e:
        exit_error(str(e))
    if criteria:
        call, call_outcomes = "choose_item", CALLS["choose_item"].outcomes
        args = {"criteria": criteria}
    else:
        answers = [o.strip() for o in outcomes.split(",") if o.strip()]
        if ESCALATE not in answers:
            answers.append(ESCALATE)
        if len(answers) < 2:
            exit_error("--outcomes needs at least one answer besides escalate")
        call, call_outcomes = "decide", tuple(answers)
        args = {"question": question}
    screens = _input_screens(session, listing, live)
    requests = [
        build_request(call, "cli", call_outcomes, args, screen) for screen in screens
    ]

    async def _go() -> None:
        pid, mid = parse_model_ref(ref)
        provider = make_provider(pid, mid)
        caller = MicroCaller(
            provider, confidence_floor=CONFIG.conductor.micro_confidence
        )
        gate = asyncio.Semaphore(4)  # bounded fan-out: replays are per-screen

        async def one(req):
            async with gate:
                return await caller.run(req)

        try:
            results = await asyncio.gather(*(one(r) for r in requests))
        finally:
            await provider.aclose()
        for i, res in enumerate(results):
            cost = (
                f"{res.usage.prompt_tokens}+{res.usage.completion_tokens}tok "
                f"{res.elapsed_ms}ms"
            )
            if res.outcome is None:
                typer.echo(f"[{i:3d}] escalate  {res.detail}  ({cost})")
                continue
            picked = f" pick={res.outcome.picked.key!r}" if res.outcome.picked else ""
            typer.echo(
                f"[{i:3d}] {res.outcome.out:10s} conf={res.outcome.confidence:.2f}"
                f"{picked}  {res.outcome.reason}  ({cost})"
            )

    asyncio.run(_go())


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
