"""`physiclaw conductor` — page-identity tooling (matcher, capture).

Offline-first: `extract` and `match --session` work from recorded
sessions with no hardware; `match --live` and `propose --live` do one
`peek` against a running `physiclaw mcp` server, same connection idiom
as macro rehearsal. Nothing here touches the engine runtime.
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
    from physiclaw.agent.conductor import corpus
    from physiclaw.common.listing import Screen

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
    from physiclaw.agent.conductor import corpus

    listings = corpus.session_listings(_resolve_sid(session))
    corpus.write_corpus(
        out, [corpus.CorpusItem(label=corpus.UNLABELED, listing=t) for t in listings]
    )
    typer.echo(f"wrote {len(listings)} listings → {out} (edit the '?' labels)")


@conductor_app.command()
def match(
    app: str = typer.Argument(help="app pack whose pages to match against"),
    session: str = typer.Option(None, "--session", help="replay a recorded session"),
    listing: Path = typer.Option(None, "--listing", help="a listing text file"),
    live: bool = typer.Option(False, "--live", help="one peek via `physiclaw mcp`"),
) -> None:
    """Print the matcher's verdict for each input screen."""
    from physiclaw.agent.conductor import pages
    from physiclaw.agent.conductor.match import match_screen

    prints = pages.prints_for_app(app)
    if not prints:
        exit_error(f"no pages declared for app {app!r} (playbooks/{app}/pages.yml)")
    for i, screen in enumerate(_input_screens(session, listing, live)):
        v = match_screen(screen, prints)
        typer.echo(
            f"[{i:3d}] {v.kind:8s} {v.page_id or '-':24s} "
            f"score={v.score:.2f} runner={v.runner_up:.2f} dy={v.dy:+.2f}  {v.detail}"
        )


@conductor_app.command()
def calibrate(
    app: str = typer.Argument(help="app pack to calibrate"),
    corpus_file: Path = typer.Argument(help="labeled corpus JSONL"),
) -> None:
    """Capture geometry + thresholds for an app from a labeled corpus.

    Lines labeled `<app>.<page>` are that page's genuine observations;
    any other non-'?' label is a hard negative. Writes
    `learned/pages/<app>.json` and prints the per-page report."""
    from physiclaw.agent.conductor import capture, corpus, pages

    decls = pages.scan_app_decls(app)
    if not decls:
        exit_error(f"no pages declared for app {app!r}")
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
    pages.yml."""
    from physiclaw.agent.conductor.capture import propose_anchors

    for i, screen in enumerate(_input_screens(session, listing, live)):
        candidates = propose_anchors(screen)
        typer.echo(f"[{i:3d}] candidates: {', '.join(repr(s) for s in candidates)}")
