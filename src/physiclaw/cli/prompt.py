"""``physiclaw prompt`` — dump the assembled SYSTEM prompt / full LLM request.

A debugging aid: `prompt system` prints the SYSTEM prompt exactly as a session
would build it; `prompt request` prints that plus the turn-0 message array (the
full request handed to the model). Both go through the engine's own assembly
(`assemble.build_prompt_bundle` / `assemble.build_initial_messages`) so the
output can't drift from what the running agent actually sends. Pass
`--save-as FILE` to write the dump to a file instead of stdout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

prompt_app = typer.Typer(
    no_args_is_help=True,
    help="Dump the SYSTEM prompt / full LLM request for inspection.",
)

_SaveAs = Annotated[
    Path | None,
    typer.Option("--save-as", help="Write the dump to this file instead of stdout."),
]


def _provider_id() -> str:
    """The active provider id, or '' when no model is configured (the SYSTEM
    prompt only uses it to pick the provider's reasoning-format fragment)."""
    from physiclaw.common.config import model_ref, parse_model_ref

    try:
        return parse_model_ref(model_ref())[0]
    except (RuntimeError, ValueError):
        return ""


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        text = getattr(block, "text", None)
        parts.append(text if text is not None else f"[{type(block).__name__}]")
    return "\n".join(parts)


def _emit(text: str, save_as: Path | None) -> None:
    """Write `text` to `save_as` (with a confirmation line), else print it to
    stdout verbatim (no added newline, so `request` output stays byte-exact)."""
    if save_as is not None:
        from physiclaw.common.text import write_text

        write_text(save_as, text.rstrip("\n") + "\n")
        typer.echo(f"Wrote {len(text)} chars to {save_as}")
    else:
        typer.echo(text, nl=False)


@prompt_app.command()
def system(save_as: _SaveAs = None) -> None:
    """Print the SYSTEM prompt (doctrine + tooling + skills + learned screen
    layout + examples + memory)."""
    from physiclaw.agent.engine import assemble

    _emit(assemble.build_prompt_bundle(_provider_id()).system_prompt, save_as)


@prompt_app.command()
def request(save_as: _SaveAs = None) -> None:
    """Print the full turn-0 request: the SYSTEM prompt plus the message array
    (wake trigger + compaction slots + plan/scratchpad/first-run tails)."""
    from physiclaw.agent.engine import assemble
    from physiclaw.agent.engine.session import Session
    from physiclaw.agent.runtime.hook import Trigger

    bundle = assemble.build_prompt_bundle(_provider_id())
    # A representative camera wake — the trigger text is runtime-specific.
    triggers = [Trigger(description="phone screen changed", source="phone")]
    messages = assemble.build_initial_messages(triggers, bundle.system_prompt)

    # Mirror `loop.drive`'s turn-0: tick the plan, then pin the same tail slots.
    session = Session()
    session.plan.tick_turn()
    messages = assemble.apply_request_tails(
        messages,
        session,
        layout_incomplete=bundle.layout_incomplete,
    )

    text = "".join(
        f"\n===== [{type(m).__name__.removesuffix('Message').lower()}] =====\n"
        f"{_content_text(m.content)}\n"
        for m in messages
    )
    _emit(text, save_as)
