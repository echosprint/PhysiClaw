"""The handover brief — the conductor's distilled report to the model.

A driver that stops mints ONE last synthesized ``[note, peek]`` turn
whose note carries these renderings, then answers None forever. Before
this, the handover reason lived only in the process log: the model took
over blind and re-derived the walk from raw synthesized turns, and
imperatives like the deny path's back-out instruction never reached it
at all. The note summary is exactly what compaction preserves, so the
brief outlives the turns it summarizes; the peek hands the model the
fresh view it would have had to take anyway (and on a dead phone its
error result is itself the evidence).

Pure text: the drivers pass state in, nothing here reads the world.
The reason string arrives verbatim; when a path owes the model an
imperative (the deny back-out, the unlock doctrine), the caller already
wrote it into the reason, so this module never invents instructions.
"""


def walk_brief(
    reason: str,
    *,
    app: str,
    playbook: str,
    node: str | None,
    idx: int,
    nodes: int,
    outputs: dict[str, str],
    consented: float | None,
) -> str:
    """A handed-over walk's report: why, where, and every piece of walk
    state the model would otherwise re-derive from raw turns."""
    where = (
        f"node {node} ({idx + 1}/{nodes})"
        if node is not None
        else f"past the last node ({nodes}/{nodes})"
    )
    parts = [
        f"conductor handing over: {reason}.",
        f"Walk {app}/{playbook} stopped at {where}.",
    ]
    if outputs:
        decided = ", ".join(f"{k}={v!r}" for k, v in outputs.items())
        parts.append(f"Decisions so far: {decided}.")
    if consented is not None:
        # Consent is CONSUMED by firing (program.py), so a consented
        # value surviving to the brief proves the payment did NOT fire.
        parts.append(
            f"The user consented to ¥{consented:g}; the payment has NOT been made."
        )
    parts.append(
        "The synthesized turns above are the walk so far; this turn's "
        "peek shows the current screen. Verify state before acting."
    )
    return " ".join(parts)


def completion_brief(
    app: str,
    playbook: str,
    nodes: int,
) -> str:
    """A completed walk's report: the task portion is done; what remains
    is the model's wrap-up (report to the user, close the session)."""
    parts = [f"conductor: walk {app}/{playbook} completed ({nodes}/{nodes} nodes)."]
    parts.append(
        "This turn's peek shows the final screen. Report the outcome to "
        "the user and wrap up."
    )
    return " ".join(parts)


def boot_brief(reason: str) -> str:
    """A quit boot's report: the overture holds no walk state, so the
    brief is the reason plus where the session stands."""
    return (
        f"conductor handing over: {reason}. The boot could not reach the "
        "user's thread; this turn's peek shows where the phone is now. "
        "Take the session from there."
    )
