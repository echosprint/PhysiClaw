"""Recorded-session addressing for the CLI — one convention, one home.

Every command that reads a recorded session (`logs`, `playbooks
replay`, `playbooks pages match/extract`, `studio --session`) takes the
id or any unique suffix of it, resolved against the engine's session
directory the same way.
"""

from physiclaw.cli._format import exit_error


def resolve_sid(suffix: str) -> str:
    """A session id from a unique suffix. Exits with the ambiguity, or
    with "no session matches", the way every CLI reader reports it."""
    from physiclaw.agent.trace.store import find_session_dirs
    from physiclaw.common import paths

    matches = find_session_dirs(paths.engine_sessions_dir(), suffix)
    if len(matches) == 1:
        return matches[0].name
    if not matches:
        exit_error(f"no session matches {suffix!r}")
    exit_error(f"ambiguous session {suffix!r}: {', '.join(m.name for m in matches)}")
