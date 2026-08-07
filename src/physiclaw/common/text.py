"""UTF-8 by default for every text I/O.

``Path.read_text`` / ``write_text`` / ``open`` without an explicit
``encoding=`` fall back to ``locale.getencoding()`` — GBK on Chinese
Windows, cp1252 on Western Windows. That's bricked PhysiClaw twice
already (config.toml in 58aa00d, SKILL.md after). Route every short-
lived text I/O through these helpers so UTF-8 is the only default.

Long-lived append handles (daily log loops in agent/claude/spawn.py,
agent/trace, agent/engine/jobs.py) keep an explicit
``open(path, "a", encoding="utf-8")`` inline — wrapping a stateful
file handle in a helper would be more indirection than it's worth.
"""

from pathlib import Path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, data: str) -> None:
    # newline="\n" pins LF: the default (newline=None) translates \n to
    # os.linesep — CRLF on Windows — breaking the "UTF-8 with LF on every
    # platform" invariant the config and session-log artifacts promise.
    path.write_text(data, encoding="utf-8", newline="\n")


def append_text(path: Path, data: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(data)
