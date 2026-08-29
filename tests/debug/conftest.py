"""Shared builders for the debug-harness tests."""

from __future__ import annotations

from physiclaw.common import paths


def write_channel_pages(anchors: tuple[str, ...] = ("MyChat",)) -> None:
    """A minimal channel pack declaring the thread page — what the
    renderer draws anchors from and the matcher scores against."""
    root = paths.playbooks_dir() / "channel"
    root.mkdir(parents=True, exist_ok=True)
    lines = "".join(f'      - "{a}"\n' for a in anchors)
    (root / "PLAYBOOK.yml").write_text(
        "name: channel\ndescription: test channel\n"
        f"pages:\n  thread:\n    anchors:\n{lines}",
        encoding="utf-8",
    )
