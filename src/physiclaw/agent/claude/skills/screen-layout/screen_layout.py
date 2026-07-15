#!/usr/bin/env python3
"""Claude-side CLI for first-run screen-layout capture.

  record — validate + persist the boxes read off a page's screenshot
  status — what's captured, what's still missing

Both are thin wrappers around `physiclaw.agent.engine.screen_layout` — the
same `record()` the native engine's `report_screen_layout` local tool calls,
so `~/.physiclaw/screen-layout/layout.json` stays identical whichever engine
is driving. No MCP tool needed: this runs under the child's `Bash(uv run:*)`
grant, exactly like the `jobs` skill.
"""

import argparse
import sys

from physiclaw.agent.engine import screen_layout


def _parse_box(spec: str) -> tuple[str, list[float]]:
    """`field=l,t,r,b` → ('field', [l, t, r, b])."""
    field, sep, coords = spec.partition("=")
    if not sep:
        raise ValueError(f"bad --box {spec!r}; expected field=l,t,r,b")
    nums = [p.strip() for p in coords.split(",")]
    if len(nums) != 4:
        raise ValueError(f"bad bbox in {spec!r}; expected 4 comma-separated numbers")
    try:
        return field.strip(), [float(n) for n in nums]
    except ValueError:
        raise ValueError(f"bad bbox in {spec!r}; the four values must be numbers")


def _cmd_record(args: argparse.Namespace) -> int:
    last = ""
    for spec in args.box:
        try:
            field, bbox = _parse_box(spec)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        # record() validates one box, merges it into layout.json, re-renders
        # layout.md, and returns a human message (layout so far + what's left,
        # or a re-measure instruction if the box failed its sanity check).
        last = screen_layout.record(args.page, field, bbox, args.app)
        print(f"[{field}] {last.splitlines()[0] if last else 'saved'}")
    if last:
        print()
        print(last)  # the trailing message already reflects cumulative state
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    if screen_layout.is_learned():
        print("Screen layout is fully learned — no first-run capture needed.")
    else:
        print(screen_layout.tail_reminder() or "Not learned; nothing captured yet.")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(
        prog="screen_layout.py",
        description="Claude-side CLI for first-run screen-layout capture.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("record", help="Validate + persist boxes read off a page")
    pr.add_argument("--page", required=True, choices=list(screen_layout.PAGES))
    pr.add_argument(
        "--app",
        help="chat app for the chat pages (wechat, whatsapp, …); omit for spotlight",
    )
    pr.add_argument(
        "--box",
        required=True,
        action="append",
        metavar="field=l,t,r,b",
        help="repeatable — one per box read off this page's screenshot",
    )
    pr.set_defaults(func=_cmd_record)

    ps = sub.add_parser("status", help="Show what's captured and what's still missing")
    ps.set_defaults(func=_cmd_status)

    args = p.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
