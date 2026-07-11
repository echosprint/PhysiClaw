#!/usr/bin/env python3
"""Claude-side CLI for the jobs subsystem — one entry point, four subcommands.

  create — append a new job to jobs.md
  list   — one line per job, optional --status filter
  get    — full details of one job
  finish — mark a job terminal (done / fail / cancel)

Each subcommand is a thin wrapper around `physiclaw.agent.engine.jobs`
and `physiclaw.agent.engine.job_store`. Those same functions back the
native engine's local tools, so the jobs.md format stays invariant
whichever engine is driving.
"""

import argparse
import sys

from physiclaw.agent.engine import jobs
from physiclaw.agent.engine.job_store import (
    KIND_ONE_TIME,
    KIND_PERIODIC,
    STATUS_CANCEL,
    STATUS_DONE,
    STATUS_FAIL,
    STATUS_FIRED,
    STATUS_PEND,
    load_jobs,
)

_ALL = "all"
_LIST_STATUSES = (
    _ALL,
    STATUS_PEND,
    STATUS_FIRED,
    STATUS_DONE,
    STATUS_FAIL,
    STATUS_CANCEL,
)


def _cmd_create(args: argparse.Namespace) -> int:
    try:
        jobs.create_job(
            id=args.id,
            description=args.description,
            schedule=args.schedule,
            context=args.context,
            kind=args.kind,
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"created {args.id}")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    try:
        js = load_jobs()
    except ValueError as e:
        print(f"error loading jobs.md: {e}", file=sys.stderr)
        return 2

    if args.status != _ALL:
        js = [j for j in js if j.status == args.status]

    if not js:
        print(f"(no jobs with status={args.status})")
        return 0

    for j in js:
        nft = j.next_fire_time or "-"
        desc = j.description.strip().splitlines()[0][:80]
        print(
            f"{j.id}  [{j.status}]  kind={j.kind}  next={nft}  "
            f"schedule={j.schedule!r}  — {desc}"
        )
    return 0


def _cmd_get(args: argparse.Namespace) -> int:
    try:
        j = jobs.get_job(args.id)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print(f"id:               {j.id}")
    print(f"kind:             {j.kind}")
    print(f"status:           {j.status}")
    print(f"schedule:         {j.schedule!r}")
    print(f"description:      {j.description}")
    print(f"context:          {j.context}")
    print(f"next fire time:   {j.next_fire_time or '-'}")
    print(f"last fire time:   {j.last_fire_time or '-'}")
    print(f"execution time:   {j.execution_time or '-'}")
    print(f"execution result: {j.execution_result or '-'}")
    return 0


def _cmd_finish(args: argparse.Namespace) -> int:
    try:
        jobs.finish_job(id=args.id, status=args.status, recap=args.recap)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"finished {args.id} as {args.status}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(
        prog="jobs.py",
        description="Claude-side CLI for the jobs subsystem.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("create", help="Append a new job to jobs.md")
    pc.add_argument("--id", required=True, help="<user>-<topic>-<YYYY-MM-DD>")
    pc.add_argument(
        "--kind", choices=[KIND_ONE_TIME, KIND_PERIODIC], default=KIND_ONE_TIME
    )
    pc.add_argument("--schedule", required=True, help="5-field cron expression")
    pc.add_argument("--description", required=True)
    pc.add_argument("--context", required=True, help="≥10 chars; injected at fire time")
    pc.set_defaults(func=_cmd_create)

    pl = sub.add_parser("list", help="List jobs as one-liners")
    pl.add_argument(
        "--status",
        default=_ALL,
        choices=_LIST_STATUSES,
        help="Filter by status (default: all)",
    )
    pl.set_defaults(func=_cmd_list)

    pg = sub.add_parser("get", help="Show all fields of one job")
    pg.add_argument("--id", required=True)
    pg.set_defaults(func=_cmd_get)

    pf = sub.add_parser("finish", help="Mark a job terminal (done/fail/cancel)")
    pf.add_argument("--id", required=True)
    pf.add_argument(
        "--status", required=True, choices=[STATUS_DONE, STATUS_FAIL, STATUS_CANCEL]
    )
    pf.add_argument("--recap", required=True, help="one-line outcome")
    pf.set_defaults(func=_cmd_finish)

    args = p.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
