"""``mlcast-datasets download``"""

from __future__ import annotations

import sys

from ._util import add_data_dir, add_name

HELP = "Download a dataset, or a time slice of it, to the local data directory."
CONFIRM_BYTES = 1e9


def add_arguments(parser) -> None:
    from ..transfer import COPY_TOOLS

    add_name(parser)
    add_data_dir(parser)
    parser.add_argument(
        "--start",
        help="first time to include, ISO 8601 (e.g. 2023-07-01 or 2023-07-01T12:00); "
        "with --start or --end only that slice is copied, to a separate store",
    )
    parser.add_argument("--end", help="time to stop at, excluded, ISO 8601")
    parser.add_argument(
        "--print-command",
        choices=COPY_TOOLS,
        metavar="TOOL",
        help="print a command that makes the full copy with "
        f"{', '.join(COPY_TOOLS)} instead of copying, e.g. for an HPC data mover",
    )
    parser.add_argument(
        "-y", "--yes", action="store_true", help="do not ask before large downloads"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing slice"
    )


def run(args) -> int:
    from ..entries import get_entry
    from ..transfer import copy_command, copy_full, copy_slice, plan_full, plan_slice
    from ._util import ProgressPrinter, confirm, data_dir, format_bytes

    entry = get_entry(args.name)
    if args.print_command:
        if args.start or args.end:
            raise ValueError("--print-command makes full copies; drop --start/--end")
        print(copy_command(entry, data_dir(args), args.print_command))
        return 0

    if args.start or args.end:
        plan = plan_slice(entry, data_dir(args), args.start, args.end)
        print(
            f"{entry.name}: {plan.first_time} to {plan.last_time}, "
            f"{plan.stop_index - plan.start_index:,} steps, {plan.n_objects:,} "
            f"objects, about {format_bytes(plan.est_bytes)}",
            file=sys.stderr,
        )
        if plan.est_bytes > CONFIRM_BYTES and not confirm("Download?", args.yes):
            return 1
        result = copy_slice(plan, args.overwrite, ProgressPrinter(plan.n_objects))
        print(
            f"Time steps {plan.start_index:,} to {plan.stop_index:,} of the full "
            f"store written to:",
            file=sys.stderr,
        )
        print(result.path)
        return 0

    plan = plan_full(entry, data_dir(args))
    todo = plan.n_objects - plan.n_present
    remaining = plan.est_bytes * todo / max(plan.n_objects, 1)
    print(
        f"{entry.name}: {plan.n_objects:,} data objects, about "
        f"{format_bytes(plan.est_bytes)}; {plan.n_present:,} already present",
        file=sys.stderr,
    )
    if remaining > CONFIRM_BYTES and not confirm(
        f"Download about {format_bytes(remaining)}?", args.yes
    ):
        return 1
    result = copy_full(entry, data_dir(args), ProgressPrinter(plan.n_objects))
    if result.absent:
        print(
            f"{result.absent:,} objects were never written in the source; "
            "they read as the fill value",
            file=sys.stderr,
        )
    print(result.path)
    return 0
