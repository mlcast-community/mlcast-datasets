"""``mlcast-datasets download``"""

from __future__ import annotations

import sys
import time

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
    from ._util import confirm, copy_progress, data_dir, format_bytes, grid, panel
    from .console import console

    entry = get_entry(args.name)
    sliced = bool(args.start or args.end)
    if args.print_command:
        if sliced:
            raise ValueError("--print-command makes full copies; drop --start/--end")
        print(copy_command(entry, data_dir(args), args.print_command))
        return 0

    with console.status(f"🔍 Planning the copy of {entry.short_name}"):
        if sliced:
            plan = plan_slice(entry, data_dir(args), args.start, args.end)
        else:
            plan = plan_full(entry, data_dir(args))
    if sliced and plan.path.exists() and not args.overwrite:
        raise FileExistsError(f"{plan.path} exists; pass --overwrite to replace it")

    info = grid()
    info.add_row("Dataset", f"📦  {entry.short_name}")
    if sliced:
        to_copy = plan.est_bytes
        info.add_row(
            "Time range",
            f"📅  {plan.first_time:%Y-%m-%d %H:%M} → {plan.last_time:%Y-%m-%d %H:%M}",
        )
        info.add_row(
            "Time steps",
            f"🕒  {plan.stop_index - plan.start_index:,}   indices "
            f"{plan.start_index:,} to {plan.stop_index:,} of the full store",
        )
        info.add_row("Objects", f"🧊  {plan.n_objects:,}, ~{format_bytes(to_copy)}")
    else:
        to_copy = plan.est_bytes * (1 - plan.n_present / max(plan.n_objects, 1))
        info.add_row(
            "Objects",
            f"🧊  {plan.n_objects:,}, ~{format_bytes(plan.est_bytes)}   "
            f"{plan.n_present:,} already downloaded",
        )
    info.add_row("Output", f"💾  {plan.path}")
    console.print(panel(info, "📥 mlcast-datasets download", subtitle=entry.url))
    if to_copy > CONFIRM_BYTES and not confirm(
        f"Download ~{format_bytes(to_copy)}?", args.yes
    ):
        return 1

    start_time = time.time()
    with copy_progress(plan.n_objects, "📥 Copying objects") as progress:
        if sliced:
            result = copy_slice(plan, args.overwrite, progress)
        else:
            result = copy_full(entry, data_dir(args), progress)

    done = grid()
    done.add_row(
        "Objects",
        f"✅  {result.copied:,} copied, {format_bytes(result.bytes)} in "
        f"{time.time() - start_time:.1f}s"
        + (f"   {result.skipped:,} already downloaded" if result.skipped else ""),
    )
    if result.absent:
        done.add_row(
            "Absent",
            f"[yellow]⚠️  {result.absent:,} objects were never written in the "
            "source; they read as the fill value[/]",
        )
    done.add_row("Output", f"💾  {result.path}")
    console.print(panel(done, "[green]🎉 download complete[/]", border="green"))
    if not sys.stdout.isatty():
        print(result.path)  # for scripts: path=$(mlcast-datasets download ...)
    return 0
