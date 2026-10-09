"""``mlcast-datasets list``"""

from __future__ import annotations

import json

HELP = "List the datasets in the catalog."


def add_arguments(parser) -> None:
    parser.add_argument(
        "--details",
        action="store_true",
        help="also open each dataset for its time range, time step and grid (slower)",
    )
    parser.add_argument("--json", action="store_true", help="print JSON")


def run(args) -> int:
    from rich import box
    from rich.table import Table
    from rich.text import Text

    from ..entries import list_entries
    from ..store import summarize
    from ._util import format_resolution, format_step
    from .console import console

    rows = []
    for entry in list_entries():
        row = {
            "name": entry.name,
            "validator_version": entry.validator_version,
            "description": entry.description,
            "url": entry.url,
        }
        if args.details:
            with console.status(f"🔍 Opening {entry.short_name}"):
                s = summarize(entry.url, entry.storage_options)
            row.update(
                time_start=f"{s['time_start']:%Y-%m-%d}",
                time_end=f"{s['time_end']:%Y-%m-%d}",
                time_step=format_step(s["time_step"]),
                n_times=s["n_times"],
                grid=" × ".join(str(n) for n in s["grid"].values()),
                resolution=format_resolution(s["resolution_m"]),
            )
        rows.append(row)

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    table = Table(
        title="📚 mlcast catalog",
        title_style="bold cyan",
        box=box.SIMPLE_HEAD,
        header_style="bold",
        pad_edge=False,
    )
    columns = ["name", "validator"]
    if args.details:
        columns += ["start", "end", "step", "grid", "resolution"]
    for column in columns:
        table.add_column(
            column, no_wrap=True, style="bold" if column == "name" else None
        )
    table.add_column("description")
    for row in rows:
        cells = [row["name"].rsplit(".", 1)[-1], row["validator_version"] or "-"]
        if args.details:
            cells += [
                row[k]
                for k in ("time_start", "time_end", "time_step", "grid", "resolution")
            ]
        # truncate rather than wrap long descriptions
        description = Text(row["description"], no_wrap=True, overflow="ellipsis")
        table.add_row(*cells, description)
    console.print(table)
    return 0
