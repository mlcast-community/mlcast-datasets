"""``mlcast-datasets list``"""

from __future__ import annotations

import json
import shutil

HELP = "List the datasets in the catalog."


def add_arguments(parser) -> None:
    parser.add_argument(
        "--details",
        action="store_true",
        help="also open each dataset for its time range, time step and grid (slower)",
    )
    parser.add_argument("--json", action="store_true", help="print JSON")


def run(args) -> int:
    from ..entries import list_entries
    from ._util import format_resolution, format_step

    entries = list_entries()
    rows = [
        {
            "name": e.name,
            "validator_version": e.validator_version,
            "description": e.description,
            "url": e.url,
        }
        for e in entries
    ]
    if args.details:
        from ..store import summarize

        for row, entry in zip(rows, entries):
            s = summarize(entry.url, entry.storage_options)
            row.update(
                time_start=str(s["time_start"]),
                time_end=str(s["time_end"]),
                time_step=format_step(s["time_step"]),
                n_times=s["n_times"],
                grid=" × ".join(str(n) for n in s["grid"].values()),
                resolution=format_resolution(s["resolution_m"]),
            )

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    columns = ["name", "validator"]
    if args.details:
        columns += ["start", "end", "step", "grid", "resolution"]
    table = [
        [row["name"].rsplit(".", 1)[-1], row["validator_version"] or "-"]
        + (
            [
                row["time_start"][:10],
                row["time_end"][:10],
                row["time_step"],
                row["grid"],
                row["resolution"],
            ]
            if args.details
            else []
        )
        for row in rows
    ]
    widths = [max(len(c), *(len(r[i]) for r in table)) for i, c in enumerate(columns)]
    width = shutil.get_terminal_size().columns
    used = sum(widths) + 2 * len(widths)
    print(
        "  ".join(c.upper().ljust(w) for c, w in zip(columns, widths)) + "  DESCRIPTION"
    )
    for row, cells in zip(rows, table):
        text = row["description"]
        room = max(width - used, 20)
        if len(text) > room:
            text = text[: room - 1] + "…"
        print("  ".join(c.ljust(w) for c, w in zip(cells, widths)) + "  " + text)
    return 0
