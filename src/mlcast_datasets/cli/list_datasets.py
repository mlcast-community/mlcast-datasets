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
    from rich.markup import escape
    from rich.text import Text
    from rich.tree import Tree

    from .. import open_catalog
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
            with console.status(f"🔍 Opening {entry.name}"):
                s = summarize(entry.url, entry.storage_options)
            row.update(
                time_start=str(s["time_start"]),
                time_end=str(s["time_end"]),
                time_step=format_step(s["time_step"]),
                n_times=s["n_times"],
                grid=" × ".join(str(n) for n in s["grid"].values()),
                resolution=format_resolution(s["resolution_m"]),
            )
        rows.append(row)

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    # one line per dataset, its columns aligned across the whole tree
    cells = [[row["name"].rsplit(".", 1)[-1]] for row in rows]
    if args.details:
        for row, row_cells in zip(rows, cells):
            row_cells += [
                f"{row['time_start'][:10]} → {row['time_end'][:10]}",
                row["time_step"],
                row["grid"],
                row["resolution"],
            ]
    widths = [max(map(len, column)) for column in zip(*cells)]

    catalog = open_catalog()
    tree = Tree("[bold cyan]📚 mlcast catalog[/]")
    nodes = {"": tree}

    def node(path: str):
        """Tree node of a sub-catalog, created along with its parents."""
        if path not in nodes:
            parent, _, label = path.rpartition(".")
            description = " ".join((catalog[path].description or "").split())
            nodes[path] = node(parent).add(
                f"[bold]{label}[/]  [dim]{escape(description)}[/]"
            )
        return nodes[path]

    for row, row_cells in zip(rows, cells):
        line = "  ".join(c.ljust(w) for c, w in zip(row_cells, widths))
        label = Text(line, no_wrap=True, overflow="ellipsis")
        label.stylize("bold", 0, widths[0])
        label.append(f"  {row['description']}", style="dim")
        node(row["name"].rpartition(".")[0]).add(label)
    console.print(tree)
    if rows:
        console.print(
            f"[dim]Use the dotted path with the other commands, "
            f"e.g. mlcast-datasets info {rows[0]['name']}[/]"
        )
    return 0
