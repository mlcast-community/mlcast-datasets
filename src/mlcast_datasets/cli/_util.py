"""Helpers shared by the commands."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

from .console import console


def add_name(parser) -> None:
    parser.add_argument(
        "name",
        help="dotted path of the dataset in the catalog, "
        "e.g. precipitation.it_dpc_sri_5min (see `mlcast-datasets list`)",
    )


def add_data_dir(parser) -> None:
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="local data directory (default: $MLCAST_DATA_DIR, or ./data)",
    )


def data_dir(args) -> Path:
    from ..transfer import default_data_dir

    return args.data_dir or default_data_dir()


def format_bytes(n: float) -> str:
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000


def format_step(step) -> str:
    if step is None:
        return "?"
    minutes = step.total_seconds() / 60
    if minutes < 60:
        return f"{minutes:g} min"
    if minutes < 1440:
        return f"{minutes / 60:g} h"
    return f"{minutes / 1440:g} days"


def format_resolution(metres) -> str:
    if metres is None:
        return "?"
    return f"{metres / 1000:g} km" if metres >= 1000 else f"{metres:g} m"


def grid():
    """Two-column grid of labels and values, as in the panels of `mlcast`."""
    from rich.table import Table

    table = Table.grid(padding=(0, 2))
    table.add_column(justify="right", style="bold cyan")
    table.add_column(overflow="fold")  # break long paths instead of cutting them
    return table


def panel(body, title: str, subtitle: str | None = None, border: str = "blue"):
    from rich.panel import Panel

    return Panel(
        body,
        title=f"[bold]{title}[/]",
        subtitle=f"[dim]{subtitle}[/]" if subtitle else None,
        border_style=border,
        expand=False,
    )


@contextmanager
def copy_progress(total: int, description: str):
    """Progress bar for a copy; yields the callback that `transfer` calls."""
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TextColumn("[cyan]{task.fields[written]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    with progress:
        task = progress.add_task(description, total=total, written="")

        def update(done: int, n_bytes: int) -> None:
            progress.update(task, completed=done, written=format_bytes(n_bytes))

        yield update


def confirm(question: str, assume_yes: bool) -> bool:
    """Ask on a terminal; elsewhere require ``--yes``."""
    from rich.prompt import Confirm

    if assume_yes:
        return True
    if not sys.stdin.isatty():
        console.print(f"[yellow]{question} Pass --yes to confirm.[/]")
        return False
    return Confirm.ask(question, console=console, default=False)
