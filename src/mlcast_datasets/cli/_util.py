"""Helpers shared by the commands."""

from __future__ import annotations

import sys
import time
from pathlib import Path


def add_name(parser) -> None:
    parser.add_argument(
        "name", help="dataset name, e.g. it_dpc_sri_5min (see `mlcast-datasets list`)"
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


class ProgressPrinter:
    """Print copy progress to stderr, at most every two seconds."""

    def __init__(self, total: int):
        self.total = total
        self.last = 0.0

    def __call__(self, done: int, n_bytes: int) -> None:
        now = time.monotonic()
        if done < self.total and now - self.last < 2:
            return
        self.last = now
        tty = sys.stderr.isatty()
        end = "\r" if tty and done < self.total else "\n"
        percent = 100 * done / max(self.total, 1)
        print(
            f"  {done:,}/{self.total:,} objects ({percent:.1f}%), "
            f"{format_bytes(n_bytes)} written",
            end=end,
            file=sys.stderr,
            flush=True,
        )


def confirm(question: str, assume_yes: bool) -> bool:
    """Ask on a terminal; elsewhere require ``--yes``."""
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        print(f"{question} Pass --yes to confirm.", file=sys.stderr)
        return False
    return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
