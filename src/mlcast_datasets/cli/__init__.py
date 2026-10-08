"""``mlcast-datasets``: list, inspect and download the datasets in the catalog."""

from __future__ import annotations

import argparse
import sys

from . import download, info, list_datasets, path

COMMANDS = {
    "list": list_datasets,
    "info": info,
    "download": download,
    "path": path,
}


def main(argv: list[str] | None = None) -> int:
    from .. import __version__
    from ..entries import DatasetNotFoundError

    parser = argparse.ArgumentParser(
        prog="mlcast-datasets",
        description="List, inspect and download the datasets in the mlcast catalog.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for name, module in COMMANDS.items():
        module.add_arguments(
            commands.add_parser(name, help=module.HELP, description=module.HELP)
        )
    args = parser.parse_args(argv)
    try:
        return COMMANDS[args.command].run(args)
    except (DatasetNotFoundError, FileExistsError, ValueError) as e:
        print(f"mlcast-datasets: error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
