"""``mlcast-datasets path``"""

from __future__ import annotations

import json

from ._util import add_data_dir, add_name

HELP = (
    "Print where to read a dataset from: the local copy if it is downloaded, "
    "otherwise the remote URL."
)


def add_arguments(parser) -> None:
    add_name(parser)
    add_data_dir(parser)
    parser.add_argument(
        "--json",
        action="store_true",
        help="print JSON with the local path, URL and storage options",
    )


def run(args) -> int:
    from ..entries import get_entry
    from ..transfer import is_complete, local_path
    from ._util import data_dir

    entry = get_entry(args.name)
    local = local_path(entry, data_dir(args))
    downloaded = is_complete(local)
    if args.json:
        out = {
            "name": entry.name,
            "local_path": str(local) if downloaded else None,
            "url": entry.url,
            "storage_options": entry.storage_options,
        }
        print(json.dumps(out, indent=2))
    else:
        print(local if downloaded else entry.url)
    return 0
