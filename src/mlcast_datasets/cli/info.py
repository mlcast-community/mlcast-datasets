"""``mlcast-datasets info``"""

from __future__ import annotations

import json

from ._util import add_data_dir, add_name

HELP = "Show what a dataset contains, how it is stored and whether it is downloaded."


def add_arguments(parser) -> None:
    add_name(parser)
    add_data_dir(parser)
    parser.add_argument("--json", action="store_true", help="print JSON")


def run(args) -> int:
    from ..entries import get_entry
    from ..store import estimate_data_size, open_dataset, read_layout, summarize
    from ..transfer import is_complete, local_path, open_store
    from ._util import data_dir, format_bytes, format_resolution, format_step

    entry = get_entry(args.name)
    summary = summarize(entry.url, entry.storage_options)
    fs, root = open_store(entry)
    layout = read_layout(fs, root)
    n_objects, est_bytes = estimate_data_size(fs, root, layout)
    block = layout.time_block_size if layout.data_arrays else None

    local = local_path(entry, data_dir(args))
    remote_version = summary["attributes"].get("mlcast_dataset_version")
    if is_complete(local):
        local_version = open_dataset(str(local)).attrs.get("mlcast_dataset_version")
        state = "downloaded"
        if local_version != remote_version:
            state += (
                f", dataset version {local_version} locally but {remote_version} "
                "remotely: download again to refresh the metadata"
            )
    elif local.exists() or local.with_name(local.name + ".partial").exists():
        state = "incomplete: download again to resume"
    else:
        state = "not downloaded"

    if args.json:
        out = {
            "name": entry.name,
            "description": entry.description,
            "url": entry.url,
            "storage_options": entry.storage_options,
            "catalog_validator_version": entry.validator_version,
            **summary,
            "zarr_format": layout.zarr_format,
            "time_steps_per_object": block,
            "n_data_objects": n_objects,
            "estimated_bytes": est_bytes,
            "local_path": str(local),
            "local_state": state,
        }
        print(json.dumps(out, indent=2, default=str))
        return 0

    attrs = summary["attributes"]
    missing = summary["n_missing_times"]
    variables = ", ".join(
        f"{v['name']} ({v['standard_name'] or '?'}, {v['units'] or '?'})"
        for v in summary["variables"]
    )
    per_object = "step" if block == 1 else f"{block} steps (shards)"
    lines = [
        ("time", f"{summary['time_start']} to {summary['time_end']}"),
        (
            "time steps",
            f"{summary['n_times']:,} every {format_step(summary['time_step'])}"
            + (f", {missing:,} missing" if missing else ""),
        ),
        (
            "grid",
            " × ".join(f"{d} {n}" for d, n in summary["grid"].items())
            + f", {format_resolution(summary['resolution_m'])}",
        ),
        ("variables", variables),
        ("dataset version", attrs.get("mlcast_dataset_version", "not set")),
        (
            "validator version",
            f"{entry.validator_version or 'not set'} in the catalog, "
            f"{attrs.get('mlcast_dataset_validator_version', 'not set')} in the dataset",
        ),
        ("license", attrs.get("license", "not set")),
        ("created by", attrs.get("mlcast_created_by", "not set")),
        ("created with", attrs.get("mlcast_created_with", "not set")),
        (
            "storage",
            f"zarr v{layout.zarr_format}, one object per {per_object}, "
            f"{n_objects:,} objects, about {format_bytes(est_bytes)}",
        ),
        ("url", entry.url),
        ("local copy", f"{local} ({state})"),
    ]
    print(entry.name)
    if entry.description:
        print(f"  {entry.description}")
    for key, value in lines:
        print(f"  {key:<18} {value}")
    return 0
