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
    from rich.markup import escape

    from ..entries import get_entry
    from ..store import estimate_data_size, open_dataset, read_layout, summarize
    from ..transfer import is_complete, local_path, open_store
    from ._util import (
        data_dir,
        format_bytes,
        format_resolution,
        format_step,
        grid,
        panel,
    )
    from .console import console

    entry = get_entry(args.name)
    with console.status(f"🔍 Reading {entry.name}"):
        summary = summarize(entry.url, entry.storage_options)
        fs, root = open_store(entry)
        layout = read_layout(fs, root)
        n_objects, est_bytes = estimate_data_size(fs, root, layout)
    block = layout.time_block_size if layout.data_arrays else None
    attrs = summary["attributes"]

    local = local_path(entry, data_dir(args))
    remote_version = attrs.get("mlcast_dataset_version")
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

    def attr(key: str) -> str:
        return escape(str(attrs[key])) if key in attrs else "[dim]not set[/]"

    missing = summary["n_missing_times"]
    variables = ", ".join(
        f"{v['name']} ({v['standard_name'] or '?'}, {v['units'] or '?'})"
        for v in summary["variables"]
    )
    per_object = "step" if block == 1 else f"{block} steps (sharded)"
    state_style = {"downloaded": "green", "not downloaded": "dim"}.get(state, "yellow")

    info = grid()
    info.add_row("Description", f"📝  {escape(entry.description)}")
    info.add_row(
        "Time range",
        f"📅  {summary['time_start']:%Y-%m-%d %H:%M} → "
        f"{summary['time_end']:%Y-%m-%d %H:%M}",
    )
    info.add_row(
        "Time step",
        f"🕒  {format_step(summary['time_step'])}   {summary['n_times']:,} steps"
        + (f", {missing:,} missing" if missing else ""),
    )
    info.add_row(
        "Grid",
        "🌍  "
        + " × ".join(f"{d}={n:,}" for d, n in summary["grid"].items())
        + f"   {format_resolution(summary['resolution_m'])}",
    )
    info.add_row("Variables", f"💧  {escape(variables)}")
    info.add_row(
        "Versions",
        f"🔖  dataset {attr('mlcast_dataset_version')}   validator "
        f"{entry.validator_version or '[dim]not set[/]'} in the catalog, "
        f"{attr('mlcast_dataset_validator_version')} in the dataset",
    )
    info.add_row("License", f"📜  {attr('license')}")
    info.add_row("Created by", f"👤  {attr('mlcast_created_by')}")
    info.add_row("Created with", f"🔧  {attr('mlcast_created_with')}")
    info.add_row(
        "Storage",
        f"🧊  zarr v{layout.zarr_format}, one object per {per_object}   "
        f"{n_objects:,} objects, ~{format_bytes(est_bytes)}",
    )
    info.add_row("URL", f"🔗  {entry.url}")
    info.add_row("Local copy", f"💾  {local}\n    [{state_style}]{state}[/]")
    console.print(panel(info, f"📦 {entry.name}"))
    return 0
