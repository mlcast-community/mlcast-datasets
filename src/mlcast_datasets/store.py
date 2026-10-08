"""Read the layout and a summary of a zarr store without loading its data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import ceil

TIME = "time"
SUMMARY_ATTRS = (
    "license",
    "mlcast_dataset_identifier",
    "mlcast_dataset_version",
    "mlcast_dataset_validator_version",
    "mlcast_created_by",
    "mlcast_created_with",
)


@dataclass(frozen=True)
class ArrayLayout:
    """One array of a store and how its objects are keyed."""

    name: str
    dims: tuple[str, ...]
    shape: tuple[int, ...]
    grid: tuple[int, ...]  # shape covered by one stored object: a chunk or a shard
    key_prefix: str
    key_separator: str

    @property
    def time_axis(self) -> int:
        return self.dims.index(TIME)

    @property
    def time_block_size(self) -> int:
        """Time steps per stored object."""
        return self.grid[self.time_axis]

    @property
    def n_time_blocks(self) -> int:
        return ceil(self.shape[self.time_axis] / self.time_block_size)

    def object_key(self, time_block: int) -> str:
        """Key of the object holding ``time_block``; the domain is a single block."""
        index = [0] * len(self.dims)
        index[self.time_axis] = time_block
        coords = self.key_separator.join(map(str, index))
        return f"{self.name}/{self.key_prefix}{coords}"


@dataclass(frozen=True)
class StoreLayout:
    """Arrays of a store, read from its consolidated metadata."""

    zarr_format: int
    arrays: dict[str, ArrayLayout]

    @property
    def data_arrays(self) -> list[ArrayLayout]:
        """Arrays over time and the 2-D domain, stored as one object per time block."""
        return [a for a in self.arrays.values() if TIME in a.dims and len(a.dims) >= 3]

    @property
    def time_block_size(self) -> int:
        sizes = {a.time_block_size for a in self.data_arrays}
        if len(sizes) != 1:
            raise ValueError(f"Data arrays use different time blocks: {sorted(sizes)}")
        return sizes.pop()

    def metadata_files(self, array: str | None = None) -> list[str]:
        """Metadata documents of the root group, or of one array."""
        if self.zarr_format == 3:
            return [f"{array}/zarr.json" if array else "zarr.json"]
        if array:
            return [f"{array}/.zarray", f"{array}/.zattrs"]
        return [".zgroup", ".zattrs", ".zmetadata"]


def read_layout(fs, root: str) -> StoreLayout:
    """Read the layout of the store at ``root`` from its consolidated metadata."""
    if fs.exists(f"{root}/.zmetadata"):
        meta = json.loads(fs.cat(f"{root}/.zmetadata"))["metadata"]
        arrays = {}
        for key, array_meta in meta.items():
            if not key.endswith("/.zarray"):
                continue
            name = key[: -len("/.zarray")]
            dims = meta.get(f"{name}/.zattrs", {}).get("_ARRAY_DIMENSIONS", [])
            arrays[name] = ArrayLayout(
                name=name,
                dims=tuple(dims),
                shape=tuple(array_meta["shape"]),
                grid=tuple(array_meta["chunks"]),
                key_prefix="",
                key_separator=array_meta.get("dimension_separator") or ".",
            )
        return StoreLayout(2, arrays)

    group = json.loads(fs.cat(f"{root}/zarr.json"))
    meta = (group.get("consolidated_metadata") or {}).get("metadata")
    if meta is None:
        raise ValueError(f"{root} has no consolidated metadata")
    arrays = {}
    for name, array_meta in meta.items():
        if array_meta.get("node_type") != "array":
            continue
        encoding = array_meta.get("chunk_key_encoding", {"name": "default"})
        default = encoding["name"] == "default"
        config = encoding.get("configuration", {})
        separator = config.get("separator", "/" if default else ".")
        arrays[name] = ArrayLayout(
            name=name,
            dims=tuple(array_meta.get("dimension_names") or ()),
            shape=tuple(array_meta["shape"]),
            grid=tuple(array_meta["chunk_grid"]["configuration"]["chunk_shape"]),
            key_prefix=f"c{separator}" if default else "",
            key_separator=separator,
        )
    return StoreLayout(3, arrays)


def check_copyable(layout: StoreLayout) -> None:
    """Fail unless the store has the layout the copy relies on (spec §4.1)."""
    if TIME not in layout.arrays or not layout.data_arrays:
        raise ValueError("The store has no data arrays over time")
    for array in layout.data_arrays:
        for axis, (grid, size) in enumerate(zip(array.grid, array.shape)):
            if axis != array.time_axis and grid != size:
                raise ValueError(
                    f"'{array.name}' is split into several chunks across the domain; "
                    "the spec requires one chunk per time step covering the whole domain"
                )
    layout.time_block_size


def estimate_data_size(
    fs, root: str, layout: StoreLayout, first: int = 0, stop: int | None = None
) -> tuple[int, int]:
    """Count the data objects in time blocks ``[first, stop)`` and estimate their size.

    The size is extrapolated from up to eight evenly spaced objects.
    """
    n_objects, n_bytes = 0, 0.0
    for array in layout.data_arrays:
        last = array.n_time_blocks if stop is None else min(stop, array.n_time_blocks)
        n = last - first
        if n <= 0:
            continue
        picks = sorted({first + round(i * (n - 1) / 7) for i in range(8)})
        sizes = []
        for block in picks:
            try:
                sizes.append(fs.info(f"{root}/{array.object_key(block)}")["size"])
            except FileNotFoundError:
                sizes.append(0)
        n_objects += n
        n_bytes += sum(sizes) / len(sizes) * n
    return n_objects, int(n_bytes)


def s3_options(url: str, storage_options: dict | None) -> dict:
    """Storage options with a larger S3 connection pool.

    The default pool of 10 connections caps small-object reads at about 170 per
    second on the catalog's bucket; 64 is about five times faster.
    """
    options = dict(storage_options or {})
    if url.startswith("s3://"):
        options.setdefault("config_kwargs", {"max_pool_connections": 64})
    return options


def open_dataset(url: str, storage_options: dict | None = None):
    """Open a store lazily with xarray, from its consolidated metadata."""
    import xarray as xr

    options = s3_options(url, storage_options) or None
    return xr.open_zarr(url, storage_options=options, consolidated=True)


def summarize(url: str, storage_options: dict | None = None) -> dict:
    """Time range, time step, grid, variables and key attributes of a store."""
    import numpy as np
    import pandas as pd

    ds = open_dataset(url, storage_options)
    times = ds.indexes[TIME]
    step = None
    if len(times) > 1:
        diffs, counts = np.unique(np.diff(times.asi8), return_counts=True)
        step = pd.Timedelta(int(diffs[counts.argmax()]))
    data_vars = [v for v in ds.data_vars.values() if TIME in v.dims and v.ndim >= 3]
    spatial = [d for d in data_vars[0].dims if d != TIME] if data_vars else []
    return {
        "time_start": times[0],
        "time_end": times[-1],
        "time_step": step,
        "n_times": len(times),
        "n_missing_times": ds.sizes.get("missing_times"),
        "grid": {d: ds.sizes[d] for d in spatial},
        "resolution_m": _resolution_m(ds),
        "variables": [
            {
                "name": v.name,
                "standard_name": v.attrs.get("standard_name"),
                "units": v.attrs.get("units"),
            }
            for v in data_vars
        ],
        "attributes": {k: ds.attrs[k] for k in SUMMARY_ATTRS if k in ds.attrs},
    }


def _resolution_m(ds) -> float | None:
    """Grid spacing along x in metres, if x has a length unit."""
    if "x" not in ds.coords or ds.sizes.get("x", 0) < 2:
        return None
    scale = {"m": 1.0, "metre": 1.0, "meter": 1.0, "km": 1000.0}.get(
        str(ds["x"].attrs.get("units", "m")).rstrip("s")
    )
    if scale is None:
        return None
    return abs(float(ds["x"][1] - ds["x"][0])) * scale
