"""Copy catalog datasets, or time slices of them, to local disk.

Data objects are copied byte for byte: chunks and shards are never decoded.
A new copy is written to ``<store>.partial`` and renamed when complete, so a
store at the final path is complete.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from math import ceil
from pathlib import Path

import fsspec

from .entries import DatasetEntry
from .store import (
    TIME,
    StoreLayout,
    check_copyable,
    estimate_data_size,
    read_layout,
    s3_options,
)

DATA_DIR_ENV = "MLCAST_DATA_DIR"
BATCH_BYTES = 256 * 2**20
COPY_TOOLS = ("s5cmd", "rclone", "aws")

Progress = Callable[[int, int], None]  # (objects done, bytes written)


@dataclass
class CopyResult:
    path: Path
    copied: int  # data objects written by this run
    skipped: int  # data objects already present locally
    absent: int  # data objects never written in the source (read as the fill value)
    bytes: int


@dataclass
class FullPlan:
    path: Path
    n_objects: int
    n_present: int  # data objects already present locally
    est_bytes: int  # estimated size of all data objects


@dataclass
class SlicePlan:
    entry: DatasetEntry
    path: Path
    start_index: int  # position in the full store, [start_index, stop_index)
    stop_index: int
    first_time: object
    last_time: object
    n_objects: int
    est_bytes: int


def default_data_dir() -> Path:
    """``$MLCAST_DATA_DIR``, or ``./data``."""
    return Path(os.environ.get(DATA_DIR_ENV, "data"))


def local_path(entry: DatasetEntry, data_dir: str | Path) -> Path:
    """Where the full copy of ``entry`` lives: its remote path below ``data_dir``."""
    return Path(data_dir) / entry.url.split("://", 1)[-1].strip("/")


def is_complete(path: Path) -> bool:
    """Whether a complete store exists at ``path``."""
    return (path / ".zmetadata").exists() or (path / "zarr.json").exists()


def open_store(entry: DatasetEntry):
    """Filesystem and root path of the remote store."""
    options = s3_options(entry.url, entry.storage_options)
    fs, root = fsspec.core.url_to_fs(entry.url, **options)
    return fs, root.rstrip("/")


def plan_full(entry: DatasetEntry, data_dir: str | Path) -> FullPlan:
    """Size of a full copy and how much of it is already local."""
    fs, root = open_store(entry)
    layout = read_layout(fs, root)
    check_copyable(layout)
    dest = local_path(entry, data_dir)
    work = _work_dir(dest)
    n_objects, est_bytes = estimate_data_size(fs, root, layout)
    present = sum(
        (work / a.object_key(b)).exists()
        for a in layout.data_arrays
        for b in range(a.n_time_blocks)
    )
    return FullPlan(dest, n_objects, present, est_bytes)


def copy_full(
    entry: DatasetEntry, data_dir: str | Path, progress: Progress | None = None
) -> CopyResult:
    """Copy the whole store byte for byte, skipping data objects already present.

    Running it again on a complete copy refreshes the metadata and the small
    arrays, e.g. after a metadata-only migration of the remote store.
    """
    fs, root = open_store(entry)
    layout = read_layout(fs, root)
    check_copyable(layout)
    dest = local_path(entry, data_dir)
    work = _work_dir(dest)
    n_objects, est_bytes = estimate_data_size(fs, root, layout)
    pairs = (
        (f"{root}/{a.object_key(b)}", work / a.object_key(b))
        for a in layout.data_arrays
        for b in range(a.n_time_blocks)
    )
    copied, skipped, absent, n_bytes = _copy(
        fs, pairs, _batch_size(est_bytes, n_objects), progress=progress
    )
    # coordinates and metadata last: the root metadata marks the copy as complete
    small = [
        p
        for a in layout.arrays.values()
        if a not in layout.data_arrays
        for p in fs.find(f"{root}/{a.name}")
    ]
    meta = [
        f"{root}/{m}" for a in layout.data_arrays for m in layout.metadata_files(a.name)
    ]
    meta += [f"{root}/{m}" for m in layout.metadata_files()]
    _copy(
        fs,
        ((p, work / p[len(root) + 1 :]) for p in small + meta),
        batch_size=100,
        skip_existing=False,
    )
    if work != dest:
        os.replace(work, dest)
    return CopyResult(dest, copied, skipped, absent, n_bytes)


def plan_slice(
    entry: DatasetEntry, data_dir: str | Path, start=None, end=None
) -> SlicePlan:
    """Plan a copy of the time steps in ``[start, end)``.

    The window is widened to whole stored objects, i.e. to whole shards in
    sharded stores.
    """
    from .store import open_dataset

    fs, root = open_store(entry)
    layout = read_layout(fs, root)
    check_copyable(layout)
    times = open_dataset(entry.url, entry.storage_options).indexes[TIME]
    i0 = 0 if start is None else int(times.searchsorted(_naive_utc(start)))
    i1 = len(times) if end is None else int(times.searchsorted(_naive_utc(end)))
    if i1 <= i0:
        raise ValueError(
            f"No time steps in [{start}, {end}); the dataset covers "
            f"{times[0]} to {times[-1]}"
        )
    block = layout.time_block_size
    t0 = i0 // block * block
    t1 = min(len(times), ceil(i1 / block) * block)
    n_objects, est_bytes = estimate_data_size(
        fs, root, layout, t0 // block, ceil(t1 / block)
    )
    path = slice_path(entry, data_dir, times[t0], times[t1 - 1])
    return SlicePlan(
        entry, path, t0, t1, times[t0], times[t1 - 1], n_objects, est_bytes
    )


def slice_path(entry: DatasetEntry, data_dir: str | Path, first, last) -> Path:
    """Path of a slice, next to the full copy, named after its first and last step."""
    base = local_path(entry, data_dir)
    name = f"{base.stem}_{first:%Y%m%dT%H%M}-{last:%Y%m%dT%H%M}{base.suffix}"
    return base.with_name(name)


def copy_slice(
    plan: SlicePlan, overwrite: bool = False, progress: Progress | None = None
) -> CopyResult:
    """Copy a planned time slice to a new, shorter store.

    Data objects are copied as they are and renumbered from zero. The root
    attributes ``mlcast_subset_*`` record the source and the slice's position.
    """
    import zarr

    entry, t0, t1 = plan.entry, plan.start_index, plan.stop_index
    fs, root = open_store(entry)
    layout = read_layout(fs, root)
    block = layout.time_block_size
    first, stop = t0 // block, ceil(t1 / block)

    dest = plan.path
    if dest.exists():
        if not overwrite:
            raise FileExistsError(f"{dest} exists; pass --overwrite to replace it")
        shutil.rmtree(dest)
    work = _work_dir(dest)
    shutil.rmtree(work, ignore_errors=True)

    subset = {
        "mlcast_subset_source": entry.url,
        "mlcast_subset_start_index": t0,
        "mlcast_subset_stop_index": t1,
    }
    if layout.zarr_format == 2:
        _write(work / ".zgroup", fs.cat(f"{root}/.zgroup"))
        attrs = {}
        if fs.exists(f"{root}/.zattrs"):
            attrs = json.loads(fs.cat(f"{root}/.zattrs"))
        _write_json(work / ".zattrs", {**attrs, **subset})
    else:
        group = json.loads(fs.cat(f"{root}/zarr.json"))
        group.pop("consolidated_metadata", None)
        group["attributes"] = {**group.get("attributes", {}), **subset}
        _write_json(work / "zarr.json", group)

    result = CopyResult(dest, 0, 0, 0, 0)
    for array in layout.arrays.values():
        if array in layout.data_arrays:
            _write_array_meta(fs, root, work, layout, array.name, {TIME: t1 - t0})
            pairs = (
                (f"{root}/{array.object_key(b)}", work / array.object_key(b - first))
                for b in range(first, stop)
            )
            copied, _, absent, n_bytes = _copy(
                fs,
                pairs,
                _batch_size(plan.est_bytes, plan.n_objects),
                progress=progress,
            )
            result.copied += copied
            result.absent += absent
            result.bytes += n_bytes
        elif TIME in array.dims:
            # small arrays over time, e.g. the time coordinate: rewrite the slice
            # with the same codecs
            size = {TIME: t1 - t0}
            _write_array_meta(fs, root, work, layout, array.name, size, grid=size)
            index = tuple(
                slice(t0, t1) if d == TIME else slice(None) for d in array.dims
            )
            values = _source_array(entry, array.name)[index]
            zarr.open_array(str(work / array.name), mode="r+")[...] = values
        elif array.name == "missing_times":
            # keep the missing steps that fall inside the slice
            source = _source_array(entry, array.name)
            values = source[...]
            missing = _decode_times(values, source.attrs)
            keep = (missing >= plan.first_time) & (missing <= plan.last_time)
            values = values[keep]
            size = {"missing_times": len(values)}
            grid = {"missing_times": max(len(values), 1)}
            _write_array_meta(fs, root, work, layout, array.name, size, grid=grid)
            if len(values):
                zarr.open_array(str(work / array.name), mode="r+")[...] = values
        else:
            for p in fs.find(f"{root}/{array.name}"):
                _write(work / p[len(root) + 1 :], fs.cat(p))

    zarr.consolidate_metadata(str(work))
    os.replace(work, dest)
    return result


def copy_command(entry: DatasetEntry, data_dir: str | Path, tool: str) -> str:
    """Shell command making the same full copy with an external tool."""
    remote = entry.url.split("://", 1)[-1].strip("/")
    endpoint = entry.storage_options.get("endpoint_url")
    anon = entry.storage_options.get("anon", False)
    dest = shlex.quote(str(local_path(entry, data_dir)))
    if tool == "s5cmd":
        flags = " --no-sign-request" if anon else ""
        flags += f" --endpoint-url {endpoint}" if endpoint else ""
        return f"s5cmd{flags} sync 's3://{remote}/*' {dest}/"
    if tool == "aws":
        flags = f" --endpoint-url {endpoint}" if endpoint else ""
        flags += " --no-sign-request" if anon else ""
        return f"aws s3 sync s3://{remote}/ {dest}/{flags}"
    if tool == "rclone":
        backend = ":s3,provider=Other" + (f",endpoint='{endpoint}'" if endpoint else "")
        return f'rclone copy --transfers 32 --checkers 64 "{backend}:{remote}" {dest}'
    raise ValueError(f"Unknown tool '{tool}', expected one of {', '.join(COPY_TOOLS)}")


def _work_dir(dest: Path) -> Path:
    """Directory a copy is written to: an existing store, else ``<store>.partial``."""
    return dest if dest.exists() else dest.with_name(dest.name + ".partial")


def _batch_size(est_bytes: int, n_objects: int) -> int:
    """Objects per request batch, keeping a batch around ``BATCH_BYTES`` in memory."""
    per_object = est_bytes / max(n_objects, 1)
    return int(max(1, min(1000, BATCH_BYTES // max(per_object, 1))))


def _copy(
    fs,
    pairs: Iterable[tuple[str, Path]],
    batch_size: int,
    skip_existing: bool = True,
    progress: Progress | None = None,
) -> tuple[int, int, int, int]:
    """Copy (remote path, local path) pairs; returns copied, skipped, absent, bytes."""
    copied = skipped = absent = n_bytes = 0
    for batch in _batches(pairs, batch_size):
        if skip_existing:
            todo = [(src, dst) for src, dst in batch if not dst.exists()]
            skipped += len(batch) - len(todo)
        else:
            todo = batch
        found = fs.cat([src for src, _ in todo], on_error="omit") if todo else {}
        for src, dst in todo:
            data = found.get(src)
            if data is None:
                absent += 1
                continue
            _write(dst, data)
            copied += 1
            n_bytes += len(data)
        if progress:
            progress(copied + skipped + absent, n_bytes)
    return copied, skipped, absent, n_bytes


def _batches(pairs: Iterable, size: int) -> Iterator[list]:
    batch = []
    for pair in pairs:
        batch.append(pair)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _write(path: Path, data: bytes) -> None:
    """Write atomically, so an interrupted copy never leaves a truncated object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _write_json(path: Path, doc: dict) -> None:
    _write(path, json.dumps(doc, indent=2).encode())


def _write_array_meta(fs, root, work, layout: StoreLayout, name, sizes, grid=None):
    """Copy an array's metadata with new sizes (and object grid) along some dims."""
    array = layout.arrays[name]
    meta_file = "zarr.json" if layout.zarr_format == 3 else ".zarray"
    meta = json.loads(fs.cat(f"{root}/{name}/{meta_file}"))
    if layout.zarr_format == 3:
        chunk_shape = meta["chunk_grid"]["configuration"]["chunk_shape"]
        codecs = meta.get("codecs", [])
        if grid and any(c.get("name") == "sharding_indexed" for c in codecs):
            raise NotImplementedError(f"Cannot re-grid the sharded array '{name}'")
    else:
        chunk_shape = meta["chunks"]
    for dim, size in sizes.items():
        meta["shape"][array.dims.index(dim)] = size
    for dim, size in (grid or {}).items():
        chunk_shape[array.dims.index(dim)] = size
    _write_json(work / name / meta_file, meta)
    if layout.zarr_format == 2 and fs.exists(f"{root}/{name}/.zattrs"):
        _write(work / name / ".zattrs", fs.cat(f"{root}/{name}/.zattrs"))


def _source_array(entry: DatasetEntry, name: str):
    import zarr

    url = f"{entry.url.rstrip('/')}/{name}"
    options = (
        s3_options(entry.url, entry.storage_options) if "://" in entry.url else None
    )
    return zarr.open_array(url, mode="r", storage_options=options or None)


def _decode_times(values, attrs):
    """Decode CF-encoded times, e.g. the raw ``missing_times`` values."""
    import xarray as xr

    attrs = {k: v for k, v in dict(attrs).items() if not k.startswith("_ARRAY")}
    ds = xr.decode_cf(xr.Dataset({"t": (("t",), values, attrs)}))
    return ds["t"].values


def _naive_utc(value):
    """A timestamp as naive UTC, the convention of the catalog's time coordinates."""
    import pandas as pd

    ts = pd.Timestamp(value)
    return ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo else ts
