import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mlcast_datasets import cli
from mlcast_datasets.entries import (
    DatasetEntry,
    DatasetNotFoundError,
    get_entry,
    list_entries,
)
from mlcast_datasets.store import summarize
from mlcast_datasets.transfer import (
    copy_command,
    copy_full,
    copy_slice,
    local_path,
    plan_full,
    plan_slice,
)

# 11 steps every 5 min, 00:25 missing
TIMES = pd.date_range("2024-01-01", periods=12, freq="5min").delete(5)
MISSING = pd.DatetimeIndex(["2024-01-01T00:25"])
FORMATS = {"v2": (2, None), "v3": (3, None), "v3-sharded": (3, 4)}


def make_store(path, zarr_format, shards=None, chunks=(1, 6, 8)):
    data = np.random.default_rng(0).random((len(TIMES), 6, 8)).astype("float32")
    ds = xr.Dataset(
        {
            "RR": (
                ("time", "y", "x"),
                data,
                {"standard_name": "rainfall_flux", "units": "mm/h"},
            )
        },
        coords={
            "time": TIMES,
            "y": np.arange(6) * 1000.0,
            "x": ("x", np.arange(8) * 1000.0, {"units": "m"}),
            "missing_times": MISSING,
        },
        attrs={"license": "CC-BY-4.0", "mlcast_dataset_version": "0.1.0"},
    )
    # time coordinate in several chunks, as in DMI's store
    encoding = {"RR": {"chunks": chunks}, "time": {"chunks": (4,)}}
    if shards:
        encoding["RR"]["shards"] = (shards, 6, 8)
    ds.to_zarr(path, zarr_format=zarr_format, encoding=encoding, consolidated=True)
    return ds, DatasetEntry(name="test.radar", url=str(path))


@pytest.fixture(params=list(FORMATS))
def store(request, tmp_path):
    zarr_format, shards = FORMATS[request.param]
    ds, entry = make_store(tmp_path / "remote" / "radar.zarr", zarr_format, shards)
    return ds, entry, shards


def _files(root: Path) -> dict:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file()
    }


def test_full_copy_is_byte_identical_and_resumes(store, tmp_path):
    _, entry, _ = store
    data_dir = tmp_path / "data"
    assert plan_full(entry, data_dir).n_present == 0

    result = copy_full(entry, data_dir)
    assert result.path == local_path(entry, data_dir)
    assert _files(result.path) == _files(Path(entry.url))

    again = copy_full(entry, data_dir)
    assert (again.copied, again.skipped) == (0, result.copied)


def test_slice_matches_source(store, tmp_path):
    ds, entry, shards = store
    plan = plan_slice(entry, tmp_path / "data", "2024-01-01T00:10", "2024-01-01T00:40")
    # [00:10, 00:40) is steps 2 to 6; sharded stores widen it to whole shards
    expected_window = (0, 8) if shards else (2, 7)
    assert (plan.start_index, plan.stop_index) == expected_window

    out = xr.open_zarr(copy_slice(plan).path)
    xr.testing.assert_identical(out["RR"], ds["RR"].isel(time=slice(*expected_window)))
    assert out.attrs["mlcast_subset_source"] == entry.url
    assert out.attrs["mlcast_subset_start_index"] == expected_window[0]
    assert out.attrs["mlcast_subset_stop_index"] == expected_window[1]
    assert out.attrs["license"] == "CC-BY-4.0"
    # 00:25 lies inside the slice
    assert list(out.indexes["missing_times"]) == list(MISSING)


def test_slice_to_the_end_drops_missing_times_outside(store, tmp_path):
    ds, entry, _ = store
    plan = plan_slice(entry, tmp_path / "data", start="2024-01-01T00:45")
    # steps 8 to 10; in the sharded store this is a partial last shard
    assert (plan.start_index, plan.stop_index) == (8, 11)

    out = xr.open_zarr(copy_slice(plan).path)
    xr.testing.assert_identical(out["RR"], ds["RR"].isel(time=slice(8, 11)))
    assert out.sizes["missing_times"] == 0


def test_empty_window_raises(store, tmp_path):
    _, entry, _ = store
    with pytest.raises(ValueError, match="No time steps"):
        plan_slice(entry, tmp_path, "2030-01-01", "2030-01-02")


def test_existing_slice_needs_overwrite(store, tmp_path):
    _, entry, _ = store
    plan = plan_slice(entry, tmp_path / "data", end="2024-01-01T00:10")
    copy_slice(plan)
    with pytest.raises(FileExistsError):
        copy_slice(plan)
    copy_slice(plan, overwrite=True)


def test_spatially_chunked_store_is_refused(tmp_path):
    _, entry = make_store(tmp_path / "tiled.zarr", 2, chunks=(1, 3, 8))
    with pytest.raises(ValueError, match="whole domain"):
        copy_full(entry, tmp_path / "data")


def test_summarize(store):
    _, entry, _ = store
    s = summarize(entry.url)
    assert (s["time_start"], s["time_end"]) == (TIMES[0], TIMES[-1])
    assert s["time_step"] == pd.Timedelta("5min")
    assert s["n_times"] == len(TIMES)
    assert s["n_missing_times"] == 1
    assert s["grid"] == {"y": 6, "x": 8}
    assert s["resolution_m"] == 1000.0


def test_copy_commands():
    entry = DatasetEntry(
        name="p.x",
        url="s3://bucket/a/x.zarr/",
        storage_options={"anon": True, "endpoint_url": "https://example.org"},
    )
    assert copy_command(entry, "data", "s5cmd") == (
        "s5cmd --numworkers 32 --no-sign-request --endpoint-url https://example.org "
        "sync 's3://bucket/a/x.zarr/*' data/bucket/a/x.zarr/"
    )
    assert copy_command(entry, "data", "aws") == (
        "aws s3 sync s3://bucket/a/x.zarr/ data/bucket/a/x.zarr/ "
        "--endpoint-url https://example.org --no-sign-request"
    )
    assert copy_command(entry, "data", "rclone") == (
        "rclone copy --transfers 32 --checkers 32 "
        "\":s3,provider=Other,endpoint='https://example.org':bucket/a/x.zarr\" "
        "data/bucket/a/x.zarr"
    )


def test_catalog_entries():
    entries = list_entries()
    assert entries
    assert all(e.url.startswith("s3://") and e.validator_version for e in entries)
    entry = get_entry("precipitation.it_dpc_sri_5min")
    assert entry.url.startswith("s3://")
    with pytest.raises(DatasetNotFoundError):
        get_entry("it_dpc_sri_5min")  # names are dotted paths


@pytest.fixture
def local_catalog(monkeypatch, tmp_path):
    _, entry = make_store(tmp_path / "remote" / "radar.zarr", 3, shards=4)
    monkeypatch.setattr("mlcast_datasets.entries.list_entries", lambda: [entry])
    return entry


def test_cli_download_and_path(local_catalog, tmp_path, capsys):
    data_dir = str(tmp_path / "data")
    assert cli.main(["path", "test.radar", "--data-dir", data_dir]) == 0
    assert capsys.readouterr().out.strip() == local_catalog.url

    assert cli.main(["download", "test.radar", "--data-dir", data_dir]) == 0
    capsys.readouterr()
    assert cli.main(["path", "test.radar", "--data-dir", data_dir]) == 0
    local = str(local_path(local_catalog, data_dir))
    assert capsys.readouterr().out.strip() == local

    args = [
        "download",
        "test.radar",
        "--data-dir",
        data_dir,
        "--start",
        "2024-01-01T00:45",
    ]
    assert cli.main(args) == 0
    assert (
        capsys.readouterr()
        .out.strip()
        .endswith("radar_20240101T0045-20240101T0055.zarr")
    )


def test_cli_info_json(local_catalog, tmp_path, capsys):
    assert cli.main(["info", "test.radar", "--data-dir", str(tmp_path), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_times"] == len(TIMES)
    assert out["n_missing_times"] == 1
    assert out["time_steps_per_object"] == 4
    assert out["resolution_m"] == 1000.0
    assert out["local_state"] == "not downloaded"


def test_cli_list_json(capsys):
    assert cli.main(["list", "--json"]) == 0
    names = [row["name"] for row in json.loads(capsys.readouterr().out)]
    assert "precipitation.it_dpc_sri_5min" in names


def test_cli_list_tree(monkeypatch, capsys):
    summary = {
        "time_start": pd.Timestamp("2020-01-01"),
        "time_end": pd.Timestamp("2021-01-01"),
        "time_step": pd.Timedelta("5min"),
        "n_times": 10,
        "grid": {"y": 2, "x": 3},
        "resolution_m": 1000.0,
    }
    monkeypatch.setattr("mlcast_datasets.store.summarize", lambda *a: summary)
    assert cli.main(["list", "--details"]) == 0
    tree = capsys.readouterr().err
    assert "precipitation" in tree and "it_dpc_sri_5min" in tree
    assert "2020-01-01 → 2021-01-01  5 min  2 × 3  1 km" in tree


def test_cli_unknown_dataset(local_catalog, capsys):
    assert cli.main(["info", "nope"]) == 1
    assert "No dataset named 'nope'" in capsys.readouterr().err
