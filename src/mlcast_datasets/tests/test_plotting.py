"""Unit tests for mlcast_datasets.plotting using a synthetic dataset."""

import matplotlib
import numpy as np
import pytest
import xarray as xr

matplotlib.use("Agg")


@pytest.fixture
def synthetic_ds():
    """Create a minimal synthetic dataset mimicking an mlcast zarr store."""
    rng = np.random.default_rng(42)
    ny, nx, nt = 20, 15, 100

    time = np.arange(
        "2020-01-01", "2020-01-01T08:20", np.timedelta64(5, "m"), dtype="datetime64[ns]"
    )[:nt]
    y = np.arange(ny, dtype=np.float64) * 1000
    x = np.arange(nx, dtype=np.float64) * 1000

    # Simple lat/lon grids (approx equirectangular)
    lon_2d, lat_2d = np.meshgrid(
        np.linspace(10, 12, nx),
        np.linspace(50, 52, ny),
    )

    data = rng.exponential(scale=2.0, size=(nt, ny, nx)).astype(np.float32)
    # Inject some NaN (missing data)
    data[:, :3, :] = np.nan

    # Use UTM 32N via epsg() — its WKT includes BBOX, so ccrs.Projection() round-trip works
    import cartopy.crs as ccrs

    crs_wkt = ccrs.epsg(32632).to_wkt()

    ds = xr.Dataset(
        {
            "RR": xr.DataArray(
                data,
                dims=["time", "y", "x"],
                attrs={
                    "grid_mapping": "crs",
                    "long_name": "Precipitation rate",
                    "standard_name": "rainfall_flux",
                    "units": "mm h-1",
                },
            ),
            "crs": xr.DataArray(
                np.nan,
                attrs={"crs_wkt": crs_wkt},
            ),
            "lat": xr.DataArray(lat_2d, dims=["y", "x"]),
            "lon": xr.DataArray(lon_2d, dims=["y", "x"]),
        },
        coords={
            "time": time,
            "y": y,
            "x": x,
        },
        attrs={
            "title": "Test Dataset",
            "license": "CC-BY-4.0",
            "mlcast_dataset_identifier": "TEST-DATASET",
        },
    )
    return ds


# ---- _metadata tests ----


def test_select_plot_variable(synthetic_ds):
    from mlcast_datasets.plotting._metadata import select_plot_variable

    assert select_plot_variable(synthetic_ds) == "RR"


def test_get_data_crs(synthetic_ds):
    from mlcast_datasets.plotting._metadata import get_data_crs

    crs = get_data_crs(synthetic_ds, "RR")
    assert crs is not None


def test_get_domain_extent(synthetic_ds):
    from mlcast_datasets.plotting._metadata import get_domain_extent

    extent = get_domain_extent(synthetic_ds, "RR")
    assert len(extent) == 4
    lon_min, lon_max, lat_min, lat_max = extent
    assert lon_min < lon_max
    assert lat_min < lat_max


def test_infer_time_step_minutes(synthetic_ds):
    from mlcast_datasets.plotting._metadata import infer_time_step_minutes

    step = infer_time_step_minutes(synthetic_ds)
    assert step == 5.0


def test_get_variable_label(synthetic_ds):
    from mlcast_datasets.plotting._metadata import get_variable_label

    label = get_variable_label(synthetic_ds, "RR")
    assert "Precipitation" in label
    assert "mm" in label


# ---- _stats tests ----


def test_compute_spatial_coverage(synthetic_ds):
    from mlcast_datasets.plotting._stats import compute_spatial_coverage

    frac = compute_spatial_coverage(synthetic_ds, n_samples=20)
    assert frac.shape == (20, 15)
    assert frac.max() <= 1.0
    assert frac.min() >= 0.0
    # First 3 rows are always NaN -> should have 0 coverage
    assert np.all(frac[:3, :] == 0.0)


def test_compute_welford_stats(synthetic_ds):
    from mlcast_datasets.plotting._stats import compute_welford_stats

    stats = compute_welford_stats(synthetic_ds, n_samples=20)
    assert "mean" in stats
    assert "std" in stats
    assert "max_val" in stats
    assert stats["mean"].shape == (20, 15)
    # Mean should be positive where there is data
    valid_mean = stats["mean"][~np.isnan(stats["mean"])]
    assert np.all(valid_mean >= 0)


def test_compute_monthly_stats(synthetic_ds):
    from mlcast_datasets.plotting._stats import compute_monthly_stats

    monthly = compute_monthly_stats(synthetic_ds, n_samples=20)
    assert 1 in monthly
    assert len(monthly[1]) > 0  # January should have data


def test_compute_spatial_coverage_n_samples(synthetic_ds):
    from mlcast_datasets.plotting._stats import compute_spatial_coverage

    frac_full = compute_spatial_coverage(synthetic_ds, n_samples=20)
    frac_fewer = compute_spatial_coverage(synthetic_ds, n_samples=10)
    assert frac_full.shape == frac_fewer.shape


# ---- Figure function tests (just check they produce a Figure) ----


def test_plot_domain_map(synthetic_ds):
    from mlcast_datasets.plotting import plot_domain_map

    fig = plot_domain_map(synthetic_ds, n_coverage_samples=3)
    assert fig is not None
    matplotlib.pyplot.close(fig)


def test_plot_spatial_coverage(synthetic_ds):
    from mlcast_datasets.plotting import plot_spatial_coverage

    fig = plot_spatial_coverage(synthetic_ds, n_samples=20)
    assert fig is not None
    matplotlib.pyplot.close(fig)


def test_plot_monthly_cycle(synthetic_ds):
    from mlcast_datasets.plotting import plot_monthly_cycle

    fig = plot_monthly_cycle(synthetic_ds, n_samples=20)
    assert fig is not None
    matplotlib.pyplot.close(fig)


def test_plot_precipitation_stats(synthetic_ds):
    from mlcast_datasets.plotting import plot_precipitation_stats

    fig_maps, fig_hist = plot_precipitation_stats(synthetic_ds, n_samples=20)
    assert fig_maps is not None
    assert fig_hist is not None
    matplotlib.pyplot.close(fig_maps)
    matplotlib.pyplot.close(fig_hist)


def test_plot_sample_precipitation(synthetic_ds):
    from mlcast_datasets.plotting import plot_sample_precipitation

    fig = plot_sample_precipitation(
        synthetic_ds,
        time_slice=slice("2020-01-01T00:00", "2020-01-01T08:00"),
        time_spacing_hours=1,
        max_frames=4,
    )
    assert fig is not None
    matplotlib.pyplot.close(fig)


def test_plot_temporal_coverage(synthetic_ds):
    from mlcast_datasets.plotting import plot_temporal_coverage

    fig = plot_temporal_coverage(synthetic_ds)
    assert fig is not None
    matplotlib.pyplot.close(fig)


def test_generate_summary_table(synthetic_ds):
    import pandas as pd

    from mlcast_datasets.plotting import generate_summary_table

    df = generate_summary_table(synthetic_ds)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Property", "Value"]
    props = dict(zip(df["Property"], df["Value"]))
    assert "RR" in props["Data variable"]
    assert props["Data type"] == "float32"
    assert "2020" in props["Time range"]
