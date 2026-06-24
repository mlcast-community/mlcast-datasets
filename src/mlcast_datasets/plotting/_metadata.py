"""Dataset introspection utilities for automatic CRS, variable, and extent detection."""

from __future__ import annotations

import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import xarray as xr


def select_plot_variable(ds: xr.Dataset) -> str:
    """Find the first 3-D (time, y, x) data variable in the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset to inspect.

    Returns
    -------
    str
        Name of the first data variable that has a ``time`` dimension and
        at least three dimensions.

    Raises
    ------
    ValueError
        If no suitable variable is found.
    """
    for var_name, da in ds.data_vars.items():
        if "time" in da.dims and da.ndim >= 3:
            return var_name
    raise ValueError("Could not find a 2D spatial variable with a time dimension.")


def get_data_crs(ds: xr.Dataset, var_name: str | None = None) -> ccrs.CRS:
    """Extract a cartopy CRS from the ``grid_mapping`` attribute of a variable.

    Falls back to ``PlateCarree`` if no ``grid_mapping`` is found.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing the variable and its grid-mapping variable.
    var_name : str or None, optional
        Name of the data variable whose ``grid_mapping`` attribute is read.
        Auto-detected via :func:`select_plot_variable` when ``None``.

    Returns
    -------
    cartopy.crs.CRS
        The coordinate reference system parsed from the ``crs_wkt``
        attribute, or ``PlateCarree`` as a fallback.
    """
    if var_name is None:
        var_name = select_plot_variable(ds)
    grid_mapping_name = ds[var_name].attrs.get("grid_mapping")
    if grid_mapping_name and grid_mapping_name in ds:
        crs_wkt = ds[grid_mapping_name].attrs.get("crs_wkt")
        if crs_wkt:
            return ccrs.Projection(crs_wkt)
    return ccrs.PlateCarree()


def get_domain_extent(ds: xr.Dataset, var_name: str | None = None) -> list[float]:
    """Compute the geographic bounding box in PlateCarree degrees.

    Uses 2-D ``lat``/``lon`` coordinate variables if present, otherwise
    reprojects the 1-D spatial coordinates from the native CRS.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with spatial coordinate variables.
    var_name : str or None, optional
        Name of the data variable used to determine spatial dimensions and
        CRS. Auto-detected via :func:`select_plot_variable` when ``None``.

    Returns
    -------
    list of float
        Bounding box as ``[lon_min, lon_max, lat_min, lat_max]`` in
        decimal degrees (PlateCarree).

    Raises
    ------
    ValueError
        If the variable does not have exactly two spatial dimensions.
    """
    if "lat" in ds and "lon" in ds:
        lat = ds["lat"].values
        lon = ds["lon"].values
        if lat.ndim == 2:
            return [
                float(np.nanmin(lon)),
                float(np.nanmax(lon)),
                float(np.nanmin(lat)),
                float(np.nanmax(lat)),
            ]
        elif lat.ndim == 1:
            return [
                float(lon.min()),
                float(lon.max()),
                float(lat.min()),
                float(lat.max()),
            ]

    # Fallback: use 1D spatial coordinates with CRS reprojection
    if var_name is None:
        var_name = select_plot_variable(ds)
    da = ds[var_name]
    spatial_dims = [d for d in da.dims if d != "time"]
    if len(spatial_dims) != 2:
        raise ValueError(f"Expected two spatial dims, got {spatial_dims}")
    y_dim, x_dim = spatial_dims

    x = ds[x_dim].values
    y = ds[y_dim].values
    data_crs = get_data_crs(ds, var_name)
    plate_carree = ccrs.PlateCarree()

    corners = np.array(
        [
            [x.min(), y.min()],
            [x.max(), y.min()],
            [x.min(), y.max()],
            [x.max(), y.max()],
        ]
    )
    transformed = plate_carree.transform_points(data_crs, corners[:, 0], corners[:, 1])
    return [
        float(transformed[:, 0].min()),
        float(transformed[:, 0].max()),
        float(transformed[:, 1].min()),
        float(transformed[:, 1].max()),
    ]


def infer_time_step_minutes(ds: xr.Dataset, sample_size: int = 128) -> float:
    """Infer the dominant time step in minutes from the first timestamps.

    Computes the median of consecutive time differences over the first
    *sample_size* timesteps.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with a ``time`` coordinate.
    sample_size : int, optional
        Number of leading timesteps to consider. Default is 128.

    Returns
    -------
    float
        Median time step in minutes, or ``np.nan`` if it cannot be
        determined (e.g. fewer than 2 timesteps).
    """
    sample = pd.DatetimeIndex(ds["time"].isel(time=slice(0, sample_size)).values)
    if len(sample) < 2:
        return np.nan
    deltas = sample.to_series().diff().dropna()
    if deltas.empty:
        return np.nan
    return deltas.median().total_seconds() / 60.0


def get_variable_label(ds: xr.Dataset, var_name: str | None = None) -> str:
    """Build a human-readable axis label from variable attributes.

    Combines ``long_name`` and ``units`` into a string such as
    ``'Precipitation rate (mm h⁻¹)'``.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    var_name : str or None, optional
        Name of the variable to label. Auto-detected via
        :func:`select_plot_variable` when ``None``.

    Returns
    -------
    str
        Label string in the form ``'{long_name} ({units})'``, or just
        ``'{long_name}'`` if ``units`` is not set.
    """
    if var_name is None:
        var_name = select_plot_variable(ds)
    attrs = ds[var_name].attrs
    long_name = attrs.get("long_name", var_name)
    units = attrs.get("units", "")
    if units:
        return f"{long_name} ({units})"
    return long_name
