"""Sample precipitation maps from a specific event window."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from ._map_helpers import (
    PLATE_CARREE,
    add_map_features,
    get_precipitation_cmap,
    savefig,
    setup_rcparams,
)
from ._metadata import get_data_crs, get_domain_extent, select_plot_variable


def plot_sample_precipitation(
    ds: xr.Dataset,
    time_slice: slice,
    var_name: str | None = None,
    time_spacing_hours: int = 3,
    max_frames: int = 6,
    title: str | None = None,
    figsize: tuple[float, float] = (12, 9),
    levels: list[float] | None = None,
    colors: list[str] | None = None,
    map_resolution: str = "50m",
    output_path: str | None = None,
) -> Figure:
    """Plot a grid of precipitation maps from a specific event window.

    Selects frames from the time window at a minimum spacing of
    *time_spacing_hours* and renders each on a separate map panel with
    a shared discrete precipitation colorbar.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with 2-D ``lat``/``lon`` coordinate variables.
    time_slice : slice
        Time range to select, e.g.
        ``slice('2023-05-16', '2023-05-17T06:00')``. **Required.**
    var_name : str or None, optional
        Data variable to plot. Auto-detected via
        :func:`~mlcast_datasets.plotting._metadata.select_plot_variable`
        when ``None``.
    time_spacing_hours : int, optional
        Minimum spacing in hours between consecutive displayed frames.
        Default is 3.
    max_frames : int, optional
        Maximum number of map panels to show. Default is 6.
    title : str or None, optional
        Figure super-title. Auto-generated from the first timestamp
        when ``None``.
    figsize : tuple of float, optional
        Figure size ``(width, height)`` in inches. Default is ``(12, 9)``.
    levels : list of float or None, optional
        Discrete colourbar boundaries. Default is
        ``[0, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]``.
    colors : list of str or None, optional
        Hex colour strings for the discrete bins (one fewer than
        *levels*). Default is a 9-colour ramp.
    map_resolution : str, optional
        Natural Earth feature resolution (``'10m'``, ``'50m'``, or
        ``'110m'``). Default is ``'50m'``.
    output_path : str or None, optional
        If given, save the figure to this file path.

    Returns
    -------
    matplotlib.figure.Figure
        The generated multi-panel precipitation figure.
    """
    setup_rcparams()
    if var_name is None:
        var_name = select_plot_variable(ds)
    data_crs = get_data_crs(ds, var_name)
    extent = get_domain_extent(ds, var_name)

    event_window = ds.sel(time=time_slice)
    all_times = event_window.time.values

    # Pick timestamps at least time_spacing_hours apart
    picked = [all_times[0]]
    for t in all_times[1:]:
        if (t - picked[-1]) >= np.timedelta64(time_spacing_hours, "h"):
            picked.append(t)
        if len(picked) == max_frames:
            break
    event = event_window.sel(time=picked)

    data = event[var_name].values
    timestamps = event.time.values

    lat = ds["lat"].values
    lon = ds["lon"].values

    cmap, norm = get_precipitation_cmap(levels=levels, colors=colors)

    if title is None:
        t0 = np.datetime_as_string(timestamps[0], unit="D")
        title = f"Precipitation Event, {t0}"

    n_frames = len(timestamps)
    ncols = min(3, n_frames)
    nrows = (n_frames + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        subplot_kw={"projection": data_crs},
        constrained_layout=True,
    )
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(nrows, ncols)

    im = None
    for idx, ax in enumerate(axes.flat):
        if idx < n_frames:
            im = ax.pcolormesh(
                lon,
                lat,
                data[idx],
                transform=PLATE_CARREE,
                cmap=cmap,
                norm=norm,
                shading="auto",
                rasterized=True,
            )
            add_map_features(ax, resolution=map_resolution)
            ax.set_extent(extent, crs=PLATE_CARREE)
            ts = np.datetime_as_string(timestamps[idx], unit="m")
            ax.set_title(ts.replace("T", " ") + " UTC", fontsize=9)
        else:
            ax.set_visible(False)

    if im is not None:
        cbar = fig.colorbar(
            im,
            ax=axes.ravel().tolist(),
            orientation="horizontal",
            shrink=0.6,
            pad=0.03,
            aspect=40,
        )
        cbar.set_label("Precipitation rate (mm h$^{-1}$)")

    fig.suptitle(title, fontsize=13, fontweight="bold")

    if output_path:
        savefig(fig, output_path, close=False)
    return fig
