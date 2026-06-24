"""Precipitation statistics: mean/max/std maps and value histogram."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

from ._map_helpers import PLATE_CARREE, add_map_features, savefig, setup_rcparams
from ._metadata import get_data_crs, get_domain_extent, select_plot_variable
from ._stats import compute_welford_stats


def plot_precipitation_stats(
    ds: xr.Dataset,
    var_name: str | None = None,
    n_samples: int = 2000,
    figsize_maps: tuple[float, float] = (15, 5.5),
    figsize_hist: tuple[float, float] = (6, 4),
    cmap_mean: str = "YlGnBu",
    cmap_max: str = "YlOrRd",
    cmap_std: str = "OrRd",
    max_vmin: float = 1,
    max_vmax: float = 300,
    hist_color: str = "#3182bd",
    hist_xlim: tuple[float, float] = (0.01, 1000),
    map_resolution: str = "50m",
    output_path_maps: str | None = None,
    output_path_hist: str | None = None,
) -> tuple[Figure, Figure]:
    """Plot 3-panel stat maps (mean, max, std) and a precipitation histogram.

    Produces two figures: a triptych of spatial maps and a log--log
    histogram of non-zero precipitation values.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with 2-D ``lat``/``lon`` coordinate variables.
    var_name : str or None, optional
        Data variable to analyse. Auto-detected via
        :func:`~mlcast_datasets.plotting._metadata.select_plot_variable`
        when ``None``.
    n_samples : int, optional
        Number of uniformly spaced timesteps to sample. Default is 2000.
    figsize_maps : tuple of float, optional
        Figure size ``(width, height)`` for the 3-panel map.
        Default is ``(15, 5.5)``.
    figsize_hist : tuple of float, optional
        Figure size ``(width, height)`` for the histogram.
        Default is ``(6, 4)``.
    cmap_mean : str, optional
        Colormap for the mean panel. Default is ``'YlGnBu'``.
    cmap_max : str, optional
        Colormap for the maximum panel. Default is ``'YlOrRd'``.
    cmap_std : str, optional
        Colormap for the standard-deviation panel. Default is ``'OrRd'``.
    max_vmin : float, optional
        Lower bound of the log-scale range for the max panel.
        Default is 1.
    max_vmax : float, optional
        Upper bound of the log-scale range for the max panel.
        Default is 300.
    hist_color : str, optional
        Bar fill colour for the histogram. Default is ``'#3182bd'``.
    hist_xlim : tuple of float, optional
        ``(xmin, xmax)`` limits for the histogram x-axis.
        Default is ``(0.01, 1000)``.
    map_resolution : str, optional
        Natural Earth feature resolution (``'10m'``, ``'50m'``, or
        ``'110m'``). Default is ``'50m'``.
    output_path_maps : str or None, optional
        If given, save the map figure to this file path.
    output_path_hist : str or None, optional
        If given, save the histogram figure to this file path.

    Returns
    -------
    fig_maps : matplotlib.figure.Figure
        The 3-panel spatial statistics figure.
    fig_hist : matplotlib.figure.Figure
        The precipitation value histogram figure.
    """
    setup_rcparams()
    if var_name is None:
        var_name = select_plot_variable(ds)
    data_crs = get_data_crs(ds, var_name)
    extent = get_domain_extent(ds, var_name)

    stats = compute_welford_stats(ds, var_name, n_samples=n_samples)

    lat = ds["lat"].values
    lon = ds["lon"].values

    # ===================== 3-panel map =====================
    fig_maps, axes = plt.subplots(
        1,
        3,
        figsize=figsize_maps,
        subplot_kw={"projection": data_crs},
        constrained_layout=True,
    )

    # (a) Mean
    im_a = axes[0].pcolormesh(
        lon,
        lat,
        stats["mean"],
        transform=PLATE_CARREE,
        cmap=cmap_mean,
        vmin=0,
        vmax=np.nanpercentile(stats["mean"], 99),
        shading="auto",
        rasterized=True,
    )
    add_map_features(axes[0], resolution=map_resolution)
    axes[0].set_extent(extent, crs=PLATE_CARREE)
    axes[0].set_title("(a) Mean", fontweight="bold")
    fig_maps.colorbar(
        im_a,
        ax=axes[0],
        orientation="horizontal",
        shrink=0.8,
        pad=0.04,
        label="mm h$^{-1}$",
    )

    # (b) Max
    im_b = axes[1].pcolormesh(
        lon,
        lat,
        stats["max_val"],
        transform=PLATE_CARREE,
        cmap=cmap_max,
        norm=LogNorm(vmin=max_vmin, vmax=max_vmax),
        shading="auto",
        rasterized=True,
    )
    add_map_features(axes[1], resolution=map_resolution)
    axes[1].set_extent(extent, crs=PLATE_CARREE)
    axes[1].set_title("(b) Maximum", fontweight="bold")
    fig_maps.colorbar(
        im_b,
        ax=axes[1],
        orientation="horizontal",
        shrink=0.8,
        pad=0.04,
        label="mm h$^{-1}$",
    )

    # (c) Std
    im_c = axes[2].pcolormesh(
        lon,
        lat,
        stats["std"],
        transform=PLATE_CARREE,
        cmap=cmap_std,
        vmin=0,
        vmax=np.nanpercentile(stats["std"], 99),
        shading="auto",
        rasterized=True,
    )
    add_map_features(axes[2], resolution=map_resolution)
    axes[2].set_extent(extent, crs=PLATE_CARREE)
    axes[2].set_title("(c) Std. dev.", fontweight="bold")
    fig_maps.colorbar(
        im_c,
        ax=axes[2],
        orientation="horizontal",
        shrink=0.8,
        pad=0.04,
        label="mm h$^{-1}$",
    )

    if output_path_maps:
        savefig(fig_maps, output_path_maps, close=False)

    # ===================== Histogram =====================
    fig_hist, ax = plt.subplots(figsize=figsize_hist)

    bins = stats["hist_bins"]
    bin_centers = np.sqrt(bins[:-1] * bins[1:])
    widths = np.diff(bins)
    ax.bar(
        bin_centers,
        stats["hist_counts"],
        width=widths,
        align="center",
        color=hist_color,
        edgecolor="none",
        alpha=0.8,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Precipitation rate (mm h$^{-1}$)")
    ax.set_ylabel("Count (from sampled timesteps)")
    ax.set_title("Distribution of Non-Zero Precipitation Values", fontweight="bold")
    ax.set_xlim(*hist_xlim)

    if output_path_hist:
        savefig(fig_hist, output_path_hist, close=False)

    return fig_maps, fig_hist
