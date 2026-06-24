"""Spatial coverage map: fraction of valid data per pixel."""

from __future__ import annotations

import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.figure import Figure

from ._map_helpers import PLATE_CARREE, add_map_features, savefig, setup_rcparams
from ._metadata import get_data_crs, get_domain_extent, select_plot_variable
from ._stats import compute_spatial_coverage


def plot_spatial_coverage(
    ds: xr.Dataset,
    var_name: str | None = None,
    n_samples: int = 1000,
    title: str = "Spatial Data Coverage",
    figsize: tuple[float, float] = (6, 8),
    cmap: str = "YlOrRd_r",
    vmin: float = 0,
    vmax: float = 1,
    contour_levels: tuple[float, ...] = (0.5, 0.9),
    contour_colors: tuple[str, ...] = ("#c0392b", "#2c3e50"),
    map_resolution: str = "10m",
    output_path: str | None = None,
) -> Figure:
    """Plot the fraction of valid observations per pixel as a map.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with a 3-D ``(time, y, x)`` data variable and
        2-D ``lat``/``lon`` coordinate variables.
    var_name : str or None, optional
        Data variable to analyse. Auto-detected via
        :func:`~mlcast_datasets.plotting._metadata.select_plot_variable`
        when ``None``.
    n_samples : int, optional
        Number of uniformly spaced timesteps to sample. Default is 1000.
    title : str, optional
        Figure title. Default is ``'Spatial Data Coverage'``.
    figsize : tuple of float, optional
        Figure size ``(width, height)`` in inches. Default is ``(6, 8)``.
    cmap : str, optional
        Matplotlib colormap name. Default is ``'YlOrRd_r'``.
    vmin : float, optional
        Lower bound of the colourbar. Default is 0.
    vmax : float, optional
        Upper bound of the colourbar. Default is 1.
    contour_levels : tuple of float, optional
        Iso-contour levels to overlay on the map.
        Default is ``(0.5, 0.9)``.
    contour_colors : tuple of str, optional
        Colours for the contour lines, matched 1-to-1 with
        *contour_levels*. Default is ``('#c0392b', '#2c3e50')``.
    map_resolution : str, optional
        Natural Earth feature resolution (``'10m'``, ``'50m'``, or
        ``'110m'``). Default is ``'10m'``.
    output_path : str or None, optional
        If given, save the figure to this file path.

    Returns
    -------
    matplotlib.figure.Figure
        The generated spatial coverage figure.
    """
    setup_rcparams()
    if var_name is None:
        var_name = select_plot_variable(ds)
    data_crs = get_data_crs(ds, var_name)
    extent = get_domain_extent(ds, var_name)

    frac = compute_spatial_coverage(ds, var_name, n_samples=n_samples).values

    lat = ds["lat"].values
    lon = ds["lon"].values

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": data_crs})

    im = ax.pcolormesh(
        lon,
        lat,
        frac,
        transform=PLATE_CARREE,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
        rasterized=True,
    )

    if contour_levels:
        ax.contour(
            lon,
            lat,
            frac,
            levels=list(contour_levels),
            colors=list(contour_colors),
            linewidths=[0.8 + 0.2 * i for i in range(len(contour_levels))],
            transform=PLATE_CARREE,
        )

    add_map_features(ax, resolution=map_resolution)
    ax.set_extent(extent, crs=PLATE_CARREE)

    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.7, pad=0.08)
    cbar.set_label("Fraction of valid observations")
    ax.set_title(title, fontsize=12, fontweight="bold")

    if output_path:
        savefig(fig, output_path, close=False)
    return fig
