"""Domain overview map with radar coverage footprint."""

from __future__ import annotations

import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.figure import Figure

from ._map_helpers import PLATE_CARREE, add_map_features, savefig, setup_rcparams
from ._metadata import get_data_crs, get_domain_extent, select_plot_variable
from ._stats import compute_spatial_coverage


def plot_domain_map(
    ds: xr.Dataset,
    var_name: str | None = None,
    n_coverage_samples: int = 100,
    title: str | None = None,
    figsize: tuple[float, float] = (6, 8),
    cmap: str = "Blues",
    alpha: float = 0.4,
    map_resolution: str = "10m",
    show_admin_boundaries: bool = False,
    output_path: str | None = None,
) -> Figure:
    """Plot a domain overview map showing the radar coverage footprint.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with 2-D ``lat``/``lon`` coordinate variables.
    var_name : str or None, optional
        Data variable used to determine coverage. Auto-detected via
        :func:`~mlcast_datasets.plotting._metadata.select_plot_variable`
        when ``None``.
    n_coverage_samples : int, optional
        Number of frames sampled to compute the domain mask.
        Default is 100.
    title : str or None, optional
        Figure title. Auto-generated from dataset attributes when ``None``.
    figsize : tuple of float, optional
        Figure size ``(width, height)`` in inches. Default is ``(6, 8)``.
    cmap : str, optional
        Matplotlib colormap name for the coverage fill.
        Default is ``'Blues'``.
    alpha : float, optional
        Opacity of the coverage fill, in ``[0, 1]``. Default is 0.4.
    map_resolution : str, optional
        Natural Earth feature resolution (``'10m'``, ``'50m'``, or
        ``'110m'``). Default is ``'10m'``.
    show_admin_boundaries : bool, optional
        If ``True``, overlay sub-national administrative boundaries.
        Default is ``False``.
    output_path : str or None, optional
        If given, save the figure to this file path.

    Returns
    -------
    matplotlib.figure.Figure
        The generated domain map figure.
    """
    setup_rcparams()
    if var_name is None:
        var_name = select_plot_variable(ds)
    data_crs = get_data_crs(ds, var_name)
    extent = get_domain_extent(ds, var_name)

    lat = ds["lat"].values
    lon = ds["lon"].values
    coverage = (
        compute_spatial_coverage(ds, var_name, n_samples=n_coverage_samples).values > 0
    )

    if title is None:
        dataset_title = ds.attrs.get(
            "title", ds.attrs.get("mlcast_dataset_identifier", var_name)
        )
        title = f"{dataset_title} Spatial Domain"

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": data_crs})

    ax.pcolormesh(
        lon,
        lat,
        coverage,
        transform=PLATE_CARREE,
        cmap=cmap,
        vmin=0,
        vmax=2,
        shading="auto",
        zorder=1,
        alpha=alpha,
        rasterized=True,
    )

    add_map_features(ax, resolution=map_resolution)

    if show_admin_boundaries:
        import cartopy.feature as cfeature

        provinces = cfeature.NaturalEarthFeature(
            "cultural",
            "admin_1_states_provinces_lines",
            map_resolution,
            facecolor="none",
            edgecolor="0.6",
            linewidth=0.3,
        )
        ax.add_feature(provinces)

    ax.set_extent(extent, crs=PLATE_CARREE)
    ax.set_title(title, fontweight="bold")

    if output_path:
        savefig(fig, output_path, close=False)
    return fig
