"""Shared plotting utilities: map features, colormaps, savefig."""

from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure

RCPARAMS = {
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}

PLATE_CARREE = ccrs.PlateCarree()


def setup_rcparams() -> None:
    """Apply publication-quality matplotlib rcParams.

    Updates ``plt.rcParams`` with font sizes, DPI, and save settings
    suitable for journal-quality figures.
    """
    plt.rcParams.update(RCPARAMS)


def add_map_features(ax, resolution: str = "50m"):
    """Add coastlines, borders, ocean/land fills, and gridlines to a cartopy axis.

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxes
        A cartopy-aware matplotlib axes to decorate.
    resolution : str, optional
        Natural Earth feature resolution, one of ``'10m'``, ``'50m'``, or
        ``'110m'``. Default is ``'50m'``.

    Returns
    -------
    cartopy.mpl.gridliner.Gridliner
        The gridliner artist added to the axes.
    """
    ax.add_feature(cfeature.OCEAN.with_scale(resolution), facecolor="#ddeeff", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale(resolution), facecolor="#f5f5f0", zorder=0)
    ax.coastlines(resolution=resolution, linewidth=0.6, color="0.2")
    ax.add_feature(
        cfeature.BORDERS.with_scale(resolution),
        linewidth=0.4,
        linestyle="--",
        edgecolor="0.4",
    )
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.7", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    return gl


def get_precipitation_cmap(
    levels: list[float] | None = None,
    colors: list[str] | None = None,
) -> tuple[ListedColormap, BoundaryNorm]:
    """Return a discrete precipitation colormap with BoundaryNorm.

    Parameters
    ----------
    levels : list of float or None, optional
        Boundary values for the discrete colour bins. Default is
        ``[0, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]``.
    colors : list of str or None, optional
        Hex colour strings, one fewer than *levels*. Default is a 9-colour
        blue-green-orange-red ramp.

    Returns
    -------
    cmap : matplotlib.colors.ListedColormap
        The discrete colormap.
    norm : matplotlib.colors.BoundaryNorm
        The norm mapping data values to colormap indices.
    """
    if levels is None:
        levels = [0, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]
    if colors is None:
        colors = [
            "#f7f7f7",
            "#c6dbef",
            "#9ecae1",
            "#6baed6",
            "#3182bd",
            "#31a354",
            "#f7dc6f",
            "#e67e22",
            "#e74c3c",
        ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels, ncolors=len(colors))
    return cmap, norm


def savefig(fig: Figure, path: str | Path, close: bool = True) -> None:
    """Save a matplotlib figure to disk, creating parent directories as needed.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    path : str or pathlib.Path
        Destination file path (e.g. ``'figures/map.pdf'``).
    close : bool, optional
        If ``True``, close the figure after saving to free memory.
        Default is ``True``.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if close:
        plt.close(fig)
    print(f"Saved: {path}")
