"""Monthly precipitation climatology box plot."""

from __future__ import annotations

import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from ._map_helpers import savefig, setup_rcparams
from ._metadata import get_variable_label, select_plot_variable
from ._stats import compute_monthly_stats

SEASON_COLORS = (
    ["#2196F3"] * 2  # Jan-Feb (winter)
    + ["#4CAF50"] * 3  # Mar-May (spring)
    + ["#FF9800"] * 3  # Jun-Aug (summer)
    + ["#F44336"] * 3  # Sep-Nov (autumn)
    + ["#2196F3"]  # Dec (winter)
)

MONTH_LABELS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


def plot_monthly_cycle(
    ds: xr.Dataset,
    var_name: str | None = None,
    n_samples: int = 3000,
    title: str = "Monthly Precipitation Climatology",
    ylabel: str | None = None,
    figsize: tuple[float, float] = (7, 4),
    show_fliers: bool = False,
    season_colors: list[str] | None = None,
    output_path: str | None = None,
) -> Figure:
    """Plot a boxplot of domain-mean values grouped by calendar month.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with a 3-D ``(time, y, x)`` data variable.
    var_name : str or None, optional
        Data variable to analyse. Auto-detected via
        :func:`~mlcast_datasets.plotting._metadata.select_plot_variable`
        when ``None``.
    n_samples : int, optional
        Number of uniformly spaced timesteps to sample. Default is 3000.
    title : str, optional
        Figure title. Default is ``'Monthly Precipitation Climatology'``.
    ylabel : str or None, optional
        Y-axis label. Auto-generated from the variable's ``long_name``
        and ``units`` attributes when ``None``.
    figsize : tuple of float, optional
        Figure size ``(width, height)`` in inches. Default is ``(7, 4)``.
    show_fliers : bool, optional
        Whether to show boxplot outlier points. Default is ``False``.
    season_colors : list of str or None, optional
        List of 12 colour strings (one per month, Jan--Dec). Defaults to
        a seasonal colour scheme (blue/green/orange/red).
    output_path : str or None, optional
        If given, save the figure to this file path.

    Returns
    -------
    matplotlib.figure.Figure
        The generated monthly-cycle boxplot figure.
    """
    setup_rcparams()
    if var_name is None:
        var_name = select_plot_variable(ds)
    if ylabel is None:
        ylabel = f"Domain-mean {get_variable_label(ds, var_name)}"
    if season_colors is None:
        season_colors = SEASON_COLORS

    monthly_data = compute_monthly_stats(ds, var_name, n_samples=n_samples)

    fig, ax = plt.subplots(figsize=figsize)

    data_lists = [monthly_data[m] for m in range(1, 13)]
    bp = ax.boxplot(
        data_lists,
        tick_labels=MONTH_LABELS,
        patch_artist=True,
        showfliers=show_fliers,
        widths=0.6,
        whiskerprops=dict(linestyle="--", linewidth=0.8),
        medianprops=dict(color="black", linewidth=1.5),
        capprops=dict(linewidth=0.8),
    )

    for patch, color in zip(bp["boxes"], season_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor("0.3")

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Month")
    ax.set_title(title, fontweight="bold")

    legend_elements = [
        Patch(facecolor="#2196F3", alpha=0.6, edgecolor="0.3", label="Winter"),
        Patch(facecolor="#4CAF50", alpha=0.6, edgecolor="0.3", label="Spring"),
        Patch(facecolor="#FF9800", alpha=0.6, edgecolor="0.3", label="Summer"),
        Patch(facecolor="#F44336", alpha=0.6, edgecolor="0.3", label="Autumn"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7)
    ax.grid(axis="y", linewidth=0.3, alpha=0.5)

    if output_path:
        savefig(fig, output_path, close=False)
    return fig
