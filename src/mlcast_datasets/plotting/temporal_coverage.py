"""Temporal data completeness heatmap and yearly bar chart."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import BoundaryNorm
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from ._map_helpers import savefig, setup_rcparams
from ._metadata import infer_time_step_minutes


def _parse_base_frequencies(
    freq_str: str,
) -> list[tuple[int, pd.Timestamp, pd.Timestamp]]:
    """Parse the ``base_frequencies`` dataset attribute into structured tuples.

    The attribute string uses the format
    ``'15min:2010-01-01/2014-06-01; 10min:2014-06-01/2020-07-01; ...'``.

    Parameters
    ----------
    freq_str : str
        Semi-colon-separated frequency bands, each as
        ``'{freq}min:{start}/{end}'``. An empty string returns an empty
        list.

    Returns
    -------
    list of tuple(int, pd.Timestamp, pd.Timestamp)
        Each tuple contains ``(freq_min, start, end)`` where *freq_min* is
        the temporal frequency in minutes. An ``end`` value of ``'None'``
        in the input is mapped to ``pd.Timestamp('2100-01-01')``.
    """
    bands = []
    if not freq_str:
        return bands
    for part in freq_str.split(";"):
        if not part.strip():
            continue
        fp, rp = part.strip().split(":", 1)
        freq_min = int(fp.replace("min", ""))
        start_s, end_s = rp.split("/")
        start = pd.Timestamp(start_s.strip())
        end = (
            pd.Timestamp(end_s.strip())
            if end_s.strip().lower() != "none"
            else pd.Timestamp("2100-01-01")
        )
        bands.append((freq_min, start, end))
    return bands


def plot_temporal_coverage(
    ds: xr.Dataset,
    title: str = "Monthly Data Completeness",
    figsize: tuple[float, float] = (8, 7.5),
    cmap=None,
    completeness_levels: list[float] | None = None,
    output_path: str | None = None,
) -> Figure:
    """Plot a heatmap of monthly data completeness and a yearly timestep bar chart.

    Gracefully handles datasets with or without ``missing_times`` and
    ``base_frequencies`` attributes.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with a ``time`` coordinate. Optionally contains a
        ``missing_times`` coordinate and a ``base_frequencies`` global
        attribute.
    title : str, optional
        Title for the heatmap panel.
        Default is ``'Monthly Data Completeness'``.
    figsize : tuple of float, optional
        Figure size ``(width, height)`` in inches. Default is ``(8, 7.5)``.
    cmap : matplotlib.colors.Colormap or None, optional
        Colormap for the completeness heatmap. Default is
        ``plt.cm.RdYlGn``.
    completeness_levels : list of float or None, optional
        Boundary values for the ``BoundaryNorm`` applied to the heatmap.
        Default is ``[0, 50, 70, 80, 90, 95, 98, 100]``.
    output_path : str or None, optional
        If given, save the figure to this file path.

    Returns
    -------
    matplotlib.figure.Figure
        The generated two-panel figure (heatmap + bar chart).
    """
    setup_rcparams()

    if cmap is None:
        cmap = plt.cm.RdYlGn
    if completeness_levels is None:
        completeness_levels = [0, 50, 70, 80, 90, 95, 98, 100]

    times = pd.DatetimeIndex(ds.time.values)

    if "missing_times" in ds:
        missing = pd.DatetimeIndex(ds.missing_times.values)
    else:
        step_min = infer_time_step_minutes(ds)
        if np.isfinite(step_min):
            expected = pd.date_range(
                times.min(), times.max(), freq=f"{int(step_min)}min"
            )
            missing = expected.difference(times)
        else:
            missing = pd.DatetimeIndex([])

    bands = _parse_base_frequencies(ds.attrs.get("base_frequencies", ""))

    years = list(range(times.year.min(), times.year.max() + 1))
    months = list(range(1, 13))

    actual_ym = times.to_series().groupby([times.year, times.month]).count()
    missing_ym = (
        missing.to_series().groupby([missing.year, missing.month]).count()
        if len(missing) > 0
        else pd.Series(dtype=int)
    )

    completeness = np.full((len(years), 12), np.nan)
    actual_counts = np.zeros((len(years), 12), dtype=int)

    for i, yr in enumerate(years):
        for j, mo in enumerate(months):
            act = actual_ym.get((yr, mo), 0)
            mis = missing_ym.get((yr, mo), 0) if len(missing_ym) > 0 else 0
            actual_counts[i, j] = act
            total = act + mis
            if total > 0:
                completeness[i, j] = act / total * 100

    # --- Figure ---
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=figsize,
        gridspec_kw={"height_ratios": [3, 1.2]},
        constrained_layout=True,
    )

    # Top: heatmap
    norm = BoundaryNorm(completeness_levels, ncolors=cmap.N)
    im = ax1.imshow(
        completeness, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest"
    )

    month_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    ax1.set_xticks(range(12))
    ax1.set_xticklabels(month_labels)
    ax1.set_yticks(range(len(years)))
    ax1.set_yticklabels(years)

    for i in range(len(years)):
        for j in range(12):
            val = completeness[i, j]
            if np.isfinite(val):
                color = "white" if val < 60 else "black"
                ax1.text(
                    j,
                    i,
                    f"{val:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=color,
                )

    cb = fig.colorbar(im, ax=ax1, orientation="vertical", shrink=0.8, pad=0.02)
    cb.set_label("Data completeness (%)")
    ax1.set_title(title, fontweight="bold")

    # Bottom: bar chart by year
    yearly_counts = actual_counts.sum(axis=1)

    freq_color_map = {5: "#FF9800", 10: "#4CAF50", 15: "#2196F3"}
    default_color = "#2196F3"

    bar_colors = []
    legend_entries = {}
    for yr in years:
        ts = pd.Timestamp(f"{yr}-07-01")
        color = default_color
        freq_label = None
        for freq_min, start, end in bands:
            if start <= ts < end:
                color = freq_color_map.get(freq_min, default_color)
                freq_label = f"{freq_min} min"
                break
        bar_colors.append(color)
        if freq_label:
            legend_entries[freq_label] = color

    ax2.bar(
        range(len(years)),
        yearly_counts,
        color=bar_colors,
        edgecolor="0.3",
        linewidth=0.3,
    )
    ax2.set_xticks(range(len(years)))
    ax2.set_xticklabels(years, rotation=45, ha="right")
    ax2.set_ylabel("Timesteps / year")
    ax2.set_xlim(-0.5, len(years) - 0.5)

    if legend_entries:
        legend_elements = [
            Patch(facecolor=c, edgecolor="0.3", label=lbl)
            for lbl, c in legend_entries.items()
        ]
        ax2.legend(
            handles=legend_elements,
            loc="upper left",
            title="Frequency",
            fontsize=7,
            title_fontsize=7,
        )

    if output_path:
        savefig(fig, output_path, close=False)
    return fig
