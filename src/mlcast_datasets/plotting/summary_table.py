"""Build a summary DataFrame of dataset metadata."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ._metadata import get_data_crs, infer_time_step_minutes, select_plot_variable


def generate_summary_table(
    ds: xr.Dataset,
    var_name: str | None = None,
    compressed_size_bytes: int | None = None,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Build a summary table of dataset metadata.

    Extracts time range, grid dimensions, CRS, compression statistics,
    and other properties into a two-column DataFrame.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset to summarise.
    var_name : str or None, optional
        Data variable to describe. Auto-detected via
        :func:`~mlcast_datasets.plotting._metadata.select_plot_variable`
        when ``None``.
    compressed_size_bytes : int or None, optional
        On-disk compressed size in bytes. When provided, compression ratio
        and compressed volume are included in the table.
    output_path : str or None, optional
        If given, save the DataFrame as a CSV file to this path.

    Returns
    -------
    pd.DataFrame
        Two-column DataFrame with columns ``['Property', 'Value']``
        containing one row per metadata property.
    """
    if var_name is None:
        var_name = select_plot_variable(ds)

    n_time = ds.sizes["time"]
    n_y = ds.sizes["y"]
    n_x = ds.sizes["x"]

    t0 = pd.Timestamp(ds.time.values[0])
    t1 = pd.Timestamp(ds.time.values[-1])
    time_range_str = f"{t0.strftime('%Y-%m-%d')} to {t1.strftime('%Y-%m-%d')}"

    if "missing_times" in ds:
        n_missing = ds.sizes["missing_times"]
        missing_pct = n_missing / (n_missing + n_time) * 100
        missing_str = f"{n_missing:,} ({missing_pct:.1f}%)"
    else:
        missing_str = "N/A"

    attrs = ds[var_name].attrs
    long_name = attrs.get("long_name", var_name)
    units = attrs.get("units", "unknown")

    data_crs = get_data_crs(ds, var_name)
    crs_name = type(data_crs).__name__
    grid_mapping_name = ds[var_name].attrs.get("grid_mapping", "")
    if grid_mapping_name and grid_mapping_name in ds:
        gm_attrs = ds[grid_mapping_name].attrs
        gm_name = gm_attrs.get("grid_mapping_name", crs_name)
        crs_display = gm_name.replace("_", " ").title()
    else:
        crs_display = crs_name

    uncompressed_bytes = n_time * n_y * n_x * 4
    uncompressed_tb = uncompressed_bytes / 1e12

    if compressed_size_bytes is not None:
        compressed_gb = compressed_size_bytes / 1e9
        compression_ratio = uncompressed_bytes / compressed_size_bytes
        compressed_str = f"{compressed_gb:.0f} GB"
        ratio_str = f"{compression_ratio:.0f}:1"
    else:
        compressed_str = "N/A"
        ratio_str = "N/A"

    freq_str = ds.attrs.get("base_frequencies", "")
    if freq_str:
        freq_bands = []
        for part in freq_str.split(";"):
            if not part.strip():
                continue
            freq_part, range_part = part.strip().split(":", 1)
            start, end = range_part.split("/")
            end_display = end.strip() if end.strip().lower() != "none" else "present"
            freq_bands.append(
                f"{freq_part.strip()} ({start.strip()[:10]}-{end_display[:10]})"
            )
        freq_display = ", ".join(freq_bands)
    else:
        step = infer_time_step_minutes(ds)
        freq_display = f"{int(step)} min" if np.isfinite(step) else "variable"

    if "x" in ds and len(ds["x"]) > 1:
        dx = abs(float(ds["x"].values[1] - ds["x"].values[0]))
        res_str = f"{dx / 1000:.0f} km" if dx > 100 else f"{dx:.0f} m"
    else:
        res_str = "N/A"

    rows = [
        ("Time range", time_range_str),
        ("Total timesteps", f"{n_time:,}"),
        ("Missing timesteps", missing_str),
        ("Grid dimensions", f"{n_y} × {n_x} pixels"),
        ("Spatial resolution", res_str),
        ("Data variable", f"{var_name} ({long_name})"),
        ("Units", units),
        ("Data type", "float32"),
        ("Uncompressed volume", f"{uncompressed_tb:.1f} TB"),
        ("Compressed volume", compressed_str),
        ("Compression ratio", ratio_str),
        ("Temporal frequency", freq_display),
        ("CRS", crs_display),
        ("License", ds.attrs.get("license", "unknown")),
    ]

    df = pd.DataFrame(rows, columns=["Property", "Value"])

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")

    return df
