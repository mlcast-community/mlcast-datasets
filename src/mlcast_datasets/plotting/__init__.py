"""Reusable plotting and diagnostic utilities for mlcast radar datasets."""

try:
    import cartopy  # noqa: F401
    import matplotlib  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "mlcast-datasets plotting extras are not installed. "
        "Run: pip install 'mlcast-datasets[plotting]'"
    ) from exc

from ._metadata import (
    get_data_crs,
    get_domain_extent,
    get_variable_label,
    infer_time_step_minutes,
    select_plot_variable,
)
from .domain_map import plot_domain_map
from .monthly_cycle import plot_monthly_cycle
from .precipitation_stats import plot_precipitation_stats
from .sample_precipitation import plot_sample_precipitation
from .spatial_coverage import plot_spatial_coverage
from .summary_table import generate_summary_table
from .temporal_coverage import plot_temporal_coverage

__all__ = [
    "plot_domain_map",
    "plot_monthly_cycle",
    "plot_precipitation_stats",
    "plot_sample_precipitation",
    "plot_spatial_coverage",
    "plot_temporal_coverage",
    "generate_summary_table",
    "select_plot_variable",
    "get_data_crs",
    "get_domain_extent",
    "infer_time_step_minutes",
    "get_variable_label",
]
