"""Look up the zarr datasets in the catalog."""

from __future__ import annotations

from dataclasses import dataclass, field


class DatasetNotFoundError(LookupError):
    """No catalog entry matches the given name."""


@dataclass(frozen=True)
class DatasetEntry:
    """A zarr dataset in the catalog."""

    name: str
    url: str
    storage_options: dict = field(default_factory=dict)
    description: str = ""
    validator_version: str | None = None


def list_entries() -> list[DatasetEntry]:
    """Return every zarr dataset in the catalog."""
    from . import open_catalog

    catalog = open_catalog()
    entries = []
    for name in catalog.walk(depth=10):
        item = catalog[name]
        if getattr(item, "container", None) == "catalog":
            continue
        reader = item.reader
        url = getattr(reader.data, "url", None)
        if url is None:
            continue
        entries.append(
            DatasetEntry(
                name=name,
                url=url,
                storage_options=dict(reader.data.storage_options or {}),
                description=" ".join((item.description or "").split()),
                validator_version=reader.metadata.get(
                    "mlcast_dataset_validator_version"
                ),
            )
        )
    return entries


def get_entry(name: str) -> DatasetEntry:
    """Find a dataset by its dotted path in the catalog, e.g.
    ``precipitation.it_dpc_sri_5min``."""
    entries = list_entries()
    for entry in entries:
        if entry.name == name:
            return entry
    known = ", ".join(e.name for e in entries)
    raise DatasetNotFoundError(f"No dataset named '{name}'. Available: {known}")
