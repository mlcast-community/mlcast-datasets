import importlib
import importlib.metadata
import shutil
import subprocess
import sys

import pytest
from loguru import logger
from packaging.version import Version

import mlcast_datasets

VALIDATOR_SPECS = {
    "precipitation": ("source_data", "radar_precipitation"),
}

VALIDATOR_PACKAGE = "mlcast-dataset-validator"
# key in the intake catalog entry metadata giving the validator version that the
# dataset conforms to
VALIDATOR_VERSION_METADATA_KEY = "mlcast_dataset_validator_version"
# global attribute on the dataset itself giving the validator version that it
# conforms to, datasets created before this attribute was introduced are assumed
# to conform to v0.3.0
VALIDATOR_VERSION_DATASET_ATTR = "mlcast_dataset_validator_version"
DEFAULT_DATASET_VALIDATOR_VERSION = Version("0.3.0")
# validator version pinned in pyproject.toml, datasets that conform to a different
# version are validated in an isolated environment using uvx
INSTALLED_VALIDATOR_VERSION = Version(importlib.metadata.version(VALIDATOR_PACKAGE))


@pytest.fixture
def catalog():
    return mlcast_datasets.open_catalog()


def all_entries():
    catalog = mlcast_datasets.open_catalog()
    return list(catalog.walk(depth=10))


@pytest.mark.parametrize("dataset_name", all_entries())
def test_get_intake_source(catalog, dataset_name):
    item = catalog[dataset_name]
    if item.container == "catalog":
        item.reload()
    else:
        logger.debug(f"Testing {dataset_name}")
        plugin = item.cat.describe()["plugin"][0]
        if plugin in ["opendap", "zarr", "netcdf"]:
            _ = item.to_dask()
        elif plugin in ["intake_esm.esm_datastore", "parquet"]:
            _ = item.get()
        elif plugin in ["json"]:
            _ = item.read()
        elif plugin == "yaml_file_cat":
            pass
        else:
            raise Exception(plugin)


def _infer_validator_spec(dataset_name: str):
    parts = dataset_name.replace("/", ".").split(".")
    if not parts:
        return None
    return VALIDATOR_SPECS.get(parts[0])


def _load_validator(spec):
    data_stage, product = spec
    module = importlib.import_module(
        f"mlcast_dataset_validator.specs.{data_stage}.{product}"
    )
    return module.validate_dataset


def _catalog_validator_version(dataset_name: str, item) -> Version:
    version = item.reader.metadata.get(VALIDATOR_VERSION_METADATA_KEY)
    if version is None:
        pytest.fail(
            f"Catalog entry '{dataset_name}' doesn't set "
            f"`metadata.{VALIDATOR_VERSION_METADATA_KEY}`, the validator version "
            "that the dataset conforms to."
        )
    return Version(str(version))


def _dataset_validator_version(ds) -> Version:
    version = ds.attrs.get(VALIDATOR_VERSION_DATASET_ATTR)
    if version is None:
        return DEFAULT_DATASET_VALIDATOR_VERSION
    return Version(str(version))


def _validate_in_process(item, ds, spec):
    validate_dataset = _load_validator(spec)

    # set storage_options explicitly on ds.attrs so that it is available to the
    # validator, which needs these when working out the zarr store path for the
    # dataset (looking for consolidated meta data), the storage options
    ds.encoding["storage_options"] = item.reader.data.storage_options
    report, _ = validate_dataset(ds)
    report.console_print(file=sys.stderr)

    if report.has_fails():
        pytest.fail(report.summarize())


def _validate_in_isolated_env(item, spec, version: Version):
    """
    Run the validator at `version` with uvx in a subprocess. Only the exit code
    is checked, so this gives pass/fail rather than a full report.
    """
    if shutil.which("uvx") is None:
        pytest.fail(
            f"uvx is required to validate '{item.name}' against "
            f"{VALIDATOR_PACKAGE}=={version}, but it wasn't found on PATH."
        )

    data_stage, product = spec
    storage_options = dict(item.reader.data.storage_options or {})
    cmd = [
        "uvx",
        "--from",
        f"{VALIDATOR_PACKAGE}=={version}",
        "mlcast.validate_dataset",
        data_stage,
        product,
        item.reader.data.url,
    ]
    endpoint_url = storage_options.pop("endpoint_url", None)
    if endpoint_url is not None:
        cmd += ["--s3-endpoint-url", endpoint_url]
    if storage_options.pop("anon", False):
        cmd.append("--s3-anon")
    if storage_options:
        pytest.fail(
            f"Storage options {storage_options} for '{item.name}' can't be passed "
            "to the validator CLI."
        )

    logger.debug(f"Running {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    print(output, file=sys.stderr)

    # the validator CLI catches exceptions with loguru and still exits with 0, so
    # look for the logged error too
    if result.returncode != 0 or "An error has been caught" in output:
        pytest.fail(
            f"Dataset '{item.name}' failed validation with "
            f"{VALIDATOR_PACKAGE}=={version} (exit code {result.returncode})."
        )


@pytest.mark.parametrize("dataset_name", all_entries())
def test_dataset_passes_validator(catalog, dataset_name):
    item = catalog[dataset_name]
    if item.container == "catalog":
        pytest.skip("Catalog entry; validator applies to datasets only.")

    spec = _infer_validator_spec(dataset_name)
    if spec is None:
        pytest.fail(f"No validator spec mapping for dataset '{dataset_name}'.")

    version = _catalog_validator_version(dataset_name, item)

    if not hasattr(item, "to_dask"):
        pytest.fail(f"Dataset '{dataset_name}' does not support to_dask().")
    ds = item.to_dask()

    dataset_version = _dataset_validator_version(ds)
    if dataset_version != version:
        if VALIDATOR_VERSION_DATASET_ATTR in ds.attrs:
            dataset_claim = (
                f"the dataset's `{VALIDATOR_VERSION_DATASET_ATTR}` attribute says "
                f"{dataset_version}"
            )
        else:
            dataset_claim = (
                f"the dataset has no `{VALIDATOR_VERSION_DATASET_ATTR}` attribute, "
                f"so it's assumed to conform to {dataset_version}"
            )
        pytest.fail(
            f"Catalog entry '{dataset_name}' says the dataset conforms to "
            f"{VALIDATOR_PACKAGE}=={version}, but {dataset_claim}."
        )

    if version == INSTALLED_VALIDATOR_VERSION:
        _validate_in_process(item, ds, spec)
    else:
        _validate_in_isolated_env(item, spec, version)


@pytest.mark.modified_on_branch
def test_make_ci_happy_if_no_test_is_selected():
    """pytest returns exit code 5 if no test is selected"""
    pass
