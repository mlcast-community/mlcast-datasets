# MLCast Community intake catalog

<!-- SPDX-License-Identifier: Apache-2.0 OR BSD-3-Clause -->

[![data-availability-check](https://github.com/mlcast-community/mlcast-datasets/actions/workflows/data_availability_check.yml/badge.svg)](https://github.com/mlcast-community/mlcast-datasets/actions/workflows/data_availability_check.yml) [![linting](https://github.com/mlcast-community/mlcast-datasets/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/mlcast-community/mlcast-datasets/actions/workflows/pre-commit.yml) [![Jupyter Book Badge](https://raw.githubusercontent.com/jupyter-book/jupyter-book/next/docs/media/images/badge.svg)](https://mlcast-community.github.io/mlcast-datasets/)

Hi! 👋

You are looking at the source data intake catalog for the MLCast community. This is a collection of datasets we have currated with the aim of making them available to build machine learning training datasets from.

The following diagram shows the intended data flow and how the intake catalog (this repository) fits into the overall architecture of the MLCast project.

![](docs/mlcast-datainfra.png)
[source for this graphic](https://docs.google.com/presentation/d/1hIlPOer4T9hlxp0mnQ8WQRggSzVUqMID/edit?slide=id.p1#slide=id.p1)


## How to use this catalog

To use the catalog, you can either a) install the necessary python packages yourself and read the catalog directly from github or b) install the most recent tagged release of the `mlcast_datasets` python package from pypi.org and read the catalog included in that release. Reading the catalog from github is useful if you want to use the most recent version of the catalog, while installing the `mlcast_datasets` package is useful if you want to use a stable version of the catalog.

### a) Reading the catalog directly from github

To read and open datasets in the catalog you will need to have the following packages installed:

```bash
pip install intake intake-xarray zarr jinja2
```

*Or*, you can installing the mlcast-datasets package directly from this
repository, which will install all the necessary dependencies:

   ```bash
   pip install git+https://github.com/mlcast-community/mlcast-datasets
   ```

The catalogue (and underlying data) can then be accessed directly from python:

```python
import intake
cat = intake.open_catalog("https://raw.githubusercontent.com/mlcast-community/mlcast-datasets/main/src/mlcast_datasets/catalog/catalog.yml")
```

### b) Installing the mlcast_datasets package

To install the most recent tagged release of the `mlcast_datasets` package, you can use pip:

```bash
pip install mlcast-datasets
```

and then read the catalog from the package:

```python
import mlcast_datasets
cat = mlcast_datasets.open_catalog()
```

### Using data within the catalog

Once you have opened the catalog, you can list the available sources with:

```python
>> list(cat)
['precipitation']

>> list(cat.precipitation)
['radklim_hourly', 'radklim_5_minutes']
```

Then load up a [dask](https://github.com/dask/dask)-backed `xarray.Dataset` so
that you have access to all the available variables and attributes in the
dataset:


```python
>> ds = cat.precipitation.radklim_5_minutes.to_dask()
>> ds
<xarray.Dataset> Size: 10TB
Dimensions:  (time: 2419200, y: 1100, x: 900)
Coordinates:
    lat      (y, x) float64 8MB dask.array<chunksize=(1100, 900), meta=np.ndarray>
    lon      (y, x) float64 8MB dask.array<chunksize=(1100, 900), meta=np.ndarray>
  * time     (time) datetime64[ns] 19MB 2001-01-01 ... 2023-12-31T23:55:00
  * x        (x) float64 7kB -443.0 -442.0 -441.0 -440.0 ... 454.0 455.0 456.0
  * y        (y) float64 9kB -4.758e+03 -4.757e+03 ... -3.66e+03 -3.659e+03
Data variables:
    RR       (time, y, x) float32 10TB dask.array<chunksize=(1, 1100, 900), meta=np.ndarray>
    crs      float64 8B ...
Attributes:
    Author:                Harald Rybka, Katharina Lengfeld
    Conventions:           CF-1.6
    history:               Created at 2021-07-09 09:10:06.385653
    institution:           Deutscher Wetterdienst (DWD)
    reference:             10.5676/DWD/RADKLIM_YW_V2017.002
    title:                 RADKLIM - radar-based precipitation climatology
    zarr_creation:         created with mlcast_dataset_radklim (https://githu...
    zarr_dataset_version:  0.1.0
```

Start using the dataset 🙂


## Contributing

We are always looking for new datasets to add to the catalog. If you have a dataset you would like to contribute, please open an issue or a pull request.

## License

This project is dual-licensed under either:

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* BSD 3-Clause License ([LICENSE-BSD](LICENSE-BSD) or https://opensource.org/licenses/BSD-3-Clause)

at your option.

See [LICENSE](LICENSE) for more details.
