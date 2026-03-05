# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/mlcast-community/mlcast-datasets)

### Added
- UK Met Office C-band rain radar 1 km composite (2005-2025, 5min, OSGB 1 km) [\#17](https://github.com/mlcast-community/mlcast-datasets/issues/17), @franchg
- Danish radar precipitation dataset covering 2016-2025 at 10min, 0.5km resolution, [\#21](https://github.com/mlcast-community/mlcast-datasets/pull/21), @arjj8
- Italian DPC SRI radar precipitation dataset (2010-2025, 1km, 5min) [\#14](https://github.com/mlcast-community/mlcast-datasets/issues/14), @franchg

### Maintenance

- Fix jupyter book badge in README and update black with pre-commit to fix ci [\#18](https://github.com/mlcast-community/mlcast-datasets/pull/18), @leifdenby
- Fix jupyter book rendering in github actions workflow by pinning jupyter book version <2 [\#24](https://github.com/mlcast-community/mlcast-datasets/pull/24), @leifdenby

### Fixed

- A bug that made non-detection values into nan was fixed in the DMI dataset [\#32](https://github.com/mlcast-community/mlcast-datasets/pull/32), @arjj8

- Update DWD-DE-radklim dataset to `v0.1.1` that fixes meta-information to match mlcast-validator requirements and rename `RR` field to `rainfall_amount` to be consistent with the units and standard name used [\#33](https://github.com/mlcast-community/mlcast-datasets/pull/33), @leifdenby


## [v0.1.1](https://github.com/mlcast-community/mlcast-datasets/releases/tag/v0.1.1) - 2025-01-03

### Added
- Dual licensing under Apache 2.0 OR BSD 3-Clause licenses
- License files (LICENSE, LICENSE-APACHE, LICENSE-BSD)

### Changed
- Updated Python version to 3.13 in CI workflow

## [v0.1.0](https://github.com/mlcast-community/mlcast-datasets/releases/tag/v0.1.0)

First release of the mlcast-datasets dataset catalog package. This release includes the RadKlim radar precipitation dataset (2001-2023) at hourly and 5-minute time resolution (version v0.1.0 of the zarr version of this dataset). This first release also includes the setup for JupyterBook documentation based setup that through github actions is built and deployed to the [mlcast-datasets documentation](https://mlcast-community.github.io/mlcast-datasets/) site.
