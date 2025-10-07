# How to Contribute

We would love to accept your patches and contributions to this project, for example by adding new datasets or improving the documentation. To do so, please follow the guidelines below.

## Adding a new dataset

> **Sharing non-public dataset**: currently we only have one bucket called `mlcast-source-datasets` (stored on S3 host `https://object-store.os-api.cci2.ecmwf.int`) which is set for public-read access. In future we expect to make a second bucket for datasets that can only be shared within the community. If you have a dataset that you would like to share with the community but and not able to make it public, please raise an issue.

Below follows a list of the general steps for preparing a new dataset for sharing with the community. If you have any issues along the way please raise an issue and the community will be happy to help.

1. Convert your dataset to zarr format. For this you will likely need tools like [xarray](https://docs.xarray.dev/en/stable/) and [kerchunk](https://github.com/fsspec/kerchunk). There is an example for the radklim dataset in https://github.com/mlcast-community/mlcast-dataset-radklim. Check-list for zarr datasets:
   - The dataset should be in zarr format
   - The dataset should follow [CF conventions](https://cfconventions.org/), including [setting the projection information](https://cfconventions.org/cf-conventions/cf-conventions.html#use-of-the-crs-well-known-text-format) in the attributes of the dataset.
   - You can validate your dataset using the [mlcast-sourcedata-validator](https://github.com/mlcast-community/mlcast-sourcedata-validator) tool to ensure it meets the required specifications before uploading.


2. Copy your zarr dataset to our European Weather Cloud S3-compatible bucket storage. There are a number of tools available for this, a quite fast command line tool is [s5cmd](https://github.com/peak/s5cmd) which supports both custom server endpoints (which we need since we use the S3 store on EWC rather than Amazon S3) and parallel uploads. The command use for uploading radklim was:

    ```bash
    s5cmd --profile ewc-eai-mlcast --endpoint-url https://object-store.os-api.cci2.ecmwf.int sync data/dst/hourly/2001_2023.zarr/ s3://mlcast-source-datasets/radklim/v0.1.0/hourly.zarr/
    ```
    To upload EWC you will need the credentials, for now you will need to contact [lcd@dmi.dk](mailto:lcd@dmi.dk) to get access to the EWC S3-compatible storage. The credentials will be provided in the form of an AWS profile, which you can add to your `~/.aws/credentials` file. The profile should look like this:
    ```ini
    [ewc-eai-mlcast]
    aws_access_key_id = <your-access-key-id>
    aws_secret_access_key = <your-secret-access-key>
    ```

3. Fork and clone your fork of this repository repository.

    ```bash
    git clone https://github.com/<your-github-username>/mlcast-datasets/
    ```

4. Edit the catalog yaml-files to add your new dataset ([src/mlcast_datasets/catalog/](src/mlcast_datasets/catalog/). We haven't defined a strict structure for the catalog yet, but for now are going with `cat.{precipitation/radiation}.{dataset_name}`.

5. Create a Jupyter notebook in `docs/` that demonstrates how to use your dataset. This will become part of the automatically built [documentation of the mlcast dataset](https://mlcast-community.github.io/mlcast-datasets/). The notebook should include:
   - A description of the dataset
   - How to load the dataset using `mlcast`
   - A few examples of how to use the dataset, including a few plots

6. Create a pull-request for your change to the main repository. Please make sure to include a description of your changes and any relevant information about the dataset you are adding. If you have added a new dataset, please include a link to the dataset in the description.

Thanks in advance for you contribution!
