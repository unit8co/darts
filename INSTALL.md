# Installation Guide

⚠️ Note: If you migrate to darts version >=0.41.0 from versions <0.41.0, refer to the [migration guidlines below](#-important-darts-pypi-package-changes-as-of-version-0410)

## From PyPI

Darts offers a modular installation system with optional dependencies. Choose the installation that fits your needs:

* **Core only** (without neural networks, Prophet, LightGBM, CatBoost, XGBoost, StatsForecast): `pip install darts`
* **Core + PyTorch** (for neural network models): `pip install "darts[torch]"`
* **Core + Prophet, LightGBM, CatBoost, XGBoost, StatsForecast** (no neural networks): `pip install "darts[notorch]"`
* **All available models**: `pip install "darts[all]"`

If the PyTorch installation fails on your platform, please follow the official installation guide for [PyTorch](https://pytorch.org/get-started/locally/), then try installing Darts again.

## From conda-forge
Create a conda environment (e.g., for Python 3.11):
(after installing [conda](https://docs.conda.io/en/latest/miniconda.html)):

    conda create --name <env-name> python=3.11

Activate the environment

    conda activate <env-name>

As some models have relatively heavy dependencies, we provide four conda-forge packages:

* **Core only** (without neural networks, Prophet, LightGBM, CatBoost, XGBoost, StatsForecast): `conda install -c conda-forge u8darts`
* **Core + PyTorch** (for neural network models): `conda install -c conda-forge -c pytorch u8darts-torch`
* **Core + Prophet, LightGBM, CatBoost, XGBoost, StatsForecast** (no neural networks): `conda install -c conda-forge u8darts-notorch`
* **All available models**: `conda install -c conda-forge -c pytorch u8darts-all`

## Other Information

### Enabling GPU support
Darts relies on PyTorch for the neural network models.
For GPU support, please follow the instructions to install CUDA in the [PyTorch installation guide](https://pytorch.org/get-started/locally/).


### From Docker
We also provide a Docker image with everything set up for you. For this setup to work you need to have a Docker service installed. You can get it at [Docker website](https://docs.docker.com/get-docker/).

Pull the latest Darts image.
```bash
docker pull unit8/darts:latest
```

To run it in interactive mode:
```bash
docker run -it -p 8888:8888 unit8/darts:latest bash
```

After that, you can also launch a Jupyter lab / notebook session:
```bash
jupyter lab --ip 0.0.0.0 --no-browser --allow-root
```

Then copy and paste the URL provided by the docker container into your browser to access Jupyter notebook.

## ⚠️ Important: Darts PyPI Package Changes As of Version 0.41.0
As of Darts version 0.41.0, we have made changes to our PyPI packages:

- `darts`: `darts` now replaces `u8darts` with all of its installation options (see section above).
- `u8darts`: we will stop maintaining the `u8darts` package in favor of `darts`. Version 0.41.0 will be the last released version.

We made these changes to simplify the installation and maintenance of Darts.

#### Migration from Darts versions <0.41.0 to >=0.41.0
No code changes are required - only package installations changes.

For `darts` users:

```
# the original `pip install darts` becomes:
pip install "darts[torch]>=0.41.0"
```

For `u8darts` users:

```
# the original `pip install u8darts[option]` becomes:
pip install "darts[option]>=0.41.0"  # or appropriate extras (e.g. darts[all])
```
