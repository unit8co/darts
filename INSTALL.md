# Installation Guide

Below, we detail how to install Darts using either `conda` or `pip`.

## From PyPI

### ⚠️ Important: Package Name Change

**If you previously used `u8darts`:** The package has been renamed to `darts` for simplicity.

**Migration:**
```bash
pip uninstall u8darts
pip install "darts[all]"  # or appropriate extras
```

**Your code doesn't need changes** - both packages import as `import darts`.

**Full guide:** [MIGRATION.md](MIGRATION.md)

---

### Installation Options

Darts offers a modular installation system with optional dependencies. Choose the installation that fits your needs:

* **Core only** (without neural networks, Prophet, LightGBM, CatBoost, XGBoost, StatsForecast): `pip install darts`
* **Core + PyTorch** (for neural network models): `pip install "darts[torch]"`
* **Core + Prophet, LightGBM, CatBoost, XGBoost, StatsForecast** (no neural networks): `pip install "darts[notorch]"`
* **All available models**: `pip install "darts[all]"`

If installation fails on your platform, please follow the official installation guide for [PyTorch](https://pytorch.org/get-started/locally/), then try installing Darts again.

## From conda-forge
Create a conda environment (e.g., for Python 3.11):
(after installing [conda](https://docs.conda.io/en/latest/miniconda.html)):

    conda create --name <env-name> python=3.11

Activate the environment

    conda activate <env-name>

As some models have relatively heavy dependencies, we provide four conda-forge packages:

* Install Darts with all available models: `conda install -c conda-forge -c pytorch darts-all`
* Install core only (without neural networks, Prophet, LightGBM, CatBoost, XGBoost, StatsForecast): `conda install -c conda-forge darts`
* Install core + Prophet + LightGBM + CatBoost + XGBoost + StatsForecast: `conda install -c conda-forge darts-notorch`
* Install core + neural networks (PyTorch): `conda install -c conda-forge -c pytorch darts-torch`


## Other Information

### Enabling Optional Dependencies
The default `darts` package includes only core dependencies. Packages Prophet, CatBoost, LightGBM, XGBoost and StatsForecast are not installed by default. We still fully support our model wrappers `Prophet`, `CatBoostModel`, `LightGBMModel`, `XGBoost` and `StatsForecast` in Darts. To use any of them, you can either:
- Install the appropriate extras: `pip install "darts[notorch]"` or `pip install "darts[all]"`
- Or manually install the corresponding packages as described below:

#### Prophet
Install the `prophet` package (version 1.1.1 or more recent) using the [Prophet install guide](https://facebook.github.io/prophet/docs/installation.html#python)

#### CatBoostModel
Install the `catboost` package (version 1.0.6 or more recent) using the [CatBoost install guide](https://catboost.ai/en/docs/concepts/python-installation)

#### LightGBMModel
Install the `lightgbm` package (version 3.2.0 or more recent) using the [LightGBM install guide](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html)

#### XGBoost
Install the `xgboost` package (version 2.1.4 or more recent) using the [XGBoost install guide](https://xgboost.readthedocs.io/en/stable/install.html)

#### StatsForecast
Install the `statsforecast` package (version 1.4 or more recent) using the [StatsForecast install guide](https://nixtlaverse.nixtla.io/statsforecast/index.html#installation)

### Enabling GPU support
Darts relies on PyTorch for the neural network models.
For GPU support, please follow the instructions to install CUDA in the [PyTorch installation guide](https://pytorch.org/get-started/locally/).


### From Docker:
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
