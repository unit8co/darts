# Installation Guide

Below, we detail how to install Darts using either `conda` or `pip`.

## From PyPI
Install Darts with all models except the ones from optional dependencies (Prophet, LightGBM, CatBoost, XGBoost, StatsForecast see more on that [here](#enabling-optional-dependencies)): `pip install darts`.

If this fails on your platform, please follow the official installation
guide for [PyTorch](https://pytorch.org/get-started/locally/), then try installing Darts again.

## From Source (Development Version)
To install the latest development version directly from GitHub:

```bash
# Install from main branch
uv pip install git+https://github.com/unit8co/darts.git@master

# Or install a specific branch (e.g., a feature branch)
uv pip install git+https://github.com/unit8co/darts.git@feature-branch-name

# Or install with optional dependencies
uv pip install "git+https://github.com/unit8co/darts.git@master#egg=u8darts[all]"
```

This is useful for testing unreleased features or contributing to development.

As some dependencies are relatively big or involve non-Python dependencies,
we also maintain the `u8darts` package, which provides the following alternate lighter install options:

* Install Darts with all available models: `pip install "u8darts[all]"`
* Install core only (without neural networks, Prophet, LightGBM, Catboost, XGBoost and StatsForecast): `pip install u8darts`
* Install core + Prophet + LightGBM + CatBoost + XGBoost + StatsForecast: `pip install "u8darts[notorch]"`
* Install core + neural networks (PyTorch): `pip install "u8darts[torch]"` (equivalent to `pip install darts`)

## From conda-forge
Create a conda environment (e.g., for Python 3.11):
(after installing [conda](https://docs.conda.io/en/latest/miniconda.html)):

    conda create --name <env-name> python=3.11

Activate the environment

    conda activate <env-name>

As some models have relatively heavy dependencies, we provide four conda-forge packages:

* Install Darts with all available models: `conda install -c conda-forge -c pytorch u8darts-all`
* Install core only (without neural networks, Prophet, LightGBM, Catboost, XGBoost and StatsForecast): `conda install -c conda-forge u8darts`
* Install core + Prophet + LightGBM + CatBoost + XGBoost + StatsForecast: `conda install -c conda-forge u8darts-notorch`
* Install core + neural networks (PyTorch): `conda install -c conda-forge -c pytorch u8darts-torch`


## Other Information

### Enabling Optional Dependencies
As of version 0.38.0, we made the default `darts` package more lightweight. Packages Prophet, CatBoost, LightGBM, XGBoost and StatsForecast will not be installed anymore. Don't worry though, we keep supporting our model wrappers `Prophet`, `CatBoostModel`, `LightGBMModel`, `XGBoost` and `StatsForecast` in Darts. If you want to use any of them, you will need to manually install the corresponding packages (or install a Darts flavor as described above).

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

#### TimesFM
To use the `TimesFMModel` wrapper for Google's TimesFM foundation model:

**Option 1: Install with Darts extras (recommended)**
```bash
pip install "darts[timesfm]"
# or with uv
uv pip install "darts[timesfm]"
```

**Option 2: Manual installation from source**
1. Install Darts with PyTorch support: `pip install "u8darts[torch]"` or `pip install darts`
2. Install TimesFM from source:
   ```bash
   git clone https://github.com/google-research/timesfm.git
   cd timesfm
   pip install -e .[torch]
   ```

**Requirements:**
- Python 3.11+ (for PyTorch version)
- PyTorch 2.0+ with MPS support (for Apple Silicon) or CUDA (for NVIDIA GPUs)

**Example:**
```python
from darts.datasets import AirPassengersDataset
from darts.models import TimesFMModel

series = AirPassengersDataset().load()
model = TimesFMModel(zero_shot=True)
model.fit(series)
forecast = model.predict(n=12, series=series)
```

For more details, see the [TimesFM GitHub repository](https://github.com/google-research/timesfm) and [HuggingFace model card](https://huggingface.co/google/timesfm-2.5-200m-pytorch).

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
