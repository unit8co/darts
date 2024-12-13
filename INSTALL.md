# Installation Guide

Below, we detail how to install Darts using either `conda` or `pip`.

## From PyPI
Install Darts with all models except the ones from optional dependencies (Prophet, LightGBM, CatBoost, see more on that [here](#enabling-optional-dependencies)): `pip install darts`.

If this fails on your platform, please follow the official installation
guide for [PyTorch](https://pytorch.org/get-started/locally/), then try installing Darts again.

As some dependencies are relatively big or involve non-Python dependencies,
we also maintain the `u8darts` package, which provides the following alternate lighter install options:

* Install Darts with all available models: `pip install "u8darts[all]"`
* Install core only (without neural networks, Prophet, LightGBM and Catboost): `pip install u8darts`
* Install core + Prophet + LightGBM + CatBoost: `pip install "u8darts[notorch]"`
* Install core + neural networks (PyTorch): `pip install "u8darts[torch]"` (equivalent to `pip install darts`)

## From conda-forge
Create a conda environment (e.g., for Python 3.10):
(after installing [conda](https://docs.conda.io/en/latest/miniconda.html)):

    conda create --name <env-name> python=3.10

Activate the environment

    conda activate <env-name>

As some models have relatively heavy dependencies, we provide four conda-forge packages:

* Install Darts with all available models: `conda install -c conda-forge -c pytorch u8darts-all`
* Install core only (without neural networks, Prophet, LightGBM and Catboost): `conda install -c conda-forge u8darts`
* Install core + Prophet + LightGBM + CatBoost: `conda install -c conda-forge u8darts-notorch`
* Install core + neural networks (PyTorch): `conda install -c conda-forge -c pytorch u8darts-torch`


## Other Information

### Enabling Optional Dependencies
As of version 0.25.0, the default `darts` package does not install Prophet, CatBoost, and LightGBM dependencies anymore, because their
build processes were too often causing issues. We continue supporting the model wrappers `Prophet`, `CatBoostModel`, and `LightGBMModel` in Darts though. If you want to use any of them, you will need to manually install the corresponding packages (or install a Darts flavor as described above).

#### Prophet
Install the `prophet` package (version 1.1.1 or more recent) using the [Prophet install guide](https://facebook.github.io/prophet/docs/installation.html#python)

#### CatBoostModel
Install the `catboost` package (version 1.0.6 or more recent) using the [CatBoost install guide](https://catboost.ai/en/docs/concepts/python-installation)

#### LightGBMModel
Install the `lightgbm` package (version 3.2.0 or more recent) using the [LightGBM install guide](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html)

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
