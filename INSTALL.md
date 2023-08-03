# Installation Guide

Below, we detail how to install Darts using either `conda` or `pip`.

## From PyPI
Install darts with all models expect the ones from optional dependencies (Prophet, LightGBM, CatBoost, see more on that [here](#enabling-optional-dependencies)): `pip install darts`.

If this fails on your platform, please follow the official installation 
guide for [PyTorch](https://pytorch.org/get-started/locally/), then try installing Darts again.

As some dependencies are relatively big or involve non-Python dependencies,
we also maintain the `u8darts` package, which provides the following alternate lighter install options:

* Install darts with all available models: `pip install u8darts[all]`
* Install core only (without neural networks, Prophet, LightGBM and Catboost): `pip install u8darts`
* Install core + Prophet + LightGBM + CatBoost: `pip install "u8darts[notorch]"`
* Install core + neural networks (PyTorch): `pip install "u8darts[torch]"` (equivalent to `pip install darts`)

## From conda-forge
Currently only the x86_64 architecture with Python 3.8-3.10
is fully supported with conda; consider using PyPI if you are running into troubles.

Create a conda environment (e.g., for Python 3.10):
(after installing [conda](https://docs.conda.io/en/latest/miniconda.html)):

    conda create --name <env-name> python=3.10

Activate the environment

    conda activate <env-name>

As some models have relatively heavy dependencies, we provide four conda-forge packages:

* Install darts with all available models: `conda install -c conda-forge -c pytorch u8darts-all`
* Install core only (without neural networks, Prophet, LightGBM and Catboost): `conda install -c conda-forge u8darts`
* Install core + Prophet + LightGBM + CatBoost: `pip install "u8darts-notorch"`
* Install core + neural networks (PyTorch): `conda install -c conda-forge -c pytorch u8darts-torch`


## Other Information

### Enabling Optional Dependencies
By default, as of 0.25.0, `darts` does not have Prophet, CatBoost, and LightGBM as dependencies anymore, because their 
build processes were too often causing issues. If you want to use any of Darts' `Prophet`, `CatBoostModel`, and 
`LightGBMModel`, you will need to manually install the corresponding packages.  

#### Prophet
Install the `prophet` package (version 1.1.1 or more recent) using the [Prophet install guide](https://facebook.github.io/prophet/docs/installation.html#python)

#### CatBoostModel
Install the `catboost` package (version 1.0.6 or more recent) using the [CatBoost install guide](https://catboost.ai/en/docs/concepts/python-installation)

#### LightGBMModel
Install the `lightgbm` package (version 3.2.0 or more recent) using the [LightGBM install guide](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html)

### Enabling GPU support
Darts relies on PyTorch for the neural network models.
For GPU support, please follow the instructions to install CUDA in the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

### Using an emulated x64 environment on Apple Silicon.
The installation of `darts` has been tested to work on Apple silicon (M1) (Python 3.10, OSX Ventura 13.2.1).

If you run into issues, you can always use rosetta to run in an (intel) emulated x64 environment:

Before you start make sure that you have rosetta2 installed by running: 

```bash
pgrep oahd
``` 

If you see a process id you are ready to go, as internally rosetta is known as oah.

If pgrep doesn't return any id then install rosetta2:

```bash
softwareupdate --install-rosetta
```

Below are the necessary instructions to create and configure the environment:
- Install conda if you haven't done so (e.g., with miniforge : `brew install miniforge`).
- Create the x_64 environment : `CONDA_SUBDIR=osx-64 conda create -n env_name python=3.10 pip`
- Activate the environment: `conda activate env_name`
- Configure the environment : `conda env config vars set CONDA_SUBDIR=osx-64`
- Deactivate and reactivate the environment:
  ```bash
  conda deactivate
  conda activate env_name
  ```
- Install darts: `pip install darts`

### Running the examples only, without installing:

If the conda setup is causing too many problems, we also provide a Docker image with everything set up for you and ready-to-use Python notebooks with demo examples.
To run the example notebooks without installing our libraries natively on your machine, you can use our Docker image:
```bash
./gradlew docker && ./gradlew dockerRun
```
If you are having M1/M2 chipset then you should change the default platform: (unfortunately not all libraries support ARM architecture)
```bash
DOCKER_DEFAULT_PLATFORM=linux/amd64 ./gradlew docker && ./gradlew dockerRun
```

Then copy and paste the URL provided by the docker container into your browser to access Jupyter notebook.

For this setup to work you need to have a Docker service installed. You can get it at [Docker website](https://docs.docker.com/get-docker/).


## Tests

The gradle setup works best when used in a python environment, but the only requirement is to have `pip` installed for Python 3+

To run all tests at once just run
```bash
./gradlew test_all
```

alternatively you can run
```bash
./gradlew unitTest_all # to run only unittests
./gradlew coverageTest # to run coverage
./gradlew lint         # to run linter
```

To run the tests for specific flavours of the library, replace `_all` with `_core`, `_prophet`, `_pmdarima` or `_torch`.

## Documentation

To build documentation locally just run
```bash
./gradlew buildDocs
```
After that docs will be available in `./docs/build/html` directory. You can just open `./docs/build/html/index.html` using your favourite browser.