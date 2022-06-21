# Installation Guide

Below, we detail how to install Darts using either `conda` or `pip`.

## From conda-forge
Currently only the x86_64 architecture with Python 3.7-3.10
is fully supported with conda; consider using PyPI if you are running into troubles.

To create a conda environment for Python 3.9
(after installing [conda](https://docs.conda.io/en/latest/miniconda.html)):

    conda create --name <env-name> python=3.9

Don't forget to activate your virtual environment

    conda activate <env-name>

As some models have relatively heavy dependencies, we provide two conda-forge packages:

* Install darts with all available models (recommended): `conda install -c conda-forge -c pytorch u8darts-all`.
* Install core + neural networks (PyTorch): `conda install -c conda-forge -c pytorch u8darts-torch`
* Install core only (without neural networks or AutoARIMA): `conda install -c conda-forge u8darts`

For GPU support, please follow the instructions to install CUDA in the [PyTorch installation guide](https://pytorch.org/get-started/locally/).


## From PyPI
Install darts with all available models: `pip install darts`.

If this fails on your platform, please follow the official installation 
guide for [PyTorch](https://pytorch.org/get-started/locally/), then try installing Darts again.

As some dependencies are relatively big or involve non-Python dependencies,
we also maintain the `u8darts` package, which provides the following alternate lighter install options:

* Install core only (without neural networks, Prophet or AutoARIMA): `pip install u8darts`
* Install core + neural networks (PyTorch): `pip install "u8darts[torch]"`
* Install core + AutoARIMA: `pip install "u8darts[pmdarima]"`

### Enabling Support for LightGBM

To enable support for LightGBM in Darts, please follow the
[installation instructions](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html) for your OS.

#### MacOS Issues with LightGBM
At the time of writing, there is an issue with ``libomp`` 12.0.1 that results in
[segmentation fault on Mac OS Big Sur](https://github.com/microsoft/LightGBM/issues/4229).
Here's the procedure to downgrade the ``libomp`` library (from the
[original Github issue](https://github.com/microsoft/LightGBM/issues/4229#issue-867528353)):
* [Install brew](https://brew.sh/) if you don't already have it.
* Install `wget` if you don't already have it : `brew install wget`.
* Run the commands below:
```
wget https://raw.githubusercontent.com/Homebrew/homebrew-core/fb8323f2b170bd4ae97e1bac9bf3e2983af3fdb0/Formula/libomp.rb
brew unlink libomp
brew install libomp.rb
```

## Enabling support for Facebook Prophet
We removed Facebook Prophet as a dependency of Darts (at least for the time being), due to its dependency
on PyStan and the complex installation this entails. In order to use the `Prophet` model in Darts, we
recommend you follow the [Prophet installation instructions](https://facebook.github.io/prophet/docs/installation.html)
and install the prophet package in your environment (the command
`from darts.models import Prophet` will work once the package is installed).
At the time of writing, this has been tested with `prophet 1.0.1`.

## Running the examples only, without installing:

If the conda setup is causing too many problems, we also provide a Docker image with everything set up for you and ready-to-use Python notebooks with demo examples.
To run the example notebooks without installing our libraries natively on your machine, you can use our Docker image:
```bash
./gradlew docker && ./gradlew dockerRun
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