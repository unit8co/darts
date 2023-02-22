# Installation Guide

Below, we detail how to install Darts using either `conda` or `pip`.

## From conda-forge
Currently only the x86_64 architecture with Python 3.7-3.10
is fully supported with conda; consider using PyPI if you are running into troubles.

Create a conda environment (e.g., for Python 3.10):
(after installing [conda](https://docs.conda.io/en/latest/miniconda.html)):

    conda create --name <env-name> python=3.10

Activate your virtual environment

    conda activate <env-name>

As some models have relatively heavy dependencies, we provide two conda-forge packages:

* Install darts with all available models (recommended): `conda install -c conda-forge -c pytorch u8darts-all`.
* Install core + neural networks (PyTorch): `conda install -c conda-forge -c pytorch u8darts-torch`
* Install core only (without neural networks): `conda install -c conda-forge u8darts`

## From PyPI
Install darts with all available models: `pip install darts`.

If this fails on your platform, please follow the official installation 
guide for [PyTorch](https://pytorch.org/get-started/locally/), then try installing Darts again.

As some dependencies are relatively big or involve non-Python dependencies,
we also maintain the `u8darts` package, which provides the following alternate lighter install options:

* Install core only (without neural networks, Prophet or AutoARIMA): `pip install u8darts`
* Install core + neural networks (PyTorch): `pip install "u8darts[torch]"`
* Install core + AutoARIMA: `pip install "u8darts[pmdarima]"`

## Other Information

### Issues with LightGBM
If you run into issues with LightGBM when installing Darts, please follow the
[installation instructions](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html) for your OS,
and then try re-installing Darts.

For instance, on MacOS you may have to run this (using [brew](https://brew.sh/)):
```
brew install cmake
brew install libomp
```
and then again: `pip install darts`.

### Enabling support for Prophet
By default, as of 0.24.0, `darts` does not have Prophet as a dependency anymore, because its build
process was too often causing issues, [notably on Apple silicon](https://github.com/facebook/prophet/issues/2002).

If you want to use Darts' `Prophet` model, you will need to install the `prophet` package (version 1.1 or more recent).
We refer to the [Prophet README install guide](https://github.com/facebook/prophet#installation-in-python---pypi-release)

### Enabling GPU support
Darts relies on PyTorch for the neural network models.
For GPU support, please follow the instructions to install CUDA in the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

### Using an emulated x64 environment on Apple Silicon.
The installation of `darts` has been tested to work on Apple silicon (Python 3.10, OSX Ventura 13.2.1).
It requires following the instructions to make LightGBM work 
[here](https://github.com/unit8co/darts/blob/master/INSTALL.md#issues-with-lightgbm).

If you still run into some issues with Apple silicon, you can consider using rosetta
to use an emulated x64 environment by following the steps below:

Before you start make sure that you have rosetta2 installed by running: 
```
pgrep oahd
``` 
If you see some process id you are ready to go, as internally rosetta is known as oah.

If pgrep doesn't return any id then install rosetta2:
```
softwareupdate --install-rosetta
```

Below are the necessary instructions to create and configure the environment:
- Start by installing conda (e.g., with miniforge : `brew install miniforge`).
- Create the x_64 environment : `CONDA_SUBDIR=osx-64 conda create -n env_name python=3.9 pip`
- Activate the created environment: `conda activate env_name`
- Configure the environment : `conda env config vars set CONDA_SUBDIR=osx-64`
- Deactivate and reactivate the environment:
  ```
  conda deactivate
  conda activate env_name
  ```
- Install darts: `pip install darts`
  - If after this you still run into issues with lightgbm having issues finding the libomp library,
  the following procedure guarantees that the correct libomp (11.1.0) library is linked.
    - Unlink the existing libomp, from terminal : `brew unlink libomp`
    - Setup a homebrew installer that is compatible with x_64 packages (follow this [blog](https://medium.com/mkdir-awesome/how-to-install-x86-64-homebrew-packages-on-apple-m1-macbook-54ba295230f) 
    post):
    ```
    cd ~/Downloads
    mkdir homebrew
    curl -L https://github.com/Homebrew/brew/tarball/master | tar xz --strip 1 -C homebrew
    sudo mv homebrew /usr/local/homebrew
    export PATH=$HOME/bin:/usr/local/bin:$PATH
    ```
    - At this point, we have a new brew command located at /usr/local/homebrew/bin/brew
    - In the following code bits we download version 11.1.0 of libomp, install it as a x_64 compatible package and link to it so that lightgbm can find it:
    ```
    wget https://raw.githubusercontent.com/Homebrew/homebrew-core/fb8323f2b170bd4ae97e1bac9bf3e2983af3fdb0/Formula/libomp.rb
    arch -x86_64 /usr/local/homebrew/bin/brew install libomp.rb
    sudo ln -s /usr/local/homebrew/Cellar/libomp/11.1.0/lib /usr/local/opt/libomp/lib
    ```
    - Verify that your lightgbm works by importing lightgbm from your python env. It should not give library loading errors. 

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