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

#### Test environment Apple M1 processor

We currently recommend to run Darts in an x_64 emulated environment on Mac computers with the Silicon M1 processor,
instead of trying to install directly with native arm64 packages, many of the dependent packages still have compatibility 
issues. The following is a proposed procedure, if you tested other procedures on similar hardware and they worked, 
please let us know about them by opening an issue or by updating this file and opening a PR. 

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
  - With this method of installation, lightgbm might still have issues finding the libomp library.
  The following procedure is to guarantee that the correct libomp (11.1.0) library is linked.
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

## Running the examples only, without installing:

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