# Installation Guide

## Preconditions

Our direct dependencies include `fbprophet` and `torch` which have non-Python dependencies.
A Conda environment is thus recommended because it will handle all of those in one go.

The following steps assume running inside a conda environment. 
If that's not possible, first follow the official instructions to install 
[fbprophet](https://facebook.github.io/prophet/docs/installation.html#python)
and [torch](https://pytorch.org/get-started/locally/), then skip to 
[Install darts](#install-darts)

To create a conda environment for Python 3.7
(after installing [conda](https://docs.conda.io/en/latest/miniconda.html)):

    conda create --name <env-name> python=3.7

Don't forget to activate your virtual environment

    conda activate <env-name>


## MAC

    conda install -c conda-forge -c pytorch pip fbprophet pytorch

## Linux and Windows

    conda install -c conda-forge -c pytorch pip fbprophet pytorch cpuonly

## Install darts

    pip install u8darts

## Running the examples only, without installing:

If the conda setup is causing too many problems, we also provide a Docker image with everything set up for you and ready-to-use python notebooks with demo examples.
To run the example notebooks without installing our libraries natively on your machine, you can use our Docker image:
```
cd scripts
./build_docker.sh && ./run_docker.sh
```

Then copy and paste the URL provided by the docker container into your browser to access Jupyter notebook.

For this setup to work you need to have a Docker service installed. You can get it at [Docker website](https://docs.docker.com/get-docker/).