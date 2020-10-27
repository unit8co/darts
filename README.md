# Time Series Made Easy in Python

![darts](https://github.com/unit8co/darts/raw/develop/static/images/darts-logo-trim.png "darts") 

---
[![PyPI version](https://badge.fury.io/py/u8darts.svg)](https://badge.fury.io/py/u8darts)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/unit8co/darts/darts%20release%20workflow/master)
![Supported versions](https://img.shields.io/badge/python-3.6+-blue.svg)
![Docker Image Version (latest by date)](https://img.shields.io/docker/v/unit8/darts?label=docker&sort=date)
![PyPI - Downloads](https://img.shields.io/pypi/dm/u8darts)
![GitHub Release Date](https://img.shields.io/github/release-date/unit8co/darts)

**darts** is a python library for easy manipulation and forecasting of time series.
It contains a variety of models, from classics such as ARIMA to neural networks.
The models can all be used in the same way, using `fit()` and `predict()` functions,
similar to scikit-learn. The library also makes it easy to backtest models,
and combine the predictions of several models and external regressors. Darts supports both
univariate and multivariate time series and models.

## Install

We recommend to first setup a clean python environment for your project with at least python 3.6 using your favorite tool ([conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html "conda-env"), [venv](https://docs.python.org/3/library/venv.html), [virtualenv](https://virtualenv.pypa.io/en/latest/) with or without [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/)).

### Quick Install

Once your environment is setup you can install darts using the pip package:

    pip install 'u8darts[all]'

### Step-by-step Install

For more detailed install instructions you can refer to our installation guide at the end of this page.

## Example Usage

Create a `TimeSeries` object from a Pandas DataFrame, and split it in train/validation series:

```python
import pandas as pd
from darts import TimeSeries

df = pd.read_csv('AirPassengers.csv', delimiter=",")
series = TimeSeries.from_dataframe(df, 'Month', '#Passengers')
train, val = series.split_after(pd.Timestamp('19580101'))
```

>The dataset used in this example can be downloaded from this [link](https://raw.githubusercontent.com/unit8co/darts/master/examples/AirPassengers.csv).

Fit an exponential smoothing model, and make a prediction over the validation series' duration:

```python
from darts.models import ExponentialSmoothing

model = ExponentialSmoothing()
model.fit(train)
prediction = model.predict(len(val))
```

Plot:
```python
import matplotlib.pyplot as plt

series.plot(label='actual')
prediction.plot(label='forecast', lw=2)
plt.legend()
plt.xlabel('Year')
```

<div style="text-align:center;">
<img src="https://github.com/unit8co/darts/raw/develop/static/images/example.png" alt="darts forecast example" />
</div>

We invite you to go over the example notebooks in the `examples` directory.

## Documentation

The documentation of the API and models is available [here](https://unit8co.github.io/darts/).

## Features

Currently, the library contains the following features: 

**Forecasting Models:**

* Exponential smoothing,
* ARIMA & auto-ARIMA,
* Facebook Prophet,
* Theta method,
* FFT (Fast Fourier Transform),
* Recurrent neural networks (vanilla RNNs, GRU, and LSTM variants),
* Temporal convolutional network.
* Transformer
* N-BEATS

**Data processing:** Tools to easily apply (and revert) common transformations on time series data (scaling, boxcox, …)

**Metrics:** A variety of metrics for evaluating time series' goodness of fit; 
from R2-scores to Mean Absolute Scaled Error.

**Backtesting:** Utilities for simulating historical forecasts, using moving time windows.

**Regressive Models:** Possibility to predict a time series from several other time series 
(e.g., external regressors), using arbitrary regressive models

**Multivariate Support:** Tools to create, manipulate and forecast multivariate time series.

## Contribute

The development is ongoing, and there are many new features that we want to add. 
We welcome pull requests and issues on GitHub.

Before working on a contribution (a new feature or a fix) make sure you can't find anything related in [issues](https://github.com/unit8co/darts/issues). If there is no on-going effort on what you plan to do then we recommend to do the following:

1. Create an issue, describe how you would attempt to solve it, and if possible wait for a discussion.
2. Fork the repository.
3. Clone the forked repository locally.
4. Create a clean python env and install requirements with pip: `pip install -r requirements/dev-all.txt`
5. Create a new branch:
    * Branch off from the **develop** branch.
    * Prefix the branch with the type of update you are making:
        * `feature/`
        * `fix/`
        * `refactor/`
        * …
    * Work on your update
6. Check that your code passes all the tests and design new unit tests if needed: `./gradlew unitTest_all`.
7. Verify your tests coverage by running `./gradlew coverageTest`
    * Additionally you can generate an xml report and use VSCode Coverage gutter to identify untested lines with `./coverage.sh xml`
8. If your contribution introduces a significant change, add it to `CHANGELOG.md` under the "Unreleased" section.
9. Create a pull request from your new branch to the **develop** branch.

## Contact Us

If what you want to tell us is not a suitable github issue, feel free to send us an email at <a href="mailto:darts@unit8.co">darts@unit8.co</a> for darts related matters or <a href="mailto:info@unit8.co">info@unit8.co</a> for any other inquiries.

## Installation Guide

### Preconditions

Some of the models depend on `fbprophet` and `torch`, which have non-Python dependencies.
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


#### MAC

    conda install -c conda-forge -c pytorch pip fbprophet pytorch

#### Linux and Windows

    conda install -c conda-forge -c pytorch pip fbprophet pytorch cpuonly

### Install darts

Install Darts with all available models: `pip install 'u8darts[all]'`.

As some models have relatively heavy (or non-Python) dependencies,
we also provide the following alternate lighter install options: 

* Install core only (without neural networks, Prophet or AutoARIMA): `pip install u8darts`
* Install core + neural networks (PyTorch): `pip install 'u8darts[torch]'`
* Install core + Facebook Prophet: `pip install 'u8darts[fbprophet]'`
* Install core + AutoARIMA: `pip install 'u8darts[pmdarima]'`
   
### Running the examples only, without installing:

If the conda setup is causing too many problems, we also provide a Docker image with everything set up for you and ready-to-use python notebooks with demo examples.
To run the example notebooks without installing our libraries natively on your machine, you can use our Docker image:
```bash
./gradlew docker && ./gradlew dockerRun
```

Then copy and paste the URL provided by the docker container into your browser to access Jupyter notebook.

For this setup to work you need to have a Docker service installed. You can get it at [Docker website](https://docs.docker.com/get-docker/).


### Tests

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

To run the tests for specific flavours of the library, replace `_all` with `_core`, `_fbprophet`, `_pmdarima` or `_torch`.

### Documentation

To build documantation locally just run
```bash
./gradlew buildDocs
```
After that docs will be available in `./docs/build/html` directory. You can just open `./docs/build/html/index.html` using your favourite browser.
