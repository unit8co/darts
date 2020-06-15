# darts: Easy manipulation and forecasting of time series

![darts](https://github.com/unit8co/darts/raw/develop/static/images/darts-logo-trim.png "darts") 

---
[![PyPI version](https://badge.fury.io/py/u8darts.svg)](https://badge.fury.io/py/u8darts)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/unit8co/darts/merge)
![Supported versions](https://img.shields.io/badge/python-3.6+-blue.svg)
![Docker Image Version (latest by date)](https://img.shields.io/docker/v/unit8/darts?label=docker&sort=date)
![PyPI - Downloads](https://img.shields.io/pypi/dm/u8darts)
![GitHub Release Date](https://img.shields.io/github/release-date/unit8co/darts)

**darts** is a python library for easy manipulation and forecasting time series.
It contains a variety of models, from classics such as ARIMA to neural networks.
The models can all be used in the same way, using `fit()` and `predict()` functions,
similar to scikit-learn. The library also makes it easy to backtest models,
and combine the predictions of several models and external regressors.

## Install

### Preconditions

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


#### MAC

    conda install -c conda-forge -c pytorch pip fbprophet pytorch

#### Linux and Windows

    conda install -c conda-forge -c pytorch pip fbprophet pytorch cpuonly

### Install darts

    pip install u8darts

### Running the examples only, without installing:

If the conda setup is causing too many problems, we also provide a Docker image with everything set up for you and ready-to-use python notebooks with demo examples.
To run the example notebooks without installing our libraries natively on your machine, you can use our Docker image:
```

cd scripts
./build_docker.sh && ./run_docker.sh
```

Then copy and paste the URL provided by the docker container into your browser to access Jupyter notebook.

For this setup to work you need to have a Docker service installed. You can get it at [Docker website](https://docs.docker.com/get-docker/).

## Example Usage

Create `TimeSeries` object from a Pandas DataFrame, and split in train/validation series:

```python
from darts import TimeSeries

df = pd.read_csv('AirPassengers.csv', delimiter=",")
series = TimeSeries.from_dataframe(df, 'Month', '#Passengers')
train, val = series.split_after(pd.Timestamp('19590101'))
```

Fit an exponential smoothing model, and make a prediction over the validation series' duration:

```python
from darts import ExponentialSmoothing

model = ExponentialSmoothing()
model.fit(train)
prediction = model.predict(len(val))
```

Plot:
```python
series.plot(label='actual', lw=3)
prediction.plot(label='forecast', lw=3)
plt.legend()
plt.xlabel('Year')
```

![example](https://github.com/unit8co/darts/raw/develop/static/images/example.png "example") { width=100% }

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

**Preprocessing:** Transformer tool for easily scaling / normalizing time series.

**Metrics:** A variety of metrics for evaluating time series' goodness of fit; 
from R2-scores to Mean Absolute Scaled Error.

**Backtesting:** Utilities for simulating historical forecasts, using moving time windows.

**Regressive Models:** Possibility to predict a time series from several other time series 
(e.g., external regressors), using arbitrary regressive models.

## Contribute

The development is ongoing, and there are many new features that we want to add. 
We welcome pull requests and issues on github.
