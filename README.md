# Time Series Made Easy in Python

![darts](https://github.com/unit8co/darts/raw/master/static/images/darts-logo-trim.png "darts")

---
[![PyPI version](https://badge.fury.io/py/u8darts.svg)](https://badge.fury.io/py/darts)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/u8darts-all.svg)](https://anaconda.org/conda-forge/u8darts-all)
![Supported versions](https://img.shields.io/badge/python-3.7+-blue.svg)
![Docker Image Version (latest by date)](https://img.shields.io/docker/v/unit8/darts?label=docker&sort=date)
![GitHub Release Date](https://img.shields.io/github/release-date/unit8co/darts)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/unit8co/darts/darts%20release%20workflow/master)
[![Downloads](https://pepy.tech/badge/u8darts)](https://pepy.tech/project/u8darts)
[![Downloads](https://pepy.tech/badge/darts)](https://pepy.tech/project/darts)
[![codecov](https://codecov.io/gh/unit8co/darts/branch/master/graph/badge.svg?token=7F1TLUFHQW)](https://codecov.io/gh/unit8co/darts)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Join the chat at https://gitter.im/u8darts/darts](https://badges.gitter.im/u8darts/darts.svg)](https://gitter.im/u8darts/darts?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

**darts** is a Python library for easy manipulation and forecasting of time series.
It contains a variety of models, from classics such as ARIMA to deep neural networks.
The models can all be used in the same way, using `fit()` and `predict()` functions,
similar to scikit-learn. The library also makes it easy to backtest models,
combine the predictions of several models, and take external data into account. 
Darts supports both univariate and multivariate time series and models. 
The ML-based models can be trained on potentially large datasets containing multiple time
series, and some of the models offer a rich support for probabilistic forecasting.

## Documentation
* [Quickstart](https://unit8co.github.io/darts/quickstart/00-quickstart.html)
* [User Guide](https://unit8co.github.io/darts/userguide.html)
* [API Reference](https://unit8co.github.io/darts/generated_api/darts.html)
* [Examples](https://unit8co.github.io/darts/examples.html)

##### High Level Introductions
* [Introductory Blog Post](https://medium.com/unit8-machine-learning-publication/darts-time-series-made-easy-in-python-5ac2947a8878)
* [Introduction video (25 minutes)](https://youtu.be/g6OXDnXEtFA)

##### Articles on Selected Topics
* [Training Models on Multiple Time Series](https://medium.com/unit8-machine-learning-publication/training-forecasting-models-on-multiple-time-series-with-darts-dc4be70b1844)
* [Using Past and Future Covariates](https://medium.com/unit8-machine-learning-publication/time-series-forecasting-using-past-and-future-external-data-with-darts-1f0539585993)
* [Temporal Convolutional Networks and Forecasting](https://medium.com/unit8-machine-learning-publication/temporal-convolutional-networks-and-forecasting-5ce1b6e97ce4)
* [Probabilistic Forecasting](https://medium.com/unit8-machine-learning-publication/probabilistic-forecasting-in-darts-e88fbe83344e)
* [Transfer Learning for Time Series Forecasting](https://medium.com/unit8-machine-learning-publication/transfer-learning-for-time-series-forecasting-87f39e375278)
* [Hierarchical Forecast Reconciliation](https://medium.com/unit8-machine-learning-publication/hierarchical-forecast-reconciliation-with-darts-8b4b058bb543)

## Quick Install

We recommend to first setup a clean Python environment for your project with Python 3.7+ using your favorite tool
([conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html "conda-env"),
[venv](https://docs.python.org/3/library/venv.html), [virtualenv](https://virtualenv.pypa.io/en/latest/) with
or without [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/)).

Once your environment is set up you can install darts using pip:

    pip install darts

For more details you can refer to our 
[installation instructions](https://github.com/unit8co/darts/blob/master/INSTALL.md).

## Example Usage

Create a `TimeSeries` object from a Pandas DataFrame, and split it in train/validation series:

```python
import pandas as pd
from darts import TimeSeries

# Read a pandas DataFrame
df = pd.read_csv("AirPassengers.csv", delimiter=",")

# Create a TimeSeries, specifying the time and value columns
series = TimeSeries.from_dataframe(df, "Month", "#Passengers")

# Set aside the last 36 months as a validation series
train, val = series[:-36], series[-36:]
```

Fit an exponential smoothing model, and make a (probabilistic) prediction over the validation series' duration:
```python
from darts.models import ExponentialSmoothing

model = ExponentialSmoothing()
model.fit(train)
prediction = model.predict(len(val), num_samples=1000)
```

Plot the median, 5th and 95th percentiles:
```python
import matplotlib.pyplot as plt

series.plot()
prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
plt.legend()
```

<div style="text-align:center;">
<img src="https://github.com/unit8co/darts/raw/master/static/images/example.png" alt="darts forecast example" />
</div>

## Features
* **Forecasting Models:** A large collection of forecasting models; from statistical models (such as
  ARIMA) to deep learning models (such as N-BEATS). See [table of models below](#forecasting-models).
* **Multivariate Support:** `TimeSeries` can be multivariate - i.e., contain multiple time-varying
  dimensions instead of a single scalar value. Many models can consume and produce multivariate series.
* **Multiple series training:** All machine learning based models (incl. all neural networks) 
  support being trained on multiple (potentially multivariate) series. This can scale to large datasets.
* **Probabilistic Support:** `TimeSeries` objects can (optionally) represent stochastic
  time series; this can for instance be used to get confidence intervals, and many models support different flavours of probabilistic forecasting (such as estimating parametric distributions 
  or quantiles).
* **Past and Future Covariates support:** Many models in Darts support past-observed and/or future-known 
  covariate (external data) time series as inputs for producing forecasts.
* **Static Covariates support:** In addition to time-dependent data, `TimeSeries` can also contain
  static data for each dimension, which can be exploited by some models.
* **Hierarchical Reconciliation:** Darts offers transformers to perform reconciliation.
  These can make the forecasts add up in a way that respects the underlying hierarchy.
* **Regression Models:** It is possible to plug-in any scikit-learn compatible model
  to obtain forecasts as functions of lagged values of the target series and covariates.
* **Explainability:** Darts has the ability to *explain* forecasting models by using Shap values.
* **Data processing:** Tools to easily apply (and revert) common transformations on
  time series data (scaling, filling missing values, boxcox, ...)
* **Metrics:** A variety of metrics for evaluating time series' goodness of fit;
  from R2-scores to Mean Absolute Scaled Error.
* **Backtesting:** Utilities for simulating historical forecasts, using moving time windows.
* **PyTorch Lightning Support:** All deep learning models are implemented using PyTorch Lightning,
  supporting among other things custom callbacks, GPUs/TPUs training and custom trainers.
* **Filtering Models:** Darts offers three filtering models: `KalmanFilter`, `GaussianProcessFilter`,
  and `MovingAverage`, which allow to filter time series, and in some cases obtain probabilistic
  inferences of the underlying states/values.
* **Datasets** The `darts.datasets` submodule contains some popular time series datasets for rapid
  experimentation.

## Forecasting Models
Here's a breakdown of the forecasting models currently implemented in Darts. We are constantly working
on bringing more models and features.

Model | Univariate | Multivariate | Probabilistic | Multiple-series training | Past-observed covariates support | Future-known covariates | Static covariates support | Reference
--- | --- | --- | --- | --- | --- | --- | --- | ---
`ARIMA` | ✅ | | ✅ | | | ✅ | |
`VARIMA` | ✅ | ✅ | | | | ✅ | |
`AutoARIMA` | ✅ | | | | | ✅ | |
`StatsForecastAutoARIMA` (faster AutoARIMA) | ✅ | | ✅ | | | ✅ | | [Nixtla's statsforecast](https://github.com/Nixtla/statsforecast)
`ExponentialSmoothing` | ✅ | | ✅ | | | | |
`StatsForecastETS` | ✅ | | | | | ✅ | | [Nixtla's statsforecast](https://github.com/Nixtla/statsforecast)
`BATS` and `TBATS` | ✅ | | ✅ | | | | | [TBATS paper](https://robjhyndman.com/papers/ComplexSeasonality.pdf)
`Theta` and `FourTheta` | ✅ | | | | | | | [Theta](https://robjhyndman.com/papers/Theta.pdf) & [4 Theta](https://github.com/Mcompetitions/M4-methods/blob/master/4Theta%20method.R)
`Prophet` (see [install notes](https://github.com/unit8co/darts/blob/master/INSTALL.md#enabling-support-for-facebook-prophet)) | ✅ | | ✅ | | | ✅ | | [Prophet repo](https://github.com/facebook/prophet)
`FFT` (Fast Fourier Transform) | ✅ | | | | | | |
`KalmanForecaster` using the Kalman filter and N4SID for system identification | ✅ | ✅ | ✅ | | | ✅ | | [N4SID paper](https://people.duke.edu/~hpgavin/SystemID/References/VanOverschee-Automatica-1994.pdf)
`Croston` method | ✅ | | | | | | |
`RegressionModel`; generic wrapper around any sklearn regression model | ✅ | ✅ | | ✅ | ✅ | ✅ | |
`RandomForest` | ✅ | ✅ | | ✅ | ✅ | ✅ | |
`LinearRegressionModel` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | |
`LightGBMModel` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | |
`CatBoostModel` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | |
`RNNModel` (incl. LSTM and GRU); equivalent to DeepAR in its probabilistic version | ✅ | ✅ | ✅ | ✅ | | ✅ | | [DeepAR paper](https://arxiv.org/abs/1704.04110)
`BlockRNNModel` (incl. LSTM and GRU) | ✅ | ✅ | ✅ | ✅ | ✅ | | |
`NBEATSModel` | ✅ | ✅ | ✅ | ✅ | ✅ | | | [N-BEATS paper](https://arxiv.org/abs/1905.10437)
`NHiTSModel` | ✅ | ✅ | ✅ | ✅ | ✅ | | | [N-HiTS paper](https://arxiv.org/abs/2201.12886)
`TCNModel` | ✅ | ✅ | ✅ | ✅ | ✅ | | | [TCN paper](https://arxiv.org/abs/1803.01271), [DeepTCN paper](https://arxiv.org/abs/1906.04397), [blog post](https://medium.com/unit8-machine-learning-publication/temporal-convolutional-networks-and-forecasting-5ce1b6e97ce4)
`TransformerModel` | ✅ | ✅ | ✅ | ✅ | ✅ | | |
`TFTModel` (Temporal Fusion Transformer) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | [TFT paper](https://arxiv.org/pdf/1912.09363.pdf), [PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io/en/latest/models.html)
Naive Baselines | ✅ | | | | | | |


## Community & Contact
Anyone is welcome to join our ~~[Discord server](https://discord.gg/Um3jBTYFsA)~~ 
[Gitter room](https://gitter.im/u8darts/darts) to
ask questions, make proposals, discuss use-cases, and more. If you spot a bug or
or have suggestions, GitHub issues are also welcome.

If what you want to tell us is not suitable for Discord or Github,
feel free to send us an email at <a href="mailto:darts@unit8.co">darts@unit8.co</a> for
darts related matters or <a href="mailto:info@unit8.co">info@unit8.co</a> for any other
inquiries.

## Contribute
The development is ongoing, and we welcome suggestions, pull requests and issues on GitHub.
All contributors will be acknowledged on the
[change log page](https://github.com/unit8co/darts/blob/master/CHANGELOG.md).

Before working on a contribution (a new feature or a fix),
[check our contribution guidelines](https://github.com/unit8co/darts/blob/master/CONTRIBUTING.md).

## Citation
If you are using Darts in your scientific work, we would appreciate citations to the following JMLR paper.

[Darts: User-Friendly Modern Machine Learning for Time Series](https://www.jmlr.org/papers/v23/21-1177.html)

Bibtex entry:
```
@article{JMLR:v23:21-1177,
  author  = {Julien Herzen and Francesco LÃ¤ssig and Samuele Giuliano Piazzetta and Thomas Neuer and LÃ©o Tafti and Guillaume Raille and Tomas Van Pottelbergh and Marek Pasieka and Andrzej Skrodzki and Nicolas Huguenin and Maxime Dumonal and Jan KoÅ›cisz and Dennis Bader and FrÃ©dÃ©rick Gusset and Mounir Benheddi and Camila Williamson and Michal Kosinski and Matej Petrik and GaÃ«l Grosch},
  title   = {Darts: User-Friendly Modern Machine Learning for Time Series},
  journal = {Journal of Machine Learning Research},
  year    = {2022},
  volume  = {23},
  number  = {124},
  pages   = {1-6},
  url     = {http://jmlr.org/papers/v23/21-1177.html}
}
```
