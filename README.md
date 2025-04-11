# Time Series Made Easy in Python

![darts](https://github.com/unit8co/darts/raw/master/static/images/darts-logo-trim.png "darts")

---
[![PyPI version](https://badge.fury.io/py/u8darts.svg)](https://badge.fury.io/py/darts)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/u8darts-all.svg)](https://anaconda.org/conda-forge/u8darts-all)
![Supported versions](https://img.shields.io/badge/python-3.9+-blue.svg)
[![Docker Image Version (latest by date)](https://img.shields.io/docker/v/unit8/darts?label=docker&sort=date)](https://hub.docker.com/r/unit8/darts)
![GitHub Release Date](https://img.shields.io/github/release-date/unit8co/darts)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/unit8co/darts/release.yml?branch=master)
[![Downloads](https://pepy.tech/badge/darts)](https://pepy.tech/project/darts)
[![Downloads](https://pepy.tech/badge/u8darts)](https://pepy.tech/project/u8darts)
[![codecov](https://codecov.io/gh/unit8co/darts/branch/master/graph/badge.svg?token=7F1TLUFHQW)](https://codecov.io/gh/unit8co/darts)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Join the chat at https://gitter.im/u8darts/darts](https://badges.gitter.im/u8darts/darts.svg)](https://gitter.im/u8darts/darts?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

**Darts** is a Python library for user-friendly forecasting and anomaly detection
on time series. It contains a variety of models, from classics such as ARIMA to
deep neural networks. The forecasting models can all be used in the same way,
using `fit()` and `predict()` functions, similar to scikit-learn.
The library also makes it easy to backtest models,
combine the predictions of several models, and take external data into account.
Darts supports both univariate and multivariate time series and models.
The ML-based models can be trained on potentially large datasets containing multiple time
series, and some of the models offer a rich support for probabilistic forecasting.

Darts also offers extensive anomaly detection capabilities.
For instance, it is trivial to apply PyOD models on time series to obtain anomaly scores,
or to wrap any of Darts forecasting or filtering models to obtain fully
fledged anomaly detection models.


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

We recommend to first setup a clean Python environment for your project with Python 3.9+ using your favorite tool
([conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html "conda-env"),
[venv](https://docs.python.org/3/library/venv.html), [virtualenv](https://virtualenv.pypa.io/en/latest/) with
or without [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/)).

Once your environment is set up you can install darts using pip:

    pip install darts

For more details you can refer to our
[installation instructions](https://github.com/unit8co/darts/blob/master/INSTALL.md).

## Example Usage

### Forecasting

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

### Anomaly Detection

Load a multivariate series, trim it, keep 2 components, split train and validation sets:

```python
from darts.datasets import ETTh2Dataset

series = ETTh2Dataset().load()[:10000][["MUFL", "LULL"]]
train, val = series.split_before(0.6)
```

Build a k-means anomaly scorer, train it on the train set
and use it on the validation set to get anomaly scores:

```python
from darts.ad import KMeansScorer

scorer = KMeansScorer(k=2, window=5)
scorer.fit(train)
anom_score = scorer.score(val)
```

Build a binary anomaly detector and train it over train scores,
then use it over validation scores to get binary anomaly classification:

```python
from darts.ad import QuantileDetector

detector = QuantileDetector(high_quantile=0.99)
detector.fit(scorer.score(train))
binary_anom = detector.detect(anom_score)
```

Plot (shifting and scaling some of the series
to make everything appear on the same figure):

```python
import matplotlib.pyplot as plt

series.plot()
(anom_score / 2. - 100).plot(label="computed anomaly score", c="orangered", lw=3)
(binary_anom * 45 - 150).plot(label="detected binary anomaly", lw=4)
```

<div style="text-align:center;">
<img src="https://github.com/unit8co/darts/raw/master/static/images/example_ad.png" alt="darts anomaly detection example" />
</div>


## Features
* **Forecasting Models:** A large collection of forecasting models; from statistical models (such as
  ARIMA) to deep learning models (such as N-BEATS). See [table of models below](#forecasting-models).

* **Anomaly Detection** The `darts.ad` module contains a collection of anomaly scorers,
  detectors and aggregators, which can all be combined to detect anomalies in time series.
  It is easy to wrap any of Darts forecasting or filtering models to build
  a fully fledged anomaly detection model that compares predictions with actuals.
  The `PyODScorer` makes it trivial to use PyOD detectors on time series.

* **Multivariate Support:** `TimeSeries` can be multivariate - i.e., contain multiple time-varying
  dimensions/columns instead of a single scalar value. Many models can consume and produce multivariate series.

* **Multiple Series Training (Global Models):** All machine learning based models (incl. all neural networks)
  support being trained on multiple (potentially multivariate) series. This can scale to large datasets too.

* **Probabilistic Support:** `TimeSeries` objects can (optionally) represent stochastic
  time series; this can for instance be used to get confidence intervals, and many models support different
  flavours of probabilistic forecasting (such as estimating parametric distributions or quantiles).
  Some anomaly detection scorers are also able to exploit these predictive distributions.

* **Conformal Prediction Support:** Our conformal prediction models allow to generate probabilistic forecasts with
  calibrated quantile intervals for any pre-trained global forecasting model.

* **Past and Future Covariates Support:** Many models in Darts support past-observed and/or future-known
  covariate (external data) time series as inputs for producing forecasts.

* **Static Covariates Support:** In addition to time-dependent data, `TimeSeries` can also contain
  static data for each dimension, which can be exploited by some models.

* **Hierarchical Reconciliation:** Darts offers transformers to perform reconciliation.
  These can make the forecasts add up in a way that respects the underlying hierarchy.

* **Regression Models:** It is possible to plug-in any scikit-learn compatible model
  to obtain forecasts as functions of lagged values of the target series and covariates.

* **Training with Sample Weights:** All global models support being trained with sample weights. They can be
  applied to each observation, forecasted time step and target column.

* **Forecast Start Shifting:** All global models support training and prediction on a shifted output window.
  This is useful for example for Day-Ahead Market forecasts, or when the covariates (or target series) are reported
  with a delay.

* **Explainability:** Darts has the ability to *explain* some forecasting models using Shap values.

* **Data Processing:** Tools to easily apply (and revert) common transformations on
  time series data (scaling, filling missing values, differencing, boxcox, ...)

* **Metrics:** A variety of metrics for evaluating time series' goodness of fit;
  from R2-scores to Mean Absolute Scaled Error.

* **Backtesting:** Utilities for simulating historical forecasts, using moving time windows.

* **PyTorch Lightning Support:** All deep learning models are implemented using PyTorch Lightning,
  supporting among other things custom callbacks, GPUs/TPUs training and custom trainers.

* **Filtering Models:** Darts offers three filtering models: `KalmanFilter`, `GaussianProcessFilter`,
  and `MovingAverageFilter`, which allow to filter time series, and in some cases obtain probabilistic
  inferences of the underlying states/values.

* **Datasets** The `darts.datasets` submodule contains some popular time series datasets for rapid
  and reproducible experimentation.

* **Compatibility with Multiple Backends:** `TimeSeries` objects can be created from and exported to various backends such as pandas, polars, numpy, pyarrow, xarray, and more, facilitating seamless integration with different data processing libraries.

## Forecasting Models
Here's a breakdown of the forecasting models currently implemented in Darts. We are constantly working
on bringing more models and features.


| Model                                                                                                                                                                                                                                                                                     | Sources                                                                                                                                                                                                                           | Target Series Support:<br/><br/>Univariate/<br/>Multivariate | Covariates Support:<br/><br/>Past-observed/<br/>Future-known/<br/>Static | Probabilistic Forecasting:<br/><br/>Sampled/<br/>Distribution Parameters | Training & Forecasting on Multiple Series |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------|--------------------------------------------------------------------------|--------------------------------------------------------------------------|-------------------------------------------|
| **Baseline Models**<br/>([LocalForecastingModel](https://unit8co.github.io/darts/userguide/covariates.html#local-forecasting-models-lfms))                                                                                                                                                |                                                                                                                                                                                                                                   |                                                              |                                                                          |                                                                          |                                           |
| [NaiveMean](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.baselines.html#darts.models.forecasting.baselines.NaiveMean)                                                                                                                                           |                                                                                                                                                                                                                                   | ✅ ✅                                                          | 🔴 🔴 🔴                                                                 | 🔴 🔴                                                                    | 🔴                                        |
| [NaiveSeasonal](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.baselines.html#darts.models.forecasting.baselines.NaiveSeasonal)                                                                                                                                   |                                                                                                                                                                                                                                   | ✅ ✅                                                          | 🔴 🔴 🔴                                                                 | 🔴 🔴                                                                    | 🔴                                        |
| [NaiveDrift](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.baselines.html#darts.models.forecasting.baselines.NaiveDrift)                                                                                                                                         |                                                                                                                                                                                                                                   | ✅ ✅                                                          | 🔴 🔴 🔴                                                                 | 🔴 🔴                                                                    | 🔴                                        |
| [NaiveMovingAverage](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.baselines.html#darts.models.forecasting.baselines.NaiveMovingAverage)                                                                                                                         |                                                                                                                                                                                                                                   | ✅ ✅                                                          | 🔴 🔴 🔴                                                                 | 🔴 🔴                                                                    | 🔴                                        |
| **Statistical / Classic Models**<br/>([LocalForecastingModel](https://unit8co.github.io/darts/userguide/covariates.html#local-forecasting-models-lfms))                                                                                                                                   |                                                                                                                                                                                                                                   |                                                              |                                                                          |                                                                          |                                           |
| [ARIMA](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.arima.html#darts.models.forecasting.arima.ARIMA)                                                                                                                                                           |                                                                                                                                                                                                                                   | ✅ 🔴                                                         | 🔴 ✅ 🔴                                                                  | ✅ 🔴                                                                     | 🔴                                        |
| [VARIMA](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.varima.html#darts.models.forecasting.varima.VARIMA)                                                                                                                                                       |                                                                                                                                                                                                                                   | 🔴 ✅                                                         | 🔴 ✅ 🔴                                                                  | ✅ 🔴                                                                     | 🔴                                        |
| [AutoARIMA](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.sf_auto_arima.html#darts.models.forecasting.sf_auto_arima.AutoARIMA) (faster AutoARIMA)                                                                                                                | [Nixtla's statsforecast](https://github.com/Nixtla/statsforecast)                                                                                                                                                                 | ✅ 🔴                                                         | 🔴 ✅ 🔴                                                                  | ✅ 🔴                                                                     | 🔴                                        |
| [ExponentialSmoothing](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.exponential_smoothing.html#darts.models.forecasting.exponential_smoothing.ExponentialSmoothing)                                                                                             |                                                                                                                                                                                                                                   | ✅ 🔴                                                         | 🔴 🔴 🔴                                                                 | ✅ 🔴                                                                     | 🔴                                        |
| [AutoETS](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.sf_auto_ets.html#darts.models.forecasting.sf_auto_ets.AutoETS)                                                                                                                                           | [Nixtla's statsforecast](https://github.com/Nixtla/statsforecast)                                                                                                                                                                 | ✅ 🔴                                                         | 🔴 ✅ 🔴                                                                  | ✅ 🔴                                                                     | 🔴                                        |
| [AutoCES](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.sf_auto_ces.html#darts.models.forecasting.sf_auto_ces.AutoCES)                                                                                                                                           | [Nixtla's statsforecast](https://github.com/Nixtla/statsforecast)                                                                                                                                                                 | ✅ 🔴                                                         | 🔴 🔴 🔴                                                                 | 🔴 🔴                                                                    | 🔴                                        |
| [AutoMFLES](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.sf_auto_mfles.html#darts.models.forecasting.sf_auto_mfles.AutoMFLES)                                                                                                                                   | [Nixtla's statsforecast](https://github.com/Nixtla/statsforecast)                                                                                                                                                                 | ✅ 🔴                                                         | 🔴 ✅ 🔴                                                                  | 🔴 🔴                                                                    | 🔴                                        |
| [BATS](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tbats_model.html#darts.models.forecasting.tbats_model.BATS) and [TBATS](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tbats_model.html#darts.models.forecasting.tbats_model.TBATS) | [TBATS paper](https://robjhyndman.com/papers/ComplexSeasonality.pdf)                                                                                                                                                              | ✅ 🔴                                                         | 🔴 🔴 🔴                                                                 | ✅ 🔴                                                                     | 🔴                                        |
| [AutoTBATS](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.sf_auto_tbats.html#darts.models.forecasting.sf_auto_tbats.AutoTBATS)                                                                                                                                   | [Nixtla's statsforecast](https://github.com/Nixtla/statsforecast)                                                                                                                                                                 | ✅ 🔴                                                         | 🔴 🔴 🔴                                                                 | ✅ 🔴                                                                     | 🔴                                        |
| [Theta](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.theta.html#darts.models.forecasting.theta.Theta) and [FourTheta](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.theta.html#darts.models.forecasting.theta.FourTheta)               | [Theta](https://robjhyndman.com/papers/Theta.pdf) & [4 Theta](https://github.com/Mcompetitions/M4-methods/blob/master/4Theta%20method.R)                                                                                          | ✅ 🔴                                                         | 🔴 🔴 🔴                                                                 | 🔴 🔴                                                                    | 🔴                                        |
| [AutoTheta](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.sf_auto_theta.html#darts.models.forecasting.sf_auto_theta.AutoTheta)                                                                                                                                   | [Nixtla's statsforecast](https://github.com/Nixtla/statsforecast)                                                                                                                                                                 | ✅ 🔴                                                         | 🔴 🔴 🔴                                                                 | ✅ 🔴                                                                     | 🔴                                        |
| [Prophet](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.prophet_model.html#darts.models.forecasting.prophet_model.Prophet)                                                                                                                                       | [Prophet repo](https://github.com/facebook/prophet)                                                                                                                                                                               | ✅ 🔴                                                         | 🔴 ✅ 🔴                                                                  | ✅ 🔴                                                                     | 🔴                                        |
| [FFT](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.fft.html#darts.models.forecasting.fft.FFT) (Fast Fourier Transform)                                                                                                                                          |                                                                                                                                                                                                                                   | ✅ 🔴                                                         | 🔴 🔴 🔴                                                                 | 🔴 🔴                                                                    | 🔴                                        |
| [KalmanForecaster](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.kalman_forecaster.html#darts.models.forecasting.kalman_forecaster.KalmanForecaster) using the Kalman filter and N4SID for system identification                                                 | [N4SID paper](https://people.duke.edu/~hpgavin/SystemID/References/VanOverschee-Automatica-1994.pdf)                                                                                                                              | ✅ ✅                                                          | 🔴 ✅ 🔴                                                                  | ✅ 🔴                                                                     | 🔴                                        |
| [Croston](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.croston.html#darts.models.forecasting.croston.Croston) method                                                                                                                                            |                                                                                                                                                                                                                                   | ✅ 🔴                                                         | 🔴 🔴 🔴                                                                 | 🔴 🔴                                                                    | 🔴                                        |
| **Global Baseline Models**<br/>([GlobalForecastingModel](https://unit8co.github.io/darts/userguide/covariates.html#global-forecasting-models-gfms))                                                                                                                                       |                                                                                                                                                                                                                                   |                                                              |                                                                          |                                                                          |                                           |
| [GlobalNaiveAggregate](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.global_baseline_models.html#darts.models.forecasting.global_baseline_models.GlobalNaiveAggregate)                                                                                           |                                                                                                                                                                                                                                   | ✅ ✅                                                          | 🔴 🔴 🔴                                                                 | 🔴 🔴                                                                    | ✅                                         |
| [GlobalNaiveDrift](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.global_baseline_models.html#darts.models.forecasting.global_baseline_models.GlobalNaiveDrift)                                                                                                   |                                                                                                                                                                                                                                   | ✅ ✅                                                          | 🔴 🔴 🔴                                                                 | 🔴 🔴                                                                    | ✅                                         |
| [GlobalNaiveSeasonal](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.global_baseline_models.html#darts.models.forecasting.global_baseline_models.GlobalNaiveSeasonal)                                                                                             |                                                                                                                                                                                                                                   | ✅ ✅                                                          | 🔴 🔴 🔴                                                                 | 🔴 🔴                                                                    | ✅                                         |
| **Regression Models**<br/>([GlobalForecastingModel](https://unit8co.github.io/darts/userguide/covariates.html#global-forecasting-models-gfms))                                                                                                                                            |                                                                                                                                                                                                                                   |                                                              |                                                                          |                                                                          |                                           |
| [RegressionModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.regression_model.html#darts.models.forecasting.regression_model.RegressionModel): generic wrapper around any sklearn regression model                                                            |                                                                                                                                                                                                                                   | ✅ ✅                                                          | ✅ ✅ ✅                                                                    | 🔴 🔴                                                                    | ✅                                         |
| [LinearRegressionModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.linear_regression_model.html#darts.models.forecasting.linear_regression_model.LinearRegressionModel)                                                                                       |                                                                                                                                                                                                                                   | ✅ ✅                                                          | ✅ ✅ ✅                                                                    | ✅ ✅                                                                      | ✅                                         |
| [RandomForest](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.random_forest.html#darts.models.forecasting.random_forest.RandomForest)                                                                                                                             |                                                                                                                                                                                                                                   | ✅ ✅                                                          | ✅ ✅ ✅                                                                    | 🔴 🔴                                                                    | ✅                                         |
| [LightGBMModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.lgbm.html#darts.models.forecasting.lgbm.LightGBMModel)                                                                                                                                             |                                                                                                                                                                                                                                   | ✅ ✅                                                          | ✅ ✅ ✅                                                                    | ✅ ✅                                                                      | ✅                                         |
| [XGBModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.xgboost.html#darts.models.forecasting.xgboost.XGBModel)                                                                                                                                                 |                                                                                                                                                                                                                                   | ✅ ✅                                                          | ✅ ✅ ✅                                                                    | ✅ ✅                                                                      | ✅                                         |
| [CatBoostModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.catboost_model.html#darts.models.forecasting.catboost_model.CatBoostModel)                                                                                                                         |                                                                                                                                                                                                                                   | ✅ ✅                                                          | ✅ ✅ ✅                                                                    | ✅ ✅                                                                      | ✅                                         |
| **PyTorch (Lightning)-based Models**<br/>([GlobalForecastingModel](https://unit8co.github.io/darts/userguide/covariates.html#global-forecasting-models-gfms))                                                                                                                             |                                                                                                                                                                                                                                   |                                                              |                                                                          |                                                                          |                                           |
| [RNNModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.rnn_model.html#darts.models.forecasting.rnn_model.RNNModel) (incl. LSTM and GRU); equivalent to DeepAR in its probabilistic version                                                                     | [DeepAR paper](https://arxiv.org/abs/1704.04110)                                                                                                                                                                                  | ✅ ✅                                                          | 🔴 ✅ 🔴                                                                  | ✅ ✅                                                                      | ✅                                         |
| [BlockRNNModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.block_rnn_model.html#darts.models.forecasting.block_rnn_model.BlockRNNModel) (incl. LSTM and GRU)                                                                                                  |                                                                                                                                                                                                                                   | ✅ ✅                                                          | ✅ 🔴 🔴                                                                  | ✅ ✅                                                                      | ✅                                         |
| [NBEATSModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.nbeats.html#darts.models.forecasting.nbeats.NBEATSModel)                                                                                                                                             | [N-BEATS paper](https://arxiv.org/abs/1905.10437)                                                                                                                                                                                 | ✅ ✅                                                          | ✅ 🔴 🔴                                                                  | ✅ ✅                                                                      | ✅                                         |
| [NHiTSModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.nhits.html#darts.models.forecasting.nhits.NHiTSModel)                                                                                                                                                 | [N-HiTS paper](https://arxiv.org/abs/2201.12886)                                                                                                                                                                                  | ✅ ✅                                                          | ✅ 🔴 🔴                                                                  | ✅ ✅                                                                      | ✅                                         |
| [TCNModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tcn_model.html#darts.models.forecasting.tcn_model.TCNModel)                                                                                                                                             | [TCN paper](https://arxiv.org/abs/1803.01271), [DeepTCN paper](https://arxiv.org/abs/1906.04397), [blog post](https://medium.com/unit8-machine-learning-publication/temporal-convolutional-networks-and-forecasting-5ce1b6e97ce4) | ✅ ✅                                                          | ✅ 🔴 🔴                                                                  | ✅ ✅                                                                      | ✅                                         |
| [TransformerModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.transformer_model.html#darts.models.forecasting.transformer_model.TransformerModel)                                                                                                             |                                                                                                                                                                                                                                   | ✅ ✅                                                          | ✅ 🔴 🔴                                                                  | ✅ ✅                                                                      | ✅                                         |
| [TFTModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tft_model.html#darts.models.forecasting.tft_model.TFTModel) (Temporal Fusion Transformer)                                                                                                               | [TFT paper](https://arxiv.org/pdf/1912.09363.pdf), [PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io/en/latest/models.html)                                                                                        | ✅ ✅                                                          | ✅ ✅ ✅                                                                    | ✅ ✅                                                                      | ✅                                         |
| [DLinearModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.dlinear.html#darts.models.forecasting.dlinear.DLinearModel)                                                                                                                                         | [DLinear paper](https://arxiv.org/pdf/2205.13504.pdf)                                                                                                                                                                             | ✅ ✅                                                          | ✅ ✅ ✅                                                                    | ✅ ✅                                                                      | ✅                                         |
| [NLinearModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.nlinear.html#darts.models.forecasting.nlinear.NLinearModel)                                                                                                                                         | [NLinear paper](https://arxiv.org/pdf/2205.13504.pdf)                                                                                                                                                                             | ✅ ✅                                                          | ✅ ✅ ✅                                                                    | ✅ ✅                                                                      | ✅                                         |
| [TiDEModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tide_model.html#darts.models.forecasting.tide_model.TiDEModel)                                                                                                                                         | [TiDE paper](https://arxiv.org/pdf/2304.08424.pdf)                                                                                                                                                                                | ✅ ✅                                                          | ✅ ✅ ✅                                                                    | ✅ ✅                                                                      | ✅                                         |
| [TSMixerModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tsmixer_model.html#darts.models.forecasting.tsmixer_model.TSMixerModel)                                                                                                                             | [TSMixer paper](https://arxiv.org/pdf/2303.06053.pdf), [PyTorch Implementation](https://github.com/ditschuk/pytorch-tsmixer)                                                                                                      | ✅ ✅                                                          | ✅ ✅ ✅                                                                    | ✅ ✅                                                                      | ✅                                         |
| **Ensemble Models**<br/>([GlobalForecastingModel](https://unit8co.github.io/darts/userguide/covariates.html#global-forecasting-models-gfms)): Model support is dependent on ensembled forecasting models and the ensemble model itself                                                    |                                                                                                                                                                                                                                   |                                                              |                                                                          |                                                                          |                                           |
| [NaiveEnsembleModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.baselines.html#darts.models.forecasting.baselines.NaiveEnsembleModel)                                                                                                                         |                                                                                                                                                                                                                                   | ✅ ✅                                                          | ✅ ✅ ✅                                                                    | ✅ ✅                                                                      | ✅                                         |
| [RegressionEnsembleModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.regression_ensemble_model.html#darts.models.forecasting.regression_ensemble_model.RegressionEnsembleModel)                                                                               |                                                                                                                                                                                                                                   | ✅ ✅                                                          | ✅ ✅ ✅                                                                    | ✅ ✅                                                                      | ✅                                         |
| **Conformal Models**<br/>([GlobalForecastingModel](https://unit8co.github.io/darts/userguide/covariates.html#global-forecasting-models-gfms)): Model support is dependent on the forecasting model used                                                                                   |                                                                                                                                                                                                                                   |                                                              |                                                                          |                                                                          |                                           |
| [ConformalNaiveModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.conformal_models.html#darts.models.forecasting.conformal_models.ConformalNaiveModel)                                                                                                         | [Conformalized Prediction](https://arxiv.org/pdf/1905.03222)                                                                                                                                                                      | ✅ ✅                                                          | ✅ ✅ ✅                                                                    | ✅ ✅                                                                      | ✅                                         |
| [ConformalQRModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.conformal_models.html#darts.models.forecasting.conformal_models.ConformalQRModel)                                                                                                               | [Conformalized Quantile Regression](https://arxiv.org/pdf/1905.03222)                                                                                                                                                             | ✅ ✅                                                          | ✅ ✅ ✅                                                                    | ✅ ✅                                                                      | ✅                                         |

## Community & Contact
Anyone is welcome to join our [Gitter room](https://gitter.im/u8darts/darts) to ask questions, make proposals,
discuss use-cases, and more. If you spot a bug or have suggestions, GitHub issues are also welcome.

If what you want to tell us is not suitable for Gitter or Github,
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
