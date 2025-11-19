:notoc:

Time Series Made Easy in Python
===============================

.. image:: _static/darts-logo-light-blue.svg
   :alt: darts
   :align: center
   :width: 100%
   :class: only-light

.. image:: _static/darts-logo-dark-yellow-new-copy.svg
   :alt: darts
   :align: center
   :width: 100%
   :class: only-dark

.. raw:: html

   <p align="center" style="margin-top: 20px;">
       <a href="https://badge.fury.io/py/darts"><img src="https://badge.fury.io/py/u8darts.svg" alt="PyPI version"></a>
       <a href="https://anaconda.org/conda-forge/u8darts-all"><img src="https://img.shields.io/conda/vn/conda-forge/u8darts-all.svg" alt="Conda Version"></a>
       <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Supported versions">
       <a href="https://hub.docker.com/r/unit8/darts"><img src="https://img.shields.io/docker/v/unit8/darts?label=docker&sort=date" alt="Docker Image Version"></a>
       <img src="https://img.shields.io/github/release-date/unit8co/darts" alt="GitHub Release Date">
       <img src="https://img.shields.io/github/actions/workflow/status/unit8co/darts/release.yml?branch=master" alt="GitHub Workflow Status">
       <a href="https://pepy.tech/project/darts"><img src="https://pepy.tech/badge/darts" alt="Downloads"></a>
       <a href="https://pepy.tech/project/u8darts"><img src="https://pepy.tech/badge/u8darts" alt="Downloads"></a>
       <a href="https://codecov.io/gh/unit8co/darts"><img src="https://codecov.io/gh/unit8co/darts/branch/master/graph/badge.svg?token=7F1TLUFHQW" alt="codecov"></a>
       <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
       <a href="https://gitter.im/u8darts/darts?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge"><img src="https://badges.gitter.im/u8darts/darts.svg" alt="Join the chat at https://gitter.im/u8darts/darts"></a>
   </p>

|

**Darts** is a Python library for user-friendly **forecasting and anomaly detection** on time series.
It contains a variety of models, from classics such as ARIMA to deep neural networks.
All models can be used in the same way, using ``fit()`` and ``predict()`` functions, similar to scikit-learn.
The library also makes it easy to backtest models, combine the predictions of several models, and take external data into account.
Darts supports both **univariate and multivariate** time series and models, and offers extensive support for **probabilistic forecasting**.

.. grid:: 1 2 3 3
    :gutter: 4
    :padding: 2 2 0 0
    :class-container: sd-text-center

    .. grid-item-card:: Quickstart
        :img-top: static/icon-quickstart.svg
        :class-card: intro-card
        :shadow: md

        Introduction to Darts' main concepts and workflow.

        +++

        .. button-ref:: quickstart/00-quickstart
            :ref-type: doc
            :click-parent:
            :color: secondary
            :expand:

            To the quickstart guide

    .. grid-item-card:: Models
        :img-top: static/icon-models.svg
        :class-card: intro-card
        :shadow: md

        Available models and supported features.

        +++

        .. button-ref:: forecasting-models
            :ref-type: ref
            :click-parent:
            :color: secondary
            :expand:

            To the models table

    .. grid-item-card:: API Reference
        :img-top: static/icon-api.svg
        :class-card: intro-card
        :shadow: md

        Detailed API documentation with methods and parameters.

        +++

        .. button-ref:: generated_api/darts
            :ref-type: doc
            :click-parent:
            :color: secondary
            :expand:

            To the API reference

    .. grid-item-card:: Examples
        :img-top: static/icon-examples.svg
        :class-card: intro-card
        :shadow: md

        Jupyter notebooks with basic and advanced tutorials.

        +++

        .. button-ref:: examples
            :ref-type: doc
            :click-parent:
            :color: secondary
            :expand:

            To the examples

    .. grid-item-card:: User Guide
        :img-top: static/icon-userguide.svg
        :class-card: intro-card
        :shadow: md

        In-depth information on key concepts and features.

        +++

        .. button-ref:: userguide
            :ref-type: doc
            :click-parent:
            :color: secondary
            :expand:

            To the user guide

    .. grid-item-card:: How to Contribute
        :img-top: static/icon-contribute.svg
        :class-card: intro-card
        :shadow: md

        Guidelines for contributing to the Darts library.

        +++

        .. button-link:: https://github.com/unit8co/darts/blob/master/CONTRIBUTING.md
            :click-parent:
            :color: secondary
            :expand:

            To the contributing guide

----

High Level Introductions
-------------------------

* `Introductory Blog Post <https://medium.com/unit8-machine-learning-publication/darts-time-series-made-easy-in-python-5ac2947a8878>`_
* `Introduction video (25 minutes) <https://youtu.be/g6OXDnXEtFA>`_

Articles on Selected Topics
---------------------------

* `Training Models on Multiple Time Series <https://medium.com/unit8-machine-learning-publication/training-forecasting-models-on-multiple-time-series-with-darts-dc4be70b1844>`_
* `Using Past and Future Covariates <https://medium.com/unit8-machine-learning-publication/time-series-forecasting-using-past-and-future-external-data-with-darts-1f0539585993>`_
* `Temporal Convolutional Networks and Forecasting <https://medium.com/unit8-machine-learning-publication/temporal-convolutional-networks-and-forecasting-5ce1b6e97ce4>`_
* `Probabilistic Forecasting <https://medium.com/unit8-machine-learning-publication/probabilistic-forecasting-in-darts-e88fbe83344e>`_
* `Transfer Learning for Time Series Forecasting <https://medium.com/unit8-machine-learning-publication/transfer-learning-for-time-series-forecasting-87f39e375278>`_
* `Hierarchical Forecast Reconciliation <https://medium.com/unit8-machine-learning-publication/hierarchical-forecast-reconciliation-with-darts-8b4b058bb543>`_

Quick Install
-------------

We recommend to first setup a clean Python environment for your project with Python 3.10+ using your favorite tool
(`conda <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_,
`venv <https://docs.python.org/3/library/venv.html>`_, `virtualenv <https://virtualenv.pypa.io/en/latest/>`_ with
or without `virtualenvwrapper <https://virtualenvwrapper.readthedocs.io/en/latest/>`_).

Once your environment is set up you can install darts using pip:

.. code-block:: bash

    pip install darts

For more details you can refer to our `installation instructions <https://github.com/unit8co/darts/blob/master/INSTALL.md>`_.

Example Usage
-------------

Forecasting
~~~~~~~~~~~

Create a ``TimeSeries`` object from a Pandas DataFrame, and split it in train/validation series:

.. code-block:: python

    import pandas as pd
    from darts import TimeSeries

    # Read a pandas DataFrame
    df = pd.read_csv("AirPassengers.csv", delimiter=",")

    # Create a TimeSeries, specifying the time and value columns
    series = TimeSeries.from_dataframe(df, "Month", "#Passengers")

    # Set aside the last 36 months as a validation series
    train, val = series[:-36], series[-36:]

Fit an exponential smoothing model, and make a (probabilistic) prediction over the validation series' duration:

.. code-block:: python

    from darts.models import ExponentialSmoothing

    model = ExponentialSmoothing()
    model.fit(train)
    prediction = model.predict(len(val), num_samples=1000)

Plot the median, 5th and 95th percentiles:

.. code-block:: python

    import matplotlib.pyplot as plt

    series.plot()
    prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    plt.legend()

.. image:: https://github.com/unit8co/darts/raw/master/static/images/example.png
   :alt: darts forecast example
   :align: center

Anomaly Detection
~~~~~~~~~~~~~~~~~

Load a multivariate series, trim it, keep 2 components, split train and validation sets:

.. code-block:: python

    from darts.datasets import ETTh2Dataset

    series = ETTh2Dataset().load()[:10000][["MUFL", "LULL"]]
    train, val = series.split_before(0.6)

Build a k-means anomaly scorer, train it on the train set
and use it on the validation set to get anomaly scores:

.. code-block:: python

    from darts.ad import KMeansScorer

    scorer = KMeansScorer(k=2, window=5)
    scorer.fit(train)
    anom_score = scorer.score(val)

Build a binary anomaly detector and train it over train scores,
then use it over validation scores to get binary anomaly classification:

.. code-block:: python

    from darts.ad import QuantileDetector

    detector = QuantileDetector(high_quantile=0.99)
    detector.fit(scorer.score(train))
    binary_anom = detector.detect(anom_score)

Plot (shifting and scaling some of the series to make everything appear on the same figure):

.. code-block:: python

    import matplotlib.pyplot as plt

    series.plot()
    (anom_score / 2. - 100).plot(label="computed anomaly score", c="orangered", lw=3)
    (binary_anom * 45 - 150).plot(label="detected binary anomaly", lw=4)

.. image:: https://github.com/unit8co/darts/raw/master/static/images/example_ad.png
   :alt: darts anomaly detection example
   :align: center


Features
--------

* **Forecasting Models:** A large collection of forecasting models for regression as well as classification tasks; from statistical models (such as ARIMA) to deep learning models (such as N-BEATS). See the forecasting models table below.

* **Anomaly Detection:** The ``darts.ad`` module contains a collection of anomaly scorers, detectors and aggregators, which can all be combined to detect anomalies in time series. It is easy to wrap any of Darts forecasting or filtering models to build a fully fledged anomaly detection model that compares predictions with actuals. The ``PyODScorer`` makes it trivial to use PyOD detectors on time series.

* **Multivariate Support:** ``TimeSeries`` can be multivariate - i.e., contain multiple time-varying dimensions/columns instead of a single scalar value. Many models can consume and produce multivariate series.

* **Multiple Series Training (Global Models):** All machine learning based models (incl. all neural networks) support being trained on multiple (potentially multivariate) series. This can scale to large datasets too.

* **Probabilistic Support:** ``TimeSeries`` objects can (optionally) represent stochastic time series; this can for instance be used to get confidence intervals, and many models support different flavours of probabilistic forecasting (such as estimating parametric distributions or quantiles). Some anomaly detection scorers are also able to exploit these predictive distributions.

* **Conformal Prediction Support:** Our conformal prediction models allow to generate probabilistic forecasts with calibrated quantile intervals for any pre-trained global forecasting model.

* **Past and Future Covariates Support:** Many models in Darts support past-observed and/or future-known covariate (external data) time series as inputs for producing forecasts.

* **Static Covariates Support:** In addition to time-dependent data, ``TimeSeries`` can also contain static data for each dimension, which can be exploited by some models.

* **Hierarchical Reconciliation:** Darts offers transformers to perform reconciliation. These can make the forecasts add up in a way that respects the underlying hierarchy.

* **Regression Models:** It is possible to plug-in any scikit-learn compatible model to obtain forecasts as functions of lagged values of the target series and covariates.

* **Training with Sample Weights:** All global models support being trained with sample weights. They can be applied to each observation, forecasted time step and target column.

* **Forecast Start Shifting:** All global models support training and prediction on a shifted output window. This is useful for example for Day-Ahead Market forecasts, or when the covariates (or target series) are reported with a delay.

* **Explainability:** Darts has the ability to *explain* some forecasting models using Shap values.

* **Data Processing:** Tools to easily apply (and revert) common transformations on time series data (scaling, filling missing values, differencing, boxcox, ...)

* **Metrics:** A variety of metrics for evaluating time series' goodness of fit; from R2-scores to Mean Absolute Scaled Error.

* **Backtesting:** Utilities for simulating historical forecasts, using moving time windows.

* **PyTorch Lightning Support:** All deep learning models are implemented using PyTorch Lightning, supporting among other things custom callbacks, GPUs/TPUs training and custom trainers.

* **Filtering Models:** Darts offers three filtering models: ``KalmanFilter``, ``GaussianProcessFilter``, and ``MovingAverageFilter``, which allow to filter time series, and in some cases obtain probabilistic inferences of the underlying states/values.

* **Datasets:** The ``darts.datasets`` submodule contains some popular time series datasets for rapid and reproducible experimentation.

* **Compatibility with Multiple Backends:** ``TimeSeries`` objects can be created from and exported to various backends such as pandas, polars, numpy, pyarrow, xarray, and more, facilitating seamless integration with different data processing libraries.

.. _forecasting-models:

Forecasting Models
------------------

Here's a breakdown of the forecasting models currently implemented in Darts. Our suite includes both regression and classification models, each tailored for specific forecasting tasks. We are committed to expanding our offerings with new models and features to enhance your forecasting capabilities.

Regression Models
~~~~~~~~~~~~~~~~~

Our regression models are designed to predict continuous numerical values, making them ideal for forecasting future trends and patterns in time series data. Utilize these models to gain insights into potential future outcomes based on historical data.

.. list-table::
   :header-rows: 1
   :widths: 30 10 10 10 10 20

   * - Model
     - Target Series Support: Univariate / Multivariate
     - Covariates Support: Past-observed / Future-known / Static
     - Probabilistic Forecasting: Sampled / Distribution Parameters
     - Training & Forecasting on Multiple Series
     - Sources
   * - **Baseline Models** (`LocalForecastingModel <https://unit8co.github.io/darts/userguide/covariates.html#local-forecasting-models-lfms>`_)
     -
     -
     -
     -
     -
   * - `NaiveMean <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.baselines.html#darts.models.forecasting.baselines.NaiveMean>`_
     - ✅ ✅
     - 🔴 🔴 🔴
     - 🔴 🔴
     - 🔴
     -
   * - `NaiveSeasonal <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.baselines.html#darts.models.forecasting.baselines.NaiveSeasonal>`_
     - ✅ ✅
     - 🔴 🔴 🔴
     - 🔴 🔴
     - 🔴
     -
   * - `NaiveDrift <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.baselines.html#darts.models.forecasting.baselines.NaiveDrift>`_
     - ✅ ✅
     - 🔴 🔴 🔴
     - 🔴 🔴
     - 🔴
     -
   * - `NaiveMovingAverage <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.baselines.html#darts.models.forecasting.baselines.NaiveMovingAverage>`_
     - ✅ ✅
     - 🔴 🔴 🔴
     - 🔴 🔴
     - 🔴
     -
   * - **Statistical / Classic Models** (`LocalForecastingModel <https://unit8co.github.io/darts/userguide/covariates.html#local-forecasting-models-lfms>`_)
     -
     -
     -
     -
     -
   * - `ARIMA <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.arima.html#darts.models.forecasting.arima.ARIMA>`_
     - ✅ 🔴
     - 🔴 ✅ 🔴
     - ✅ 🔴
     - 🔴
     -
   * - `VARIMA <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.varima.html#darts.models.forecasting.varima.VARIMA>`_
     - 🔴 ✅
     - 🔴 ✅ 🔴
     - ✅ 🔴
     - 🔴
     -
   * - `ExponentialSmoothing <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.exponential_smoothing.html#darts.models.forecasting.exponential_smoothing.ExponentialSmoothing>`_
     - ✅ 🔴
     - 🔴 🔴 🔴
     - ✅ 🔴
     - 🔴
     -
   * - `Theta <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.theta.html#darts.models.forecasting.theta.Theta>`_ and `FourTheta <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.theta.html#darts.models.forecasting.theta.FourTheta>`_
     - ✅ 🔴
     - 🔴 🔴 🔴
     - 🔴 🔴
     - 🔴
     - `Theta paper <https://robjhyndman.com/papers/Theta.pdf>`_ & `4 Theta source <https://github.com/Mcompetitions/M4-methods/blob/master/4Theta%20method.R>`_
   * - `Prophet <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.prophet_model.html#darts.models.forecasting.prophet_model.Prophet>`_
     - ✅ 🔴
     - 🔴 ✅ 🔴
     - ✅ 🔴
     - 🔴
     - `Prophet repo <https://github.com/facebook/prophet>`_
   * - `FFT <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.fft.html#darts.models.forecasting.fft.FFT>`_ (Fast Fourier Transform)
     - ✅ 🔴
     - 🔴 🔴 🔴
     - 🔴 🔴
     - 🔴
     -
   * - `KalmanForecaster <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.kalman_forecaster.html#darts.models.forecasting.kalman_forecaster.KalmanForecaster>`_ using the Kalman filter and N4SID for system identification
     - ✅ ✅
     - 🔴 ✅ 🔴
     - ✅ 🔴
     - 🔴
     - `N4SID paper <https://people.duke.edu/~hpgavin/SystemID/References/VanOverschee-Automatica-1994.pdf>`_
   * - `TBATS <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.sf_tbats.html#darts.models.forecasting.sf_tbats.TBATS>`_
     - ✅ 🔴
     - 🔴 ✅ 🔴
     - ✅ ✅
     - 🔴
     - `TBATS paper <https://robjhyndman.com/papers/ComplexSeasonality.pdf>`_
   * - `Croston <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.sf_croston.html#darts.models.forecasting.sf_croston.Croston>`_ method
     - ✅ 🔴
     - 🔴 ✅ 🔴
     - ✅ ✅
     - 🔴
     -
   * - `StatsForecastModel <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.sf_model.html#darts.models.forecasting.sf_model.StatsForecastModel>`_ wrapper around any `StatsForecast <https://nixtlaverse.nixtla.io/statsforecast/index.html#models>`_ model
     - ✅ 🔴
     - 🔴 ✅ 🔴
     - ✅ ✅
     - 🔴
     - `Nixtla's statsforecast <https://github.com/Nixtla/statsforecast>`_
   * - `AutoARIMA <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.sf_auto_arima.html#darts.models.forecasting.sf_auto_arima.AutoARIMA>`_
     - ✅ 🔴
     - 🔴 ✅ 🔴
     - ✅ ✅
     - 🔴
     - `Nixtla's statsforecast <https://github.com/Nixtla/statsforecast>`_
   * - `AutoETS <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.sf_auto_ets.html#darts.models.forecasting.sf_auto_ets.AutoETS>`_
     - ✅ 🔴
     - 🔴 ✅ 🔴
     - ✅ ✅
     - 🔴
     - `Nixtla's statsforecast <https://github.com/Nixtla/statsforecast>`_
   * - `AutoCES <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.sf_auto_ces.html#darts.models.forecasting.sf_auto_ces.AutoCES>`_
     - ✅ 🔴
     - 🔴 ✅ 🔴
     - ✅ ✅
     - 🔴
     - `Nixtla's statsforecast <https://github.com/Nixtla/statsforecast>`_
   * - `AutoMFLES <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.sf_auto_mfles.html#darts.models.forecasting.sf_auto_mfles.AutoMFLES>`_
     - ✅ 🔴
     - 🔴 ✅ 🔴
     - ✅ ✅
     - 🔴
     - `Nixtla's statsforecast <https://github.com/Nixtla/statsforecast>`_
   * - `AutoTBATS <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.sf_auto_tbats.html#darts.models.forecasting.sf_auto_tbats.AutoTBATS>`_
     - ✅ 🔴
     - 🔴 ✅ 🔴
     - ✅ ✅
     - 🔴
     - `Nixtla's statsforecast <https://github.com/Nixtla/statsforecast>`_
   * - `AutoTheta <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.sf_auto_theta.html#darts.models.forecasting.sf_auto_theta.AutoTheta>`_
     - ✅ 🔴
     - 🔴 ✅ 🔴
     - ✅ ✅
     - 🔴
     - `Nixtla's statsforecast <https://github.com/Nixtla/statsforecast>`_
   * - **Global Baseline Models** (`GlobalForecastingModel <https://unit8co.github.io/darts/userguide/covariates.html#global-forecasting-models-gfms>`_)
     -
     -
     -
     -
     -
   * - `GlobalNaiveAggregate <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.global_baseline_models.html#darts.models.forecasting.global_baseline_models.GlobalNaiveAggregate>`_
     - ✅ ✅
     - 🔴 🔴 🔴
     - 🔴 🔴
     - ✅
     -
   * - `GlobalNaiveDrift <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.global_baseline_models.html#darts.models.forecasting.global_baseline_models.GlobalNaiveDrift>`_
     - ✅ ✅
     - 🔴 🔴 🔴
     - 🔴 🔴
     - ✅
     -
   * - `GlobalNaiveSeasonal <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.global_baseline_models.html#darts.models.forecasting.global_baseline_models.GlobalNaiveSeasonal>`_
     - ✅ ✅
     - 🔴 🔴 🔴
     - 🔴 🔴
     - ✅
     -
   * - **Regression Models** (`GlobalForecastingModel <https://unit8co.github.io/darts/userguide/covariates.html#global-forecasting-models-gfms>`_)
     -
     -
     -
     -
     -
   * - `SKLearnModel <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.sklearn_model.html#darts.models.forecasting.sklearn_model.SKLearnModel>`_: wrapper around any scikit-learn-like regression model
     - ✅ ✅
     - ✅ ✅ ✅
     - 🔴 🔴
     - ✅
     -
   * - `LinearRegressionModel <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.linear_regression_model.html#darts.models.forecasting.linear_regression_model.LinearRegressionModel>`_
     - ✅ ✅
     - ✅ ✅ ✅
     - ✅ ✅
     - ✅
     -
   * - `RandomForestModel <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.random_forest.html#darts.models.forecasting.random_forest.RandomForestModel>`_
     - ✅ ✅
     - ✅ ✅ ✅
     - 🔴 🔴
     - ✅
     -
   * - `CatBoostModel <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.catboost_model.html#darts.models.forecasting.catboost_model.CatBoostModel>`_
     - ✅ ✅
     - ✅ ✅ ✅
     - ✅ ✅
     - ✅
     -
   * - `LightGBMModel <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.lgbm.html#darts.models.forecasting.lgbm.LightGBMModel>`_
     - ✅ ✅
     - ✅ ✅ ✅
     - ✅ ✅
     - ✅
     -
   * - `XGBModel <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.xgboost.html#darts.models.forecasting.xgboost.XGBModel>`_
     - ✅ ✅
     - ✅ ✅ ✅
     - ✅ ✅
     - ✅
     -
   * - **PyTorch (Lightning)-based Models** (`GlobalForecastingModel <https://unit8co.github.io/darts/userguide/covariates.html#global-forecasting-models-gfms>`_)
     -
     -
     -
     -
     -
   * - `RNNModel <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.rnn_model.html#darts.models.forecasting.rnn_model.RNNModel>`_ (incl. LSTM and GRU); equivalent to DeepAR in its probabilistic version
     - ✅ ✅
     - 🔴 ✅ 🔴
     - ✅ ✅
     - ✅
     - `DeepAR paper <https://arxiv.org/abs/1704.04110>`_
   * - `BlockRNNModel <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.block_rnn_model.html#darts.models.forecasting.block_rnn_model.BlockRNNModel>`_ (incl. LSTM and GRU)
     - ✅ ✅
     - ✅ ✅ ✅
     - ✅ ✅
     - ✅
     -
   * - `NBEATSModel <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.nbeats.html#darts.models.forecasting.nbeats.NBEATSModel>`_
     - ✅ ✅
     - ✅ 🔴 🔴
     - ✅ ✅
     - ✅
     - `N-BEATS paper <https://arxiv.org/abs/1905.10437>`_
   * - `NHiTSModel <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.nhits.html#darts.models.forecasting.nhits.NHiTSModel>`_
     - ✅ ✅
     - ✅ 🔴 🔴
     - ✅ ✅
     - ✅
     - `N-HiTS paper <https://arxiv.org/abs/2201.12886>`_
   * - `TCNModel <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tcn_model.html#darts.models.forecasting.tcn_model.TCNModel>`_
     - ✅ ✅
     - ✅ 🔴 🔴
     - ✅ ✅
     - ✅
     - `TCN paper <https://arxiv.org/abs/1803.01271>`_, `DeepTCN paper <https://arxiv.org/abs/1906.04397>`_, `blog post <https://medium.com/unit8-machine-learning-publication/temporal-convolutional-networks-and-forecasting-5ce1b6e97ce4>`_
   * - `TransformerModel <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.transformer_model.html#darts.models.forecasting.transformer_model.TransformerModel>`_
     - ✅ ✅
     - ✅ 🔴 🔴
     - ✅ ✅
     - ✅
     -
   * - `TFTModel <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tft_model.html#darts.models.forecasting.tft_model.TFTModel>`_ (Temporal Fusion Transformer)
     - ✅ ✅
     - ✅ ✅ ✅
     - ✅ ✅
     - ✅
     - `TFT paper <https://arxiv.org/pdf/1912.09363.pdf>`_, `PyTorch Forecasting <https://pytorch-forecasting.readthedocs.io/en/latest/models.html>`_
   * - `DLinearModel <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.dlinear.html#darts.models.forecasting.dlinear.DLinearModel>`_
     - ✅ ✅
     - ✅ ✅ ✅
     - ✅ ✅
     - ✅
     - `DLinear paper <https://arxiv.org/pdf/2205.13504.pdf>`_
   * - `NLinearModel <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.nlinear.html#darts.models.forecasting.nlinear.NLinearModel>`_
     - ✅ ✅
     - ✅ ✅ ✅
     - ✅ ✅
     - ✅
     - `NLinear paper <https://arxiv.org/pdf/2205.13504.pdf>`_
   * - `TiDEModel <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tide_model.html#darts.models.forecasting.tide_model.TiDEModel>`_
     - ✅ ✅
     - ✅ ✅ ✅
     - ✅ ✅
     - ✅
     - `TiDE paper <https://arxiv.org/pdf/2304.08424.pdf>`_
   * - `TSMixerModel <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tsmixer_model.html#darts.models.forecasting.tsmixer_model.TSMixerModel>`_
     - ✅ ✅
     - ✅ ✅ ✅
     - ✅ ✅
     - ✅
     - `TSMixer paper <https://arxiv.org/pdf/2303.06053.pdf>`_, `PyTorch Implementation <https://github.com/ditschuk/pytorch-tsmixer>`_
   * - **Foundation Models** (`GlobalForecastingModel <https://unit8co.github.io/darts/userguide/covariates.html#global-forecasting-models-gfms>`_): No training required
     -
     -
     -
     -
     -
   * - `Chronos2Model <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.chronos2_model.html#darts.models.forecasting.chronos2_model.Chronos2Model>`_
     - ✅ ✅
     - ✅ ✅ 🔴
     - ✅ ✅
     - ✅
     - `Chronos-2 report <https://arxiv.org/abs/2510.15821>`_, `Amazon blog post <https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting>`_
   * - **Ensemble Models** (`GlobalForecastingModel <https://unit8co.github.io/darts/userguide/covariates.html#global-forecasting-models-gfms>`_): Model support is dependent on ensembled forecasting models and the ensemble model itself
     -
     -
     -
     -
     -
   * - `NaiveEnsembleModel <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.baselines.html#darts.models.forecasting.baselines.NaiveEnsembleModel>`_
     - ✅ ✅
     - ✅ ✅ ✅
     - ✅ ✅
     - ✅
     -
   * - `RegressionEnsembleModel <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.regression_ensemble_model.html#darts.models.forecasting.regression_ensemble_model.RegressionEnsembleModel>`_
     - ✅ ✅
     - ✅ ✅ ✅
     - ✅ ✅
     - ✅
     -
   * - **Conformal Models** (`GlobalForecastingModel <https://unit8co.github.io/darts/userguide/covariates.html#global-forecasting-models-gfms>`_): Model support is dependent on the forecasting model used
     -
     -
     -
     -
     -
   * - `ConformalNaiveModel <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.conformal_models.html#darts.models.forecasting.conformal_models.ConformalNaiveModel>`_
     - ✅ ✅
     - ✅ ✅ ✅
     - ✅ ✅
     - ✅
     - `Conformalized Prediction <https://arxiv.org/pdf/1905.03222>`_
   * - `ConformalQRModel <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.conformal_models.html#darts.models.forecasting.conformal_models.ConformalQRModel>`_
     - ✅ ✅
     - ✅ ✅ ✅
     - ✅ ✅
     - ✅
     - `Conformalized Quantile Regression <https://arxiv.org/pdf/1905.03222>`_

Classification Models
~~~~~~~~~~~~~~~~~~~~~

Classification models in Darts are designed to predict categorical class labels, enabling effective time series labeling and future class prediction. These models are perfect for scenarios where identifying distinct categories or states over time is crucial.

.. list-table::
   :header-rows: 1
   :widths: 30 10 10 10 10 20

   * - Model
     - Target Series Support: Univariate / Multivariate
     - Covariates Support: Past-observed / Future-known / Static
     - Probabilistic Forecasting: Sampled / Distribution Parameters
     - Training & Forecasting on Multiple Series
     - Sources
   * - **Regression Models** (`GlobalForecastingModel <https://unit8co.github.io/darts/userguide/covariates.html#global-forecasting-models-gfms>`_)
     -
     -
     -
     -
     -
   * - `SKLearnClassifierModel <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.sklearn_model.html#darts.models.forecasting.sklearn_model.SKLearnClassifierModel>`_: wrapper around any scikit-learn-like classification model
     - ✅ ✅
     - ✅ ✅ ✅
     - ✅ ✅
     - ✅
     -
   * - `CatBoostClassifierModel <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.catboost_model.html#darts.models.forecasting.catboost_model.CatBoostClassifierModel>`_
     - ✅ ✅
     - ✅ ✅ ✅
     - ✅ ✅
     - ✅
     -
   * - `LightGBMClassifierModel <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.lgbm.html#darts.models.forecasting.lgbm.LightGBMClassifierModel>`_
     - ✅ ✅
     - ✅ ✅ ✅
     - ✅ ✅
     - ✅
     -
   * - `XGBClassifierModel <https://unit8co.github.io/darts/generated_api/darts.models.forecasting.xgboost.html#darts.models.forecasting.xgboost.XGBClassifierModel>`_
     - ✅ ✅
     - ✅ ✅ ✅
     - ✅ ✅
     - ✅
     -

Community & Contact
-------------------

Anyone is welcome to join our `Gitter room <https://gitter.im/u8darts/darts>`_ to ask questions, make proposals, discuss use-cases, and more. If you spot a bug or have suggestions, GitHub issues are also welcome.

If what you want to tell us is not suitable for Gitter or Github, feel free to send us an email at darts@unit8.co for darts related matters or info@unit8.co for any other inquiries.

Contribute
----------

The development is ongoing, and we welcome suggestions, pull requests and issues on GitHub. All contributors will be acknowledged on the `change log page <https://github.com/unit8co/darts/blob/master/CHANGELOG.md>`_.

Before working on a contribution (a new feature or a fix), `check our contribution guidelines <https://github.com/unit8co/darts/blob/master/CONTRIBUTING.md>`_.

Citation
--------

If you are using Darts in your scientific work, we would appreciate citations to the following JMLR paper.

`Darts: User-Friendly Modern Machine Learning for Time Series <https://www.jmlr.org/papers/v23/21-1177.html>`_

Bibtex entry:

.. code-block:: bibtex

    @article{JMLR:v23:21-1177,
      author  = {Julien Herzen and Francesco Lässig and Samuele Giuliano Piazzetta and Thomas Neuer and Léo Tafti and Guillaume Raille and Tomas Van Pottelbergh and Marek Pasieka and Andrzej Skrodzki and Nicolas Huguenin and Maxime Dumonal and Jan Kościsz and Dennis Bader and Frédérick Gusset and Mounir Benheddi and Camila Williamson and Michal Kosinski and Matej Petrik and Gaël Grosch},
      title   = {Darts: User-Friendly Modern Machine Learning for Time Series},
      journal = {Journal of Machine Learning Research},
      year    = {2022},
      volume  = {23},
      number  = {124},
      pages   = {1-6},
      url     = {http://jmlr.org/papers/v23/21-1177.html}
    }

.. toctree::
   :hidden:

   Quickstart<quickstart/00-quickstart.ipynb>

.. toctree::
   :hidden:

   User Guide<userguide>

.. toctree::
   :hidden:

   API Reference<generated_api/darts>

.. toctree::
   :hidden:

   Examples<examples>

.. toctree::
   :hidden:

   Release Notes<release_notes/RELEASE_NOTES>
