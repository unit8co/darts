.. role:: raw-html-m2r(raw)
   :format: html


Time Series Made Easy in Python
===============================


.. image:: https://github.com/unit8co/darts/raw/master/static/images/darts-logo-trim.png
   :target: https://github.com/unit8co/darts/raw/master/static/images/darts-logo-trim.png
   :alt: darts


----


.. image:: https://badge.fury.io/py/u8darts.svg
   :target: https://badge.fury.io/py/darts
   :alt: PyPI version


.. image:: https://img.shields.io/conda/vn/conda-forge/u8darts-all.svg
   :target: https://anaconda.org/conda-forge/u8darts-all
   :alt: Conda Version


.. image:: https://img.shields.io/badge/python-3.7+-blue.svg
   :target: https://img.shields.io/badge/python-3.7+-blue.svg
   :alt: Supported versions


.. image:: https://img.shields.io/docker/v/unit8/darts?label=docker&sort=date
   :target: https://img.shields.io/docker/v/unit8/darts?label=docker&sort=date
   :alt: Docker Image Version (latest by date)


.. image:: https://img.shields.io/github/release-date/unit8co/darts
   :target: https://img.shields.io/github/release-date/unit8co/darts
   :alt: GitHub Release Date


.. image:: https://img.shields.io/github/workflow/status/unit8co/darts/darts%20release%20workflow/master
   :target: https://img.shields.io/github/workflow/status/unit8co/darts/darts%20release%20workflow/master
   :alt: GitHub Workflow Status


.. image:: https://pepy.tech/badge/u8darts
   :target: https://pepy.tech/project/u8darts
   :alt: Downloads


.. image:: https://pepy.tech/badge/darts
   :target: https://pepy.tech/project/darts
   :alt: Downloads


.. image:: https://codecov.io/gh/unit8co/darts/branch/master/graph/badge.svg?token=7F1TLUFHQW
   :target: https://codecov.io/gh/unit8co/darts
   :alt: codecov


.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: black


**darts** is a Python library for easy manipulation and forecasting of time series.
It contains a variety of models, from classics such as ARIMA to deep neural networks.
The models can all be used in the same way, using ``fit()`` and ``predict()`` functions,
similar to scikit-learn. The library also makes it easy to backtest models,
combine the predictions of several models, and take external data into account. 
Darts supports both univariate and multivariate time series and models. 
The ML-based models can be trained on potentially large datasets containing multiple time
series, and some of the models offer a rich support for probabilistic forecasting.

Documentation
-------------


* `Quickstart <https://unit8co.github.io/darts/quickstart/00-quickstart.html>`_
* `API Reference <https://unit8co.github.io/darts/generated_api/darts.html>`_
* `Examples <https://unit8co.github.io/darts/examples.html>`_

High Level Introductions
""""""""""""""""""""""""


* `Introductory Blog Post <https://medium.com/unit8-machine-learning-publication/darts-time-series-made-easy-in-python-5ac2947a8878>`_
* `Introduction to Darts at PyData Global 2021 <https://youtu.be/g6OXDnXEtFA>`_

Articles on Selected Topics
"""""""""""""""""""""""""""


* `Training Models on Multiple Time Series <https://medium.com/unit8-machine-learning-publication/training-forecasting-models-on-multiple-time-series-with-darts-dc4be70b1844>`_
* `Using Past and Future Covariates <https://medium.com/unit8-machine-learning-publication/time-series-forecasting-using-past-and-future-external-data-with-darts-1f0539585993>`_
* `Temporal Convolutional Networks and Forecasting <https://medium.com/unit8-machine-learning-publication/temporal-convolutional-networks-and-forecasting-5ce1b6e97ce4>`_
* `Probabilistic Forecasting <https://medium.com/unit8-machine-learning-publication/probabilistic-forecasting-in-darts-e88fbe83344e>`_

Quick Install
-------------

We recommend to first setup a clean Python environment for your project with at least Python 3.7 using your favorite tool
(\ :raw-html-m2r:`<a href="https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html" title="conda-env">conda</a>`\ ,
`venv <https://docs.python.org/3/library/venv.html>`_\ , `virtualenv <https://virtualenv.pypa.io/en/latest/>`_ with
or without `virtualenvwrapper <https://virtualenvwrapper.readthedocs.io/en/latest/>`_\ ).

Once your environment is set up you can install darts using pip:

.. code-block::

   pip install darts


For more details you can refer to our 
`installation instructions <https://github.com/unit8co/darts/blob/master/INSTALL.md>`_.

Example Usage
-------------

Create a ``TimeSeries`` object from a Pandas DataFrame, and split it in train/validation series:

.. code-block:: python

   import pandas as pd
   from darts import TimeSeries

   # Read a pandas DataFrame
   df = pd.read_csv('AirPassengers.csv', delimiter=",")

   # Create a TimeSeries, specifying the time and value columns
   series = TimeSeries.from_dataframe(df, 'Month', '#Passengers')

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
   prediction.plot(label='forecast', low_quantile=0.05, high_quantile=0.95)
   plt.legend()


.. raw:: html

   <div style="text-align:center;">
   <img src="https://github.com/unit8co/darts/raw/master/static/images/example.png" alt="darts forecast example" />
   </div>


Features
--------


* **Forecasting Models:** A large collection of forecasting models; from statistical models (such as
  ARIMA) to deep learning models (such as N-BEATS). See table of models below.
* **Data processing:** Tools to easily apply (and revert) common transformations on
  time series data (scaling, boxcox, ...)
* **Metrics:** A variety of metrics for evaluating time series' goodness of fit;
  from R2-scores to Mean Absolute Scaled Error.
* **Backtesting:** Utilities for simulating historical forecasts, using moving time windows.
* **Regression Models:** Possibility to predict a time series from lagged versions of itself
  and of some external covariate series, using arbitrary regression models (e.g. scikit-learn models).
* **Multiple series training:** All machine learning based models (incl.\ all neural networks) 
  support being trained on multiple series.
* **Past and Future Covariates support:** Some models support past-observed and/or future-known covariate time series
  as inputs for producing forecasts.
* **Multivariate Support:** Tools to create, manipulate and forecast multivariate time series.
* **Probabilistic Support:** ``TimeSeries`` objects can (optionally) represent stochastic
  time series; this can for instance be used to get confidence intervals, and several models
  support different flavours of probabilistic forecasting.
* **PyTorch Lightning Support:** All deep learning models are implemented using PyTorch Lightning,
  supporting among other things custom callbacks, GPUs/TPUs training and custom trainers.
* **Filtering Models:** Darts offers three filtering models: ``KalmanFilter``\ , ``GaussianProcessFilter``\ ,
  and ``MovingAverage``\ , which allow to filter time series, and in some cases obtain probabilistic
  inferences of the underlying states/values.

Forecasting Models
------------------

Here's a breakdown of the forecasting models currently implemented in Darts. We are constantly working
on bringing more models and features.

.. list-table::
   :header-rows: 1

   * - Model
     - Univariate
     - Multivariate
     - Probabilistic
     - Multiple-series training
     - Past-observed covariates support
     - Future-known covariates support
     - Reference
   * - ``ARIMA``
     - ✅
     - 
     - ✅
     - 
     - 
     - ✅
     - 
   * - ``VARIMA``
     - ✅
     - ✅
     - 
     - 
     - 
     - ✅
     - 
   * - ``AutoARIMA``
     - ✅
     - 
     - 
     - 
     - 
     - ✅
     - 
   * - ``ExponentialSmoothing``
     - ✅
     - 
     - ✅
     - 
     - 
     - 
     - 
   * - ``BATS`` and ``TBATS``
     - ✅
     - 
     - ✅
     - 
     - 
     - 
     - `TBATS paper <https://robjhyndman.com/papers/ComplexSeasonality.pdf>`_
   * - ``Theta`` and ``FourTheta``
     - ✅
     - 
     - 
     - 
     - 
     - 
     - `Theta <https://robjhyndman.com/papers/Theta.pdf>`_ & `4 Theta <https://github.com/Mcompetitions/M4-methods/blob/master/4Theta%20method.R>`_
   * - ``Prophet``
     - ✅
     - 
     - ✅
     - 
     - 
     - ✅
     - `Prophet repo <https://github.com/facebook/prophet>`_
   * - ``FFT`` (Fast Fourier Transform)
     - ✅
     - 
     - 
     - 
     - 
     - 
     - 
   * - ``KalmanForecaster`` using the Kalman filter and N4SID for system identification
     - ✅
     - ✅
     - ✅
     - 
     - 
     - ✅
     - `N4SID paper <https://people.duke.edu/~hpgavin/SystemID/References/VanOverschee-Automatica-1994.pdf>`_
   * - ``RegressionModel``\ ; generic wrapper around any sklearn regression model
     - ✅
     - ✅
     - 
     - ✅
     - ✅
     - ✅
     - 
   * - ``RandomForest``
     - ✅
     - ✅
     - 
     - ✅
     - ✅
     - ✅
     - 
   * - ``LinearRegressionModel``
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - 
   * - ``LightGBMModel``
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - 
   * - ``RNNModel`` (incl. LSTM and GRU); equivalent to DeepAR in its probabilistic version
     - ✅
     - ✅
     - ✅
     - ✅
     - 
     - ✅
     - `DeepAR paper <https://arxiv.org/abs/1704.04110>`_
   * - ``BlockRNNModel`` (incl. LSTM and GRU)
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - 
     - 
   * - ``NBEATSModel``
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - 
     - `N-BEATS paper <https://arxiv.org/abs/1905.10437>`_
   * - ``TCNModel``
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - 
     - `TCN paper <https://arxiv.org/abs/1803.01271>`_\ , `DeepTCN paper <https://arxiv.org/abs/1906.04397>`_\ , `blog post <https://medium.com/unit8-machine-learning-publication/temporal-convolutional-networks-and-forecasting-5ce1b6e97ce4>`_
   * - ``TransformerModel``
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - 
     - 
   * - ``TFTModel`` (Temporal Fusion Transformer)
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - `TFT paper <https://arxiv.org/pdf/1912.09363.pdf>`_\ , `PyTorch Forecasting <https://pytorch-forecasting.readthedocs.io/en/latest/models.html>`_
   * - Naive Baselines
     - ✅
     - 
     - 
     - 
     - 
     - 
     - 


Community & Contact
-------------------

Anyone is welcome to join our `Discord server <https://discord.gg/Um3jBTYFsA>`_ to
ask questions, make proposals, discuss use-cases, and more. If you spot a bug or
or have a feature request, Github issues are also welcome.

If what you want to tell us is not suitable for Discord or Github,
feel free to send us an email at :raw-html-m2r:`<a href="mailto:darts@unit8.co">darts@unit8.co</a>` for
darts related matters or :raw-html-m2r:`<a href="mailto:info@unit8.co">info@unit8.co</a>` for any other
inquiries.

Contribute
----------

The development is ongoing, and we welcome suggestions, pull requests and issues on Github.
All contributors will be acknowledged on the
`change log page <https://github.com/unit8co/darts/blob/master/CHANGELOG.md>`_.

Before working on a contribution (a new feature or a fix),
`check our contribution guidelines <https://github.com/unit8co/darts/blob/master/CONTRIBUTING.md>`_.

Citation
--------

If you are using Darts in your scientific work, we would appreciate citations to the following paper.

`Darts: User-Friendly Modern Machine Learning for Time Series <https://arxiv.org/abs/2110.03224>`_

Bibtex entry:

.. code-block::

   @misc{herzen2021darts,
         title={Darts: User-Friendly Modern Machine Learning for Time Series},
         author={Julien Herzen and Francesco Lässig and Samuele Giuliano Piazzetta and Thomas Neuer and Léo Tafti and Guillaume Raille and Tomas Van Pottelbergh and Marek Pasieka and Andrzej Skrodzki and Nicolas Huguenin and Maxime Dumonal and Jan Kościsz and Dennis Bader and Frédérick Gusset and Mounir Benheddi and Camila Williamson and Michal Kosinski and Matej Petrik and Gaël Grosch},
         year={2021},
         eprint={2110.03224},
         archivePrefix={arXiv},
         primaryClass={cs.LG}
   }
