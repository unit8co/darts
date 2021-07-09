.. role:: raw-html-m2r(raw)
   :format: html


Time Series Made Easy in Python
===============================


.. image:: https://github.com/unit8co/darts/raw/develop/static/images/darts-logo-trim.png
   :target: https://github.com/unit8co/darts/raw/develop/static/images/darts-logo-trim.png
   :alt: darts


----


.. image:: https://badge.fury.io/py/u8darts.svg
   :target: https://badge.fury.io/py/darts
   :alt: PyPI version


.. image:: https://img.shields.io/github/workflow/status/unit8co/darts/darts%20release%20workflow/master
   :target: https://img.shields.io/github/workflow/status/unit8co/darts/darts%20release%20workflow/master
   :alt: GitHub Workflow Status


.. image:: https://img.shields.io/badge/python-3.7+-blue.svg
   :target: https://img.shields.io/badge/python-3.7+-blue.svg
   :alt: Supported versions


.. image:: https://img.shields.io/docker/v/unit8/darts?label=docker&sort=date
   :target: https://img.shields.io/docker/v/unit8/darts?label=docker&sort=date
   :alt: Docker Image Version (latest by date)


.. image:: https://img.shields.io/github/release-date/unit8co/darts
   :target: https://img.shields.io/github/release-date/unit8co/darts
   :alt: GitHub Release Date


.. image:: https://pepy.tech/badge/u8darts
   :target: https://pepy.tech/project/u8darts
   :alt: Downloads


.. image:: https://pepy.tech/badge/darts
   :target: https://pepy.tech/project/darts
   :alt: Downloads


**darts** is a Python library for easy manipulation and forecasting of time series.
It contains a variety of models, from classics such as ARIMA to deep neural networks.
The models can all be used in the same way, using ``fit()`` and ``predict()`` functions,
similar to scikit-learn. The library also makes it easy to backtest models,
and combine the predictions of several models and external regressors. Darts supports both
univariate and multivariate time series and models. The neural networks can be trained
on multiple time series, and some of the models offer probabilistic forecasts.

Documentation
-------------


* `Examples & Tutorials <https://unit8co.github.io/darts/examples.html>`_
* `API Documentation <https://unit8co.github.io/darts/generated_api/darts.html>`_

High Level Introductions
""""""""""""""""""""""""


* `Introductory Blog Post <https://medium.com/unit8-machine-learning-publication/darts-time-series-made-easy-in-python-5ac2947a8878>`_
* `Introductory Video <https://www.youtube.com/watch?v=Sx-uI-PypmU&t=8s&ab_channel=Unit8>`_

Install
-------

We recommend to first setup a clean Python environment for your project with at least Python 3.7 using your favorite tool (\ :raw-html-m2r:`<a href="https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html" title="conda-env">conda</a>`\ , `venv <https://docs.python.org/3/library/venv.html>`_\ , `virtualenv <https://virtualenv.pypa.io/en/latest/>`_ with or without `virtualenvwrapper <https://virtualenvwrapper.readthedocs.io/en/latest/>`_\ ).

Once your environment is set up you can install darts using pip:

.. code-block::

   pip install darts


For more detailed install instructions you can refer to our installation guide at the end of this page.

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
   <img src="https://github.com/unit8co/darts/raw/develop/static/images/example.png" alt="darts forecast example" />
   </div>


We invite you to go over the example and tutorial notebooks in
the `examples <https://github.com/unit8co/darts/tree/master/examples>`_ directory.

Features
--------

Currently, the library contains the following features:

**Forecasting Models:** A large collection of forecasting models; from statistical models (such as
ARIMA) to deep learning models (such as N-BEATS). See table of models below.

**Data processing:** Tools to easily apply (and revert) common transformations on time series data (scaling, boxcox, …)

**Metrics:** A variety of metrics for evaluating time series' goodness of fit;
from R2-scores to Mean Absolute Scaled Error.

**Backtesting:** Utilities for simulating historical forecasts, using moving time windows.

**Regressive Models:** Possibility to predict a time series from lagged versions of itself
and of some external covariate series, using arbitrary regression models (e.g. scikit-learn models)

**Multivariate Support:** Tools to create, manipulate and forecast multivariate time series.

**Probabilistic Support:** ``TimeSeries`` objects can (optionally) represent stochastic
time series; this can for instance be used to get confidence intervals.

**Filtering Models:** Darts offers three filtering models: ``KalmanFilter``\ , ``GaussianProcessFilter``\ ,
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
   * - ``ARIMA``
     - x
     - 
     - x
     - 
     - 
     - 
     - 
   * - ``VARIMA``
     - x
     - x
     - 
     - 
     - 
     - 
     - 
   * - ``AutoARIMA``
     - x
     - 
     - 
     - 
     - 
     - 
     - 
   * - ``ExponentialSmoothing``
     - x
     - 
     - x
     - 
     - 
     - 
     - 
   * - ``Theta`` and ``FourTheta``
     - x
     - 
     - 
     - 
     - 
     - 
     - 
   * - ``Prophet``
     - x
     - 
     - 
     - 
     - 
     - 
     - 
   * - ``FFT`` (Fast Fourier Transform)
     - x
     - 
     - 
     - 
     - 
     - 
     - 
   * - Regression Models (incl ``RandomForest`` and ``LinearRegressionModel``\ )
     - x
     - 
     - 
     - 
     - 
     - 
     - 
   * - ``RNNModel`` (incl. LSTM and GRU); equivalent to DeepAR in its probabilistic version
     - x
     - x
     - x
     - x
     - x
     - x
     - 
   * - ``BlockRNNModel`` (incl. LSTM and GRU)
     - x
     - x
     - 
     - x
     - x
     - (x)
     - 
   * - ``NBEATSModel``
     - x
     - x
     - 
     - x
     - x
     - (x)
     - 
   * - ``TCNModel``
     - x
     - x
     - x
     - x
     - x
     - (x)
     - 
   * - ``TransformerModel``
     - x
     - x
     - 
     - x
     - x
     - (x)
     - 
   * - Naive Baselines
     - x
     - 
     - 
     - 
     - 
     - 
     - 


Contribute
----------

The development is ongoing, and there are many new features that we want to add.
We welcome pull requests and issues on GitHub.

Before working on a contribution (a new feature or a fix), `\ **check our contribution guidelines** <CONTRIBUTE.md>`_.

Contact Us
----------

If what you want to tell us is not a suitable github issue, feel free to send us an email at :raw-html-m2r:`<a href="mailto:darts@unit8.co">darts@unit8.co</a>` for darts related matters or :raw-html-m2r:`<a href="mailto:info@unit8.co">info@unit8.co</a>` for any other inquiries.

Installation Guide
------------------

Preconditions
^^^^^^^^^^^^^

Some of the models depend on ``fbprophet`` and ``torch``\ , which have non-Python dependencies.
A Conda environment is thus recommended because it will handle all of those in one go.

The following steps assume running inside a conda environment.
If that's not possible, first follow the official instructions to install
`fbprophet <https://facebook.github.io/prophet/docs/installation.html#python>`_
and `torch <https://pytorch.org/get-started/locally/>`_\ , then skip to
`Install darts <#install-darts>`_

To create a conda environment for Python 3.7
(after installing `conda <https://docs.conda.io/en/latest/miniconda.html>`_\ ):

.. code-block::

   conda create --name <env-name> python=3.7


Don't forget to activate your virtual environment

.. code-block::

   conda activate <env-name>



MAC
~~~

.. code-block::

   conda install -c conda-forge -c pytorch pip fbprophet pytorch


Linux and Windows
~~~~~~~~~~~~~~~~~

.. code-block::

   conda install -c conda-forge -c pytorch pip fbprophet pytorch cpuonly


Install darts
^^^^^^^^^^^^^

Install Darts with all available models: ``pip install darts``.

As some models have relatively heavy (or non-Python) dependencies,
we also maintain the ``u8darts`` package, which provides the following alternate lighter install options:


* Install core only (without neural networks, Prophet or AutoARIMA): ``pip install u8darts``
* Install core + neural networks (PyTorch): ``pip install 'u8darts[torch]'``
* Install core + Facebook Prophet: ``pip install 'u8darts[fbprophet]'``
* Install core + AutoARIMA: ``pip install 'u8darts[pmdarima]'``

Running the examples only, without installing:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the conda setup is causing too many problems, we also provide a Docker image with everything set up for you and ready-to-use Python notebooks with demo examples.
To run the example notebooks without installing our libraries natively on your machine, you can use our Docker image:

.. code-block:: bash

   ./gradlew docker && ./gradlew dockerRun

Then copy and paste the URL provided by the docker container into your browser to access Jupyter notebook.

For this setup to work you need to have a Docker service installed. You can get it at `Docker website <https://docs.docker.com/get-docker/>`_.

Tests
^^^^^

The gradle setup works best when used in a python environment, but the only requirement is to have ``pip`` installed for Python 3+

To run all tests at once just run

.. code-block:: bash

   ./gradlew test_all

alternatively you can run

.. code-block:: bash

   ./gradlew unitTest_all # to run only unittests
   ./gradlew coverageTest # to run coverage
   ./gradlew lint         # to run linter

To run the tests for specific flavours of the library, replace ``_all`` with ``_core``\ , ``_fbprophet``\ , ``_pmdarima`` or ``_torch``.

Documentation
^^^^^^^^^^^^^

To build documantation locally just run

.. code-block:: bash

   ./gradlew buildDocs

After that docs will be available in ``./docs/build/html`` directory. You can just open ``./docs/build/html/index.html`` using your favourite browser.
