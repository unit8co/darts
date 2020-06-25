.. role:: raw-html-m2r(raw)
   :format: html


Time Series Made Easy in Python
===============================


.. image:: https://github.com/unit8co/darts/raw/develop/static/images/darts-logo-trim.png
   :target: https://github.com/unit8co/darts/raw/develop/static/images/darts-logo-trim.png
   :alt: darts
 

----


.. image:: https://badge.fury.io/py/u8darts.svg
   :target: https://badge.fury.io/py/u8darts
   :alt: PyPI version


.. image:: https://img.shields.io/github/workflow/status/unit8co/darts/darts%20PR%20merge%20workflow/develop
   :target: https://img.shields.io/github/workflow/status/unit8co/darts/darts%20PR%20merge%20workflow/develop
   :alt: GitHub Workflow Status


.. image:: https://img.shields.io/badge/python-3.6+-blue.svg
   :target: https://img.shields.io/badge/python-3.6+-blue.svg
   :alt: Supported versions


.. image:: https://img.shields.io/docker/v/unit8/darts?label=docker&sort=date
   :target: https://img.shields.io/docker/v/unit8/darts?label=docker&sort=date
   :alt: Docker Image Version (latest by date)


.. image:: https://img.shields.io/pypi/dm/u8darts
   :target: https://img.shields.io/pypi/dm/u8darts
   :alt: PyPI - Downloads


.. image:: https://img.shields.io/github/release-date/unit8co/darts
   :target: https://img.shields.io/github/release-date/unit8co/darts
   :alt: GitHub Release Date


**darts** is a python library for easy manipulation and forecasting of time series.
It contains a variety of models, from classics such as ARIMA to neural networks.
The models can all be used in the same way, using ``fit()`` and ``predict()`` functions,
similar to scikit-learn. The library also makes it easy to backtest models,
and combine the predictions of several models and external regressors.

Install
-------

We recommend to first setup a clean python environment for your project with at least python 3.6 using your favorite tool (\ :raw-html-m2r:`<a href="https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html" title="conda-env">conda</a>`\ , `venv <https://docs.python.org/3/library/venv.html>`_\ , `virtualenv <https://virtualenv.pypa.io/en/latest/>`_ with or without `virtualenvwrapper <https://virtualenvwrapper.readthedocs.io/en/latest/>`_\ ).

Quick Install
^^^^^^^^^^^^^

Once your environement is setup you can install darts using the pip package:

.. code-block::

   pip install u8darts


Step-by-step Install
^^^^^^^^^^^^^^^^^^^^

For more detailed install instructions you can refer to our installation guide at the end of this page.

Example Usage
-------------

Create ``TimeSeries`` object from a Pandas DataFrame, and split in train/validation series:

.. code-block:: python

   import pandas as pd
   from darts import TimeSeries

   df = pd.read_csv('AirPassengers.csv', delimiter=",")
   series = TimeSeries.from_dataframe(df, 'Month', '#Passengers')
   train, val = series.split_after(pd.Timestamp('19590101'))

..

   The dataset used in this example can be downloaded from this `link <https://raw.githubusercontent.com/unit8co/darts/master/examples/AirPassengers.csv>`_.


Fit an exponential smoothing model, and make a prediction over the validation series' duration:

.. code-block:: python

   from darts.models import ExponentialSmoothing

   model = ExponentialSmoothing()
   model.fit(train)
   prediction = model.predict(len(val))

Plot:

.. code-block:: python

   import matplotlib.pyplot as plt

   series.plot(label='actual', lw=3)
   prediction.plot(label='forecast', lw=3)
   plt.legend()
   plt.xlabel('Year')


.. raw:: html

   <div style="text-align:center;">
   <img src="https://github.com/unit8co/darts/raw/develop/static/images/example.png" alt="darts forecast example" />
   </div>


We invite you to go over the example notebooks in the ``examples`` directory.

Documentation
-------------

The documentation of the API and models is available `here <https://unit8co.github.io/darts/>`_.

Features
--------

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

**Multivariate Support:** Tools to create, manipulate and forecast multivariate time series.

Contribute
----------

The development is ongoing, and there are many new features that we want to add. 
We welcome pull requests and issues on github.

Before working on a contribution (a new feature or a fix) make sure you can't find anything related in `issues <https://github.com/unit8co/darts/issues>`_. If there is no on-going effort on what you plan to do then we recommend to do the following:


#. Create an issue, describe how you would attempt to solve it, and if possible wait for a discussion.
#. Fork the repository.
#. Clone the forked repository locally.
#. Create a clean python env and install requirements with pip: ``pip install -r requirements/main.txt -r requirements/dev.txt -r requirements/release.txt``
#. Create a new branch with your fix / feature from the **develop** branch.
#. Create a pull request from your new branch to the **develop** branch.

Contact Us
----------

If what you want to tell us is not a suitable github issue, feel free to send us an email at :raw-html-m2r:`<a href="mailto:darts@unit8.co">darts@unit8.co</a>` for darts related matters or :raw-html-m2r:`<a href="mailto:info@unit8.co">info@unit8.co</a>` for any other inquiries.

Installation Guide
------------------

Preconditions
^^^^^^^^^^^^^

Our direct dependencies include ``fbprophet`` and ``torch`` which have non-Python dependencies.
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
^^^

.. code-block::

   conda install -c conda-forge -c pytorch pip fbprophet pytorch


Linux and Windows
^^^^^^^^^^^^^^^^^

.. code-block::

   conda install -c conda-forge -c pytorch pip fbprophet pytorch cpuonly


Install darts
^^^^^^^^^^^^^

.. code-block::

   pip install u8darts


Running the examples only, without installing:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the conda setup is causing too many problems, we also provide a Docker image with everything set up for you and ready-to-use python notebooks with demo examples.
To run the example notebooks without installing our libraries natively on your machine, you can use our Docker image:

.. code-block::

   cd scripts
   ./build_docker.sh && ./run_docker.sh

Then copy and paste the URL provided by the docker container into your browser to access Jupyter notebook.

For this setup to work you need to have a Docker service installed. You can get it at `Docker website <https://docs.docker.com/get-docker/>`_.
