:notoc:

Time Series Made Easy in Python
===============================

.. image:: _static/darts-logo-trim.png
   :alt: darts
   :align: center
   :width: 100%
   :class: only-light

.. image:: _static/darts-logo-dark.png
   :alt: darts
   :align: center
   :width: 100%
   :class: only-dark

.. image:: https://badge.fury.io/py/u8darts.svg
   :target: https://badge.fury.io/py/darts
   :alt: PyPI version

.. image:: https://img.shields.io/conda/vn/conda-forge/u8darts-all.svg
   :target: https://anaconda.org/conda-forge/u8darts-all
   :alt: Conda Version

.. image:: https://img.shields.io/badge/python-3.10+-blue.svg
   :alt: Supported versions

.. image:: https://img.shields.io/docker/v/unit8/darts?label=docker&sort=date
   :target: https://hub.docker.com/r/unit8/darts
   :alt: Docker Image Version

.. image:: https://img.shields.io/github/release-date/unit8co/darts
   :alt: GitHub Release Date

.. image:: https://img.shields.io/github/actions/workflow/status/unit8co/darts/release.yml?branch=master
   :alt: GitHub Workflow Status

.. image:: https://pepy.tech/badge/darts
   :target: https://pepy.tech/project/darts
   :alt: Downloads

.. image:: https://pepy.tech/badge/u8darts
   :target: https://pepy.tech/project/u8darts
   :alt: Downloads

.. image:: https://codecov.io/gh/unit8co/darts/branch/master/graph/badge.svg?token=7F1TLUFHQW
   :target: https://codecov.io/gh/unit8co/darts
   :alt: codecov

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: black

.. image:: https://badges.gitter.im/u8darts/darts.svg
   :target: https://gitter.im/u8darts/darts?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
   :alt: Join the chat at https://gitter.im/u8darts/darts

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

    .. grid-item-card:: Home
        :img-top: static/icon-home.svg
        :class-card: intro-card
        :shadow: md

        Features, installation, and usage examples.

        +++

        .. button-ref:: README
            :ref-type: doc
            :click-parent:
            :color: secondary
            :expand:

            To the home page

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

.. toctree::
   :hidden:

   Home<README>

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
