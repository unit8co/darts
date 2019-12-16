Models
======

This section treats the different models that have been implemented as of now.

.. automodule:: u8timeseries.models
   
    .. rubric:: Models summary
    
    .. autosummary::
       
       u8timeseries.Arima
       u8timeseries.AutoArima
       u8timeseries.AutoRegressiveModel
       u8timeseries.ExponentialSmoothing
       u8timeseries.KthValueAgoBaseline
       u8timeseries.Prophet
       u8timeseries.StandardRegressiveModel

Arima
-----

.. currentmodule:: u8timeseries

.. autoclass:: Arima
   :show-inheritance:

   .. rubric:: Methods Summary

   .. autosummary::

      ~Arima.fit
      ~Arima.predict

   .. rubric:: Methods Documentation

   .. automethod:: fit
   .. automethod:: predict
   
   
AutoArima
---------

.. currentmodule:: u8timeseries

.. autoclass:: AutoArima
   :show-inheritance:

   .. rubric:: Methods Summary

   .. autosummary::

      ~AutoArima.fit
      ~AutoArima.predict

   .. rubric:: Methods Documentation

   .. automethod:: fit
   .. automethod:: predict
   
 
AutoRegressiveModel
-------------------

.. currentmodule:: u8timeseries

.. autoclass:: AutoRegressiveModel

   .. rubric:: Methods Summary

   .. autosummary::

      ~AutoRegressiveModel.fit
      ~AutoRegressiveModel.predict

   .. rubric:: Methods Documentation

   .. automethod:: fit
   .. automethod:: predict 


ExponentialSmoothing
--------------------

.. currentmodule:: u8timeseries

.. autoclass:: ExponentialSmoothing
   :show-inheritance:

   .. rubric:: Methods Summary

   .. autosummary::

      ~ExponentialSmoothing.fit
      ~ExponentialSmoothing.predict

   .. rubric:: Methods Documentation

   .. automethod:: fit
   .. automethod:: predict


KthValueAgoBaseline
-------------------

.. currentmodule:: u8timeseries

.. autoclass:: KthValueAgoBaseline
   :show-inheritance:

   .. rubric:: Methods Summary

   .. autosummary::

      ~KthValueAgoBaseline.fit
      ~KthValueAgoBaseline.predict

   .. rubric:: Methods Documentation

   .. automethod:: fit
   .. automethod:: predict
   

Prophet
-------

.. currentmodule:: u8timeseries

.. autoclass:: Prophet
   :show-inheritance:

   .. rubric:: Methods Summary

   .. autosummary::

      ~Prophet.fit
      ~Prophet.predict

   .. rubric:: Methods Documentation

   .. automethod:: fit
   .. automethod:: predict


StandardRegressiveModel
-----------------------

.. currentmodule:: u8timeseries

.. autoclass:: StandardRegressiveModel
   :show-inheritance:

   .. rubric:: Methods Summary

   .. autosummary::

      ~StandardRegressiveModel.fit
      ~StandardRegressiveModel.predict

   .. rubric:: Methods Documentation

   .. automethod:: fit
   .. automethod:: predict   
