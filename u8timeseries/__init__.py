"""
u8timeseries
------------
"""
from u8timeseries.models.arima import Arima, AutoArima
from u8timeseries.models.baselines import KthValueAgoBaseline
from u8timeseries.models.exponential_smoothing import ExponentialSmoothing
from u8timeseries.models.prophet import Prophet
from u8timeseries.models.standard_regression_model import StandardRegressionModel
from u8timeseries.models.forecasting_model import ForecastingModel
from u8timeseries.models.theta import Theta
from u8timeseries.models.RNN_model import RNNModule, RNNModel
from .timeseries import TimeSeries
from u8timeseries.preprocessing.transformer import Transformer

import os

path = os.path.join(os.path.dirname(__file__), 'VERSION')
__version__ = open(path, "r").read()
