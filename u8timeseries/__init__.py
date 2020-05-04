"""
u8timeseries
------------
"""
import os

from u8timeseries.models.RNN_model import RNNModule, RNNModel
from u8timeseries.models.arima import Arima, AutoArima
from u8timeseries.models.autoregressive_model import AutoRegressiveModel
from u8timeseries.models.baselines import KthValueAgoBaseline
from u8timeseries.models.exponential_smoothing import ExponentialSmoothing
from u8timeseries.models.prophet import Prophet
from u8timeseries.models.standard_regressive_model import StandardRegressiveModel
from u8timeseries.models.theta import Theta
from u8timeseries.preprocessing.transformer import Transformer
from .timeseries import TimeSeries

path = os.path.join(os.path.dirname(__file__), 'VERSION')
__version__ = open(path, "r").read()
