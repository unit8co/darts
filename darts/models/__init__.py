"""
Models
------
"""

# Forecasting
from .arima import ARIMA, AutoARIMA
from .baselines import NaiveMean, NaiveSeasonal, NaiveDrift
from .prophet import Prophet
from .exponential_smoothing import ExponentialSmoothing
from .rnn_model import RNNModel
from .tcn_model import TCNModel
from .theta import Theta, FourTheta
from .fft import FFT

# Regression
from .standard_regression_model import StandardRegressionModel

# Enums
from enum import Enum


class Season(Enum):
    MULTIPLICATIVE = 'multiplicative'
    ADDITIVE = 'additive'
    NONE = None


class Trend(Enum):
    LINEAR = 'linear'
    EXPONENTIAL = 'exponential'


class Model(Enum):
    MULTIPLICATIVE = 'multiplicative'
    ADDITIVE = 'additive'
