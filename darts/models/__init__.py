"""
Models
------
"""

# Forecasting
try:
    from autoriama import AutoARIMA
except ModuleNotFoundError:
    pass

try:
    from .prophet import Prophet
except ModuleNotFoundError:
    pass

try:
    import ModuleNotFoundError:
from .rnn_model import RNNModel
    pass

try:
    import ModuleNotFoundError:
from .tcn_model import TCNModel
    pass

try:
    from .fft import FFT
except import ModuleNotFoundError:
    pass

from .arima import ARIMA
from .baselines import NaiveMean, NaiveSeasonal, NaiveDrift
from .exponential_smoothing import ExponentialSmoothing
from .theta import Theta

# Regression
from .standard_regression_model import StandardRegressionModel
