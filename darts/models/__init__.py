"""
Models
------
"""

from ..logging import get_logger

logger = get_logger(__name__)

# Forecasting
try:
    from arima import AutoARIMA
except ModuleNotFoundError:
    logger.warning("Support pmdarima based models not available. To enable it install darts[pmdarima]")

try:
    from .prophet import Prophet
except ModuleNotFoundError:
    logger.warning("Support Prophet based models not available. To enable it install darts[prophet]")

try:
    from .rnn_model import RNNModel
except ModuleNotFoundError:
    logger.warning("Support Torch based models not available. To enable it install darts[torch]")

try:
    from .tcn_model import TCNModel
except ModuleNotFoundError:
    logger.warning("Support of Torch based models not available. To enable it install darts[torch]")


from .arima import ARIMA
from .baselines import NaiveMean, NaiveSeasonal, NaiveDrift
from .exponential_smoothing import ExponentialSmoothing
from .fft import FFT
from .theta import Theta

# Regression
from .standard_regression_model import StandardRegressionModel
