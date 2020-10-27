"""
Models
------
"""
from ..logging import get_logger

logger = get_logger(__name__)

# Forecasting
from .baselines import NaiveMean, NaiveSeasonal, NaiveDrift
from .exponential_smoothing import ExponentialSmoothing
from .theta import Theta, FourTheta
from .arima import ARIMA
from .fft import FFT

try:
    from .auto_arima import AutoARIMA
except ModuleNotFoundError:
    logger.warning("Support for AutoARIMA is not available. To enable it, install u8darts[pmdarima] or u8darts[all].")

try:
    from .prophet import Prophet
except ModuleNotFoundError:
    logger.warning("Support Facebook Prophet is not available. "
                   "To enable it, install u8darts[fbprophet] or u8darts[all].")

try:
    from .rnn_model import RNNModel
    from .tcn_model import TCNModel
    from .nbeats import NBEATSModel
    from .transformer_model import TransformerModel

except ModuleNotFoundError:
    logger.warning("Support Torch based models not available. To enable it, install u8darts[torch] or u8darts[all].")

# Regression
from .standard_regression_model import StandardRegressionModel
