"""
Models
------
"""

from ..logging import get_logger

logger = get_logger(__name__)

# Forecasting
from darts.models.forecasting.baselines import NaiveMean, NaiveSeasonal, NaiveDrift
from darts.models.forecasting.exponential_smoothing import ExponentialSmoothing
from darts.models.forecasting.theta import Theta, FourTheta
from darts.models.forecasting.arima import ARIMA
from darts.models.forecasting.fft import FFT
from darts.models.forecasting.varima import VARIMA

try:
    from darts.models.forecasting.auto_arima import AutoARIMA
except ModuleNotFoundError:
    logger.warning("Support for AutoARIMA is not available. To enable it, install u8darts[pmdarima] or u8darts[all].")

try:
    from darts.models.forecasting.prophet import Prophet
except ModuleNotFoundError:
    logger.warning("Support Facebook Prophet is not available. "
                   "To enable it, install u8darts[prophet] or u8darts[all].")

try:
    from darts.models.forecasting.block_rnn_model import BlockRNNModel
    from darts.models.forecasting.rnn_model import RNNModel
    from darts.models.forecasting.tcn_model import TCNModel
    from darts.models.forecasting.nbeats import NBEATSModel
    from darts.models.forecasting.transformer_model import TransformerModel

except ModuleNotFoundError:
    logger.warning("Support Torch based models not available. To enable it, install u8darts[torch] or u8darts[all].")

# Regression
from darts.models.forecasting.lgbm_model import LinearRegressionModel
from darts.models.forecasting.random_forest import RandomForest
from darts.models.forecasting.regression_model import RegressionModel

try:
    from darts.models.forecasting.gradient_boosted_model import LightGBMModel
except:
    logger.warning("Support for LightGBM not available. To enable LightGBM support in Darts, follow the detailed "
                   "install instructions for LightGBM in the README: "
                   "https://github.com/unit8co/darts/blob/master/README.md")

# Ensembling
from darts.models.forecasting.ensemble_model import EnsembleModel
from darts.models.forecasting.baselines import NaiveEnsembleModel
from darts.models.forecasting.regression_ensemble_model import RegressionEnsembleModel

# Filtering
from darts.models.filtering.moving_average import MovingAverage
from darts.models.filtering.kalman_filter import KalmanFilter
from darts.models.filtering.gaussian_process_filter import GaussianProcessFilter
