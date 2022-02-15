"""
Models
------
"""

from darts.logging import get_logger

logger = get_logger(__name__)

from darts.models.forecasting.arima import ARIMA

# Forecasting
from darts.models.forecasting.baselines import NaiveDrift, NaiveMean, NaiveSeasonal
from darts.models.forecasting.exponential_smoothing import ExponentialSmoothing
from darts.models.forecasting.fft import FFT
from darts.models.forecasting.kalman_forecaster import KalmanForecaster
from darts.models.forecasting.theta import FourTheta, Theta
from darts.models.forecasting.varima import VARIMA

try:
    from darts.models.forecasting.auto_arima import AutoARIMA
except ModuleNotFoundError:
    logger.warning(
        "Support for AutoARIMA is not available."
        "To enable it, install u8darts[pmdarima] or u8darts[all]."
    )

try:
    from darts.models.forecasting.prophet_model import Prophet
except ModuleNotFoundError:
    logger.warning(
        "Support for Facebook Prophet is not available. "
        'To enable it, install "darts", "u8darts[prophet]" or "u8darts[all]" (with pip);'
        'or "u8darts-all" (with conda).'
    )

try:
    from darts.models.forecasting.block_rnn_model import BlockRNNModel
    from darts.models.forecasting.nbeats import NBEATSModel
    from darts.models.forecasting.rnn_model import RNNModel
    from darts.models.forecasting.tcn_model import TCNModel
    from darts.models.forecasting.tft_model import TFTModel
    from darts.models.forecasting.transformer_model import TransformerModel

except ModuleNotFoundError:
    logger.warning(
        "Support for Torch based models not available. "
        'To enable them, install "darts", "u8darts[torch]" or "u8darts[all]" (with pip); '
        'or "u8darts-torch" or "u8darts-all" (with conda).'
    )

# Regression & RegressionEnsemble:
try:
    from darts.models.forecasting.linear_regression_model import LinearRegressionModel
    from darts.models.forecasting.random_forest import RandomForest
    from darts.models.forecasting.regression_ensemble_model import (
        RegressionEnsembleModel,
    )
    from darts.models.forecasting.regression_model import RegressionModel
except ModuleNotFoundError:
    logger.warning(
        "Support for Regression based models (incl. RegressionEnsembleModel) not available, "
        "as they depend on PyTorch Datasets. "
        'To enable them, install "darts", "u8darts[torch]" or "u8darts[all]" (with pip); '
        'or "u8darts-torch" or "u8darts-all" (with conda).'
    )

try:
    from darts.models.forecasting.gradient_boosted_model import LightGBMModel
except ModuleNotFoundError:
    logger.warning(
        "Support for LightGBM not available."
        "To enable LightGBM support in Darts, follow the detailed "
        "install instructions for LightGBM in the README: "
        "https://github.com/unit8co/darts/blob/master/README.md"
    )

from darts.models.filtering.gaussian_process_filter import GaussianProcessFilter
from darts.models.filtering.kalman_filter import KalmanFilter

# Filtering
from darts.models.filtering.moving_average import MovingAverage
from darts.models.forecasting.baselines import NaiveEnsembleModel

# Ensembling
from darts.models.forecasting.ensemble_model import EnsembleModel
