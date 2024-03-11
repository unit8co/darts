"""
Models
------
"""

from darts.logging import get_logger

logger = get_logger(__name__)

# Forecasting
from darts.models.forecasting.arima import ARIMA
from darts.models.forecasting.auto_arima import AutoARIMA
from darts.models.forecasting.baselines import (
    NaiveDrift,
    NaiveMean,
    NaiveMovingAverage,
    NaiveSeasonal,
)
from darts.models.forecasting.exponential_smoothing import ExponentialSmoothing
from darts.models.forecasting.fft import FFT
from darts.models.forecasting.kalman_forecaster import KalmanForecaster
from darts.models.forecasting.linear_regression_model import LinearRegressionModel
from darts.models.forecasting.random_forest import RandomForest
from darts.models.forecasting.regression_ensemble_model import RegressionEnsembleModel
from darts.models.forecasting.regression_model import RegressionModel
from darts.models.forecasting.tbats_model import BATS, TBATS
from darts.models.forecasting.theta import FourTheta, Theta
from darts.models.forecasting.varima import VARIMA
from darts.models.utils import NotImportedModule

try:
    from darts.models.forecasting.block_rnn_model import BlockRNNModel
    from darts.models.forecasting.dlinear import DLinearModel
    from darts.models.forecasting.global_baseline_models import (
        GlobalNaiveAggregate,
        GlobalNaiveDrift,
        GlobalNaiveSeasonal,
    )
    from darts.models.forecasting.nbeats import NBEATSModel
    from darts.models.forecasting.nhits import NHiTSModel
    from darts.models.forecasting.nlinear import NLinearModel
    from darts.models.forecasting.rnn_model import RNNModel
    from darts.models.forecasting.tcn_model import TCNModel
    from darts.models.forecasting.tft_model import TFTModel
    from darts.models.forecasting.tide_model import TiDEModel
    from darts.models.forecasting.transformer_model import TransformerModel
except ModuleNotFoundError:
    logger.warning(
        "Support for Torch based models not available. "
        'To enable them, install "darts", "u8darts[torch]" or "u8darts[all]" (with pip); '
        'or "u8darts-torch" or "u8darts-all" (with conda).'
    )

try:
    from darts.models.forecasting.lgbm import LightGBMModel
except ModuleNotFoundError:
    LightGBMModel = NotImportedModule(module_name="LightGBM", warn=False)

try:
    from darts.models.forecasting.prophet_model import Prophet
except ImportError:
    Prophet = NotImportedModule(module_name="Prophet", warn=False)

try:
    from darts.models.forecasting.catboost_model import CatBoostModel
except ModuleNotFoundError:
    CatBoostModel = NotImportedModule(module_name="CatBoost", warn=False)

try:
    from darts.models.forecasting.croston import Croston
    from darts.models.forecasting.sf_auto_arima import StatsForecastAutoARIMA
    from darts.models.forecasting.sf_auto_ces import StatsForecastAutoCES
    from darts.models.forecasting.sf_auto_ets import StatsForecastAutoETS
    from darts.models.forecasting.sf_auto_theta import StatsForecastAutoTheta

except ImportError:
    logger.warning(
        "The StatsForecast module could not be imported. "
        "To enable support for the StatsForecastAutoARIMA, "
        "StatsForecastAutoETS and Croston models, please consider "
        "installing it."
    )
    Croston = NotImportedModule(module_name="StatsForecast", warn=False)
    StatsForecastAutoARIMA = NotImportedModule(module_name="StatsForecast", warn=False)
    StatsForecastAutoCES = NotImportedModule(module_name="StatsForecast", warn=False)
    StatsForecastAutoETS = NotImportedModule(module_name="StatsForecast", warn=False)
    StatsForecastAutoTheta = NotImportedModule(module_name="StatsForecast", warn=False)

try:
    from darts.models.forecasting.xgboost import XGBModel
except ImportError:
    XGBModel = NotImportedModule(module_name="XGBoost")

from darts.models.filtering.gaussian_process_filter import GaussianProcessFilter
from darts.models.filtering.kalman_filter import KalmanFilter

# Filtering
from darts.models.filtering.moving_average_filter import MovingAverageFilter
from darts.models.forecasting.baselines import NaiveEnsembleModel

# Ensembling
from darts.models.forecasting.ensemble_model import EnsembleModel
