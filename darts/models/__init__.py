"""
Models
------
"""

from darts.logging import get_logger

logger = get_logger(__name__)

# Forecasting
from darts.models.forecasting.arima import ARIMA
from darts.models.forecasting.auto_arima import AutoARIMA
from darts.models.forecasting.baselines import NaiveDrift, NaiveMean, NaiveSeasonal
from darts.models.forecasting.exponential_smoothing import ExponentialSmoothing
from darts.models.forecasting.fft import FFT
from darts.models.forecasting.kalman_forecaster import KalmanForecaster
from darts.models.forecasting.linear_regression_model import LinearRegressionModel
from darts.models.forecasting.random_forest import RandomForest
from darts.models.forecasting.regression_ensemble_model import RegressionEnsembleModel
from darts.models.forecasting.regression_model import RegressionModel
from darts.models.forecasting.tbats import BATS, TBATS
from darts.models.forecasting.theta import FourTheta, Theta
from darts.models.forecasting.varima import VARIMA

try:
    from darts.models.forecasting.block_rnn_model import BlockRNNModel
    from darts.models.forecasting.dlinear import DLinearModel
    from darts.models.forecasting.nbeats import NBEATSModel
    from darts.models.forecasting.nhits import NHiTSModel
    from darts.models.forecasting.nlinear import NLinearModel
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

try:
    from darts.models.forecasting.lgbm import LightGBMModel
except ModuleNotFoundError:
    logger.warning(
        "Support for LightGBM not available. "
        "To enable LightGBM support in Darts, follow the detailed "
        "install instructions for LightGBM in the README: "
        "https://github.com/unit8co/darts/blob/master/INSTALL.md"
    )
    # TODO: simpler option would be to write LightGBMModel=None
    # Prevent a second ImportError that would interrupt the import
    class NotImportedLightGBM:
        usable = False

    LightGBMModel = NotImportedLightGBM()

try:
    from darts.models.forecasting.prophet_model import Prophet
except ImportError:
    logger.warning(
        "The prophet module could not be imported. "
        "To enable support for Prophet model, follow "
        "the instruction in the README: "
        "https://github.com/unit8co/darts/blob/master/INSTALL.md"
    )

    class NotImportedProphet:
        usable = False

    Prophet = NotImportedProphet()

try:
    from darts.models.forecasting.catboost_model import CatBoostModel
except ModuleNotFoundError:
    logger.warning(
        "The catboost module could not be imported. "
        "To enable support for CatBoostModel model, "
        "follow the instruction in the README: "
        "https://github.com/unit8co/darts/blob/master/INSTALL.md"
    )

    class NotImportedCatBoostModel:
        usable = False

    CatBoostModel = NotImportedCatBoostModel()

try:
    from darts.models.forecasting.croston import Croston
    from darts.models.forecasting.sf_auto_arima import StatsForecastAutoARIMA
    from darts.models.forecasting.sf_ets import StatsForecastETS
except ImportError:
    logger.warning(
        "The statsforecast module could not be imported. "
        "To enable support for the StatsForecastAutoARIMA, "
        "StatsForecastETS and Croston models, please consider "
        "installing it."
    )

    class NotImportedStatsForecastAutoARIMA:
        usable = False

    StatsForecastAutoARIMA = NotImportedStatsForecastAutoARIMA()

    class NotImportedStatsForecastETS:
        usable = False

    StatsForecastETS = NotImportedStatsForecastETS()

    class NotImportedCroston:
        usable = False

    Croston = NotImportedCroston()

try:
    from darts.models.forecasting.xgboost import XGBModel
except ImportError:
    logger.warning(
        "The xgboost module could not be imported. "
        "To enable support for XGBoost model, install the xgboost package."
    )

    class NotImportedXGBModel:
        usable = False

    XGBModel = NotImportedXGBModel()

from darts.models.filtering.gaussian_process_filter import GaussianProcessFilter
from darts.models.filtering.kalman_filter import KalmanFilter

# Filtering
from darts.models.filtering.moving_average import MovingAverage
from darts.models.forecasting.baselines import NaiveEnsembleModel

# Ensembling
from darts.models.forecasting.ensemble_model import EnsembleModel
