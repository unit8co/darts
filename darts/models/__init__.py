"""
Models
------

A comprehensive collection of forecasting and filtering models, including baseline models
(NaiveSeasonal, NaiveMovingAverage, ...), statistical models (ARIMA, exponential smoothing, ...),
machine learning models (LightGBM, CatBoost, sklearn-based, ...), neural network models (RNN,
N-BEATS, TiDE...), and foundation models (Chronos-2, TimesFM 2.5).
"""

from typing import TYPE_CHECKING

from darts.utils._lazy import setup_lazy_imports

if TYPE_CHECKING:
    from darts.models.filtering.gaussian_process_filter import (
        GaussianProcessFilter as GaussianProcessFilter,
    )
    from darts.models.filtering.kalman_filter import KalmanFilter as KalmanFilter
    from darts.models.filtering.moving_average_filter import (
        MovingAverageFilter as MovingAverageFilter,
    )
    from darts.models.forecasting.arima import ARIMA as ARIMA
    from darts.models.forecasting.baselines import NaiveDrift as NaiveDrift
    from darts.models.forecasting.baselines import NaiveMean as NaiveMean
    from darts.models.forecasting.baselines import (
        NaiveMovingAverage as NaiveMovingAverage,
    )
    from darts.models.forecasting.baselines import NaiveSeasonal as NaiveSeasonal
    from darts.models.forecasting.block_rnn_model import BlockRNNModel as BlockRNNModel
    from darts.models.forecasting.catboost_model import (
        CatBoostClassifierModel as CatBoostClassifierModel,
    )
    from darts.models.forecasting.catboost_model import CatBoostModel as CatBoostModel
    from darts.models.forecasting.chronos2_model import Chronos2Model as Chronos2Model
    from darts.models.forecasting.conformal_models import (
        ConformalNaiveModel as ConformalNaiveModel,
    )
    from darts.models.forecasting.conformal_models import (
        ConformalQRModel as ConformalQRModel,
    )
    from darts.models.forecasting.dlinear import DLinearModel as DLinearModel
    from darts.models.forecasting.ensemble_model import EnsembleModel as EnsembleModel
    from darts.models.forecasting.exponential_smoothing import (
        ExponentialSmoothing as ExponentialSmoothing,
    )
    from darts.models.forecasting.fft import FFT as FFT
    from darts.models.forecasting.global_baseline_models import (
        GlobalNaiveAggregate as GlobalNaiveAggregate,
    )
    from darts.models.forecasting.global_baseline_models import (
        GlobalNaiveDrift as GlobalNaiveDrift,
    )
    from darts.models.forecasting.global_baseline_models import (
        GlobalNaiveSeasonal as GlobalNaiveSeasonal,
    )
    from darts.models.forecasting.kalman_forecaster import (
        KalmanForecaster as KalmanForecaster,
    )
    from darts.models.forecasting.lgbm import (
        LightGBMClassifierModel as LightGBMClassifierModel,
    )
    from darts.models.forecasting.lgbm import LightGBMModel as LightGBMModel
    from darts.models.forecasting.linear_regression_model import (
        LinearRegressionModel as LinearRegressionModel,
    )
    from darts.models.forecasting.naive_ensemble_model import (
        NaiveEnsembleModel as NaiveEnsembleModel,
    )
    from darts.models.forecasting.nbeats import NBEATSModel as NBEATSModel
    from darts.models.forecasting.nf_model import (
        NeuralForecastModel as NeuralForecastModel,
    )
    from darts.models.forecasting.nhits import NHiTSModel as NHiTSModel
    from darts.models.forecasting.nlinear import NLinearModel as NLinearModel
    from darts.models.forecasting.prophet_model import Prophet as Prophet
    from darts.models.forecasting.random_forest import RandomForest as RandomForest
    from darts.models.forecasting.random_forest import (
        RandomForestModel as RandomForestModel,
    )
    from darts.models.forecasting.regression_ensemble_model import (
        RegressionEnsembleModel as RegressionEnsembleModel,
    )
    from darts.models.forecasting.rnn_model import RNNModel as RNNModel
    from darts.models.forecasting.sf_auto_arima import AutoARIMA as AutoARIMA
    from darts.models.forecasting.sf_auto_ces import AutoCES as AutoCES
    from darts.models.forecasting.sf_auto_ets import AutoETS as AutoETS
    from darts.models.forecasting.sf_auto_mfles import AutoMFLES as AutoMFLES
    from darts.models.forecasting.sf_auto_tbats import AutoTBATS as AutoTBATS
    from darts.models.forecasting.sf_auto_theta import AutoTheta as AutoTheta
    from darts.models.forecasting.sf_croston import Croston as Croston
    from darts.models.forecasting.sf_model import (
        StatsForecastModel as StatsForecastModel,
    )
    from darts.models.forecasting.sf_tbats import TBATS as TBATS
    from darts.models.forecasting.sklearn_model import (
        RegressionModel as RegressionModel,
    )
    from darts.models.forecasting.sklearn_model import (
        SKLearnClassifierModel as SKLearnClassifierModel,
    )
    from darts.models.forecasting.sklearn_model import SKLearnModel as SKLearnModel
    from darts.models.forecasting.tcn_model import TCNModel as TCNModel
    from darts.models.forecasting.tft_model import TFTModel as TFTModel
    from darts.models.forecasting.theta import FourTheta as FourTheta
    from darts.models.forecasting.theta import Theta as Theta
    from darts.models.forecasting.tide_model import TiDEModel as TiDEModel
    from darts.models.forecasting.timesfm2p5_model import (
        TimesFM2p5Model as TimesFM2p5Model,
    )
    from darts.models.forecasting.transformer_model import (
        TransformerModel as TransformerModel,
    )
    from darts.models.forecasting.tsmixer_model import TSMixerModel as TSMixerModel
    from darts.models.forecasting.varima import VARIMA as VARIMA
    from darts.models.forecasting.xgboost import (
        XGBClassifierModel as XGBClassifierModel,
    )
    from darts.models.forecasting.xgboost import XGBModel as XGBModel

# mapping of public name -> (module_path, optional_dependency_name | None);
# when optional_dependency_name is not None, a missing dependency yields a
# ``NotImportedModule`` stub instead of propagating the ImportError
_LAZY_IMPORTS: dict[str, tuple[str, str | None]] = {
    # --- Forecasting: always-available models ---
    "ARIMA": ("darts.models.forecasting.arima", None),
    "NaiveDrift": ("darts.models.forecasting.baselines", None),
    "NaiveMean": ("darts.models.forecasting.baselines", None),
    "NaiveMovingAverage": ("darts.models.forecasting.baselines", None),
    "NaiveSeasonal": ("darts.models.forecasting.baselines", None),
    "NaiveEnsembleModel": ("darts.models.forecasting.naive_ensemble_model", None),
    "ConformalNaiveModel": ("darts.models.forecasting.conformal_models", None),
    "ConformalQRModel": ("darts.models.forecasting.conformal_models", None),
    "EnsembleModel": ("darts.models.forecasting.ensemble_model", None),
    "ExponentialSmoothing": ("darts.models.forecasting.exponential_smoothing", None),
    "FFT": ("darts.models.forecasting.fft", None),
    "KalmanForecaster": ("darts.models.forecasting.kalman_forecaster", None),
    "LinearRegressionModel": ("darts.models.forecasting.linear_regression_model", None),
    "RandomForest": ("darts.models.forecasting.random_forest", None),
    "RandomForestModel": ("darts.models.forecasting.random_forest", None),
    "RegressionEnsembleModel": (
        "darts.models.forecasting.regression_ensemble_model",
        None,
    ),
    "RegressionModel": ("darts.models.forecasting.sklearn_model", None),
    "SKLearnClassifierModel": ("darts.models.forecasting.sklearn_model", None),
    "SKLearnModel": ("darts.models.forecasting.sklearn_model", None),
    "FourTheta": ("darts.models.forecasting.theta", None),
    "Theta": ("darts.models.forecasting.theta", None),
    "VARIMA": ("darts.models.forecasting.varima", None),
    # --- Forecasting: LightGBM (import first to avoid segfault) ---
    "LightGBMModel": ("darts.models.forecasting.lgbm", "LightGBM"),
    "LightGBMClassifierModel": ("darts.models.forecasting.lgbm", "LightGBM"),
    # --- Forecasting: Torch-based models ---
    "BlockRNNModel": ("darts.models.forecasting.block_rnn_model", "(Py)Torch"),
    "DLinearModel": ("darts.models.forecasting.dlinear", "(Py)Torch"),
    "GlobalNaiveAggregate": (
        "darts.models.forecasting.global_baseline_models",
        "(Py)Torch",
    ),
    "GlobalNaiveDrift": (
        "darts.models.forecasting.global_baseline_models",
        "(Py)Torch",
    ),
    "GlobalNaiveSeasonal": (
        "darts.models.forecasting.global_baseline_models",
        "(Py)Torch",
    ),
    "NBEATSModel": ("darts.models.forecasting.nbeats", "(Py)Torch"),
    "NHiTSModel": ("darts.models.forecasting.nhits", "(Py)Torch"),
    "NLinearModel": ("darts.models.forecasting.nlinear", "(Py)Torch"),
    "RNNModel": ("darts.models.forecasting.rnn_model", "(Py)Torch"),
    "TCNModel": ("darts.models.forecasting.tcn_model", "(Py)Torch"),
    "TFTModel": ("darts.models.forecasting.tft_model", "(Py)Torch"),
    "TiDEModel": ("darts.models.forecasting.tide_model", "(Py)Torch"),
    "TransformerModel": ("darts.models.forecasting.transformer_model", "(Py)Torch"),
    "TSMixerModel": ("darts.models.forecasting.tsmixer_model", "(Py)Torch"),
    # --- Forecasting: Foundation models (Torch-based) ---
    "Chronos2Model": ("darts.models.forecasting.chronos2_model", "(Py)Torch"),
    "TimesFM2p5Model": ("darts.models.forecasting.timesfm2p5_model", "(Py)Torch"),
    # --- Forecasting: NeuralForecast ---
    "NeuralForecastModel": ("darts.models.forecasting.nf_model", "NeuralForecast"),
    # --- Forecasting: Prophet ---
    "Prophet": ("darts.models.forecasting.prophet_model", "Prophet"),
    # --- Forecasting: CatBoost ---
    "CatBoostModel": ("darts.models.forecasting.catboost_model", "CatBoost"),
    "CatBoostClassifierModel": ("darts.models.forecasting.catboost_model", "CatBoost"),
    # --- Forecasting: StatsForecast ---
    "AutoARIMA": ("darts.models.forecasting.sf_auto_arima", "StatsForecast"),
    "AutoCES": ("darts.models.forecasting.sf_auto_ces", "StatsForecast"),
    "AutoETS": ("darts.models.forecasting.sf_auto_ets", "StatsForecast"),
    "AutoMFLES": ("darts.models.forecasting.sf_auto_mfles", "StatsForecast"),
    "AutoTBATS": ("darts.models.forecasting.sf_auto_tbats", "StatsForecast"),
    "AutoTheta": ("darts.models.forecasting.sf_auto_theta", "StatsForecast"),
    "Croston": ("darts.models.forecasting.sf_croston", "StatsForecast"),
    "StatsForecastModel": ("darts.models.forecasting.sf_model", "StatsForecast"),
    "TBATS": ("darts.models.forecasting.sf_tbats", "StatsForecast"),
    # --- Forecasting: XGBoost ---
    "XGBModel": ("darts.models.forecasting.xgboost", "XGBoost"),
    "XGBClassifierModel": ("darts.models.forecasting.xgboost", "XGBoost"),
    # --- Filtering ---
    "GaussianProcessFilter": ("darts.models.filtering.gaussian_process_filter", None),
    "KalmanFilter": ("darts.models.filtering.kalman_filter", None),
    "MovingAverageFilter": ("darts.models.filtering.moving_average_filter", None),
}

__all__, __getattr__, __dir__ = setup_lazy_imports(_LAZY_IMPORTS, __name__, globals())
