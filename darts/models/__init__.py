"""
Models
------

A comprehensive collection of forecasting and filtering models, including baseline models
(NaiveSeasonal, NaiveMovingAverage, ...), statistical models (ARIMA, exponential smoothing, ...),
machine learning models (LightGBM, CatBoost, sklearn-based, ...), neural network models (RNN,
N-BEATS, TiDE...), and foundation models (Chronos-2, TimesFM 2.5).
"""

from darts.utils._lazy import setup_lazy_imports

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
