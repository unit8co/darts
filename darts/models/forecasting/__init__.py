"""
Forecasting Models
------------------

Baseline Models (`LocalForecastingModel <https://unit8co.github.io/darts/userguide/covariates.html#local-forecasting-models-lfms>`_)
    - :class:`NaiveMean <darts.models.forecasting.baselines.NaiveMean>`
    - :class:`NaiveSeasonal <darts.models.forecasting.baselines.NaiveSeasonal>`
    - :class:`NaiveDrift <darts.models.forecasting.baselines.NaiveDrift>`
    - :class:`NaiveMovingAverage <darts.models.forecasting.baselines.NaiveMovingAverage>`
Statistical Models (`LocalForecastingModel <https://unit8co.github.io/darts/userguide/covariates.html#local-forecasting-models-lfms>`_)
    - :class:`ARIMA <darts.models.forecasting.arima.ARIMA>`
    - :class:`VARIMA <darts.models.forecasting.varima.VARIMA>`
    - :class:`AutoARIMA <darts.models.forecasting.auto_arima.AutoARIMA>`
    - :class:`StatsForecastAutoARIMA <darts.models.forecasting.sf_auto_arima.StatsForecastAutoARIMA>`
    - :class:`ExponentialSmoothing <darts.models.forecasting.exponential_smoothing.ExponentialSmoothing>`
    - :class:`StatsForecastAutoETS <darts.models.forecasting.sf_auto_ets.StatsForecastAutoETS>`
    - :class:`StatsForecastAutoCES <darts.models.forecasting.sf_auto_ces.StatsForecastAutoCES>`
    - :class:`BATS <darts.models.forecasting.tbats_model.BATS>`
    - :class:`TBATS <darts.models.forecasting.tbats_model.TBATS>`
    - :class:`Theta <darts.models.forecasting.theta.Theta>`
    - :class:`FourTheta <darts.models.forecasting.theta.FourTheta>`
    - :class:`StatsForecastAutoTheta <darts.models.forecasting.sf_auto_theta.StatsForecastAutoTheta>`
    - :class:`Prophet <darts.models.forecasting.prophet_model.Prophet>`
    - :class:`FFT (Fast Fourier Transform) <darts.models.forecasting.fft.FFT>`
    - :class:`KalmanForecaster <darts.models.forecasting.kalman_forecaster.KalmanForecaster>`
    - :class:`Croston <darts.models.forecasting.croston.Croston>`
Regression Models (`GlobalForecastingModel <https://unit8co.github.io/darts/userguide/covariates.html#global-forecasting-models-gfms>`_)
    - :class:`RegressionModel <darts.models.forecasting.regression_model.RegressionModel>`
    - :class:`LinearRegressionModel <darts.models.forecasting.linear_regression_model.LinearRegressionModel>`
    - :class:`RandomForest <darts.models.forecasting.random_forest.RandomForest>`
    - :class:`LightGBMModel <darts.models.forecasting.lgbm.LightGBMModel>`
    - :class:`XGBModel <darts.models.forecasting.xgboost.XGBModel>`
    - :class:`CatBoostModel <darts.models.forecasting.catboost_model.CatBoostModel>`
PyTorch (Lightning)-based Models (`GlobalForecastingModel <https://unit8co.github.io/darts/userguide/covariates.html#global-forecasting-models-gfms>`_)
    - :class:`RNNModel <darts.models.forecasting.rnn_model.RNNModel>`
    - :class:`BlockRNNModel <darts.models.forecasting.block_rnn_model.BlockRNNModel>`
    - :class:`NBEATSModel <darts.models.forecasting.nbeats.NBEATSModel>`
    - :class:`NHiTSModel <darts.models.forecasting.nhits.NHiTSModel>`
    - :class:`TCNModel <darts.models.forecasting.tcn_model.TCNModel>`
    - :class:`TransformerModel <darts.models.forecasting.transformer_model.TransformerModel>`
    - :class:`TFTModel <darts.models.forecasting.tft_model.TFTModel>`
    - :class:`DLinearModel <darts.models.forecasting.dlinear.DLinearModel>`
    - :class:`NLinearModel <darts.models.forecasting.nlinear.NLinearModel>`
    - :class:`TiDEModel <darts.models.forecasting.tide_model.TiDEModel>`
Ensemble Models (`GlobalForecastingModel <https://unit8co.github.io/darts/userguide/covariates.html#global-forecasting-models-gfms>`_)
    - :class:`NaiveEnsembleModel <darts.models.forecasting.baselines.NaiveEnsembleModel>`
    - :class:`RegressionEnsembleModel <darts.models.forecasting.regression_ensemble_model.RegressionEnsembleModel>`
"""
