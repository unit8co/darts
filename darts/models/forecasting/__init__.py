"""
Forecasting Models
------------------

Baseline Models (`LocalForecastingModel <https://unit8co.github.io/darts/userguide/covariates.html#local-forecasting-models-lfms>`_)
    - :class:`~darts.models.forecasting.baselines.NaiveMean`
    - :class:`~darts.models.forecasting.baselines.NaiveSeasonal`
    - :class:`~darts.models.forecasting.baselines.NaiveDrift`
    - :class:`~darts.models.forecasting.baselines.NaiveMovingAverage`
Global Baseline Models (`GlobalForecastingModel <https://unit8co.github.io/darts/userguide/covariates.html#global-forecasting-models-gfms>`_)
    - :class:`~darts.models.forecasting.global_baseline_models.GlobalNaiveAggregate`
    - :class:`~darts.models.forecasting.global_baseline_models.GlobalNaiveDrift`
    - :class:`~darts.models.forecasting.global_baseline_models.GlobalNaiveSeasonal`
Statistical Models (`LocalForecastingModel <https://unit8co.github.io/darts/userguide/covariates.html#local-forecasting-models-lfms>`_)
    - :class:`~darts.models.forecasting.arima.ARIMA`
    - :class:`~darts.models.forecasting.varima.VARIMA`
    - :class:`~darts.models.forecasting.auto_arima.AutoARIMA`
    - :class:`~darts.models.forecasting.sf_auto_arima.StatsForecastAutoARIMA`
    - :class:`~darts.models.forecasting.exponential_smoothing.ExponentialSmoothing`
    - :class:`~darts.models.forecasting.sf_auto_ets.StatsForecastAutoETS`
    - :class:`~darts.models.forecasting.sf_auto_ces.StatsForecastAutoCES`
    - :class:`~darts.models.forecasting.tbats_model.BATS`
    - :class:`~darts.models.forecasting.tbats_model.TBATS`
    - :class:`~darts.models.forecasting.sf_auto_tbats.StatsForecastAutoTBATS`
    - :class:`~darts.models.forecasting.theta.Theta`
    - :class:`~darts.models.forecasting.theta.FourTheta`
    - :class:`~darts.models.forecasting.sf_auto_theta.StatsForecastAutoTheta`
    - :class:`~darts.models.forecasting.prophet_model.Prophet`
    - :class:`~Fast Fourier Transform) <darts.models.forecasting.fft.FFT`
    - :class:`~darts.models.forecasting.kalman_forecaster.KalmanForecaster`
    - :class:`~darts.models.forecasting.croston.Croston`
Regression Models (`GlobalForecastingModel <https://unit8co.github.io/darts/userguide/covariates.html#global-forecasting-models-gfms>`_)
    - :class:`~darts.models.forecasting.regression_model.RegressionModel`
    - :class:`~darts.models.forecasting.linear_regression_model.LinearRegressionModel`
    - :class:`~darts.models.forecasting.random_forest.RandomForest`
    - :class:`~darts.models.forecasting.lgbm.LightGBMModel`
    - :class:`~darts.models.forecasting.xgboost.XGBModel`
    - :class:`~darts.models.forecasting.catboost_model.CatBoostModel`
PyTorch (Lightning)-based Models (`GlobalForecastingModel <https://unit8co.github.io/darts/userguide/covariates.html#global-forecasting-models-gfms>`_)
    - :class:`~darts.models.forecasting.rnn_model.RNNModel`
    - :class:`~darts.models.forecasting.block_rnn_model.BlockRNNModel`
    - :class:`~darts.models.forecasting.nbeats.NBEATSModel`
    - :class:`~darts.models.forecasting.nhits.NHiTSModel`
    - :class:`~darts.models.forecasting.tcn_model.TCNModel`
    - :class:`~darts.models.forecasting.transformer_model.TransformerModel`
    - :class:`~darts.models.forecasting.tft_model.TFTModel`
    - :class:`~darts.models.forecasting.dlinear.DLinearModel`
    - :class:`~darts.models.forecasting.nlinear.NLinearModel`
    - :class:`~darts.models.forecasting.tide_model.TiDEModel`
    - :class:`~darts.models.forecasting.tsmixer_model.TSMixerModel`
Ensemble Models (`GlobalForecastingModel <https://unit8co.github.io/darts/userguide/covariates.html#global-forecasting-models-gfms>`_)
    - :class:`~darts.models.forecasting.baselines.NaiveEnsembleModel`
    - :class:`~darts.models.forecasting.regression_ensemble_model.RegressionEnsembleModel`
Conformal Models  (`GlobalForecastingModel <https://unit8co.github.io/darts/userguide/covariates.html#global-forecasting-models-gfms>`_)
    - :class:`~darts.models.forecasting.conformal_models.ConformalNaiveModel`
    - :class:`~darts.models.forecasting.conformal_models.ConformalQRModel`
"""
