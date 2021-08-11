"""
Regression ensemble model
-------------------------

An ensemble model which uses a regression model to compute the ensemble forecast.
"""
from typing import List
from darts.timeseries import TimeSeries
from darts.logging import get_logger, raise_if
from darts.models.forecasting_model import ForecastingModel
from darts.models import EnsembleModel, LinearRegressionModel, RegressionModel

logger = get_logger(__name__)


class RegressionEnsembleModel(EnsembleModel):
    def __init__(self,
                 forecasting_models: List[ForecastingModel],
                 regression_train_n_points: int,
                 regression_model=None):
        """
        Class for ensemble models using a regression model for ensembling individual models' predictions.
        The provided regression model must implement fit() and predict() methods
        (e.g. scikit-learn regression models). Note that here the regression model is used to learn how to
        best ensemble the individual forecasting models' forecasts. It is not the same usage of regression
        as in `RegressionModel`, where the regression model is used to produce forecasts based on the
        lagged series.

        Parameters
        ----------
        forecasting_models
            List of forecasting models whose predictions to ensemble
        regression_train_n_points
            The number of points to use to train the regression model
        regression_model
            Any regression model with predict() and fit() methods (e.g. from scikit-learn)
            Default: `darts.model.LinearRegressionModel(fit_intercept=False)`
        """
        super().__init__(forecasting_models)
        if regression_model is None:
            regression_model = LinearRegressionModel(lags=None, lags_future_covariates=[0], fit_intercept=False)
        elif isinstance(regression_model, RegressionModel):
            regression_model = regression_model
        else:
            # scikit-learn like model
            regression_model = RegressionModel(lags_future_covariates=[0], model=regression_model)

        raise_if(
            regression_model.lags is not None and regression_model.lags_historical_covariates is not None
            and regression_model.lags_past_covariates is not None and regression_model.lags_future_covariates != [0],
            (f"`lags`, `lags_historical_covariates` and `lags_past_covariates` of regression model must be `None`"
             f"and `lags_future_covariates` must be [0]. Given:\n`lags`: {regression_model.lags},"
             f"`lags_historical_covariates`: {regression_model.lags_historical_covariates},"
             f"`lags_past_covariates`: {regression_model.lags} and `lags_future_covariates`"
             f"{regression_model.lags_future_covariates}.")
        )

        self.regression_model = regression_model
        self.train_n_points = regression_train_n_points

    def fit(self, series: TimeSeries) -> None:
        super().fit(series)

        # spare train_n_points points to serve as regression target
        raise_if(len(self.training_series) <= self.train_n_points,
                 "regression_train_n_points parameter too big (must be smaller or equal to the number of points in "
                 "training_series)", logger)

        forecast_training = self.training_series[:-self.train_n_points]
        regression_target = self.training_series[-self.train_n_points:]

        # fit the forecasting models
        for model in self.models:
            model.fit(forecast_training)

        # predict train_n_points points for each model
        predictions = self.models[0].predict(self.train_n_points)
        for model in self.models[1:]:
            predictions = predictions.stack(model.predict(self.train_n_points))

        # train the regression model on the individual models' predictions
        self.regression_model.fit(series=regression_target, future_covariates=predictions)

        # prepare the forecasting models for further predicting by fitting
        # them with the entire data

        # Some models (incl. Neural-Network based models) may need to be 'reset'
        # to allow being retrained from scratch
        self.models = [model.untrained_model() if hasattr(model, "untrained_model") else model
                       for model in self.models]

        # fit the forecasting models
        for model in self.models:
            model.fit(self.training_series)

    def ensemble(self, predictions: TimeSeries) -> TimeSeries:
        return self.regression_model.predict(n=len(predictions), future_covariates=predictions)
