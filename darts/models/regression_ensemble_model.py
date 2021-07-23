"""
Regression ensemble model
-------------------------
"""
from typing import Optional, List, Union, Sequence, Tuple

from darts.timeseries import TimeSeries
from darts.logging import get_logger, raise_if
from darts.models.forecasting_model import ForecastingModel, GlobalForecastingModel
from darts.models import (
    EnsembleModel, LinearRegressionModel, RegressionModel,
    RandomForest
)

logger = get_logger(__name__)


class RegressionEnsembleModel(EnsembleModel):
    def __init__(self,
                 forecasting_models: Union[List[ForecastingModel], List[GlobalForecastingModel]],
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
            regression_model = LinearRegressionModel(lags_exog=0, fit_intercept=False)

        regression_model = RegressionModel(lags_exog=0, model=regression_model)
        raise_if(regression_model.lags is not None and regression_model.lags_exog != [0], (
            "`lags` of regression model must be `None` and `lags_exog` must be [0]. Given: {} and {}"
            .format(regression_model.lags, regression_model.lags_exog)
            )
        )
        self.regression_model = regression_model
        self.train_n_points = regression_train_n_points

    def _split_multi_ts_sequence(self, n: int, ts_sequence: Sequence[TimeSeries]
                                 ) -> Tuple[Sequence[TimeSeries], Sequence[TimeSeries]]:
        left = [ts[:-n] for ts in ts_sequence]
        right = [ts[-n:] for ts in ts_sequence]
        return left, right

    def fit(self,
            series: Union[TimeSeries, Sequence[TimeSeries]],
            covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None
            ) -> None:

        super().fit(series, covariates)

        # spare train_n_points points to serve as regression target
        if isinstance(series, TimeSeries):
            train_n_points_too_big = len(self.training_series) <= self.train_n_points
        else:
            train_n_points_too_big = any([len(s) <= self.train_n_points for s in series])

        raise_if(train_n_points_too_big,
                 "regression_train_n_points parameter too big (must be smaller or equal" +
                 " to the number of points in training_series)",
                 logger)

        if isinstance(series, TimeSeries):
            forecast_training = self.training_series[:-self.train_n_points]
            regression_target = self.training_series[-self.train_n_points:]
        else:
            forecast_training, regression_target = self._split_multi_ts_sequence(self.train_n_points, series)

        if covariates is not None:
            if isinstance(covariates, TimeSeries):
                forecast_covariates = self.covariate_series[:-self.train_n_points]
                regression_covariates = self.covariate_series[-self.train_n_points:]  # TODO do we need it at all?
            else:
                forecast_covariates, regression_covariates = \
                    self._split_multi_ts_sequence(-self.train_n_points, covariates)  # TODO do we need it at all?
        else:
            forecast_covariates=None
            regression_covariates=None  # TODO do we need it at all?

        # fit the forecasting models
        for model in self.models:
            if self.is_global_ensemble:
                model.fit(forecast_training, forecast_covariates)
            else:
                model.fit(forecast_training)

        # predict train_n_points points for each model
        if self.is_global_ensemble:
            predictions = self._ts_sequence_to_multivariate_ts(
                self.models[0].predict(n=self.train_n_points, series=forecast_training, covariates=forecast_covariates))
        else:
            predictions = self.models[0].predict(self.train_n_points)

        if len(self.models) > 1:
            for model in self.models[1:]:
                if self.is_global_ensemble:
                    prediction = self._ts_sequence_to_multivariate_ts(
                        model.predict(n=self.train_n_points, series=forecast_training, covariates=forecast_covariates))
                else:
                    prediction = model.predict(self.train_n_points)

                predictions = predictions.stack(prediction)

        # train the regression model on the individual models' predictions
        self.regression_model.fit(series=regression_target, exog=predictions)

        # prepare the forecasting models for further predicting by fitting
        # them with the entire data

        # Some models (incl. Neural-Network based models) may need to be 'reset'
        # to allow being retrained from scratch
        self.models = [model.untrained_model() if hasattr(model, 'untrained_model') else model
                       for model in self.models]

        # fit the forecasting models
        for model in self.models:
            model.fit(self.training_series)

    def ensemble(self, predictions: TimeSeries) -> TimeSeries:
        return self.regression_model.predict(n=len(predictions), exog=predictions)
