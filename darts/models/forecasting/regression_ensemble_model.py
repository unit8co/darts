"""
Regression ensemble model
-------------------------

An ensemble model which uses a regression model to compute the ensemble forecast.
"""
from typing import Optional, List, Union, Sequence, Tuple
from darts.timeseries import TimeSeries
from darts.logging import get_logger, raise_if

from darts.models.forecasting.forecasting_model import ForecastingModel, GlobalForecastingModel
from darts.models.forecasting.ensemble_model import EnsembleModel
from darts.models.forecasting.linear_regression_model import LinearRegressionModel
from darts.models.forecasting.regression_model import RegressionModel

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

    def _split_multi_ts_sequence(self, n: int, ts_sequence: Sequence[TimeSeries]
                                 ) -> Tuple[Sequence[TimeSeries], Sequence[TimeSeries]]:
        left = [ts[:-n] for ts in ts_sequence]
        right = [ts[-n:] for ts in ts_sequence]
        return left, right

    def fit(self,
            series: Union[TimeSeries, Sequence[TimeSeries]],
            past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
            future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
            ) -> None:

        super().fit(series, past_covariates=past_covariates, future_covariates=future_covariates)

        # spare train_n_points points to serve as regression target
        if self.is_single_series:
            train_n_points_too_big = len(self.training_series) <= self.train_n_points
        else:
            train_n_points_too_big = any([len(s) <= self.train_n_points for s in series])

        raise_if(train_n_points_too_big,
                 "regression_train_n_points parameter too big (must be smaller or equal to the number of points in "
                 "training_series)", logger)

        if self.is_single_series:
            forecast_training = self.training_series[:-self.train_n_points]
            regression_target = self.training_series[-self.train_n_points:]
        else:
            forecast_training, regression_target = self._split_multi_ts_sequence(self.train_n_points, series)

        for model in self.models:
            model._fit_wrapper(
                series=forecast_training,
                past_covariates=past_covariates,
                future_covariates=future_covariates
            )

        predictions = self._make_multiple_predictions(
            n=self.train_n_points,
            series=forecast_training,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            num_samples=1
        )

        # train the regression model on the individual models' predictions
        self.regression_model.fit(series=regression_target, future_covariates=predictions)

        # prepare the forecasting models for further predicting by fitting them with the entire data

        # Some models (incl. Neural-Network based models) may need to be 'reset' to allow being retrained from scratch
        self.models = [model.untrained_model() for model in self.models]

        for model in self.models:
            if self.is_global_ensemble:
                kwargs = dict(series=series)
                if model.uses_past_covariates:
                    kwargs['past_covariates'] = past_covariates
                if model.uses_future_covariates:
                    kwargs['future_covariates'] = future_covariates
                model.fit(**kwargs)

            else:
                model.fit(self.training_series)

    def ensemble(self,
                 predictions: Union[TimeSeries, Sequence[TimeSeries]],
                 series: Optional[Sequence[TimeSeries]] = None) -> Union[TimeSeries, Sequence[TimeSeries]]:
        if self.is_single_series:
            predictions = [predictions]
            series = [series]

        ensembled = [self.regression_model.predict(n=len(prediction), series=serie, future_covariates=prediction)
                     for serie, prediction in zip(series, predictions)]

        return ensembled[0] if self.is_single_series else ensembled
