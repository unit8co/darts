"""
Regression ensemble model
-------------------------
"""
from sklearn.linear_model import LinearRegression
from typing import Optional, List

from darts.timeseries import TimeSeries
from darts.models import EnsembleModel, StandardRegressionModel
from darts.models.forecasting_model import ForecastingModel
from darts.logging import get_logger, raise_if

logger = get_logger(__name__)


class RegressionEnsembleModel(EnsembleModel):
    def __init__(self,
                 forecasting_models: List[ForecastingModel],
                 regression_train_n_points: int,
                 regression_model=LinearRegression(n_jobs=-1, fit_intercept=False)):
        """
        Class for ensemble models using a regression model for ensembling individual models' predictions
        The provided regression model must implement fit() and predict() methods
        (e.g. scikit-learn regression models)

        Parameters
        ----------
        forecasting_models
            List of forecasting models whose predictions to ensemble
        regression_train_n_points
            The number of points to use to train the regression model
        regression_model
            Any regression model with predict() and fit() methods (e.g. from scikit-learn)
            Default: `sklearn.linear_model.LinearRegression(n_jobs=-1, fit_intercept=False)`
        """
        super().__init__(forecasting_models)

        # wrap provided regression_model in a StandardRegressionModel (if not already the case)
        if isinstance(regression_model, StandardRegressionModel):
            # raise exception if train_n_points value is ambiguous
            model_train_n_points = regression_model.train_n_points
            raise_if(model_train_n_points is not None and regression_train_n_points != model_train_n_points,
                     "Provided StandardRegressionModel.train_n_points parameter doesn't match specified"
                     " regression_train_n_points parameter.",
                     logger)

            # if it was None, set regression_model.train_n_points to regression_train_n_points
            regression_model.train_n_points = regression_train_n_points
        else:
            regression_model = StandardRegressionModel(regression_train_n_points, regression_model)

        self.regression_model = regression_model

    def fit(self, series: TimeSeries) -> None:
        super().fit(series)

        # spare train_n_points points to serve as regression target
        raise_if(len(self.training_series) <= self.regression_model.train_n_points,
                 "regression_train_n_points parameter too big (must be smaller or equal" +
                 " to the number of points in training_series)",
                 logger)
        forecast_training = self.training_series[:-self.regression_model.train_n_points]
        regression_target = self.training_series[-self.regression_model.train_n_points:]

        # fit the forecasting models
        for model in self.models:
            model.fit(forecast_training)

        # predict train_n_points points for each model
        predictions = []
        for model in self.models:
            predictions.append(model.predict(self.regression_model.train_n_points))

        # train the regression model on the individual models' predictions
        self.regression_model.fit(train_features=predictions, train_target=regression_target)

        # prepare the forecasting models for further predicting by fitting
        # them with the entire data

        # Some models (incl. Neural-Network based models) may need to be 'reset'
        # to allow being retrained from scratch
        self.models = [model.untrained_model() if hasattr(model, 'untrained_model') else model
                       for model in self.models]

        # fit the forecasting models
        for model in self.models:
            model.fit(self.training_series)

    def ensemble(self, predictions: List[TimeSeries]) -> TimeSeries:
        return self.regression_model.predict(predictions)
