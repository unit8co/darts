"""
Regression ensemble model
-----------------------
"""
from sklearn.linear_model import LinearRegression
from typing import Optional, List

from darts.timeseries import TimeSeries
from darts.models import EnsembleModel, StandardRegressionModel
from darts.models.forecasting_model import ForecastingModel
from darts.models.torch_forecasting_model import TorchForecastingModel
from darts.logging import get_logger, raise_if, raise_if_not

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

            regression_model.train_n_points = regression_train_n_points
        else:
            regression_model = StandardRegressionModel(regression_train_n_points, regression_model)

        self.regression_model = regression_model

    def fit(self, training_series: TimeSeries, target_series: Optional[TimeSeries] = None) -> None:
        # TODO: Factor this out, same logic is used in 3 places already. Need to find an appropiate name
        if target_series is None:
            target_series = training_series
        raise_if_not(all(training_series.time_index() == target_series.time_index()),
                     "training and target series must have same time indices.",
                     logger)

        # spare train_n_points points to serve as regression target
        forecast_training = training_series[:-self.regression_model.train_n_points]
        forecast_target = target_series[:-self.regression_model.train_n_points]

        regression_target = target_series[-self.regression_model.train_n_points:]

        # fit the forecasting models
        super().fit(forecast_training, forecast_target)

        # predict train_n_points points for each model
        predictions = []
        for model in self.models:
            predictions.append(model.predict(self.regression_model.train_n_points))

        # train the regression model on the individual models' predictions
        self.regression_model.fit(train_features=predictions, train_target=regression_target)

        # prepare the forecasting models for further predicting by fitting
        # them with the entire data

        # Neural-Network based models need to be retrained from scratch.
        self.models = [model.untrained_model() if isinstance(model, TorchForecastingModel) else model
                       for model in self.models]

        super().fit(training_series, target_series)

    def ensemble(self, predictions: List[TimeSeries]) -> TimeSeries:
        return self.regression_model.predict(predictions)
