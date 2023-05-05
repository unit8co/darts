"""
Regression ensemble model
-------------------------

An ensemble model which uses a regression model to compute the ensemble forecast.
"""
from typing import List, Optional, Sequence, Tuple, Union

from darts.logging import get_logger, raise_if, raise_if_not
from darts.models.forecasting.ensemble_model import EnsembleModel
from darts.models.forecasting.forecasting_model import (
    GlobalForecastingModel,
    LocalForecastingModel,
)
from darts.models.forecasting.linear_regression_model import LinearRegressionModel
from darts.models.forecasting.regression_model import RegressionModel
from darts.timeseries import TimeSeries
from darts.utils.utils import seq2series, series2seq

logger = get_logger(__name__)

# arbitrary threshold to raise warning for probabilistic forecasting models
SAMPLES_WARNING_THRESHOLD = 1e12


class RegressionEnsembleModel(EnsembleModel):
    def __init__(
        self,
        forecasting_models: Union[
            List[LocalForecastingModel], List[GlobalForecastingModel]
        ],
        regression_train_n_points: int,
        regression_model=None,
        regression_train_num_samples: Optional[int] = 1,
        regression_train_samples_reduction: Optional[Union[str, float]] = "mean",
    ):
        """
        Use a regression model for ensembling individual models' predictions using the stacking technique [1]_.

        The provided regression model must implement ``fit()`` and ``predict()`` methods
        (e.g. scikit-learn regression models). Note that here the regression model is used to learn how to
        best ensemble the individual forecasting models' forecasts. It is not the same usage of regression
        as in :class:`RegressionModel`, where the regression model is used to produce forecasts based on the
        lagged series.

        Parameters
        ----------
        forecasting_models
            List of forecasting models whose predictions to ensemble
        regression_train_n_points
            The number of points to use to train the regression model
        regression_model
            Any regression model with ``predict()`` and ``fit()`` methods (e.g. from scikit-learn)
            Default: ``darts.model.LinearRegressionModel(fit_intercept=False)``

            .. note::
                if `regression_model` is probabilistic, the `RegressionEnsembleModel` will also be probabilistic.
            ..
        regression_train_num_samples
            Number of times a prediction is sampled from each forecasting models to train the regression
            model (samples are averaged). Should be left set to 1 for deterministic models. Default: 1.
        regression_train_samples_reduction
            If `forecasting models` are probabilistic and `regression_train_num_samples` > 1, method used to
            reduce the samples before passing them to the regression model. Possible values: "mean", "median"
            or float value corresponding to the desired quantile. Default: "mean"
        References
        ----------
        .. [1] D. H. Wolpert, “Stacked generalization”, Neural Networks, vol. 5, no. 2, pp. 241–259, Jan. 1992
        """
        super().__init__(forecasting_models)
        if regression_model is None:
            regression_model = LinearRegressionModel(
                lags=None, lags_future_covariates=[0], fit_intercept=False
            )
        elif isinstance(regression_model, RegressionModel):
            regression_model = regression_model
        else:
            # scikit-learn like model
            regression_model = RegressionModel(
                lags_future_covariates=[0], model=regression_model
            )

        # check lags of the regression model
        raise_if_not(
            regression_model.lags == {"future": [0]},
            f"`lags` and `lags_past_covariates` of regression model must be `None`"
            f"and `lags_future_covariates` must be [0]. Given:\n"
            f"{regression_model.lags}",
        )

        # check the reduction method
        if isinstance(regression_train_samples_reduction, float):
            # this is already checked by `ts.quantile()`, maybe too redundant
            raise_if(
                regression_train_samples_reduction > 1.0
                or regression_train_samples_reduction < 0,
                f"`regression_train_samples_reduction` should be comprised between "
                f"0 and 1 ({regression_train_samples_reduction}).",
                logger,
            )
        elif isinstance(regression_train_samples_reduction, str):
            supported_reduction = ["mean", "median"]
            raise_if(
                regression_train_samples_reduction not in supported_reduction,
                f"`regression_train_samples_reduction` should be one of {supported_reduction}, "
                f"received ({regression_train_samples_reduction})",
                logger,
            )
        else:
            logger.exception(
                f"`regression_train_samples_reduction` type not supported "
                f"({regression_train_samples_reduction}). Must be either `str` or `float`."
            )

        raise_if(
            regression_train_num_samples > 1 and not self._models_are_probabilistic(),
            "`regression_train_num_samples` is greater than 1 but the `RegressionEnsembleModel` "
            "contains at least one non-probabilistic forecasting model.",
            logger,
        )

        if (
            regression_train_num_samples
            * regression_train_n_points
            * len(forecasting_models)
            > SAMPLES_WARNING_THRESHOLD
        ):
            logger.warning(
                f"Considering the number of models present in this ensemble ({len(forecasting_models)}), "
                f"`regression_train_n_points` ({regression_train_n_points}) and `regression_train_num_samples` "
                f"({regression_train_num_samples}) the number of sampled values to train the regression model "
                f"will be very large (>{SAMPLES_WARNING_THRESHOLD})."
            )

        self.regression_model = regression_model
        self.train_n_points = regression_train_n_points
        self.regression_train_num_samples = regression_train_num_samples
        self.regression_train_samples_reduction = regression_train_samples_reduction

    def _split_multi_ts_sequence(
        self, n: int, ts_sequence: Sequence[TimeSeries]
    ) -> Tuple[Sequence[TimeSeries], Sequence[TimeSeries]]:
        left = [ts[:-n] for ts in ts_sequence]
        right = [ts[-n:] for ts in ts_sequence]
        return left, right

    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    ):

        super().fit(
            series, past_covariates=past_covariates, future_covariates=future_covariates
        )

        # spare train_n_points points to serve as regression target
        is_single_series = isinstance(series, TimeSeries)
        if is_single_series:
            train_n_points_too_big = len(self.training_series) <= self.train_n_points
        else:
            train_n_points_too_big = any(
                [len(s) <= self.train_n_points for s in series]
            )

        raise_if(
            train_n_points_too_big,
            "`regression_train_n_points` parameter too big (must be smaller or "
            "equal to the number of points in training_series)",
            logger,
        )

        if is_single_series:
            forecast_training = self.training_series[: -self.train_n_points]
            regression_target = self.training_series[-self.train_n_points :]
        else:
            forecast_training, regression_target = self._split_multi_ts_sequence(
                self.train_n_points, series
            )

        for model in self.models:
            model._fit_wrapper(
                series=forecast_training,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
            )

        predictions = self._make_multiple_predictions(
            n=self.train_n_points,
            series=forecast_training,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            num_samples=self.regression_train_num_samples,
        )

        # component-wise reduction of the probabilistic forecasting models predictions
        if predictions[0].n_samples > 1:
            if self.regression_train_samples_reduction == "mean":
                predictions = predictions.mean(axis=2)
            elif self.regression_train_samples_reduction == "median":
                predictions = predictions.median(axis=2)
            else:
                predictions = predictions.quantile(
                    self.regression_train_samples_reduction
                )

        # train the regression model on the individual models' predictions
        self.regression_model.fit(
            series=regression_target, future_covariates=predictions
        )

        # prepare the forecasting models for further predicting by fitting them with the entire data

        # Some models (incl. Neural-Network based models) may need to be 'reset' to allow being retrained from scratch
        self.models = [model.untrained_model() for model in self.models]

        for model in self.models:
            kwargs = dict(series=series)
            if model.supports_past_covariates:
                kwargs["past_covariates"] = past_covariates
            if model.supports_future_covariates:
                kwargs["future_covariates"] = future_covariates
            model.fit(**kwargs)
        return self

    def ensemble(
        self,
        predictions: Union[TimeSeries, Sequence[TimeSeries]],
        series: Optional[Sequence[TimeSeries]] = None,
        num_samples: int = 1,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:

        is_single_series = isinstance(series, TimeSeries) or series is None
        predictions = series2seq(predictions)
        series = series2seq(series) if series is not None else [None]

        ensembled = [
            self.regression_model.predict(
                n=len(prediction),
                series=serie,
                future_covariates=prediction,
                num_samples=num_samples
                if self.regression_model._is_probabilistic()
                else 1,
            )
            for serie, prediction in zip(series, predictions)
        ]
        return seq2series(ensembled) if is_single_series else ensembled

    @property
    def extreme_lags(
        self,
    ) -> Tuple[
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
    ]:
        extreme_lags_ = super().extreme_lags
        return (extreme_lags_[0] - self.train_n_points,) + extreme_lags_[1:]

    def _is_probabilistic(self) -> bool:
        """
        A RegressionEnsembleModel is probabilistic if its regression
        model is probabilistic (ensembling layer)
        """
        return self.regression_model._is_probabilistic()
