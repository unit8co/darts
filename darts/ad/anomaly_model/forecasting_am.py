"""
Forecasting Anomaly Model
-------------------------

A ``ForecastingAnomalyModel`` wraps around a Darts forecasting model and one or several anomaly
scorer(s) to compute anomaly scores by comparing how actuals deviate from the model's forecasts.
"""

# TODO:
#     - put start default value to its minimal value (wait for the release of historical_forecast)
import sys
from typing import Dict, Optional, Sequence, Union

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import pandas as pd

from darts.ad.anomaly_model.anomaly_model import AnomalyModel
from darts.ad.scorers.scorers import AnomalyScorer
from darts.logging import get_logger, raise_log
from darts.models.forecasting.forecasting_model import (
    ForecastingModel,
    GlobalForecastingModel,
)
from darts.timeseries import TimeSeries
from darts.utils.utils import n_steps_between

logger = get_logger(__name__)


class ForecastingAnomalyModel(AnomalyModel):
    def __init__(
        self,
        model: ForecastingModel,
        scorer: Union[AnomalyScorer, Sequence[AnomalyScorer]],
    ):
        """Forecasting-based Anomaly Detection Model

        The forecasting model may or may not be already fitted. The underlying assumption is that `model`
        should be able to accurately forecast the series in the absence of anomalies. For this reason,
        it is recommended to either provide a model that has already been fitted and evaluated to work
        appropriately on a series without anomalies, or to ensure that a simple call to the :func:`fit()`
        method of the model will be sufficient to train it to satisfactory performance on a series without anomalies.

        Calling :func:`fit()` on the anomaly model will fit the underlying forecasting model only
        if ``allow_model_training`` is set to ``True`` upon calling ``fit()``.
        In addition, calling :func:`fit()` will also fit the fittable scorers, if any.

        Parameters
        ----------
        model
            An instance of a Darts forecasting model.
        scorer
            One or multiple scorer(s) that will be used to compare the actual and predicted time series in order
            to obtain an anomaly score ``TimeSeries``.
            If a list of `N` scorers is given, the anomaly model will call each
            one of the scorers and output a list of `N` anomaly scores ``TimeSeries``.
        """
        if not isinstance(model, ForecastingModel):
            raise_log(
                ValueError(
                    f"Model must be a darts ForecastingModel not a {type(model)}."
                ),
                logger=logger,
            )
        self.model = model
        super().__init__(model=model, scorer=scorer)

    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        allow_model_training: bool = False,
        forecast_horizon: int = 1,
        start: Union[pd.Timestamp, float, int] = 0.5,
        num_samples: int = 1,
        **model_fit_kwargs,
    ) -> Self:
        """Fit the underlying forecasting model (if applicable) and the fittable scorers, if any.

        Train the forecasting model (if not already fitted and `allow_model_training` is `True`) and the fittable
        scorer(s) on the given time series.

        We use the trained forecasting model to compute historical forecasts for the input `series`.
        The scorer(s) are then trained on these forecasts along with the input `series`.

        Parameters
        ----------
        series
            The (sequence of) series to train on (generally assumed to be anomaly-free).
        past_covariates
            Optionally, a (sequence of) past-observed covariate series or sequence of series. This applies only to
            models that support past covariates.
        future_covariates
            Optionally, a (sequence of) future-known covariate series or sequence of series. This applies only to
            models that support future covariates.
        allow_model_training
            Whether the forecasting model should be fitted on the given series. If `False`, the model must already be
            fitted.
        forecast_horizon
            The forecast horizon for the predictions.
        start
            The first point of time at which a prediction is computed for a future time.
            This parameter supports 3 different data types: ``float``, ``int`` and ``pandas.Timestamp``.
            In the case of ``float``, the parameter will be treated as the proportion of the time series
            that should lie before the first prediction point.
            In the case of ``int``, the parameter will be treated as an integer index to the time index of
            `series` that will be used as first prediction time.
            In case of ``pandas.Timestamp``, this time stamp will be used to determine the first prediction time
            directly.
        num_samples
            Number of times a prediction is sampled from a probabilistic model. Should be left set to 1 for
            deterministic models.
        model_fit_kwargs
            Parameters to be passed on to the forecast model ``fit()`` method.

        Returns
        -------
        self
            Fitted model
        """
        return super().fit(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            allow_model_training=allow_model_training,
            forecast_horizon=forecast_horizon,
            start=start,
            num_samples=num_samples,
            **model_fit_kwargs,
        )

    def score(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        forecast_horizon: int = 1,
        start: Union[pd.Timestamp, float, int] = 0.5,
        num_samples: int = 1,
        return_model_prediction: bool = False,
    ) -> Union[TimeSeries, Sequence[TimeSeries], Sequence[Sequence[TimeSeries]]]:
        """Compute anomaly score(s) for the given series.

        Predicts the given target time series with the forecasting model, and applies the scorer(s)
        on the prediction and the target input time series.

        Parameters
        ----------
        series
            The (sequence of) series to score on.
        past_covariates
            Optionally, a (sequence of) past-observed covariate series or sequence of series. This applies only to
            models that support past covariates.
        future_covariates
            Optionally, a (sequence of) future-known covariate series or sequence of series. This applies only to
            models that support future covariates.
        forecast_horizon
            The forecast horizon for the predictions.
        start
            The first point of time at which a prediction is computed for a future time.
            This parameter supports 3 different data types: ``float``, ``int`` and ``pandas.Timestamp``.
            In the case of ``float``, the parameter will be treated as the proportion of the time series
            that should lie before the first prediction point.
            In the case of ``int``, the parameter will be treated as an integer index to the time index of
            `series` that will be used as first prediction time.
            In case of ``pandas.Timestamp``, this time stamp will be used to determine the first prediction time
            directly.
        num_samples
            Number of times a prediction is sampled from a probabilistic model. Should be left set to 1 for
            deterministic models.
        return_model_prediction
            Whether to return the forecasting model prediction along with the anomaly scores.

        Returns
        -------
        TimeSeries
            A single `TimeSeries` for a single `series` with a single anomaly scorers.
        Sequence[TimeSeries]
            A sequence of `TimeSeries` for:

            - a single `series` with multiple anomaly scorers.
            - a sequence of `series` with a single anomaly scorer.
        Sequence[Sequence[TimeSeries]]
            A sequence of sequences of `TimeSeries` for a sequence of `series` and multiple anomaly scorers.
            The outer sequence is over the series, and inner sequence is over the scorers.
        """
        return super().score(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            forecast_horizon=forecast_horizon,
            start=start,
            num_samples=num_samples,
            return_model_prediction=return_model_prediction,
        )

    def eval_metric(
        self,
        actual_anomalies: Union[TimeSeries, Sequence[TimeSeries]],
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        forecast_horizon: int = 1,
        start: Union[pd.Timestamp, float, int] = 0.5,
        num_samples: int = 1,
        metric: str = "AUC_ROC",
    ) -> Union[
        Dict[str, float],
        Dict[str, Sequence[float]],
        Sequence[Dict[str, float]],
        Sequence[Dict[str, Sequence[float]]],
    ]:
        """Compute the accuracy of the anomaly scores computed by the model.

        Predicts the `series` with the forecasting model, and applies the scorer(s) on the predicted time series and
        the given target time series. Returns the score(s) of an agnostic threshold metric, based on the anomaly score
        given by the scorer(s).

        Parameters
        ----------
        actual_anomalies
            The (sequence of) ground truth binary anomaly series (`1` if it is an anomaly and `0` if not).
        series
            The (sequence of) series to predict anomalies on.
        past_covariates
            Optionally, a (sequence of) past-observed covariate series or sequence of series. This applies only to
            models that support past covariates.
        future_covariates
            Optionally, a (sequence of) future-known covariate series or sequence of series. This applies only to
            models that support future covariates.
        forecast_horizon
            The forecast horizon for the predictions.
        start
            The first point of time at which a prediction is computed for a future time.
            This parameter supports 3 different data types: ``float``, ``int`` and ``pandas.Timestamp``.
            In the case of ``float``, the parameter will be treated as the proportion of the time series
            that should lie before the first prediction point.
            In the case of ``int``, the parameter will be treated as an integer index to the time index of
            `series` that will be used as first prediction time.
            In case of ``pandas.Timestamp``, this time stamp will be used to determine the first prediction time
            directly.
        num_samples
            Number of times a prediction is sampled from a probabilistic model. Should be left set to 1 for
            deterministic models.
        metric
            The name of the scoring function to use. Must be one of "AUC_ROC" (Area Under the
            Receiver Operating Characteristic Curve) and "AUC_PR" (Average Precision from scores).
            Default: "AUC_ROC"

        Returns
        -------
        Dict[str, float]
            A dictionary with the resulting metrics for single univariate `series`, with keys representing the
            anomaly scorer(s), and values representing the metric values.
        Dict[str, Sequence[float]]
            Same as for `Dict[str, float]` but for multivariate `series`, and anomaly scorers that treat series
            components/columns independently (by nature of the scorer or if `component_wise=True`).
        Sequence[Dict[str, float]]
            Same as for `Dict[str, float]` but for a sequence of univariate series.
        Sequence[Dict[str, Sequence[float]]]
            Same as for `Dict[str, float]` but for a sequence of multivariate series.
        """
        return super().eval_metric(
            actual_anomalies=actual_anomalies,
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            forecast_horizon=forecast_horizon,
            start=start,
            num_samples=num_samples,
            metric=metric,
        )

    def show_anomalies(
        self,
        series: TimeSeries,
        past_covariates: Optional[TimeSeries] = None,
        future_covariates: Optional[TimeSeries] = None,
        forecast_horizon: int = 1,
        start: Union[pd.Timestamp, float, int] = 0.5,
        num_samples: int = 1,
        actual_anomalies: TimeSeries = None,
        names_of_scorers: Union[str, Sequence[str]] = None,
        title: str = None,
        metric: str = None,
        **score_kwargs,
    ):
        """Plot the results of the anomaly model.

        Computes the score on the given series input and shows the different anomaly scores with respect to time.

        The plot will be composed of the following:

        - the series itself with the output of the forecasting model.
        - the anomaly score for each scorer. The scorers with different windows will be separated.
        - the actual anomalies, if given.

        It is possible to:

        - add a title to the figure with the parameter `title`
        - give personalized names for the scorers with `names_of_scorers`
        - show the results of a metric for each anomaly score (AUC_ROC or AUC_PR), if the actual anomalies are provided.

        Parameters
        ----------
        series
            The series to visualize anomalies from.
        past_covariates
            Optionally, a past-observed covariate series or sequence of series. This applies only to
            models that support past covariates.
        future_covariates
            Optionally, a future-known covariate series or sequence of series. This applies only to models that support
            future covariates.
        forecast_horizon
            The forecast horizon for the predictions.
        start
            The first point of time at which a prediction is computed for a future time. This parameter supports 3
            different data types: ``float``, ``int`` and ``pandas.Timestamp``.
            In the case of ``float``, the parameter will be treated as the proportion of the time series that should
            lie before the first prediction point.
            In the case of ``int``, the parameter will be treated as an integer index to the time index of `series`
            that will be used as first prediction time.
            In case of ``pandas.Timestamp``, this time stamp will be used to determine the first prediction time
            directly.
        num_samples
            Number of times a prediction is sampled from a probabilistic model. Should be left set to 1 for
            deterministic models.
        actual_anomalies
            The ground truth of the anomalies (1 if it is an anomaly and 0 if not).
        names_of_scorers
            Name of the scores. Must be a list of length equal to the number of scorers in the anomaly_model.
        title
            Title of the figure.
        metric
            Optionally, Scoring function to use. Must be one of "AUC_ROC" and "AUC_PR". Default: "AUC_ROC".
        score_kwargs
            parameters for the ``score()`` method.
        """
        predict_kwargs = {
            "past_covariates": past_covariates,
            "future_covariates": future_covariates,
            "forecast_horizon": forecast_horizon,
            "start": start,
            "num_samples": num_samples,
        }
        return super().show_anomalies(
            series=series,
            predict_kwargs=predict_kwargs,
            actual_anomalies=actual_anomalies,
            names_of_scorers=names_of_scorers,
            title=title,
            metric=metric,
            **score_kwargs,
        )

    def _fit_core(
        self,
        series: Sequence[TimeSeries],
        past_covariates: Optional[Sequence[TimeSeries]] = None,
        future_covariates: Optional[Sequence[TimeSeries]] = None,
        allow_model_training: bool = False,
        forecast_horizon: int = 1,
        start: Union[pd.Timestamp, float, int] = 0.5,
        num_samples: int = 1,
        **model_fit_kwargs,
    ):
        """Fit the forecasting model (if applicable) and scorers."""
        # fit forecasting model
        if allow_model_training:
            series_ = series
            past_covariates_ = past_covariates
            future_covariates_ = future_covariates
            # for local models extract single series
            if not isinstance(self.model, GlobalForecastingModel):
                if len(series) > 1:
                    raise_log(
                        ValueError(
                            f"Forecasting model {self.model.__class__.__name__} only accepts a single "
                            f"time series for the training phase and not a sequence of multiple of time series."
                        ),
                        logger=logger,
                    )
                series_ = series[0]
                past_covariates_ = (
                    past_covariates[0]
                    if past_covariates is not None
                    else past_covariates
                )
                future_covariates_ = (
                    future_covariates[0]
                    if future_covariates is not None
                    else future_covariates
                )
            self.model._fit_wrapper(
                series=series_,
                past_covariates=past_covariates_,
                future_covariates=future_covariates_,
                **model_fit_kwargs,
            )
        elif not self.model._fit_called:
            raise_log(
                ValueError(
                    f"With `allow_model_training=False`, the underlying model `{self.model.__class__.__name__}` "
                    f"must have already been trained. Either train it before or set `allow_model_training=True` "
                    f"(model will trained from scratch on the provided series)."
                ),
                logger=logger,
            )

        # generate the historical_forecast() prediction of the model on the train set
        if self.scorers_are_trainable:
            historical_forecasts = self._predict_series(
                series,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
                forecast_horizon=forecast_horizon,
                start=start,
                num_samples=num_samples,
            )
            # fit the scorers
            self._fit_scorers(series, historical_forecasts)

    def _predict_series(
        self,
        series: Sequence[TimeSeries],
        past_covariates: Optional[Sequence[TimeSeries]] = None,
        future_covariates: Optional[Sequence[TimeSeries]] = None,
        forecast_horizon: int = 1,
        start: Union[pd.Timestamp, float, int] = None,
        num_samples: int = 1,
    ) -> Sequence[TimeSeries]:
        """Compute the historical forecasts that would have been obtained by this model on the `series`.

        `retrain` is set to False if possible (this is not supported by all models). If set to True, it will always
        re-train the model on the entire available history,
        """
        if not self.model._fit_called:
            raise_log(
                ValueError(
                    f"Forecasting `model` {self.model} has not been trained yet. Call ``fit()`` before."
                ),
                logger=logger,
            )

        # check if the window size of the scorers are lower than the max size allowed
        self._check_window_size(series, start)

        # TODO: raise an exception. We only support models that do not need retrain
        # checks if model accepts to not be retrained in the historical_forecasts()
        if self.model._supports_non_retrainable_historical_forecasts:
            # default: set to False. Allows a faster computation.
            retrain = False
        else:
            retrain = True

        historical_forecasts_param = {
            "past_covariates": past_covariates,
            "future_covariates": future_covariates,
            "forecast_horizon": forecast_horizon,
            "start": start,
            "retrain": retrain,
            "num_samples": num_samples,
            "stride": 1,
            "last_points_only": True,
            "verbose": False,
        }
        return self.model.historical_forecasts(series, **historical_forecasts_param)

    def _check_window_size(
        self, series: Sequence[TimeSeries], start: Union[pd.Timestamp, float, int]
    ):
        """Checks if the parameters `window` of the scorers are smaller than the maximum window size allowed.
        The maximum size allowed is equal to the output length of the .historical_forecast() applied on `series`.
        It is defined by the parameter `start` and the series’ length.

        Parameters
        ----------
        series
            The series given to the .historical_forecast()
        start
            Parameter of the .historical_forecast(): first point of time at which a prediction is computed
            for a future time.
        """
        # biggest window of the anomaly_model scorers
        max_window = max(scorer.window for scorer in self.scorers)
        for s in series:
            max_possible_window = (
                n_steps_between(
                    end=s.end_time(), start=s.get_timestamp_at_point(start), freq=s.freq
                )
                + 1
            )
            if max_window > max_possible_window:
                raise_log(
                    ValueError(
                        f"Window size {max_window} is greater than the targeted series length {max_possible_window}, "
                        f"must be lower or equal. Reduce window size, or reduce start value (start: {start})."
                    ),
                    logger=logger,
                )
