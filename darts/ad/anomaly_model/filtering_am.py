"""
Filtering Anomaly Model
-----------------------

A `FilteringAnomalyModel` wraps around a Darts filtering model and one or
several anomaly scorer(s) to compute anomaly scores
by comparing how actuals deviate from the model's predictions (filtered series).
"""

import sys
from collections.abc import Sequence
from typing import Literal, Optional, Union

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from darts.ad.anomaly_model.anomaly_model import AnomalyModel
from darts.ad.scorers.scorers import AnomalyScorer
from darts.logging import get_logger, raise_log
from darts.models.filtering.filtering_model import FilteringModel
from darts.timeseries import TimeSeries

logger = get_logger(__name__)


class FilteringAnomalyModel(AnomalyModel):
    def __init__(
        self,
        model: FilteringModel,
        scorer: Union[AnomalyScorer, Sequence[AnomalyScorer]],
    ):
        """Filtering-based Anomaly Detection Model

        The filtering model may or may not be already fitted. The underlying assumption is that this model
        should be able to adequately filter the series in the absence of anomalies. For this reason,
        it is recommended to either provide a model that has already been fitted and evaluated to work
        appropriately on a series without anomalies, or to ensure that a simple call to the :func:`fit()`
        function of the model will be sufficient to train it to satisfactory performance on series without anomalies.

        Calling :func:`fit()` on the anomaly model will fit the underlying filtering model only
        if `allow_model_training` is set to `True` upon calling `fit()`.
        In addition, calling :func:`fit()` will also fit the fittable scorers, if any.

        Parameters
        ----------
        model
            A Darts `FilteringModel` used to filter the actual time series.
        scorer
            One or multiple scorer(s) used to compare the actual and predicted time series in order to obtain an
            anomaly score `TimeSeries`. If a list of scorers,
            :meth:`~darts.ad.anomaly_model.filtering_am.FilteringAnomalyModel.score` will output anomaly scores for
            each scorer.
        """
        if not isinstance(model, FilteringModel):
            raise_log(
                ValueError("`model` must be a Darts `FilteringModel`."),
                logger=logger,
            )
        super().__init__(model=model, scorer=scorer)

    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        allow_model_training: bool = False,
        **filter_fit_kwargs,
    ) -> Self:
        """Fit the underlying filtering model (if applicable) and the fittable scorers, if any.

        Train the filter (if not already fitted and `allow_model_training` is `True`) and the fittable scorer(s) on the
        given time series.

        The filter model will be applied to the given series, and the results will be used
        to train the scorer(s).

        Parameters
        ----------
        series
            The (sequence of) series to train on (generally assumed to be anomaly-free).
        allow_model_training
            Whether the filtering model should be fitted on the given series. If `False`, the model must already be
            fitted.
        **filter_fit_kwargs
            Additional parameters passed to the filtering model's `fit()` method.

        Returns
        -------
        self
            Fitted model.
        """
        return super().fit(
            series=series,
            allow_model_training=allow_model_training,
            **filter_fit_kwargs,
        )

    def score(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        return_model_prediction: bool = False,
        **filter_kwargs,
    ) -> Union[TimeSeries, Sequence[TimeSeries], Sequence[Sequence[TimeSeries]]]:
        """Compute the anomaly score(s) for the given (sequence of) series.

        Predicts the given target time series with the filtering model, and applies the scorer(s)
        to compare the predicted (filtered) series and the provided series.

        Parameters
        ----------
        series
            The (sequence of) series to score on.
        return_model_prediction
            Whether to return the filtering model prediction along with the anomaly scores.
        **filter_kwargs
            Additional parameters passed to the filtering model's `filter()` method.

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
            return_model_prediction=return_model_prediction,
            **filter_kwargs,
        )

    def predict_series(
        self, series: Sequence[TimeSeries], **kwargs
    ) -> Sequence[TimeSeries]:
        """Filters the given sequence of target time series with the filtering model.

        Parameters
        ----------
        series
            The sequence of series to filter.
        **kwargs
            Additional parameters passed to the filtering model's `filter()` method.
        """
        return [self.model.filter(s, **kwargs) for s in series]

    def eval_metric(
        self,
        anomalies: Union[TimeSeries, Sequence[TimeSeries]],
        series: Union[TimeSeries, Sequence[TimeSeries]],
        metric: Literal["AUC_ROC", "AUC_PR"] = "AUC_ROC",
        **filter_kwargs,
    ) -> Union[
        dict[str, float],
        dict[str, Sequence[float]],
        Sequence[dict[str, float]],
        Sequence[dict[str, Sequence[float]]],
    ]:
        """Compute a metric for the anomaly scores computed by the model.

        Predicts the `series` with the filtering model, and applies the scorer(s) on the filtered time series
        and the given target time series. Returns the score(s) of an agnostic threshold metric, based on the anomaly
        score given by the scorer(s).

        Parameters
        ----------
        anomalies
            The (sequence of) ground truth binary anomaly series (`1` if it is an anomaly and `0` if not).
        series
            The (sequence of) series to predict anomalies on.
        metric
            The name of the metric function to use. Must be one of "AUC_ROC" (Area Under the
            Receiver Operating Characteristic Curve) and "AUC_PR" (Average Precision from scores).
            Default: "AUC_ROC".
        **filter_kwargs
            Additional parameters passed to the filtering model's `filter()` method.

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
            anomalies=anomalies,
            series=series,
            metric=metric,
            **filter_kwargs,
        )

    def show_anomalies(
        self,
        series: TimeSeries,
        anomalies: TimeSeries = None,
        names_of_scorers: Union[str, Sequence[str]] = None,
        title: str = None,
        metric: Optional[Literal["AUC_ROC", "AUC_PR"]] = None,
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
        anomalies
            The ground truth of the anomalies (1 if it is an anomaly and 0 if not).
        names_of_scorers
            Name of the scores. Must be a list of length equal to the number of scorers in the anomaly_model.
        title
            Title of the figure.
        metric
            Optionally, the name of the metric function to use. Must be one of "AUC_ROC" (Area Under the
            Receiver Operating Characteristic Curve) and "AUC_PR" (Average Precision from scores).
            Default: "AUC_ROC".
        score_kwargs
            parameters for the `score()` method.
        """
        return super().show_anomalies(
            series=series,
            anomalies=anomalies,
            predict_kwargs=None,
            names_of_scorers=names_of_scorers,
            title=title,
            metric=metric,
            **score_kwargs,
        )

    def _fit_core(
        self,
        series: Sequence[TimeSeries],
        allow_model_training: bool,
        **model_fit_kwargs,
    ):
        """Fit the filters (if applicable) and scorers."""
        # TODO: add support for covariates (see eg. Kalman Filter)
        if allow_model_training and hasattr(self.model, "fit"):
            # TODO: check if filter is already fitted (for now fit it regardless -> only Kalman)
            if len(series) > 1:
                raise_log(
                    ValueError(
                        f"Filter model {self.model.__class__.__name__} can only be fitted "
                        f"on a single time series, but multiple are provided."
                    ),
                    logger=logger,
                )
            self.model.fit(series[0], **model_fit_kwargs)
        else:
            # TODO: check if Kalman is fitted or not
            # if not raise error "fit filter before, or set `allow_model_training` to TRUE"
            pass

        if self.scorers_are_trainable:
            pred = self.predict_series(series)
            # fit the scorers
            self._fit_scorers(series, pred)
