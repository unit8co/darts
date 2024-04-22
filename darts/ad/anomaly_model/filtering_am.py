"""
Filtering Anomaly Model
-----------------------

A ``FilteringAnomalyModel`` wraps around a Darts filtering model and one or
several anomaly scorer(s) to compute anomaly scores
by comparing how actuals deviate from the model's predictions (filtered series).
"""

from typing import Dict, Sequence, Union

from darts.ad.anomaly_model.anomaly_model import AnomalyModel
from darts.ad.scorers.scorers import AnomalyScorer
from darts.ad.utils import _assert_same_length, series2seq
from darts.logging import get_logger, raise_if_not, raise_log
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
        if ``allow_model_training`` is set to ``True`` upon calling ``fit()``.
        In addition, calling :func:`fit()` will also fit the fittable scorers, if any.

        Parameters
        ----------
        filter
            A filtering model from Darts that will be used to filter the actual time series
        scorer
            One or multiple scorer(s) that will be used to compare the actual and predicted time series in order
            to obtain an anomaly score ``TimeSeries``.
            If a list of `N` scorer is given, the anomaly model will call each
            one of the scorers and output a list of `N` anomaly scores ``TimeSeries``.
        """

        raise_if_not(
            isinstance(model, FilteringModel),
            f"`model` must be a darts.models.filtering not a {type(model)}.",
        )
        self.filter = model

        super().__init__(model=model, scorer=scorer)

    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        allow_model_training: bool = False,
        **filter_fit_kwargs,
    ):
        """Fit the underlying filtering model (if applicable) and the fittable scorers, if any.

        Train the filter (if not already fitted and `allow_model_training` is set to True)
        and the scorer(s) on the given time series.

        The filter model will be applied to the given series, and the results will be used
        to train the scorer(s).

        Parameters
        ----------
        series
            The (sequence of) series to be trained on.
        allow_model_training
            Boolean value that indicates if the filtering model needs to be fitted on the given series.
            If set to False, the model needs to be already fitted.
            Default: False
        filter_fit_kwargs
            Parameters to be passed on to the filtering model ``fit()`` method.

        Returns
        -------
        self
            Fitted model
        """
        # TODO: add support for covariates (see eg. Kalman Filter)
        super().fit(series=series, allow_model_training=allow_model_training)

        # interrupt training if nothing to fit
        if not allow_model_training and not self.scorers_are_trainable:
            logger.warning(
                f"The filtering model {self.model.__class__.__name__} won't be trained"
                + " because the parameter `allow_model_training` is set to False, and no scorer"
                + " is fittable. ``.fit()`` method has no effect."
            )
            return

        list_series = series2seq(series)

        if allow_model_training:
            # fit filtering model
            if hasattr(self.filter, "fit"):
                # TODO: check if filter is already fitted (for now fit it regardless -> only Kalman)
                raise_if_not(
                    len(list_series) == 1,
                    f"Filter model {self.model.__class__.__name__} can only be fitted on a"
                    + " single time series, but multiple are provided.",
                )

                self.filter.fit(list_series[0], **filter_fit_kwargs)
            else:
                raise_log(
                    ValueError(
                        "`allow_model_training` was set to True, but the filter"
                        + f" {self.model.__class__.__name__} has no fit() method."
                    ),
                    logger,
                )
        else:
            # TODO: check if Kalman is fitted or not
            # if not raise error "fit filter before, or set `allow_model_training` to TRUE"
            pass

        if self.scorers_are_trainable:
            list_pred = [self.filter.filter(series) for series in list_series]

            # fit the scorers
            self._fit_scorers(list_series, list_pred)

        return self

    def score(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        return_model_prediction: bool = False,
        **filter_kwargs,
    ):
        """Compute the anomaly score(s) for the given series.

        Predicts the given target time series with the filtering model, and applies the scorer(s)
        to compare the predicted (filtered) series and the provided series.

        Outputs the anomaly score(s) of the provided time series.

        Parameters
        ----------
        series
            The (sequence of) series to score.
        return_model_prediction
            Boolean value indicating if the prediction of the model should be returned along the anomaly score
            Default: False
        filter_kwargs
            parameters of the Darts `.filter()` filtering model

        Returns
        -------
        Union[TimeSeries, Sequence[TimeSeries], Sequence[Sequence[TimeSeries]]]
            Anomaly scores series generated by the anomaly model scorers

                - ``TimeSeries`` if `series` is a series, and the anomaly model contains one scorer.
                - ``Sequence[TimeSeries]``

                    * If `series` is a series, and the anomaly model contains multiple scorers,
                    returns one series per scorer.
                    * If `series` is a sequence, and the anomaly model contains one scorer,
                    returns one series per series in the sequence.
                - ``Sequence[Sequence[TimeSeries]]`` if `series` is a sequence, and the anomaly
                model contains multiple scorers.
                The outer sequence is over the series, and inner sequence is over the scorers.
        """
        raise_if_not(
            type(return_model_prediction) is bool,  # noqa: E721
            f"`return_model_prediction` must be Boolean, found type: {type(return_model_prediction)}.",
        )

        list_series = series2seq(series)

        # TODO: vectorize this call later on if we have any filtering models allowing this
        list_pred = [self.filter.filter(s, **filter_kwargs) for s in list_series]

        scores = list(
            zip(
                *[
                    sc.score_from_prediction(list_series, list_pred)
                    for sc in self.scorers
                ]
            )
        )

        if len(scores) == 1 and not isinstance(series, Sequence):
            # there's only one series
            scores = scores[0]
            if len(scores) == 1:
                # there's only one scorer
                scores = scores[0]

        if len(list_pred) == 1:
            list_pred = list_pred[0]

        if return_model_prediction:
            return scores, list_pred
        else:
            return scores

    def eval_metric(
        self,
        actual_anomalies: Union[TimeSeries, Sequence[TimeSeries]],
        series: Union[TimeSeries, Sequence[TimeSeries]],
        metric: str = "AUC_ROC",
        **filter_kwargs,
    ) -> Union[
        Dict[str, float],
        Dict[str, Sequence[float]],
        Sequence[Dict[str, float]],
        Sequence[Dict[str, Sequence[float]]],
    ]:
        """Compute the accuracy of the anomaly scores computed by the model.

        Predicts the `series` with the filtering model, and applies the
        scorer(s) on the filtered time series and the given target time series. Returns the
        score(s) of an agnostic threshold metric, based on the anomaly score given by the scorer(s).

        Parameters
        ----------
        actual_anomalies
            The (sequence of) ground truth of the anomalies (1 if it is an anomaly and 0 if not)
        series
            The (sequence of) series to predict anomalies on.
        metric
            Optionally, Scoring function to use. Must be one of "AUC_ROC" and "AUC_PR".
            Default: "AUC_ROC"
        filter_kwargs
            parameters of the Darts `.filter()` filtering model

        Returns
        -------
        Union[Dict[str, float], Dict[str, Sequence[float]], Sequence[Dict[str, float]],
        Sequence[Dict[str, Sequence[float]]]]
            Score for the time series.
            A (sequence of) dictionary with the keys being the name of the scorers, and the values being the
            metric results on the (sequence of) `series`. If the scorer treats every dimension independently
            (by nature of the scorer or if its component_wise is set to True), the values of the dictionary
            will be a Sequence containing the score for each dimension.
        """
        list_series, list_actual_anomalies = series2seq(series), series2seq(
            actual_anomalies
        )

        raise_if_not(
            all([isinstance(s, TimeSeries) for s in list_series]),
            "all input `series` must be of type Timeseries.",
        )

        raise_if_not(
            all([isinstance(s, TimeSeries) for s in list_actual_anomalies]),
            "all input `actual_anomalies` must be of type Timeseries.",
        )

        _assert_same_length(list_series, list_actual_anomalies)
        self._check_univariate(list_actual_anomalies)

        list_anomaly_scores = self.score(series=list_series, **filter_kwargs)

        acc_anomaly_scores = self._eval_metric_from_scores(
            list_actual_anomalies=list_actual_anomalies,
            list_anomaly_scores=list_anomaly_scores,
            metric=metric,
        )

        if len(acc_anomaly_scores) == 1 and not isinstance(series, Sequence):
            return acc_anomaly_scores[0]
        else:
            return acc_anomaly_scores
