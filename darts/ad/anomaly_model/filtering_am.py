"""
Filtering Anomaly Model
-----------------------

A ``FilteringAnomalyModel`` wraps around a Darts filtering model and one or
several anomaly scorer(s) to compute anomaly scores
by comparing how actuals deviate from the model's predictions.
"""

from typing import Dict, Sequence, Union

from darts.ad.anomaly_model.anomaly_model import AnomalyModel
from darts.ad.scorers.scorers import AnomalyScorer
from darts.ad.utils import _same_length, _to_list, eval_accuracy_from_scores
from darts.logging import get_logger, raise_if_not
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
        should be able to acurately predict the series in the absence of anomalies. For this reason,
        it is recommend to either provide a model that has already been fitted and evaluated to work
        appropriately on a series without anomalies, or to ensure that a simple call to the :func:`fit()`
        method of the model will be sufficient to train it to satisfactory performance on a series without anomalies.

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
        allow_filter_training: bool = False,
        **filter_fit_kwargs,
    ):
        """Fit the underlying filtering model (if applicable) and the fittable scorers, if any.

        Train the filter (if not already fitted and ``allow_filter_training`` is set to True)
        and the scorer(s) on the given time series.

        The filter model will be applied to the given series, and the results will be used
        to train the scorer(s).

        Parameters
        ----------
        series
            The (sequence of) series to be trained on.
        allow_filter_training
            Boolean value that indicates if the filtering model needs to be fitted on the given series.
            If set to False, the model needs to be already fitted.
            Default: False
        filter_fit_kwargs
            Parameters to be passed on to the filtering model ``fit()`` method.

        Returns
        -------
        self
            Fitted Anomaly model (filtering model and scorer(s))
        """

        raise_if_not(
            type(allow_filter_training) is bool,
            f"`allow_filter_training` must be Boolean, found type: {type(allow_filter_training)}.",
        )

        # checks if model does not need training and all scorer(s) are not fittable
        if not allow_filter_training and not self.scorers_are_trainable:

            logger.warning(
                f"The filtering model {self.model.__class__.__name__} is not required to be trained"
                + " because the parameter `allow_filter_training` is set to False, and all scorers are"
                + " not fittable. No need to call the ``.fit()`` function."
            )

        list_series = _to_list(series)

        raise_if_not(
            all([isinstance(s, TimeSeries) for s in list_series]),
            "all input `series` must be of type Timeseries.",
        )

        if allow_filter_training:
            # fit filtering model
            if hasattr(self.filter, "fit"):
                # TODO: check if filter is already fitted (for now fit it regardless -> only Kalman)
                raise_if_not(
                    len(list_series) == 1,
                    f"Filter model {self.model.__class__.__name__} can only be fitted on a"
                    + " time series and not a list of time series.",
                )

                self.filter.fit(list_series[0], **filter_fit_kwargs)
            else:
                raise ValueError(
                    "`allow_filter_training` was set to True, but the filter"
                    + f" {self.model.__class__.__name__} is not fittable."
                )
        else:
            # TODO: check if Kalman is fitted or not
            # if not raise error "fit filter before, or set `allow_filter_training` to TRUE"
            pass

        if self.scorers_are_trainable:
            list_pred = []
            for series in list_series:
                list_pred.append(self.filter.filter(series))

        # fit the scorers
        for scorer in self.scorers:
            if hasattr(scorer, "fit"):
                scorer.fit_from_prediction(list_series, list_pred)

    def show_anomalies(
        self,
        series: TimeSeries,
        actual_anomalies: TimeSeries = None,
        names_of_scorers: Union[str, Sequence[str]] = None,
        title: str = None,
        metric: str = None,
        **score_kwargs,
    ):
        """Plot the results of the anomaly model.

        Computes the score on the given series input. And shows the different anomaly scores with respect to time.

        The plot will be composed of the following:
            - the series itself with the output of the filtering model
            - the anomaly score of each scorer. The scorer with different windows will be separated.
            - the actual anomalies, if given.

        It is possible to:
            - add a title to the figure with the parameter `title`
            - give personalized names for the scorers with `names_of_scorers`
            - show the results of a metric for each anomaly score (AUC_ROC or AUC_PR), if the actual anomalies is given

        Parameters
        ----------
        series
            The series to visualize anomalies from.
        actual_anomalies
            The ground truth of the anomalies (1 if it is an anomaly and 0 if not)
        names_of_scorers
            Name of the scores. Must be a list of length equal to the number of scorers in the anomaly_model.
        title
            Title of the figure
        metric
            Optionally, Scoring function to use. Must be one of "AUC_ROC" and "AUC_PR".
            Default: "AUC_ROC"
        score_kwargs
            parameters for the `.score()` function
        """

        if isinstance(series, Sequence):
            raise_if_not(
                len(series) == 1,
                f"`show_anomalies` expects one series, found a sequence of length {len(series)} as input.",
            )

            series = series[0]

        anomaly_scores, model_output = self.score(
            series, return_model_prediction=True, **score_kwargs
        )

        return self._show_anomalies(
            series,
            model_output=model_output,
            anomaly_scores=anomaly_scores,
            names_of_scorers=names_of_scorers,
            actual_anomalies=actual_anomalies,
            title=title,
            metric=metric,
        )

    def score(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        return_model_prediction: bool = False,
        **filter_kwargs,
    ):
        """Compute anomaly score(s) for the given series.

        Predicts the given target time series with the filtering model, and applies the scorer(s)
        on the prediction and the provided series. Outputs the anomaly score of the provided time series.

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

                * If `series` is a series, and the anomaly model contains multiple scorers.
                  returns one series per scorer.
                * If `series` is a sequence, and the anomaly model contains one scorer.
                  returns one series per series in the sequence.
            - ``Sequence[Sequence[TimeSeries]]`` if `series` is a sequence, and the anomaly
              model contains multiple scorers.
              The outer sequence is over the series, and inner sequence is over the scorers.
        """
        raise_if_not(
            type(return_model_prediction) is bool,
            f"`return_model_prediction` must be Boolean, found type: {type(return_model_prediction)}.",
        )

        list_series = _to_list(series)

        preds = [self.filter.filter(s, **filter_kwargs) for s in list_series]
        scores = [
            [sc.score_from_prediction(s, p) for sc in self.scorers]
            for s, p in zip(list_series, preds)
        ]

        if len(scores) == 1 and not isinstance(series, Sequence):
            # there's only one series
            scores = scores[0]
            if len(scores) == 1:
                # there's only one scorer
                scores = scores[0]

        if len(preds) == 1:
            preds = preds[0]

        if return_model_prediction:
            return scores, preds
        else:
            return scores

    def eval_accuracy(
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
        list_series, list_actual_anomalies = _to_list(series), _to_list(
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

        _same_length(list_series, list_actual_anomalies)
        self.check_returns_UTS(list_actual_anomalies)

        list_anomaly_scores = self.score(series=list_series, **filter_kwargs)

        acc_anomaly_scores = []
        for anomalies, scores in zip(list_actual_anomalies, list_anomaly_scores):

            scorer_results = {}
            for idx, scorer in enumerate(self.scorers):
                name = scorer.__str__() + "_w" + str(scorer.window)

                if name in scorer_results:
                    i = 1
                    new_name = name + "_" + str(i)
                    while new_name in scorer_results:
                        i = i + 1
                        new_name = name + "_" + str(i)
                    name = new_name

                scorer_results[name] = eval_accuracy_from_scores(
                    actual_anomalies=anomalies,
                    anomaly_score=scores[idx],
                    window=scorer.window,
                    metric=metric,
                )
            acc_anomaly_scores.append(scorer_results)

        if len(acc_anomaly_scores) == 1 and not isinstance(series, Sequence):
            return acc_anomaly_scores[0]
        else:
            return acc_anomaly_scores
