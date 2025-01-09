"""
Anomaly models base classes
"""

import sys
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal, Optional, Union

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from darts.ad.scorers.scorers import AnomalyScorer
from darts.ad.utils import (
    _assert_same_length,
    _check_input,
    eval_metric_from_scores,
    show_anomalies_from_scores,
)
from darts.logging import get_logger, raise_log
from darts.timeseries import TimeSeries

logger = get_logger(__name__)


class AnomalyModel(ABC):
    """Base class for all anomaly models."""

    def __init__(self, model, scorer):
        self.scorers = [scorer] if not isinstance(scorer, Sequence) else scorer
        if not all([isinstance(s, AnomalyScorer) for s in self.scorers]):
            raise_log(
                ValueError(
                    "all scorers must be of instance `darts.ad.scorers.AnomalyScorer`."
                ),
                logger=logger,
            )
        self.model = model

    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        allow_model_training: bool,
        **kwargs,
    ) -> Self:
        """Fit the underlying forecasting/filtering model (if applicable) and the fittable scorers."""
        # interrupt training if nothing to fit
        if not allow_model_training and not self.scorers_are_trainable:
            return self

        # check input series and covert to sequences
        series, kwargs = self._process_input_series(series, **kwargs)
        self._fit_core(
            series=series, allow_model_training=allow_model_training, **kwargs
        )
        return self

    @abstractmethod
    def score(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        return_model_prediction: bool = False,
        **kwargs,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Compute anomaly score(s) for the given series.

        Predicts the given target time series with the forecasting model, and applies the scorer(s)
        on the prediction and the target input time series.

        Parameters
        ----------
        series
            The (sequence of) series to score on.
        return_model_prediction
            Whether to return the forecasting/filtering model prediction along with the anomaly scores.
        **kwargs
            Additional parameters passed to `AnomalyModel.predict_series()`

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
        called_with_single_series = isinstance(series, TimeSeries)
        # check input series and covert to sequences
        series, kwargs = self._process_input_series(series, **kwargs)
        # predict / filter `series`
        pred = self.predict_series(series=series, **kwargs)

        scores = list(
            zip(*[sc.score_from_prediction(series, pred) for sc in self.scorers])
        )

        if called_with_single_series:
            scores = scores[0]
            if len(scores) == 1:
                # there's only one scorer
                scores = scores[0]
            pred = pred[0]

        if return_model_prediction:
            return scores, pred

        return scores

    @abstractmethod
    def predict_series(
        self, series: Sequence[TimeSeries], **kwargs
    ) -> Sequence[TimeSeries]:
        """Abstract method to implement the generation of predictions for the input `series`."""
        pass

    def eval_metric(
        self,
        anomalies: Union[TimeSeries, Sequence[TimeSeries]],
        series: Union[TimeSeries, Sequence[TimeSeries]],
        metric: Literal["AUC_ROC", "AUC_PR"] = "AUC_ROC",
        **kwargs,
    ) -> Union[
        dict[str, float],
        dict[str, Sequence[float]],
        Sequence[dict[str, float]],
        Sequence[dict[str, Sequence[float]]],
    ]:
        """Compute the accuracy of the anomaly scores computed by the model.

        Predicts the `series` with the underlying forecasting/filtering model, and applies the scorer(s) on the
        predicted time series and the given target time series. Returns the score(s) of an agnostic threshold metric,
        based on the anomaly score given by the scorer(s).

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
        **kwargs
            Additional parameters passed to the `score()` method.

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

        def _check_univariate(s: TimeSeries):
            """Checks if `anomalies` contains only univariate series, which
            is required if any of the scorers returns a univariate score.
            """
            if self.scorers_are_univariate and not s.width == 1:
                raise_log(
                    ValueError(
                        f"Anomaly model contains scorer {[s.__str__() for s in self.scorers if s.is_univariate]} "
                        f"that will return a univariate anomaly score series (width=1). Found a multivariate "
                        f"`anomalies`. The evaluation of the accuracy cannot be computed. If applicable, "
                        f"think about setting the scorer parameter `componenet_wise` to True."
                    ),
                    logger=logger,
                )

        called_with_single_series = isinstance(series, TimeSeries)
        # deterministic `series`
        series = _check_input(
            series,
            name="series",
            check_deterministic=True,
        )
        # deterministic, binary anomalies, (possibly univariate)
        anomalies = _check_input(
            anomalies,
            name="anomalies",
            check_deterministic=True,
            check_binary=True,
            extra_checks=_check_univariate,
        )
        _assert_same_length(series, anomalies, "series", "anomalies")

        pred_scores = self.score(series=series, **kwargs)

        # compute metric for anomaly scores
        windows = [s.window for s in self.scorers]

        # create a list of unique names for each scorer that
        # will be used as keys for the dictionary containing
        # the accuracy of each scorer.
        name_scorers = []
        for scorer in self.scorers:
            name = scorer.__str__() + "_w=" + str(scorer.window)

            if name in name_scorers:
                i = 1
                new_name = name + "_" + str(i)
                while new_name in name_scorers:
                    i = i + 1
                    new_name = name + "_" + str(i)
                name = new_name

            name_scorers.append(name)

        metric_vals = []
        for anomalies, scores in zip(anomalies, pred_scores):
            metric_vals.append(
                eval_metric_from_scores(
                    anomalies=anomalies,
                    pred_scores=scores,
                    window=windows,
                    metric=metric,
                )
            )
        metric_vals_pred_scores = [
            dict(zip(name_scorers, scorer_values)) for scorer_values in metric_vals
        ]

        return (
            metric_vals_pred_scores[0]
            if called_with_single_series
            else metric_vals_pred_scores
        )

    def show_anomalies(
        self,
        series: TimeSeries,
        anomalies: TimeSeries = None,
        predict_kwargs: Optional[dict] = None,
        names_of_scorers: Union[str, Sequence[str]] = None,
        title: str = None,
        metric: Optional[Literal["AUC_ROC", "AUC_PR"]] = None,
        component_wise: bool = False,
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
        predict_kwargs
            Optionally, some additional parameters passed to `AnomalyModel.predict_series()`.
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
        component_wise
            If True, will separately plot each component in case of multivariate anomaly detection.
        """
        series = _check_input(series, name="series", num_series_expected=1)[0]
        predict_kwargs = predict_kwargs if predict_kwargs is not None else {}
        pred_scores, pred_series = self.score(
            series,
            return_model_prediction=True,
            **predict_kwargs,
            **score_kwargs,
        )

        if title is None:
            title = f"Anomaly results ({self.model.__class__.__name__})"

        if names_of_scorers is None:
            names_of_scorers = [s.__str__() for s in self.scorers]

        list_window = [s.window for s in self.scorers]

        return show_anomalies_from_scores(
            series=series,
            anomalies=anomalies,
            pred_series=pred_series,
            pred_scores=pred_scores,
            window=list_window,
            names_of_scorers=names_of_scorers,
            title=title,
            metric=metric,
            component_wise=component_wise,
        )

    @property
    def scorers_are_univariate(self):
        """Whether any of the Scorers is univariate."""
        return any(s.is_univariate for s in self.scorers)

    @property
    def scorers_are_trainable(self):
        """Whether any of the Scorers is trainable."""
        return any(s.is_trainable for s in self.scorers)

    @abstractmethod
    def _fit_core(
        self,
        series: Sequence[TimeSeries],
        allow_model_training: bool,
        **kwargs,
    ):
        """Abstract method to implement the model and scorer training."""
        pass

    def _fit_scorers(
        self, list_series: Sequence[TimeSeries], list_pred: Sequence[TimeSeries]
    ):
        """Train the fittable scorers using model forecasts"""
        for scorer in self.scorers:
            if scorer.is_trainable:
                scorer.fit_from_prediction(list_series, list_pred)

    @staticmethod
    def _process_input_series(
        series: Union[TimeSeries, Sequence[TimeSeries]], **kwargs
    ):
        """Checks input series and coverts series and covariates in `kwargs` to sequences."""
        series = _check_input(series, name="series")
        for cov_name in ["past_covariates", "future_covariates"]:
            cov = kwargs.pop(cov_name, None)
            if cov is not None:
                cov = _check_input(cov, name=cov_name)
                _assert_same_length(series, cov, "series", cov_name)
                kwargs[cov_name] = cov
        return series, kwargs
