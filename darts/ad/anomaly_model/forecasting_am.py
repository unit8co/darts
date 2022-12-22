"""
Forecasting Anomaly Model
-------------------------

A ``ForecastingAnomalyModel`` wraps around a Darts forecasting model and one or several anomaly
scorer(s) to compute anomaly scores by comparing how actuals deviate from the model's forecasts.
"""

# TODO:
#     - put start default value to its minimal value (wait for the release of historical_forecast)

import inspect
from typing import Dict, Optional, Sequence, Union

import pandas as pd

from darts.ad.anomaly_model.anomaly_model import AnomalyModel
from darts.ad.scorers.scorers import AnomalyScorer
from darts.ad.utils import _assert_same_length, _assert_timeseries, _to_list
from darts.logging import get_logger, raise_if_not
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.timeseries import TimeSeries

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

        raise_if_not(
            isinstance(model, ForecastingModel),
            f"Model must be a darts ForecastingModel not a {type(model)}.",
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
    ):
        """Fit the underlying forecasting model (if applicable) and the fittable scorers, if any.

        Train the model (if not already fitted and ``allow_model_training`` is set to True) and the
        scorer(s) (if fittable) on the given time series.

        Once the model is fitted, the series historical forecasts are computed,
        representing what would have been forecasted by this model on the series.

        The prediction and the series are then used to train the scorer(s).

        Parameters
        ----------
        series
            One or multiple (if the model supports it) target series to be
            trained on (generally assumed to be anomaly-free).
        past_covariates
            Optional past-observed covariate series or sequence of series. This applies only if the model
            supports past covariates.
        future_covariates
            Optional future-known covariate series or sequence of series. This applies only if the model
            supports future covariates.
        allow_model_training
            Boolean value that indicates if the forecasting model needs to be fitted on the given series.
            If set to False, the model needs to be already fitted.
            Default: False
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
            Default: 0.5
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

        raise_if_not(
            type(allow_model_training) is bool,
            f"`allow_model_training` must be Boolean, found type: {type(allow_model_training)}.",
        )

        # checks if model does not need training and all scorer(s) are not fittable
        if not allow_model_training and not self.scorers_are_trainable:
            logger.warning(
                f"The forecasting model {self.model.__class__.__name__} won't be trained"
                + " because the parameter `allow_model_training` is set to False, and no scorer"
                + " is fittable. ``.fit()`` method has no effect."
            )
            return

        list_series = _to_list(series)

        raise_if_not(
            all([isinstance(s, TimeSeries) for s in list_series]),
            "all input `series` must be of type Timeseries.",
        )

        list_past_covariates = self._prepare_covariates(
            past_covariates, list_series, "past"
        )
        list_future_covariates = self._prepare_covariates(
            future_covariates, list_series, "future"
        )

        model_fit_kwargs["past_covariates"] = list_past_covariates
        model_fit_kwargs["future_covariates"] = list_future_covariates

        # fit forecasting model
        if allow_model_training:
            # the model has not been trained yet

            fit_signature_series = (
                inspect.signature(self.model.fit).parameters["series"].annotation
            )

            # checks if model can be trained on multiple time series or only on a time series
            # TODO: check if model can accept multivariate timeseries, raise error if given and model cannot
            if "Sequence[darts.timeseries.TimeSeries]" in str(fit_signature_series):
                self.model.fit(series=list_series, **model_fit_kwargs)
            else:
                raise_if_not(
                    len(list_series) == 1,
                    f"Forecasting model {self.model.__class__.__name__} only accepts a single time series"
                    + " for the training phase and not a sequence of multiple of time series.",
                )
                self.model.fit(series=list_series[0], **model_fit_kwargs)
        else:
            raise_if_not(
                self.model._fit_called,
                f"Model {self.model.__class__.__name__} needs to be trained, consider training "
                + "it beforehand or setting "
                + "`allow_model_training` to True (default: False). "
                + "The model will then be trained on the provided series.",
            )

        # generate the historical_forecast() prediction of the model on the train set
        if self.scorers_are_trainable:
            # check if the window size of the scorers are lower than the max size allowed
            self._check_window_size(list_series, start)

            list_pred = []
            for idx, series in enumerate(list_series):

                if list_past_covariates is not None:
                    past_covariates = list_past_covariates[idx]

                if list_future_covariates is not None:
                    future_covariates = list_future_covariates[idx]

                list_pred.append(
                    self._predict_with_forecasting(
                        series,
                        past_covariates=past_covariates,
                        future_covariates=future_covariates,
                        forecast_horizon=forecast_horizon,
                        start=start,
                        num_samples=num_samples,
                    )
                )

        # fit the scorers
        for scorer in self.scorers:
            if hasattr(scorer, "fit"):
                scorer.fit_from_prediction(list_series, list_pred)

        return self

    def _prepare_covariates(
        self,
        covariates: Union[TimeSeries, Sequence[TimeSeries]],
        series: Sequence[TimeSeries],
        name_covariates: str,
    ) -> Sequence[TimeSeries]:
        """Convert `covariates` into Sequence, if not already, and checks if their length is equal to the one of `series`.

        Parameters
        ----------
        covariates
            Covariate ("future" or "past") of `series`.
        series
            The series to be trained on.
        name_covariates
            Internal parameter for error message, a string indicating if it is a "future" or "past" covariates.

        Returns
        -------
        Sequence[TimeSeries]
            Covariate time series
        """

        if covariates is not None:
            list_covariates = _to_list(covariates)

            for covariates in list_covariates:
                _assert_timeseries(
                    covariates, name_covariates + "_covariates input series"
                )

            raise_if_not(
                len(list_covariates) == len(series),
                f"Number of {name_covariates}_covariates must match the number of given "
                + f"series, found length {len(list_covariates)} and expected {len(series)}.",
            )

        return list_covariates if covariates is not None else None

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
        - show the results of a metric for each anomaly score (AUC_ROC or AUC_PR),
            if the actual anomalies are provided.

        Parameters
        ----------
        series
            The series to visualize anomalies from.
        past_covariates
            An optional past-observed covariate series or sequence of series. This applies only if the model
            supports past covariates.
        future_covariates
            An optional future-known covariate series or sequence of series. This applies only if the model
            supports future covariates.
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
        actual_anomalies
            The ground truth of the anomalies (1 if it is an anomaly and 0 if not)
        names_of_scorers
            Name of the scores. Must be a list of length equal to the number of scorers in the anomaly_model.
        title
            Title of the figure
        metric
            Optionally, Scoring function to use. Must be one of "AUC_ROC" and "AUC_PR".
            Default: "AUC_ROC"
        """

        if isinstance(series, Sequence):
            raise_if_not(
                len(series) == 1,
                f"`show_anomalies` expects one series, found a list of length {len(series)} as input.",
            )

            series = series[0]

        raise_if_not(
            isinstance(series, TimeSeries),
            f"`show_anomalies` expects an input of type TimeSeries, found type: {type(series)}.",
        )

        anomaly_scores, model_output = self.score(
            series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            forecast_horizon=forecast_horizon,
            start=start,
            num_samples=num_samples,
            return_model_prediction=True,
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
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        forecast_horizon: int = 1,
        start: Union[pd.Timestamp, float, int] = 0.5,
        num_samples: int = 1,
        return_model_prediction: bool = False,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Compute anomaly score(s) for the given series.

        Predicts the given target time series with the forecasting model, and applies the scorer(s)
        on the prediction and the target input time series. Outputs the anomaly score of the given
        input time series.

        Parameters
        ----------
        series
            The (sequence of) series to score on.
        past_covariates
            An optional past-observed covariate series or sequence of series. This applies only if the model
            supports past covariates.
        future_covariates
            An optional future-known covariate series or sequence of series. This applies only if the model
            supports future covariates.
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
            directly. Default: 0.5
        num_samples
            Number of times a prediction is sampled from a probabilistic model. Should be left set to 1 for
            deterministic models.
        return_model_prediction
            Boolean value indicating if the prediction of the model should be returned along the anomaly score
            Default: False

        Returns
        -------
        Union[TimeSeries, Sequence[TimeSeries], Sequence[Sequence[TimeSeries]]]
            Anomaly scores series generated by the anomaly model scorers

                - ``TimeSeries`` if `series` is a series, and the anomaly model contains one scorer.
                - ``Sequence[TimeSeries]``

                    * if `series` is a series, and the anomaly model contains multiple scorers,
                      returns one series per scorer.
                    * if `series` is a sequence, and the anomaly model contains one scorer,
                      returns one series per series in the sequence.
                - ``Sequence[Sequence[TimeSeries]]`` if `series` is a sequence, and the anomaly
                  model contains multiple scorers. The outer sequence is over the series,
                  and inner sequence is over the scorers.
        """
        raise_if_not(
            type(return_model_prediction) is bool,
            f"`return_model_prediction` must be Boolean, found type: {type(return_model_prediction)}.",
        )

        raise_if_not(
            self.model._fit_called,
            f"Model {self.model} has not been trained. Please call ``.fit()``.",
        )

        list_series = _to_list(series)

        list_past_covariates = self._prepare_covariates(
            past_covariates, list_series, "past"
        )
        list_future_covariates = self._prepare_covariates(
            future_covariates, list_series, "future"
        )

        # check if the window size of the scorers are lower than the max size allowed
        self._check_window_size(list_series, start)

        list_pred = []
        for idx, s in enumerate(list_series):

            if list_past_covariates is not None:
                past_covariates = list_past_covariates[idx]

            if list_future_covariates is not None:
                future_covariates = list_future_covariates[idx]

            list_pred.append(
                self._predict_with_forecasting(
                    s,
                    past_covariates=past_covariates,
                    future_covariates=future_covariates,
                    forecast_horizon=forecast_horizon,
                    start=start,
                    num_samples=num_samples,
                )
            )

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
                len(s.drop_before(s.get_timestamp_at_point(start))) + 1
            )
            raise_if_not(
                max_window <= max_possible_window,
                f"Window size {max_window} is greater than the targeted series length {max_possible_window},"
                + f" must be lower or equal. Reduce window size, or reduce start value (start: {start}).",
            )

    def _predict_with_forecasting(
        self,
        series: TimeSeries,
        past_covariates: Optional[TimeSeries] = None,
        future_covariates: Optional[TimeSeries] = None,
        forecast_horizon: int = 1,
        start: Union[pd.Timestamp, float, int] = None,
        num_samples: int = 1,
    ) -> TimeSeries:

        """Compute the historical forecasts that would have been obtained by this model on the `series`.

        `retrain` is set to False if possible (this is not supported by all models). If set to True, it will always
        re-train the model on the entire available history,

        Parameters
        ----------
        series
            The target time series to use to successively train and evaluate the historical forecasts.
        past_covariates
            An optional past-observed covariate series or sequence of series. This applies only if the model
            supports past covariates.
        future_covariates
            An optional future-known covariate series or sequence of series. This applies only if the model
            supports future covariates.
        forecast_horizon
            The forecast horizon for the predictions
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

        Returns
        -------
        TimeSeries
            Single ``TimeSeries`` instance created from the last point of each individual forecast.
        """

        # TODO: raise an exception. We only support models that do not need retrain
        # checks if model accepts to not be retrained in the historical_forecasts()
        if self.model._supports_non_retrainable_historical_forecasts():
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

    def eval_accuracy(
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

        Predicts the `series` with the forecasting model, and applies the
        scorer(s) on the predicted time series and the given target time series. Returns the
        score(s) of an agnostic threshold metric, based on the anomaly score given by the scorer(s).

        Parameters
        ----------
        actual_anomalies
            The (sequence of) ground truth of the anomalies (1 if it is an anomaly and 0 if not)
        series
            The (sequence of) series to predict anomalies on.
        past_covariates
            An optional past-observed covariate series or sequence of series. This applies only
            if the model supports past covariates.
        future_covariates
            An optional future-known covariate series or sequence of series. This applies only
            if the model supports future covariates.
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
            Optionally, Scoring function to use. Must be one of "AUC_ROC" and "AUC_PR".
            Default: "AUC_ROC"

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

        list_actual_anomalies = _to_list(actual_anomalies)
        list_series = _to_list(series)

        raise_if_not(
            all([isinstance(s, TimeSeries) for s in list_series]),
            "all input `series` must be of type Timeseries.",
        )

        raise_if_not(
            all([isinstance(s, TimeSeries) for s in list_actual_anomalies]),
            "all input `actual_anomalies` must be of type Timeseries.",
        )

        _assert_same_length(list_actual_anomalies, list_series)
        self._check_univariate(list_actual_anomalies)

        list_anomaly_scores = self.score(
            series=list_series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            forecast_horizon=forecast_horizon,
            start=start,
            num_samples=num_samples,
        )

        acc_anomaly_scores = self._eval_accuracy_from_scores(
            list_actual_anomalies=list_actual_anomalies,
            list_anomaly_scores=list_anomaly_scores,
            metric=metric,
        )

        if len(acc_anomaly_scores) == 1 and not isinstance(series, Sequence):
            return acc_anomaly_scores[0]
        else:
            return acc_anomaly_scores
