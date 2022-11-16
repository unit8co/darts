"""
Anomaly Model
------------

An anomaly model expect a model and a scorer. It offers a ``score()``function, which takes
as input a time series and returns its anomaly score as a ``TimeSeries``.

The model can be a forecasting method (:class:`ForecastingAnomalyModel`) or a filtering method
(:class:`FilteringAnomalyModel`).
The main functions are ``fit()`` (only for the trainable model/scorer), ``score()`` and
``eval_accuracy()``. ``fit()`` trains the model and/or the scorer over the history of one or multiple series.
``score()`` applies the model on the time series input and and calls the scorer to return an anomaly score of
the input series.

The function ``eval_accuracy()` is the same as ``score()``, but outputs the score of an agnostic
threshold metric (AUC-ROC or AUC-PR), between the predicted anomaly score time series, and some known binary
ground-truth time series indicating the presence of actual anomalies.

TODO:
    - check if warning is the right way to do: with import warnings
"""

import inspect
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Union

import pandas as pd

from darts.ad.scorers import AnomalyScorer
from darts.ad.utils import (
    _convert_to_list,
    eval_accuracy_from_scores,
    show_anomalies_from_scores,
)
from darts.logging import raise_if_not
from darts.models.filtering.filtering_model import FilteringModel
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.timeseries import TimeSeries


class AnomalyModel(ABC):
    "Base class for all anomaly models"

    def __init__(self, model, scorer):

        self.scorers = [scorer] if not isinstance(scorer, Sequence) else scorer

        self.scorers_are_trainable = False
        for index, scorer in enumerate(self.scorers):
            raise_if_not(
                isinstance(scorer, AnomalyScorer),
                f"Scorer must be a darts.ad.scorers not a {type(scorer)}",
            )

            if scorer.trainable:
                self.scorers_are_trainable = True

        self.model = model

    @abstractmethod
    def fit(
        self, series: Union[TimeSeries, Sequence[TimeSeries]]
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        pass

    @abstractmethod
    def score(
        self, series: Union[TimeSeries, Sequence[TimeSeries]]
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        pass

    @abstractmethod
    def eval_accuracy(
        self, series: Union[TimeSeries, Sequence[TimeSeries]]
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        pass

    @abstractmethod
    def show_anomalies(self, series: TimeSeries):
        pass

    def _show_anomalies(
        self,
        series: TimeSeries,
        model_output: TimeSeries = None,
        anomaly_scores: Union[TimeSeries, Sequence[TimeSeries]] = None,
        name_of_scorers: Union[str, Sequence[str]] = None,
        actual_anomalies: TimeSeries = None,
        title: str = None,
        save_png: str = None,
        metric: str = None,
    ):
        """Internal function that plots the results of the anomaly model.
        Called by the function show_anomalies().
        """

        if title is None:
            title = f"Anomaly results by model {self.model.__class__.__name__}"

        if name_of_scorers is None:
            name_of_scorers = []
            for scorer in self.scorers:
                name_of_scorers.append(scorer.__str__())

        list_window = []
        for scorer in self.scorers:
            list_window.append(scorer.window)

        return show_anomalies_from_scores(
            series,
            model_output=model_output,
            anomaly_scores=anomaly_scores,
            window=list_window,
            name_of_scorers=name_of_scorers,
            actual_anomalies=actual_anomalies,
            title=title,
            save_png=save_png,
            metric=metric,
        )


class ForecastingAnomalyModel(AnomalyModel):
    def __init__(
        self,
        model: ForecastingModel,
        scorer: Union[AnomalyScorer, Sequence[AnomalyScorer]],
    ):
        """Forecasting-based Anomaly Model

        Wraps around a fitted Darts forecasting model and an anomaly scorer to compute anomaly scores
        by comparing how actuals deviate from the model's predictions.

        By assumption, the Darts forecasting model should be able to forecast accuratly the considered
        scenario when there are no anomalies.

        Parameters
        ----------
        model
            A fitted Darts forecasting model.
        scorer
            A scorer that will be used to convert the actual and predicted time series to
            an anomaly score ``TimeSeries``. If a list of `N` scorer is given, the anomaly model will call each
            one of the scorers and output a list of `N` anomaly scores ``TimeSeries``.
        """

        raise_if_not(
            isinstance(model, ForecastingModel),
            f"Model must be a darts.models.forecasting not a {type(model)}",
        )
        self.model = model

        super().__init__(model=model, scorer=scorer)

    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        allow_model_training: bool = False,
        model_fit_params: Optional[Dict[str, Any]] = None,
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        forecast_horizon: int = None,
        start: Union[pd.Timestamp, float, int] = None,
        num_samples: int = None,
    ):
        """Train the model (if not already fitted and allow_model_training is set to True) and the
        scorer(s) (if fittable) on the given time series.

        Once the model is fitted, the series historical forecasts is computed,
        representing what would have been obtained by this model on the series.

        The prediction and the series will then be used to train the scorer(s).

        Parameters
        ----------
        series
            The series to be trained on (anomaly free).
        allow_model_training
            Boolean value that indicates if the forecasting model needs to be fitted on the given series.
            If set to False, the model needs to be already fitted.
            Default: False
        model_fit_params
            Parameters to be passed on to the forecast model ``fit()`` method.
        past_covariates
            An optional past-observed covariate series. This applies only if the model supports past covariates.
        future_covariates
            An optional future-known covariate series. This applies only if the model supports future covariates.
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

        Returns
        -------
        self
            Fitted Anomaly model (forecasting model and scorer(s))
        """

        # checks if model does not need training and all scorer(s) are not fittable
        if not allow_model_training and not self.scorers_are_trainable:
            warnings.warn(
                f"Warning: the forecasting model {self.model.__class__.__name__} is not required to be \
            trained because the parameter allow_model_training is set to False, and all scorers are not fittable.\
            No need to call the .fit() function"
            )

        list_series = [series] if not isinstance(series, Sequence) else series

        list_past_covariates = self._check_covariates(
            past_covariates, list_series, "past"
        )
        list_future_covariates = self._check_covariates(
            future_covariates, list_series, "future"
        )

        if model_fit_params is None:
            model_fit_params = {}

        raise_if_not(
            isinstance(model_fit_params, dict),
            f"model_fit_params must be of type dictionary, found {type(model_fit_params)}",
        )

        model_fit_params["past_covariates"] = list_past_covariates
        model_fit_params["future_covariates"] = list_future_covariates

        # remove None element from dictionary model_fit_params
        model_fit_params = {k: v for k, v in model_fit_params.items() if v is not None}

        # fit forecasting model
        if allow_model_training:
            # the model has not been trained yet

            fit_signature_series = (
                inspect.signature(self.model.fit).parameters["series"].annotation
            )

            # checks if model can be trained on a list of time series or only on a time series
            # TODO: check if model can accept multivariate timeseries, raise error if given and model cannot
            if "Sequence[darts.timeseries.TimeSeries]" in str(fit_signature_series):
                self.model.fit(series=list_series, **model_fit_params)
            else:
                raise_if_not(
                    len(list_series) == 1,
                    f"Forecasting model {self.model.__class__.__name__} only accepts a time series for the training \
                    phase and not a list of time series.",
                )
                self.model.fit(series=list_series[0], **model_fit_params)
        else:
            raise_if_not(
                self.model._fit_called,
                f"Model {self.model.__class__.__name__} needs to be trained, otherwise set parameter 'allow_model_training' to True \
                (default: False). The model will then be trained on the given input series.",
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
                scorer.fit(list_pred, list_series)

    def _check_covariates(
        self,
        covariates: Union[TimeSeries, Sequence[TimeSeries]],
        series: Sequence[TimeSeries],
        name: str,
    ) -> Sequence[TimeSeries]:
        """Converts 'covariates' into Sequence, if not already, and checks if its length is equal to the one of 'series'.

        Parameters
        ----------
        covariates
            Covariate ("future" or "past") of 'series'.
        series
            The series to be trained on.
        name
            Internal parameter for error message, a string indicating if it is a "future" or "past" covariates.

        Returns
        -------
        Sequence[TimeSeries]
            Covariate time series
        """

        if covariates is not None:
            covariates = (
                [covariates] if not isinstance(covariates, Sequence) else covariates
            )

            raise_if_not(
                len(covariates) == len(series),
                f"Number of {name}_covariates must match the number of given series, \
                found length: {len(covariates)} and {len(series)}",
            )

        return covariates

    def show_anomalies(
        self,
        series=TimeSeries,
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        forecast_horizon: int = None,
        start: Union[pd.Timestamp, float, int] = None,
        num_samples: int = None,
        actual_anomalies=TimeSeries,
        name_of_scorers: Union[str, Sequence[str]] = None,
        title: str = None,
        save_png: str = None,
        metric: str = None,
    ):
        """Plot the results of the anomaly model.

        Computes the score on the given series input. And shows the different anomaly scores with respect to time.

        The plot will be composed of the following:
            - the series itself with the output of the forecasting model.
            - the anomaly score of each scorer. The scorer with different windows will be separated.
            - the actual anomalies, if given.

        Possible to:
            - add a title to the figure with the parameter 'title'
            - give personalized names for the scorers with 'name_of_scorers'
            - save the plot as a png at the path 'save_png'
            - show the results of a metric for each anomaly score (AUC_ROC or AUC_PR), if the actual anomalies is given.

        Parameters
        ----------
        series
            The series to visualize anomalies from.
        past_covariates
            An optional past-observed covariate series. This applies only if the model supports past covariates.
        future_covariates
            An optional future-known covariate series. This applies only if the model supports future covariates.
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
        name_of_scorers
            Name of the scores. Must be a list of length equal to the number of scorers in the anomaly_model.
        title
            Title of the figure
        save_png
            Path to where the plot in format png should be saved
            Default: None (the plot will not be saved)
        metric
            Optionally, Scoring function to use. Must be one of "AUC_ROC" and "AUC_PR".
            Default: "AUC_ROC"
        """

        if isinstance(series, Sequence):
            raise_if_not(
                len(series) == 1,
                f"'visualize_anomalies' expects one series, found a list of length {len(series)} as input.",
            )

            series = series[0]

        raise_if_not(
            isinstance(series, TimeSeries),
            f"'visualize_anomalies' expects an input of type TimeSeries, found type: {type(series)}.",
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
            name_of_scorers=name_of_scorers,
            actual_anomalies=actual_anomalies,
            title=title,
            save_png=save_png,
            metric=metric,
        )

    def score(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        forecast_horizon: int = None,
        start: Union[pd.Timestamp, float, int] = None,
        num_samples: int = None,
        return_model_prediction: bool = False,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Predicts the given input time series with the forecasting model, and applies the scorer(s)
        on the prediction and the given input time series. Outputs the anomaly score of the given
        input time series.

        Parameters
        ----------
        series
            The series to score on.
        past_covariates
            An optional past-observed covariate series. This applies only if the model supports past covariates.
        future_covariates
            An optional future-known covariate series. This applies only if the model supports future covariates.
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
            Boolean value indicating if the prediction of the model should be returned along the anomaly score
            Default: False

        Returns
        -------
        Union[TimeSeries, Sequence[TimeSeries]]
            Anomaly score time series generated by a scorer (list of anomaly score if more than one scorer)
        """

        raise_if_not(
            self.model._fit_called,
            f"Model {self.model} has not been trained. Please call .fit()",
        )

        list_series = [series] if not isinstance(series, Sequence) else series

        list_past_covariates = self._check_covariates(
            past_covariates, list_series, "past"
        )
        list_future_covariates = self._check_covariates(
            future_covariates, list_series, "future"
        )

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

        list_anomaly_scores = []
        for scorer in self.scorers:
            list_anomaly_scores.append(
                scorer.score(
                    list_pred if len(list_pred) > 1 else list_pred[0],
                    list_series if len(list_series) > 1 else list_series[0],
                )
            )

        if len(list_anomaly_scores) == 1 and not isinstance(series, Sequence):
            list_anomaly_scores = list_anomaly_scores[0]

        if len(list_pred) == 1:
            list_pred = list_pred[0]

        if return_model_prediction:
            return list_anomaly_scores, list_pred
        else:
            return list_anomaly_scores

    def _check_window_size(
        self, series: Sequence[TimeSeries], start: Union[pd.Timestamp, float, int]
    ):
        """Checks if the parameters 'window' of the scorers are smaller than the maximum window size allowed.
        The maximum size allowed is equal to the output length of the .historical_forecast() applied on 'series'.
        It is defined by the parameter 'start' and the series’ length.

        Parameters
        ----------
        series
            The series given to the .historical_forecast()
        start
            Parameter of the .historical_forecast(): first point of time at which a prediction is computed
            for a future time.
        """
        if start is None:
            start = 0.5

        for scorer in self.scorers:
            for s in series:
                max_possible_window = len(s) - len(
                    s.drop_after(s.get_timestamp_at_point(start))
                )
                raise_if_not(
                    scorer.window <= max_possible_window,
                    f"Window size {scorer.window} is greater than the targeted series length {max_possible_window}, \
                    must be lower or equal. Reduce window size, or reduce start value (start value: 0.5).",
                )

    def _predict_with_forecasting(
        self,
        series: TimeSeries,
        past_covariates: Optional[TimeSeries] = None,
        future_covariates: Optional[TimeSeries] = None,
        forecast_horizon: int = None,
        start: Union[pd.Timestamp, float, int] = None,
        num_samples=None,
    ) -> TimeSeries:

        """Compute the historical forecasts that would have been obtained by this model on the `series`.

        `retrain` is set to False if possible (this is not supported by all models). If set to True, it will always
        re-train the model on the entire available history,

        Parameters
        ----------
        series
            The target time series to use to successively train and evaluate the historical forecasts.
        past_covariates
            An optional past-observed covariate series. This applies only if the model supports past covariates.
        future_covariates
            An optional future-known covariate series. This applies only if the model supports future covariates.
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

        if forecast_horizon is None:
            forecast_horizon = 1

        if start is None:
            # TODO: set 'start' to its minimal possible value
            start = 0.5

        if num_samples is None:
            num_samples = 1

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

        # remove None element from dictionary historical_forecasts_param
        historical_forecasts_param = {
            k: v for k, v in historical_forecasts_param.items() if v is not None
        }

        return self.model.historical_forecasts(series, **historical_forecasts_param)

    def eval_accuracy(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        actual_anomalies: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        forecast_horizon: int = None,
        start: Union[pd.Timestamp, float, int] = None,
        num_samples=None,
        metric: str = "AUC_ROC",
    ) -> Union[float, Sequence[float], Sequence[Sequence[float]]]:
        """Predicts the 'series' with the forecasting model, and applies the
        scorer(s) on the filtered time series and the given input time series. Returns the
        score(s) of an agnostic threshold metric, based on the anomaly score given by the scorer(s).

        Parameters
        ----------
        series
            The series to predict anomalies on.
        actual_anomalies
            The ground truth of the anomalies (1 if it is an anomaly and 0 if not)
        past_covariates
            An optional past-observed covariate series. This applies only if the model supports past covariates.
        future_covariates
            An optional future-known covariate series. This applies only if the model supports future covariates.
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
        Union[float, Sequence[float], Sequence[Sequence[float]]]
            Score for the time series
        """

        list_series, list_actual_anomalies = _convert_to_list(series, actual_anomalies)

        anomaly_scores = self.score(
            series=list_series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            forecast_horizon=forecast_horizon,
            start=start,
            num_samples=num_samples,
        )

        list_anomaly_scores = (
            [anomaly_scores]
            if not isinstance(anomaly_scores, Sequence)
            else anomaly_scores
        )

        acc_anomaly_scores = []
        for idx, scorer in enumerate(self.scorers):
            acc_anomaly_scores.append(
                eval_accuracy_from_scores(
                    anomaly_score=list_anomaly_scores[idx],
                    actual_anomalies=list_actual_anomalies,
                    window=scorer.window,
                    metric=metric,
                )
            )

        if len(acc_anomaly_scores) == 1 and not isinstance(series, Sequence):
            return acc_anomaly_scores[0]
        else:
            return acc_anomaly_scores


class FilteringAnomalyModel(AnomalyModel):
    def __init__(
        self,
        filter: FilteringModel,
        scorer: Union[AnomalyScorer, Sequence[AnomalyScorer]],
    ):
        """Filtering-based Anomaly Model

        Wraps around a fitted Darts filtering model and an anomaly scorer to compute anomaly scores
        by comparing how actuals deviate from the model's output.

        Parameters
        ----------
        filter
            A filtering model from Darts that will be used to filter the actual time series
        scorer
            A scorer that will be used to convert the actual and filtered time series to
            an anomaly score ``TimeSeries``. If a list of `N` scorer is given, the anomaly model will call each
            one of the scorers and output a list of `N` anomaly scores ``TimeSeries``.
        """

        raise_if_not(
            isinstance(filter, FilteringModel),
            f"Filter must be a darts.models.filtering not a {type(filter)}",
        )
        self.filter = filter

        super().__init__(model=filter, scorer=scorer)

    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        allow_model_training: bool = False,
        filter_fit_params: Optional[Dict[str, Any]] = None,
    ):
        """Train the filter (if not already fitted and allow_model_training is set to True)
        and the scorer(s) on the given time series.

        The filter model will be applied to the given series, and the results will be used
        to train the scorer(s).

        Parameters
        ----------
        series
            The series to be trained on.
        allow_model_training
            Boolean value that indicates if the filtering model needs to be fitted on the given series.
            If set to False, the model needs to be already fitted.
            Default: False
        filter_fit_params
            Parameters to be passed on to the filtering model ``fit()`` method.

        Returns
        -------
        self
            Fitted Anomaly model (filtering model and scorer(s))
        """

        list_series = [series] if not isinstance(series, Sequence) else series

        if filter_fit_params is None:
            filter_fit_params = {}

        raise_if_not(
            isinstance(filter_fit_params, dict),
            f"filter_fit_params must be of type dictionary, found {type(filter_fit_params)}",
        )

        if allow_model_training:
            # fit filtering model
            if hasattr(self.filter, "fit"):
                # TODO: check if filter is already fitted (for now fit it regardless -> only Kallman)
                raise_if_not(
                    len(list_series) == 1,
                    f"Filter model {self.model.__class__.__name__} can only be fitted on a \
                    time series and not a list of time series",
                )

                self.filter.fit(list_series[0], **filter_fit_params)
        else:
            # TODO: check if Kallman is fitted or not
            # if not raise error "fit filter before, or set 'allow_model_training' to TRUE"
            pass

        if self.scorers_are_trainable:
            list_pred = []
            for series in list_series:
                list_pred.append(self.filter.filter(series))

        # fit the scorers
        for scorer in self.scorers:
            if hasattr(scorer, "fit"):
                scorer.fit(list_pred, list_series)

    def show_anomalies(
        self,
        series=TimeSeries,
        filter_params: Optional[Dict[str, Any]] = None,
        actual_anomalies=TimeSeries,
        name_of_scorers: Union[str, Sequence[str]] = None,
        title: str = None,
        save_png: str = None,
        metric: str = None,
    ):
        """Plot the results of the anomaly model.

        Computes the score on the given series input. And shows the different anomaly scores with respect to time.

        The plot will be composed of the following:
            - the series itself with the output of the filtering model
            - the anomaly score of each scorer. The scorer with different windows will be separated.
            - the actual anomalies, if given.

        Possible to:
            - add a title to the figure with the parameter 'title'
            - give personalized names for the scorers with 'name_of_scorers'
            - save the plot as a png at the path 'save_png'
            - show the results of a metric for each anomaly score (AUC_ROC or AUC_PR), if the actual anomalies is given

        Parameters
        ----------
        series
            The series to visualize anomalies from.
        filter_params
            parameters of the Darts `.filter()` filtering model
        actual_anomalies
            The ground truth of the anomalies (1 if it is an anomaly and 0 if not)
        name_of_scorers
            Name of the scores. Must be a list of length equal to the number of scorers in the anomaly_model.
        title
            Title of the figure
        save_png
            Path to where the plot in format png should be saved
            Default: None (the plot will not be saved)
        metric
            Optionally, Scoring function to use. Must be one of "AUC_ROC" and "AUC_PR".
            Default: "AUC_ROC"
        """

        if isinstance(series, Sequence):
            raise_if_not(
                len(series) == 1,
                f"'visualize_anomalies' expects one series, found a list of length {len(series)} as input.",
            )

            series = series[0]

        anomaly_scores, model_output = self.score(
            series, filter_params=filter_params, return_model_prediction=True
        )

        return self._show_anomalies(
            series,
            model_output=model_output,
            anomaly_scores=anomaly_scores,
            name_of_scorers=name_of_scorers,
            actual_anomalies=actual_anomalies,
            title=title,
            save_png=save_png,
            metric=metric,
        )

    def score(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        filter_params: Optional[Dict[str, Any]] = None,
        return_model_prediction: bool = False,
    ):
        """Filters the given input time series with the filtering model, and applies the scorer(s)
        on the filtered time series and the given input time series. Outputs the anomaly score of
        the given input time series.

        Parameters
        ----------
        series
            The series to score on.
        filter_params
            parameters of the Darts `.filter()` filtering model
        return_model_prediction
            Boolean value indicating if the prediction of the model should be returned along the anomaly score
            Default: False

        Returns
        -------
        Union[TimeSeries, Sequence[TimeSeries]]
            Anomaly score time series generated by a scorer (list of anomaly score if more than one scorer)
        """

        list_series = [series] if not isinstance(series, Sequence) else series

        if filter_params is None:
            filter_params = {}

        raise_if_not(
            isinstance(filter_params, dict),
            f"filter_fit_params must be of type dictionary, found {type(filter_params)}",
        )

        list_pred = []
        for s in list_series:
            list_pred.append(self.filter.filter(s, **filter_params))

        list_anomaly_scores = []
        for scorer in self.scorers:
            list_anomaly_scores.append(
                scorer.score(
                    list_pred if len(list_pred) > 1 else list_pred[0],
                    list_series if len(list_series) > 1 else list_series[0],
                )
            )

        if len(list_anomaly_scores) == 1 and not isinstance(series, Sequence):
            list_anomaly_scores = list_anomaly_scores[0]

        if len(list_pred) == 1:
            list_pred = list_pred[0]

        if return_model_prediction:
            return list_anomaly_scores, list_pred
        else:
            return list_anomaly_scores

    def eval_accuracy(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        actual_anomalies: Union[TimeSeries, Sequence[TimeSeries]],
        filter_params: Optional[Dict[str, Any]] = None,
        metric: str = "AUC_ROC",
    ):
        """Filters the given input time series with the filtering model, and applies the scorer(s)
        on the filtered time series and the given input time series. Returns the score(s)
        of an agnostic threshold metric, based on the anomaly score given by the scorer(s).

        Parameters
        ----------
        series
            The series to predict anomalies on.
        actual_anomalies
            The ground truth of the anomalies (1 if it is an anomaly and 0 if not)
        filter_params: dict, optional
            parameters of the Darts `.filter()` filtering model
        metric
            Optionally, Scoring function to use. Must be one of "AUC_ROC" and "AUC_PR".
            Default: "AUC_ROC"

        Returns
        -------
        Union[float, Sequence[float], Sequence[Sequence[float]]]
            Score for the time series
        """

        list_series, list_actual_anomalies = _convert_to_list(series, actual_anomalies)

        anomaly_scores = self.score(series=list_series, filter_params=filter_params)

        list_anomaly_scores = (
            [anomaly_scores]
            if not isinstance(anomaly_scores, Sequence)
            else anomaly_scores
        )

        acc_anomaly_scores = []
        for idx, scorer in enumerate(self.scorers):

            acc_anomaly_scores.append(
                eval_accuracy_from_scores(
                    anomaly_score=list_anomaly_scores[idx],
                    actual_anomalies=list_actual_anomalies,
                    window=scorer.window,
                    metric=metric,
                )
            )

        if len(acc_anomaly_scores) == 1 and not isinstance(series, Sequence):
            return acc_anomaly_scores[0]
        else:
            return acc_anomaly_scores
