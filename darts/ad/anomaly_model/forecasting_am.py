"""
Forecasting Anomaly Model
-------------------------

A `ForecastingAnomalyModel` wraps around a Darts forecasting model and one or several anomaly
scorer(s) to compute anomaly scores by comparing how actuals deviate from the model's forecasts.
"""

# TODO:
#     - put start default value to its minimal value (wait for the release of historical_forecast)
import sys
from collections.abc import Sequence
from typing import Literal, Optional, Union

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import pandas as pd

from darts.ad.anomaly_model.anomaly_model import AnomalyModel
from darts.ad.scorers.scorers import AnomalyScorer
from darts.logging import get_logger, raise_log
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from darts.timeseries import TimeSeries

logger = get_logger(__name__)


class ForecastingAnomalyModel(AnomalyModel):
    def __init__(
        self,
        model: GlobalForecastingModel,
        scorer: Union[AnomalyScorer, Sequence[AnomalyScorer]],
    ):
        """Forecasting-based Anomaly Detection Model

        The forecasting model must be a `GlobalForecastingModel` that may or may not be already fitted. The
        underlying assumption is that `model` should be able to accurately forecast the series in the absence of
        anomalies. For this reason, it is recommended to either provide a model that has already been fitted and
        evaluated to work appropriately on a series without anomalies, or to ensure that a simple call to the
        :func:`fit()` method of the model will be sufficient to train it to satisfactory performance on a series
        without anomalies. The pre-trained model will be used to generate forecasts when calling :func:`score()`.

        Calling :func:`fit()` on the anomaly model will fit the underlying forecasting model only if
        `allow_model_training` is set to `True` upon calling `fit()`.
        In addition, calling :func:`fit()` will also fit the fittable scorers, if any.

        Parameters
        ----------
        model
            An instance of a Darts forecasting model.
        scorer
            One or multiple scorer(s) that will be used to compare the actual and predicted time series in order
            to obtain an anomaly score `TimeSeries`.
            If a list of `N` scorers is given, the anomaly model will call each
            one of the scorers and output a list of `N` anomaly scores `TimeSeries`.
        """
        if not isinstance(model, GlobalForecastingModel):
            raise_log(
                ValueError("`model` must be a Darts `GlobalForecastingModel`."),
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
        start: Union[pd.Timestamp, float, int] = None,
        start_format: Literal["position", "value"] = "value",
        num_samples: int = 1,
        verbose: bool = False,
        show_warnings: bool = True,
        enable_optimization: bool = True,
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
            This parameter supports 3 different data types: `float`, `int` and `pandas.Timestamp`.
            In the case of `float`, the parameter will be treated as the proportion of the time series
            that should lie before the first prediction point.
            In the case of `int`, the parameter will be treated as an integer index to the time index of
            `series` that will be used as first prediction time.
            In case of `pandas.Timestamp`, this time stamp will be used to determine the first prediction time
            directly.
        start_format
            Defines the `start` format. Only effective when `start` is an integer and `series` is indexed with a
            `pd.RangeIndex`.
            If set to 'position', `start` corresponds to the index position of the first predicted point and can range
            from `(-len(series), len(series) - 1)`.
            If set to 'value', `start` corresponds to the index value/label of the first predicted point. Will raise
            an error if the value is not in `series`' index. Default: `'value'`
        num_samples
            Number of times a prediction is sampled from a probabilistic model. Must be `1` for deterministic models.
        verbose
            Whether to print the progress.
        show_warnings
            Whether to show warnings related to historical forecasts optimization, or parameters `start` and
            `train_length`.
        enable_optimization
            Whether to use the optimized version of historical_forecasts when supported and available.
            Default: ``True``.
        model_fit_kwargs
            Parameters to be passed on to the forecast model `fit()` method.

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
            start_format=start_format,
            num_samples=num_samples,
            verbose=verbose,
            show_warnings=show_warnings,
            enable_optimization=enable_optimization,
            **model_fit_kwargs,
        )

    def score(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        forecast_horizon: int = 1,
        start: Union[pd.Timestamp, float, int] = None,
        start_format: Literal["position", "value"] = "value",
        num_samples: int = 1,
        verbose: bool = False,
        show_warnings: bool = True,
        enable_optimization: bool = True,
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
            This parameter supports 3 different data types: `float`, `int` and `pandas.Timestamp`.
            In the case of `float`, the parameter will be treated as the proportion of the time series
            that should lie before the first prediction point.
            In the case of `int`, the parameter will be treated as an integer index to the time index of
            `series` that will be used as first prediction time.
            In case of `pandas.Timestamp`, this time stamp will be used to determine the first prediction time
            directly.
        start_format
            Defines the `start` format. Only effective when `start` is an integer and `series` is indexed with a
            `pd.RangeIndex`.
            If set to 'position', `start` corresponds to the index position of the first predicted point and can range
            from `(-len(series), len(series) - 1)`.
            If set to 'value', `start` corresponds to the index value/label of the first predicted point. Will raise
            an error if the value is not in `series`' index. Default: `'value'`
        num_samples
            Number of times a prediction is sampled from a probabilistic model. Must be `1` for deterministic models.
        verbose
            Whether to print the progress.
        show_warnings
            Whether to show warnings related to historical forecasts optimization, or parameters `start` and
            `train_length`.
        enable_optimization
            Whether to use the optimized version of historical_forecasts when supported and available.
            Default: ``True``.
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
            start_format=start_format,
            num_samples=num_samples,
            verbose=verbose,
            show_warnings=show_warnings,
            enable_optimization=enable_optimization,
            return_model_prediction=return_model_prediction,
        )

    def predict_series(
        self,
        series: Sequence[TimeSeries],
        past_covariates: Optional[Sequence[TimeSeries]] = None,
        future_covariates: Optional[Sequence[TimeSeries]] = None,
        forecast_horizon: int = 1,
        start: Union[pd.Timestamp, float, int] = None,
        start_format: Literal["position", "value"] = "value",
        num_samples: int = 1,
        verbose: bool = False,
        show_warnings: bool = True,
        enable_optimization: bool = True,
    ) -> Sequence[TimeSeries]:
        """Computes the historical forecasts that would have been obtained by the underlying forecasting model
        on `series`.

        `retrain` is set to `False` if possible (this is not supported by all models). If set to `True`, it will always
        re-train the model on the entire available history,

        Parameters
        ----------
        series
            The sequence of series to score on.
        past_covariates
            Optionally, a sequence of past-observed covariate series or sequence of series. This applies only to
            models that support past covariates.
        future_covariates
            Optionally, a sequence of future-known covariate series or sequence of series. This applies only to
            models that support future covariates.
        forecast_horizon
            The forecast horizon for the predictions.
        start
            The first point of time at which a prediction is computed for a future time.
            This parameter supports 3 different data types: `float`, `int` and `pandas.Timestamp`.
            In the case of `float`, the parameter will be treated as the proportion of the time series
            that should lie before the first prediction point.
            In the case of `int`, the parameter will be treated as an integer index to the time index of
            `series` that will be used as first prediction time.
            In case of `pandas.Timestamp`, this time stamp will be used to determine the first prediction time
            directly.
        start_format
            Defines the `start` format. Only effective when `start` is an integer and `series` is indexed with a
            `pd.RangeIndex`.
            If set to 'position', `start` corresponds to the index position of the first predicted point and can range
            from `(-len(series), len(series) - 1)`.
            If set to 'value', `start` corresponds to the index value/label of the first predicted point. Will raise
            an error if the value is not in `series`' index. Default: `'value'`
        num_samples
            Number of times a prediction is sampled from a probabilistic model. Must be `1` for deterministic models.
        verbose
            Whether to print the progress.
        show_warnings
            Whether to show warnings related to historical forecasts optimization, or parameters `start` and
            `train_length`.
        enable_optimization
            Whether to use the optimized version of historical_forecasts when supported and available.
            Default: ``True``.

        Returns
        -------
        Sequence[TimeSeries]
            A sequence of `TimeSeries` with the historical forecasts for each series (with `last_points_only=True`).
        """
        if not self.model._fit_called:
            raise_log(
                ValueError(
                    f"Forecasting `model` {self.model} has not been trained yet. Call `fit()` before."
                ),
                logger=logger,
            )
        return self.model.historical_forecasts(
            series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            forecast_horizon=forecast_horizon,
            stride=1,
            retrain=False,
            last_points_only=True,
            start=start,
            start_format=start_format,
            num_samples=num_samples,
            verbose=verbose,
            show_warnings=show_warnings,
            enable_optimization=enable_optimization,
        )

    def eval_metric(
        self,
        anomalies: Union[TimeSeries, Sequence[TimeSeries]],
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        forecast_horizon: int = 1,
        start: Union[pd.Timestamp, float, int] = None,
        start_format: Literal["position", "value"] = "value",
        num_samples: int = 1,
        verbose: bool = False,
        show_warnings: bool = True,
        enable_optimization: bool = True,
        metric: Literal["AUC_ROC", "AUC_PR"] = "AUC_ROC",
    ) -> Union[
        dict[str, float],
        dict[str, Sequence[float]],
        Sequence[dict[str, float]],
        Sequence[dict[str, Sequence[float]]],
    ]:
        """Compute the accuracy of the anomaly scores computed by the model.

        Predicts the `series` with the forecasting model, and applies the scorer(s) on the predicted time series
        and the given target time series. Returns the score(s) of an agnostic threshold metric, based on the anomaly
        score given by the scorer(s).

        Parameters
        ----------
        anomalies
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
            This parameter supports 3 different data types: `float`, `int` and `pandas.Timestamp`.
            In the case of `float`, the parameter will be treated as the proportion of the time series
            that should lie before the first prediction point.
            In the case of `int`, the parameter will be treated as an integer index to the time index of
            `series` that will be used as first prediction time.
            In case of `pandas.Timestamp`, this time stamp will be used to determine the first prediction time
            directly.
        start_format
            Defines the `start` format. Only effective when `start` is an integer and `series` is indexed with a
            `pd.RangeIndex`.
            If set to 'position', `start` corresponds to the index position of the first predicted point and can range
            from `(-len(series), len(series) - 1)`.
            If set to 'value', `start` corresponds to the index value/label of the first predicted point. Will raise
            an error if the value is not in `series`' index. Default: `'value'`
        num_samples
            Number of times a prediction is sampled from a probabilistic model. Must be `1` for deterministic models.
        verbose
            Whether to print the progress.
        show_warnings
            Whether to show warnings related to historical forecasts optimization, or parameters `start` and
            `train_length`.
        enable_optimization
            Whether to use the optimized version of historical_forecasts when supported and available.
            Default: ``True``.
        metric
            The name of the metric function to use. Must be one of "AUC_ROC" (Area Under the
            Receiver Operating Characteristic Curve) and "AUC_PR" (Average Precision from scores).
            Default: "AUC_ROC".

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
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            forecast_horizon=forecast_horizon,
            start=start,
            start_format=start_format,
            num_samples=num_samples,
            verbose=verbose,
            show_warnings=show_warnings,
            enable_optimization=enable_optimization,
            metric=metric,
        )

    def show_anomalies(
        self,
        series: TimeSeries,
        past_covariates: Optional[TimeSeries] = None,
        future_covariates: Optional[TimeSeries] = None,
        forecast_horizon: int = 1,
        start: Union[pd.Timestamp, float, int] = None,
        start_format: Literal["position", "value"] = "value",
        num_samples: int = 1,
        verbose: bool = False,
        show_warnings: bool = True,
        enable_optimization: bool = True,
        anomalies: TimeSeries = None,
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
        past_covariates
            Optionally, a past-observed covariate series or sequence of series. This applies only to
            models that support past covariates.
        future_covariates
            Optionally, a future-known covariate series or sequence of series. This applies only to models that support
            future covariates.
        forecast_horizon
            The forecast horizon for the predictions.
        start
            The first point of time at which a prediction is computed for a future time.
            This parameter supports 3 different data types: `float`, `int` and `pandas.Timestamp`.
            In the case of `float`, the parameter will be treated as the proportion of the time series
            that should lie before the first prediction point.
            In the case of `int`, the parameter will be treated as an integer index to the time index of
            `series` that will be used as first prediction time.
            In case of `pandas.Timestamp`, this time stamp will be used to determine the first prediction time
            directly.
        start_format
            Defines the `start` format. Only effective when `start` is an integer and `series` is indexed with a
            `pd.RangeIndex`.
            If set to 'position', `start` corresponds to the index position of the first predicted point and can range
            from `(-len(series), len(series) - 1)`.
            If set to 'value', `start` corresponds to the index value/label of the first predicted point. Will raise
            an error if the value is not in `series`' index. Default: `'value'`
        num_samples
            Number of times a prediction is sampled from a probabilistic model. Must be `1` for deterministic models.
        verbose
            Whether to print the progress.
        show_warnings
            Whether to show warnings related to historical forecasts optimization, or parameters `start` and
            `train_length`.
        enable_optimization
            Whether to use the optimized version of historical_forecasts when supported and available.
            Default: ``True``.
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
        component_wise
            If True, will separately plot each component in case of multivariate anomaly detection.
        score_kwargs
            parameters for the `score()` method.
        """
        predict_kwargs = {
            "past_covariates": past_covariates,
            "future_covariates": future_covariates,
            "forecast_horizon": forecast_horizon,
            "start": start,
            "start_format": start_format,
            "num_samples": num_samples,
            "verbose": verbose,
            "show_warnings": show_warnings,
            "enable_optimization": enable_optimization,
        }
        return super().show_anomalies(
            series=series,
            anomalies=anomalies,
            predict_kwargs=predict_kwargs,
            names_of_scorers=names_of_scorers,
            title=title,
            metric=metric,
            component_wise=component_wise,
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
        start_format: Literal["position", "value"] = "value",
        num_samples: int = 1,
        verbose: bool = False,
        show_warnings: bool = True,
        enable_optimization: bool = True,
        **model_fit_kwargs,
    ):
        """Fit the forecasting model (if applicable) and scorers."""
        # fit forecasting model
        if allow_model_training:
            self.model._fit_wrapper(
                series=series,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
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
            historical_forecasts = self.predict_series(
                series=series,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
                forecast_horizon=forecast_horizon,
                start=start,
                start_format=start_format,
                num_samples=num_samples,
                verbose=verbose,
                show_warnings=show_warnings,
                enable_optimization=enable_optimization,
            )
            # fit the scorers
            self._fit_scorers(series, historical_forecasts)
