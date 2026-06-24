"""
Regression Ensemble Model
-------------------------

An ensemble model which uses a regression model to compute the ensemble forecast.
"""

import math

from darts import TimeSeries, concatenate
from darts.logging import get_logger, raise_log
from darts.models.forecasting.ensemble_model import EnsembleModel
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.models.forecasting.linear_regression_model import LinearRegressionModel
from darts.models.forecasting.sklearn_model import SKLearnModel
from darts.typing import TimeSeriesLike
from darts.utils import n_steps_between
from darts.utils.ts_utils import (
    get_series_seq_type,
    get_single_series,
    seq2series,
    series2seq,
)

logger = get_logger(__name__)


class RegressionEnsembleModel(EnsembleModel):
    def __init__(
        self,
        forecasting_models: list[ForecastingModel],
        regression_train_n_points: int,
        regression_model=None,
        regression_train_num_samples: int = 1,
        regression_train_samples_reduction: str | float | None = "median",
        train_forecasting_models: bool = True,
        train_using_historical_forecasts: bool = False,
        show_warnings: bool = True,
    ):
        """
        Use a regression model for ensembling individual models' predictions using the stacking technique [1]_.

        The provided regression model must implement ``fit()`` and ``predict()`` methods
        (e.g. scikit-learn regression models). Note that here the regression model is used to learn how to
        best ensemble the individual forecasting models' forecasts. It is not the same usage of regression
        as in :class:`SKLearnModel`, where the regression model is used to produce forecasts based on the
        lagged series.

        If `future_covariates` or `past_covariates` are provided at training or inference time,
        they will be passed only to the forecasting models supporting them.

        If `forecasting_models` contains exclusively GlobalForecastingModels, they can be pre-trained. Otherwise,
        the `forecasting_models` must be untrained.

        The regression model does not leverage the covariates passed to ``fit()`` and ``predict()``.

        Parameters
        ----------
        forecasting_models
            List of forecasting models whose predictions to ensemble
        regression_train_n_points
            The number of points per series to use to train the regression model. Can be set to `-1` to use the
            entire series to train the regressor if `forecasting_models` are already fitted and
            `train_forecasting_models=False`.
        regression_model
            Any regression model with ``predict()`` and ``fit()`` methods (e.g. from scikit-learn)
            Default: ``darts.models.LinearRegressionModel(fit_intercept=False)``

            .. note::
                if `regression_model` is probabilistic, the `RegressionEnsembleModel` will also be probabilistic.
            ..
        regression_train_num_samples
            Number of prediction samples from each forecasting model to train the regression model (samples are
            averaged). Should be set to 1 for deterministic models. Default: 1.

            .. note::
                if `forecasting_models` contains a mix of probabilistic and deterministic models,
                `regression_train_num_samples` will be passed only to the probabilistic ones.
            ..
        regression_train_samples_reduction
            If `forecasting_models` are probabilistic and `regression_train_num_samples` > 1, method used to
            reduce the samples before passing them to the regression model. Possible values: "mean", "median"
            or float value corresponding to the desired quantile. Default: "median"
        train_forecasting_models
            If set to `False`, the `forecasting_models` are not retrained when calling `fit()` (only supported
            if all the `forecasting_models` are pre-trained `GlobalForecastingModels`). Default: ``True``.
        train_using_historical_forecasts
            If set to `True`, use `historical_forecasts()` to generate the forecasting models' predictions used to
            train the regression model in `fit()`. Available when `forecasting_models` contains only
            `GlobalForecastingModels`. Recommended when `regression_train_n_points` is greater than
            `output_chunk_length` of the underlying `forecasting_models`.
            Default: ``False``.
        show_warnings
            Whether to show warnings related to forecasting_models covariates support.
        References
        ----------
        .. [1] D. H. Wolpert, “Stacked generalization”, Neural Networks, vol. 5, no. 2, pp. 241–259, Jan. 1992

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.models import RegressionEnsembleModel, NaiveSeasonal, LinearRegressionModel
        >>> series = AirPassengersDataset().load()
        >>> model = RegressionEnsembleModel(
        >>>     forecasting_models = [
        >>>         NaiveSeasonal(K=12),
        >>>         LinearRegressionModel(lags=4)
        >>>     ],
        >>>     regression_train_n_points=20
        >>> )
        >>> model.fit(series)
        >>> pred = model.predict(6)
        >>> print(pred.values())
        [[494.24050364]
         [464.3869697 ]
         [496.53180506]
         [544.82269341]
         [557.35256055]
         [630.24334385]]
        """
        if not isinstance(forecasting_models, list) or len(forecasting_models) == 0:
            raise_log(
                ValueError(
                    "`forecasting_models` must be a non-empty list of forecasting models."
                ),
                logger,
            )

        shifts = [model.output_chunk_shift for model in forecasting_models]
        if len(set(shifts)) != 1:
            raise_log(
                ValueError(
                    "All base forecasting models must have the same `output_chunk_shift`. "
                    f"Observed shifts: {set(shifts)}."
                )
            )

        output_chunk_shift = shifts[0] or 0
        if output_chunk_shift > 0:
            if not train_using_historical_forecasts:
                raise_log(
                    ValueError(
                        "`train_using_historical_forecasts` must be `True` when base models use `output_chunk_shift>0`."
                    ),
                    logger,
                )

            output_chunk_length = min([
                model.output_chunk_length or 1 for model in forecasting_models
            ])
            lags_future_covariates = list(range(output_chunk_length))
        else:
            output_chunk_length = 1
            lags_future_covariates = [0]

        if regression_model is None:
            regression_model = LinearRegressionModel(
                lags=None,
                lags_future_covariates=lags_future_covariates,
                output_chunk_length=output_chunk_length,
                output_chunk_shift=output_chunk_shift,
                fit_intercept=False,
            )
        elif isinstance(regression_model, SKLearnModel):
            regression_model = regression_model
        else:
            # scikit-learn like model
            regression_model = SKLearnModel(
                lags=None,
                lags_future_covariates=lags_future_covariates,
                output_chunk_length=output_chunk_length,
                output_chunk_shift=output_chunk_shift,
                model=regression_model,
            )

        if not regression_model.multi_models:
            raise_log(
                ValueError(
                    "Cannot use `regression_model` that was created with `multi_models = False`."
                ),
                logger,
            )
        if regression_model.output_chunk_shift != output_chunk_shift:
            raise_log(
                ValueError(
                    f"`regression_model` must use the same `output_chunk_shift` as the base "
                    f"forecasting models. "
                    f"Observed `output_chunk_shift`: `{regression_model.output_chunk_shift}`, "
                    f"expected: {output_chunk_shift}."
                ),
                logger,
            )
        if regression_train_n_points > 0 and (
            regression_model.output_chunk_length
            + regression_model.min_train_samples
            - 1
            > regression_train_n_points
        ):
            raise_log(
                ValueError(
                    f"`regression_train_n_points` ({regression_train_n_points}) must be "
                    f"`>={regression_model.output_chunk_length + regression_model.min_train_samples - 1}`, given by "
                    f"`(regression_model.output_chunk_length + regression_model.min_train_samples - 1)`."
                ),
                logger,
            )

        if output_chunk_shift > 0 and (
            regression_model.output_chunk_length != output_chunk_length
        ):
            raise_log(
                ValueError(
                    f"With `output_chunk_shift>0`, `regression_model` must use the minimum "
                    f"`output_chunk_length` of all base forecasting models. "
                    f"Observed `output_chunk_length`: `{regression_model.output_chunk_length}`, "
                    f"expected: {output_chunk_length}."
                ),
                logger,
            )

        # check lags of the regression model
        if list(regression_model.lags.keys()) != ["future"]:
            raise_log(
                ValueError(
                    "`lags` and `lags_past_covariates` of `regression_model` must be `None`."
                ),
                logger,
            )
        min_lag, max_lag = 0, regression_model.output_chunk_length - 1
        # adjust model lags by `output_chunk_shift` to get original lags
        lags_observed = [
            lag - output_chunk_shift for lag in regression_model.lags["future"]
        ]
        if any([not (min_lag <= lag <= max_lag) for lag in lags_observed]):
            raise_log(
                ValueError(
                    f"All lags in `lags_future_covariates` must be `0<=lag<={max_lag}`, "
                    f"where the upper bound is given by `(regression_model.output_chunk_length-1)`. "
                    f"Received lags: {lags_observed}."
                ),
                logger,
            )

        super().__init__(
            forecasting_models=forecasting_models,
            ensemble_model=regression_model,
            train_num_samples=regression_train_num_samples,
            train_samples_reduction=regression_train_samples_reduction,
            train_forecasting_models=train_forecasting_models,
            train_n_points=regression_train_n_points,
            show_warnings=show_warnings,
        )

        if train_using_historical_forecasts and not self.is_global_ensemble:
            raise_log(
                ValueError(
                    "`train_using_historical_forecasts=True` is only available when all "
                    "`forecasting_models` are global models."
                ),
                logger,
            )

        self.train_using_historical_forecasts = train_using_historical_forecasts

    def _make_multiple_historical_forecasts(
        self,
        train_n_points: int,
        series: TimeSeriesLike,
        past_covariates: TimeSeriesLike | None = None,
        future_covariates: TimeSeriesLike | None = None,
        num_samples: int = 1,
        verbose: bool | None = None,
    ) -> TimeSeriesLike:
        """
        For GlobalForecastingModel, when predicting n > output_chunk_length, `historical_forecasts()` generally
        produce better forecasts than `predict()`.

        To get as close as possible to the predictions generated by the forecasting models during inference,
        `historical_forecasts` forecast horizon is equal to each model output_chunk_length.

        train_n_points are generated, starting from the end of the series.
        """
        verbose = verbose or False
        sequence_type_in = get_series_seq_type(series)
        series = series2seq(series)
        past_covariates = series2seq(past_covariates)
        future_covariates = series2seq(future_covariates)

        predictions: list[list[TimeSeries]] = []
        for m_idx, model in enumerate(self.forecasting_models):
            # we start historical forecasts at multiple of the output length before the end
            n_ocl_back = math.ceil(train_n_points / model.output_chunk_length)

            start_hist_forecasts = (
                n_ocl_back * model.output_chunk_length + self.output_chunk_shift
            )
            hfc = model.historical_forecasts(
                series=series,
                past_covariates=(
                    past_covariates if model.supports_past_covariates else None
                ),
                future_covariates=(
                    future_covariates if model.supports_future_covariates else None
                ),
                forecast_horizon=model.output_chunk_length,
                stride=model.output_chunk_length,
                num_samples=(
                    num_samples if model.supports_probabilistic_prediction else 1
                ),
                start=-start_hist_forecasts,
                start_format="position",
                retrain=False,
                overlap_end=False,
                last_points_only=False,
                show_warnings=self.show_warnings,
                predict_likelihood_parameters=False,
                verbose=verbose,
            )

            predictions_tmp: list[TimeSeries] = []
            for idx, (series_, series_hfc) in enumerate(zip(series, hfc)):
                # check that all forecasts end at the end of the target series
                if (
                    n_steps_between(
                        end=series_.end_time(),
                        start=series_hfc[-1].end_time(),
                        freq=series_.freq,
                    )
                    != 0
                ):
                    raise_log(
                        ValueError(
                            f"Some covariates do not extend far enough into the future "
                            f"to generate all required historical forecasts for the series "
                            f"at index {idx}"
                        ),
                        logger,
                    )

                # concatenate the stridden predictions
                predictions_tmp.append(concatenate(series_hfc, axis=0))
            predictions.append(predictions_tmp)

        # postprocess the forecasts
        forecasts: list[TimeSeries] = []
        for idx, series_forecasts in enumerate(zip(*predictions)):
            # make sure all model forecasts share the same time index per series
            min_length = min(len(forecast) for forecast in series_forecasts)
            series_forecasts: list[TimeSeries] = [
                forecast[-min_length:] if len(forecast) != min_length else forecast
                for forecast in series_forecasts
            ]

            if (
                len(get_single_series(series_forecasts)) < train_n_points
                and self.show_warnings
            ):
                logger.warning(
                    f"Generated fewer forecasts than the requested {train_n_points} "
                    f"for the series at index {idx}."
                )

            # reduce the probabilistics series
            if self.train_samples_reduction is not None and self.train_num_samples > 1:
                series_forecasts = self._predictions_reduction(series_forecasts)

            # stack individual model predictions into multivariate series
            forecasts.append(self._stack_ts_seq(series_forecasts))

        return series2seq(forecasts, seq_type_out=sequence_type_in)

    def fit(
        self,
        series: TimeSeriesLike,
        past_covariates: TimeSeriesLike | None = None,
        future_covariates: TimeSeriesLike | None = None,
        sample_weight: TimeSeriesLike | str | None = None,
        verbose: bool | None = None,
    ):
        """
        Fits the forecasting models with the entire series except the last `regression_train_n_points` values, which
        are used to train the regression model.

        If `forecasting_models` contains fitted `GlobalForecastingModels` and `train_forecasting_model=False`,
        only the regression model will be trained.

        Parameters
        ----------
        series
            TimeSeries or Sequence[TimeSeries] object containing the target values.
        past_covariates
            Optionally, a series or sequence of series specifying past-observed covariates passed to the
            forecasting models
        future_covariates
            Optionally, a series or sequence of series specifying future-known covariates passed to the
            forecasting models
        sample_weight
            Optionally, some sample weights to apply to the target `series` labels. They are applied per observation,
            per label (each step in `output_chunk_length`), and per component.
            If a series or sequence of series, then those weights are used. If the weight series only have a single
            component / column, then the weights are applied globally to all components in `series`. Otherwise, for
            component-specific weights, the number of components must match those of `series`.
            If a string, then the weights are generated using built-in weighting functions. The available options are
            `"linear"` or `"exponential"` decay - the further in the past, the lower the weight. The weights are
            computed globally based on the length of the longest series in `series`. Then for each series, the weights
            are extracted from the end of the global weights. This gives a common time weighting across all series.
        verbose
            Optionally, set the fit verbosity. Not effective for all models.
        """
        super().fit(
            series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            verbose=verbose,
        )

        # at this point, we know that all target series are long enough
        is_single_series = isinstance(series, TimeSeries)

        # determine the actual number of training points to use
        if self.train_n_points == -1:
            input_shift = self._target_window_lengths[0]
            min_series_length = (
                len(series) if is_single_series else min(len(ts) for ts in series)
            )
            self.train_n_points = min_series_length - input_shift

        # spare train_n_points points to serve as regression target
        if is_single_series:
            forecast_training = series[: -self.train_n_points]
            regression_target = series[-self.train_n_points :]
        else:
            forecast_training, regression_target = [], []
            for ts in series:
                forecast_training.append(ts[: -self.train_n_points])
                regression_target.append(ts[-self.train_n_points :])

        if self.train_forecasting_models:
            for model in self.forecasting_models:
                # maximize covariate usage
                model._fit_wrapper(
                    series=forecast_training,
                    past_covariates=(
                        past_covariates if model.supports_past_covariates else None
                    ),
                    future_covariates=(
                        future_covariates if model.supports_future_covariates else None
                    ),
                    sample_weight=sample_weight
                    if model.supports_sample_weight
                    else None,
                    verbose=verbose,
                )

        # we can call direct prediction in any case. Even if we overwrite with historical
        # forecasts later on, it serves as input validation
        if not self.train_using_historical_forecasts:
            predictions = self._make_multiple_predictions(
                n=self.train_n_points,
                series=forecast_training,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
                num_samples=self.train_num_samples,
                verbose=verbose,
            )

        else:
            predictions = self._make_multiple_historical_forecasts(
                train_n_points=self.train_n_points,
                series=series,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
                num_samples=self.train_num_samples,
                verbose=verbose,
            )

        # train the regression model on the individual models' predictions
        self.ensemble_model.fit(
            series=regression_target,
            future_covariates=predictions,
            sample_weight=sample_weight,
            verbose=verbose,
        )

        # prepare the forecasting models for further predicting by fitting them with the entire data
        if self.train_forecasting_models:
            # Some models may need to be 'reset' to allow being retrained from scratch, especially torch-based models
            self.forecasting_models: list[ForecastingModel] = [
                model.untrained_model() for model in self.forecasting_models
            ]
            for model in self.forecasting_models:
                model._fit_wrapper(
                    series=series,
                    past_covariates=(
                        past_covariates if model.supports_past_covariates else None
                    ),
                    future_covariates=(
                        future_covariates if model.supports_future_covariates else None
                    ),
                    sample_weight=sample_weight
                    if model.supports_sample_weight
                    else None,
                    verbose=verbose,
                )
        return self

    def ensemble(
        self,
        predictions: TimeSeriesLike,
        series: TimeSeriesLike,
        n: int,
        num_samples: int = 1,
        predict_likelihood_parameters: bool = False,
        random_state: int | None = None,
        verbose: bool | None = None,
    ) -> TimeSeriesLike:
        is_single_series = isinstance(series, TimeSeries) or series is None
        predictions = series2seq(predictions)
        series = series2seq(series) if series is not None else [None]

        ensembled = [
            self.ensemble_model.predict(
                n=n,
                series=serie,
                future_covariates=prediction,
                num_samples=num_samples,
                predict_likelihood_parameters=predict_likelihood_parameters,
                random_state=random_state,
                verbose=verbose,
            )
            for serie, prediction in zip(series, predictions)
        ]
        return seq2series(ensembled) if is_single_series else ensembled

    @property
    def supports_likelihood_parameter_prediction(self) -> bool:
        # likelihood parameters predictions are supported if the regression model supports it (ensembling layer)
        return self.ensemble_model.supports_likelihood_parameter_prediction

    @property
    def supports_probabilistic_prediction(self) -> bool:
        # probabilistic predictions are supported if the regression model supports it (ensembling layer)
        return self.ensemble_model.supports_probabilistic_prediction
