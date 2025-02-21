"""
Regression ensemble model
-------------------------

An ensemble model which uses a regression model to compute the ensemble forecast.
"""

from collections.abc import Sequence
from typing import Optional, Union

from darts.logging import get_logger, raise_if, raise_if_not
from darts.models.forecasting.ensemble_model import EnsembleModel
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.models.forecasting.linear_regression_model import LinearRegressionModel
from darts.models.forecasting.regression_model import RegressionModel
from darts.timeseries import TimeSeries, concatenate
from darts.utils.ts_utils import seq2series, series2seq

logger = get_logger(__name__)


class RegressionEnsembleModel(EnsembleModel):
    def __init__(
        self,
        forecasting_models: list[ForecastingModel],
        regression_train_n_points: int,
        regression_model=None,
        regression_train_num_samples: int = 1,
        regression_train_samples_reduction: Optional[Union[str, float]] = "median",
        train_forecasting_models: bool = True,
        train_using_historical_forecasts: bool = False,
        show_warnings: bool = True,
    ):
        """
        Use a regression model for ensembling individual models' predictions using the stacking technique [1]_.

        The provided regression model must implement ``fit()`` and ``predict()`` methods
        (e.g. scikit-learn regression models). Note that here the regression model is used to learn how to
        best ensemble the individual forecasting models' forecasts. It is not the same usage of regression
        as in :class:`RegressionModel`, where the regression model is used to produce forecasts based on the
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
                `regression_train_num_samples will be passed only to the probabilistic ones.
            ..
        regression_train_samples_reduction
            If `forecasting_models` are probabilistic and `regression_train_num_samples` > 1, method used to
            reduce the samples before passing them to the regression model. Possible values: "mean", "median"
            or float value corresponding to the desired quantile. Default: "median"
        train_forecasting_models
            If set to `False`, the `forecasting_models` are not retrained when calling `fit()` (only supported
            if all the `forecasting_models` are pretrained `GlobalForecastingModels`). Default: ``True``.
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
        >>> pred.values()
        array([[494.24050364],
               [464.3869697 ],
               [496.53180506],
               [544.82269341],
               [557.35256055],
               [630.24334385]])
        """
        super().__init__(
            forecasting_models=forecasting_models,
            train_num_samples=regression_train_num_samples,
            train_samples_reduction=regression_train_samples_reduction,
            train_forecasting_models=train_forecasting_models,
            show_warnings=show_warnings,
        )

        if regression_model is None:
            regression_model = LinearRegressionModel(
                lags=None, lags_future_covariates=[0], fit_intercept=False
            )
        elif isinstance(regression_model, RegressionModel):
            raise_if_not(
                regression_model.multi_models,
                "Cannot use `regression_model` that was created with `multi_models = False`.",
                logger,
            )
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

        self.regression_model: RegressionModel = regression_model

        raise_if(
            regression_train_n_points == -1
            and not (self.all_trained and (not train_forecasting_models)),
            "`regression_train_n_points` can only be `-1` if `retrain_forecasting_model=False` and "
            "all `forecasting_models` are already fitted.",
            logger,
        )

        # converted to List[int] if regression_train_n_points=-1 and ensemble is trained with multiple series
        self.train_n_points: Union[int, list[int]] = regression_train_n_points

        raise_if(
            train_using_historical_forecasts and not self.is_global_ensemble,
            "`train_using_historical_forecasts=True` is only available when all "
            "`forecasting_models` are global models.",
            logger,
        )

        self.train_using_historical_forecasts = train_using_historical_forecasts

    def _split_multi_ts_sequence(
        self, n: Union[int, list[int]], ts_sequence: Sequence[TimeSeries]
    ) -> tuple[Sequence[TimeSeries], Sequence[TimeSeries]]:
        if isinstance(n, int):
            n = [n] * len(ts_sequence)
        left = [ts[:-n_] for ts, n_ in zip(ts_sequence, n)]
        right = [ts[-n_:] for ts, n_ in zip(ts_sequence, n)]
        return left, right

    def _make_multiple_historical_forecasts(
        self,
        train_n_points: int,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        direct_predictions: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        num_samples: int = 1,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """
        For GlobalForecastingModel, when predicting n > output_chunk_length, `historical_forecasts()` generally
        produce better forecasts than `predict()`.

        To get as close as possible to the predictions generated by the forecasting models during inference,
        `historical_forecasts` forecast horizon is equal to each model output_chunk_length.

        train_n_points are generated, starting from the end of the series.
        """
        is_single_series = isinstance(series, TimeSeries)
        series = series2seq(series)
        direct_predictions = series2seq(direct_predictions)
        past_covariates = series2seq(past_covariates)
        future_covariates = series2seq(future_covariates)

        n_components = series[0].n_components
        model_predict_cols = direct_predictions[0].columns.tolist()

        predictions = []
        for m_idx, model in enumerate(self.forecasting_models):
            # we start historical fc at multiple of the output length before the end.
            n_ocl_back = train_n_points // model.output_chunk_length
            start_hist_forecasts = n_ocl_back * model.output_chunk_length

            # we use the precomputed `direct_prediction` to fill any missing prediction
            # timesteps at the beginning (if train_n_points is not perfectly divisible by output length)
            missing_steps = train_n_points % model.output_chunk_length

            tmp_pred = model.historical_forecasts(
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
            )
            # concatenate the strided predictions of output_chunk_length values each
            tmp_pred = [concatenate(sub_pred, axis=0) for sub_pred in tmp_pred]

            # add the missing steps at beginning by taking the first values of precomputed predictions
            if missing_steps:
                # add the missing steps at beginning by taking the first values of precomputed predictions
                # get the model's direct (uni/multivariate) predictions
                pred_cols = model_predict_cols[
                    m_idx * n_components : (m_idx + 1) * n_components
                ]
                hfc_cols = tmp_pred[0].columns.tolist()
                tmp_pred = [
                    concatenate(
                        [
                            preds_dir[:missing_steps][pred_cols].with_columns_renamed(
                                pred_cols, hfc_cols
                            ),
                            preds_hfc,
                        ],
                        axis=0,
                    )
                    for preds_dir, preds_hfc in zip(direct_predictions, tmp_pred)
                ]
            predictions.append(tmp_pred)

        tmp_predictions = []
        # slice the forecasts, training series-wise, to align them
        for prediction in predictions:
            tmp_predictions.append([ts for idx, ts in enumerate(prediction)])
        predictions = [seq2series(prediction) for prediction in tmp_predictions]

        # reduce the probabilistics series
        if self.train_samples_reduction is not None and self.train_num_samples > 1:
            predictions = [
                self._predictions_reduction(prediction) for prediction in predictions
            ]

        return (
            self._stack_ts_seq(predictions)
            if is_single_series
            else self._stack_ts_multiseq(predictions)
        )

    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        sample_weight: Optional[Union[TimeSeries, Sequence[TimeSeries], str]] = None,
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
        """
        super().fit(
            series, past_covariates=past_covariates, future_covariates=future_covariates
        )

        # spare train_n_points points to serve as regression target
        is_single_series = isinstance(series, TimeSeries)
        if self.train_n_points == -1:
            if is_single_series:
                train_n_points = [len(series)]
            else:
                # maximize each series usage
                train_n_points = [len(ts) for ts in series]

            # shift by the forecasting models' largest input length
            all_shifts = []
            # when it's not clearly defined, extreme_lags returns
            # `min_train_series_length` for the LocalForecastingModels
            for model in self.forecasting_models:
                min_target_lag, _, _, _, _, _, _, _ = model.extreme_lags
                if min_target_lag is not None:
                    all_shifts.append(-min_target_lag)

            input_shift = max(all_shifts)
            idx_series_too_short = []
            tmp_train_n_points = []
            for idx, ts_length in enumerate(train_n_points):
                ajusted_length = ts_length - input_shift
                if ajusted_length < 0:
                    idx_series_too_short.append(idx)
                else:
                    tmp_train_n_points.append(ajusted_length)

            raise_if(
                len(idx_series_too_short) > 0,
                f"TimeSeries at indexes {idx_series_too_short} of `series` are too short to train the regression "
                f"model due to the number of values necessary to produce one prediction : {input_shift}.",
                logger,
            )

            if is_single_series:
                self.train_n_points = tmp_train_n_points[0]
            else:
                self.train_n_points = tmp_train_n_points

            train_n_points_too_big = False
        else:
            # self.train_n_points is necessarily an integer
            if is_single_series:
                train_n_points_too_big = len(series) <= self.train_n_points
            else:
                train_n_points_too_big = any([
                    len(s) <= self.train_n_points for s in series
                ])

        raise_if(
            train_n_points_too_big,
            "`regression_train_n_points` parameter too big (must be strictly smaller than "
            "the number of points in training_series)",
            logger,
        )

        if is_single_series:
            forecast_training = series[: -self.train_n_points]
            regression_target = series[-self.train_n_points :]
        else:
            forecast_training, regression_target = self._split_multi_ts_sequence(
                self.train_n_points, series
            )

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
                )

        # we can call direct prediction in any case. Even if we overwrite with historical
        # forecasts later on, it serves as a input validation
        predictions = self._make_multiple_predictions(
            n=self.train_n_points,
            series=forecast_training,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            num_samples=self.train_num_samples,
        )

        if self.train_using_historical_forecasts:
            predictions = self._make_multiple_historical_forecasts(
                train_n_points=self.train_n_points,
                series=series,
                direct_predictions=predictions,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
                num_samples=self.train_num_samples,
            )

        # train the regression model on the individual models' predictions
        self.regression_model.fit(
            series=regression_target,
            future_covariates=predictions,
            sample_weight=sample_weight,
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
                )
        return self

    def ensemble(
        self,
        predictions: Union[TimeSeries, Sequence[TimeSeries]],
        series: Union[TimeSeries, Sequence[TimeSeries]],
        num_samples: int = 1,
        predict_likelihood_parameters: bool = False,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        is_single_series = isinstance(series, TimeSeries) or series is None
        predictions = series2seq(predictions)
        series = series2seq(series) if series is not None else [None]

        ensembled = [
            self.regression_model.predict(
                n=len(prediction),
                series=serie,
                future_covariates=prediction,
                num_samples=num_samples,
                predict_likelihood_parameters=predict_likelihood_parameters,
            )
            for serie, prediction in zip(series, predictions)
        ]
        return seq2series(ensembled) if is_single_series else ensembled

    @property
    def extreme_lags(
        self,
    ) -> tuple[
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        int,
        Optional[int],
    ]:
        extreme_lags_ = super().extreme_lags
        # shift min_target_lag in the past to account for the regression model training set
        if extreme_lags_[0] is None:
            return (-self.train_n_points,) + extreme_lags_[1:]
        else:
            return (extreme_lags_[0] - self.train_n_points,) + extreme_lags_[1:]

    @property
    def output_chunk_length(self) -> int:
        """Return the `output_chunk_length` of the regression model (ensembling layer)"""
        return self.regression_model.output_chunk_length

    @property
    def supports_likelihood_parameter_prediction(self) -> bool:
        """RegressionEnsembleModel supports likelihood parameters predictions if its regression model does"""
        return self.regression_model.supports_likelihood_parameter_prediction

    @property
    def supports_multivariate(self) -> bool:
        return (
            super().supports_multivariate
            and self.regression_model.supports_multivariate
        )

    @property
    def supports_probabilistic_prediction(self) -> bool:
        """
        A RegressionEnsembleModel is probabilistic if its regression
        model is probabilistic (ensembling layer)
        """
        return self.regression_model.supports_probabilistic_prediction
