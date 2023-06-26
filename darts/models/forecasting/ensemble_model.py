"""
Ensemble Model Base Class
"""

from abc import abstractmethod
from functools import reduce
from typing import List, Optional, Sequence, Tuple, Union

from darts.logging import get_logger, raise_if, raise_if_not, raise_log
from darts.models.forecasting.forecasting_model import (
    ForecastingModel,
    GlobalForecastingModel,
    LocalForecastingModel,
)
from darts.timeseries import TimeSeries
from darts.utils.utils import series2seq

logger = get_logger(__name__)


class EnsembleModel(GlobalForecastingModel):
    """
    Abstract base class for ensemble models.
    Ensemble models take in a list of forecasting models and ensemble their predictions
    to make a single one according to the rule defined by their `ensemble()` method.

    If `future_covariates` or `past_covariates` are provided at training or inference time,
    they will be passed only to the models supporting them.

    Parameters
    ----------
    forecasting_models
        List of forecasting models whose predictions to ensemble

        .. note::
                if all the models are probabilistic, the `EnsembleModel` will also be probabilistic.
        ..
    train_num_samples
        Number of prediction samples from each forecasting model for multi-level ensembles. The n_samples
        dimension will be reduced using the `train_samples_reduction` method.
    train_samples_reduction
        If `forecasting_models` are probabilistic and `train_num_samples` > 1, method used to reduce the
        samples dimension to 1. Possible values: "mean", "median" or float value corresponding to the
        desired quantile.
    retrain_forecasting_models
        If set to `False`, the `forecasting_models` are not retrained when calling `fit()` (only supported
        if all the `forecasting_models` are pretrained `GlobalForecastingModels`). Default: ``True``.
    show_warnings
        Whether to show warnings related to models covariates support.
    """

    def __init__(
        self,
        forecasting_models: List[ForecastingModel],
        train_num_samples: int,
        train_samples_reduction: Optional[Union[str, float]],
        retrain_forecasting_models: bool = True,
        show_warnings: bool = True,
    ):
        raise_if_not(
            isinstance(forecasting_models, list) and forecasting_models,
            "Cannot instantiate EnsembleModel with an empty list of `forecasting_models`",
            logger,
        )

        is_local_model = [
            isinstance(model, LocalForecastingModel) for model in forecasting_models
        ]
        is_global_model = [
            isinstance(model, GlobalForecastingModel) for model in forecasting_models
        ]

        self.is_local_ensemble = all(is_local_model)
        self.is_global_ensemble = all(is_global_model)

        raise_if_not(
            all(
                [
                    local_model or global_model
                    for local_model, global_model in zip(
                        is_local_model, is_global_model
                    )
                ]
            ),
            "All models must be of type `GlobalForecastingModel`, or `LocalForecastingModel`. "
            "Also, make sure that all `forecasting_models` are instantiated.",
            logger,
        )

        model_fit_status = [m._fit_called for m in forecasting_models]
        self.all_trained = all(model_fit_status)
        some_trained = any(model_fit_status)

        raise_if(
            (not self.is_global_ensemble and some_trained)
            or (self.is_global_ensemble and not (self.all_trained or not some_trained)),
            "Cannot instantiate EnsembleModel with a mixture of unfitted and fitted `forecasting_models`. "
            "Consider resetting all models with `my_model.untrained_model()` or using only trained "
            "GlobalForecastingModels together with `retrain_forecasting_models=False`.",
            logger,
        )

        if retrain_forecasting_models:
            # prevent issues with pytorch-lightning trainer during retraining
            raise_if(
                some_trained,
                "`retrain_forecasting_models=True` but some `forecasting_models` were already fitted. "
                "Consider resetting all the `forecasting_models` with `my_model.untrained_model()` "
                "before passing them to the `EnsembleModel`.",
                logger,
            )
        else:
            raise_if_not(
                self.is_global_ensemble and self.all_trained,
                "`retrain_forecasting_models=False` is supported only if all the `forecasting_models` are "
                "already trained `GlobalForecastingModels`.",
                logger,
            )

        raise_if(
            train_num_samples > 1
            and all([not m._is_probabilistic() for m in forecasting_models]),
            "`train_num_samples` is greater than 1 but the `RegressionEnsembleModel` "
            "contains only deterministic `forecasting_models`.",
            logger,
        )

        supported_reduction = ["mean", "median"]
        if train_samples_reduction is None:
            pass
        elif isinstance(train_samples_reduction, float):
            raise_if_not(
                0.0 < train_samples_reduction < 1.0,
                f"if a float, `train_samples_reduction` must be between "
                f"0 and 1, received ({train_samples_reduction})",
                logger,
            )
        elif isinstance(train_samples_reduction, str):
            raise_if(
                train_samples_reduction not in supported_reduction,
                f"if a string, `train_samples_reduction` must be one of {supported_reduction}, "
                f"received ({train_samples_reduction})",
                logger,
            )
        else:
            raise_log(
                ValueError(
                    f"`train_samples_reduction` type not supported "
                    f"({train_samples_reduction}). Must be `float` "
                    f" or one of {supported_reduction}."
                ),
                logger,
            )

        super().__init__()
        self.forecasting_models = forecasting_models
        self.train_num_samples = train_num_samples
        self.train_samples_reduction = train_samples_reduction
        self.retrain_forecasting_models = retrain_forecasting_models

        if show_warnings:
            if (
                self.supports_past_covariates
                and not self._full_past_covariates_support()
            ):
                logger.warning(
                    "Some `forecasting_models` in the ensemble do not support past covariates, the past covariates "
                    "will be provided only to the models supporting them when calling fit()` or `predict()`. "
                    "To hide these warnings, set `show_warnings=False`."
                )

            if (
                self.supports_future_covariates
                and not self._full_future_covariates_support()
            ):
                logger.warning(
                    "Some `forecasting_models` in the ensemble do not support future covariates, the future covariates"
                    " will be provided only to the models supporting them when calling `fit()` or `predict()`. "
                    "To hide these warnings, set `show_warnings=False`."
                )

    @abstractmethod
    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    ):
        """
        Fits the model on the provided series.
        Note that `EnsembleModel.fit()` does NOT call `fit()` on each of its constituent forecasting models.
        It is left to classes inheriting from EnsembleModel to do so appropriately when overriding `fit()`
        """

        is_single_series = isinstance(series, TimeSeries)

        # local models OR mix of local and global models
        raise_if(
            not self.is_global_ensemble and not is_single_series,
            "The `forecasting_models` contain at least one LocalForecastingModel, which does not support training "
            "on multiple series.",
            logger,
        )

        # check that if timeseries is single series, that covariates are as well and vice versa
        error_past_cov = False
        error_future_cov = False

        if past_covariates is not None:
            error_past_cov = is_single_series != isinstance(past_covariates, TimeSeries)

        if future_covariates is not None:
            error_future_cov = is_single_series != isinstance(
                future_covariates, TimeSeries
            )

        raise_if(
            error_past_cov or error_future_cov,
            "Both series and covariates have to be either single TimeSeries or sequences of TimeSeries.",
            logger,
        )

        self._verify_past_future_covariates(past_covariates, future_covariates)

        super().fit(series, past_covariates, future_covariates)

        return self

    def _stack_ts_seq(self, predictions):
        # stacks list of predictions into one multivariate timeseries
        return reduce(lambda a, b: a.stack(b), predictions)

    def _stack_ts_multiseq(self, predictions_list):
        # stacks multiple sequences of timeseries elementwise
        return [self._stack_ts_seq(ts_list) for ts_list in zip(*predictions_list)]

    def _model_encoder_settings(self):
        raise NotImplementedError(
            "Encoders are not supported by EnsembleModels. Instead add encoder to the underlying `forecasting_models`."
        )

    def _make_multiple_predictions(
        self,
        n: int,
        series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        num_samples: int = 1,
    ):
        is_single_series = isinstance(series, TimeSeries) or series is None
        # maximize covariate usage
        predictions = [
            model._predict_wrapper(
                n=n,
                series=series,
                past_covariates=past_covariates
                if model.supports_past_covariates
                else None,
                future_covariates=future_covariates
                if model.supports_future_covariates
                else None,
                num_samples=num_samples if model._is_probabilistic() else 1,
            )
            for model in self.forecasting_models
        ]

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

    def predict(
        self,
        n: int,
        series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        num_samples: int = 1,
        verbose: bool = False,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:

        super().predict(
            n=n,
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            num_samples=num_samples,
            verbose=verbose,
        )

        # for multi-level models, forecasting models can generate arbitrary number of samples
        if self.train_samples_reduction is None:
            pred_num_samples = num_samples
        else:
            pred_num_samples = self.train_num_samples

        self._verify_past_future_covariates(past_covariates, future_covariates)

        predictions = self._make_multiple_predictions(
            n=n,
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            num_samples=pred_num_samples,
        )

        return self.ensemble(predictions, series=series, num_samples=num_samples)

    @abstractmethod
    def ensemble(
        self,
        predictions: Union[TimeSeries, Sequence[TimeSeries]],
        series: Optional[Sequence[TimeSeries]] = None,
        num_samples: int = 1,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """
        Defines how to ensemble the individual models' predictions to produce a single prediction.

        Parameters
        ----------
        predictions
            Individual predictions to ensemble
        series
            Sequence of timeseries to predict on. Optional, since it only makes sense for sequences of timeseries -
            local models retain timeseries for prediction.

        Returns
        -------
        TimeSeries or Sequence[TimeSeries]
            The predicted ``TimeSeries`` or sequence of ``TimeSeries`` obtained by ensembling the individual predictions
        """
        pass

    def _predictions_reduction(self, predictions: TimeSeries) -> TimeSeries:
        """Reduce the sample dimension of the forecasting models predictions"""
        is_single_series = isinstance(predictions, TimeSeries)
        predictions = series2seq(predictions)
        if self.train_samples_reduction == "median":
            predictions = [pred.median(axis=2) for pred in predictions]
        elif self.train_samples_reduction == "mean":
            predictions = [pred.mean(axis=2) for pred in predictions]
        else:
            predictions = [
                pred.quantile(self.train_samples_reduction) for pred in predictions
            ]
        return predictions[0] if is_single_series else predictions

    @property
    def min_train_series_length(self) -> int:
        return max(model.min_train_series_length for model in self.forecasting_models)

    @property
    def min_train_samples(self) -> int:
        return max(model.min_train_samples for model in self.forecasting_models)

    @property
    def extreme_lags(
        self,
    ) -> Tuple[
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
    ]:
        def find_max_lag_or_none(lag_id, aggregator) -> Optional[int]:
            max_lag = None
            for model in self.forecasting_models:
                curr_lag = model.extreme_lags[lag_id]
                if max_lag is None:
                    max_lag = curr_lag
                elif curr_lag is not None:
                    max_lag = aggregator(max_lag, curr_lag)
            return max_lag

        lag_aggregators = (min, max, min, max, min, max)
        return tuple(
            find_max_lag_or_none(i, agg) for i, agg in enumerate(lag_aggregators)
        )

    def _models_are_probabilistic(self) -> bool:
        return all([model._is_probabilistic() for model in self.forecasting_models])

    def _is_probabilistic(self) -> bool:
        return self._models_are_probabilistic()

    @property
    def supports_past_covariates(self) -> bool:
        return any(
            [model.supports_past_covariates for model in self.forecasting_models]
        )

    @property
    def supports_future_covariates(self) -> bool:
        return any(
            [model.supports_future_covariates for model in self.forecasting_models]
        )

    def _full_past_covariates_support(self) -> bool:
        return all(
            [model.supports_past_covariates for model in self.forecasting_models]
        )

    def _full_future_covariates_support(self) -> bool:
        return all(
            [model.supports_future_covariates for model in self.forecasting_models]
        )

    def _verify_past_future_covariates(self, past_covariates, future_covariates):
        """
        Verify that any non-None covariates comply with the model type.
        """
        raise_if(
            past_covariates is not None and not self.supports_past_covariates,
            "`past_covariates` were provided to an `EnsembleModel` but none of its "
            "`forecasting_models` support such covariates.",
            logger,
        )
        raise_if(
            future_covariates is not None and not self.supports_future_covariates,
            "`future_covariates` were provided to an `EnsembleModel` but none of its "
            "`forecasting_models` support such covariates.",
            logger,
        )
