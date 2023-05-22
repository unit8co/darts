"""
Ensemble Model Base Class
"""

from abc import abstractmethod
from functools import reduce
from typing import List, Optional, Sequence, Tuple, Union

from darts.logging import get_logger, raise_if, raise_if_not
from darts.models.forecasting.forecasting_model import (
    ForecastingModel,
    GlobalForecastingModel,
    LocalForecastingModel,
)
from darts.timeseries import TimeSeries

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
    models
        List of forecasting models whose predictions to ensemble
    """

    def __init__(self, models: List[ForecastingModel]):
        raise_if_not(
            isinstance(models, list) and models,
            "Cannot instantiate EnsembleModel with an empty list of models",
            logger,
        )

        is_local_model = [isinstance(model, LocalForecastingModel) for model in models]
        is_global_model = [
            isinstance(model, GlobalForecastingModel) for model in models
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
            "Also, make sure that all models in `models` are instantiated.",
            logger,
        )

        raise_if(
            any([m._fit_called for m in models]),
            "Cannot instantiate EnsembleModel with trained/fitted models. "
            "Consider resetting all models with `my_model.untrained_model()`",
            logger,
        )

        super().__init__()
        self.models = models

        if self.supports_past_covariates and not self._full_past_covariates_support():
            logger.info(
                "Some models in the ensemble do not support past covariates, the past covariates will be "
                "provided only to the models supporting them when calling fit/predict."
            )

        if (
            self.supports_future_covariates
            and not self._full_future_covariates_support()
        ):
            logger.info(
                "Some models in the ensemble do not support future covariates, the future covariates will be "
                "provided only to the models supporting them when calling fit/predict."
            )

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
            "The models contain at least one LocalForecastingModel, which does not support training on multiple "
            "series.",
            logger,
        )
        raise_if(
            self.is_local_ensemble and past_covariates is not None,
            "The models are of type LocalForecastingModel, which does not support past covariates.",
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
            "Encoders are not supported by EnsembleModels. Instead add encoder to the underlying `models`."
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
                num_samples=num_samples,
            )
            for model in self.models
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

        self._verify_past_future_covariates(past_covariates, future_covariates)

        predictions = self._make_multiple_predictions(
            n=n,
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            num_samples=num_samples,
        )
        return self.ensemble(predictions, series=series)

    @abstractmethod
    def ensemble(
        self,
        predictions: Union[TimeSeries, Sequence[TimeSeries]],
        series: Optional[Sequence[TimeSeries]] = None,
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

    @property
    def min_train_series_length(self) -> int:
        return max(model.min_train_series_length for model in self.models)

    @property
    def min_train_samples(self) -> int:
        return max(model.min_train_samples for model in self.models)

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
            for model in self.models:
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

    def _is_probabilistic(self) -> bool:
        return all([model._is_probabilistic() for model in self.models])

    @property
    def supports_past_covariates(self) -> bool:
        return any([model.supports_past_covariates for model in self.models])

    @property
    def supports_future_covariates(self) -> bool:
        return any([model.supports_future_covariates for model in self.models])

    def _full_past_covariates_support(self) -> bool:
        return all([model.supports_past_covariates for model in self.models])

    def _full_future_covariates_support(self) -> bool:
        return all([model.supports_future_covariates for model in self.models])

    def _verify_past_future_covariates(self, past_covariates, future_covariates):
        """
        Verify that any non-None covariates comply with the model type.
        """
        raise_if(
            past_covariates is not None and not self.supports_past_covariates,
            "Some past_covariates have been provided to a EnsembleModel containing no models "
            "supporting such covariates.",
            logger,
        )
        raise_if(
            future_covariates is not None and not self.supports_future_covariates,
            "Some future_covariates have been provided to a Ensemble model containing no models "
            "supporting such covariates.",
            logger,
        )
